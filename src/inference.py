"""
Inference pipeline for VLA Hallucination Mitigation.

Combines all components for production inference:
1. Detector grounding (YOLOv8 simulated)
2. Multi-view consistency check
3. Chain-of-Thought reasoning
4. Self-verification

When the model is uncertain, it triggers multi-view
checking across 4 camera angles before answering.
"""

import re
import torch
from typing import List, Dict, Any, Optional

from .objects import AI2THOR_OBJECTS
from .multi_view import MultiViewConsistency
from .reward import HallucinationReward
from .data_generator import SYSTEM_PROMPT
from .config import MultiViewConfig


class VLAInferencePipeline:
    """Full inference pipeline with multi-view and CoT.

    Workflow (matches dissertation Fig. 3):
      Scene → Detector → Multi-View → VLM + CoT → Self-Verify → Answer
    """

    def __init__(
        self,
        model,
        processor,
        mv_checker: Optional[MultiViewConsistency] = None,
        reward_fn: Optional[HallucinationReward] = None,
        confidence_threshold: float = 0.7,
    ):
        self.model = model
        self.processor = processor
        self.mv_checker = mv_checker or MultiViewConsistency()
        self.reward_fn = reward_fn or HallucinationReward()
        self.confidence_threshold = confidence_threshold
        self.device = next(model.parameters()).device

    # ── Detector simulation ─────────────────────────────────────────

    @staticmethod
    def simulate_detector(
        gt_objects: List[Dict[str, Any]],
    ) -> str:
        """Simulate YOLOv8 detector output.

        In production, this would call ultralytics YOLOv8.
        Here we use ground-truth + noise for training/testing.
        """
        import random

        detections = []
        for obj in gt_objects:
            name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', obj['name']).lower()
            conf = obj.get('confidence', round(random.uniform(0.75, 0.99), 2))
            detections.append(f"{name} (conf: {conf:.2f})")
        return "Detected objects: " + ", ".join(detections)

    # ── Multi-view check ────────────────────────────────────────────

    def run_multiview_check(
        self,
        gt_objects: List[str],
    ) -> Dict[str, Any]:
        """Run multi-view consistency check.

        Returns consistent objects and per-object view scores.
        """
        views = MultiViewConsistency.simulate_views(gt_objects)
        result = self.mv_checker.check_consistency(views)
        return {
            'consistent': result['consistent'],
            'inconsistent': result['inconsistent'],
            'scores': result['scores'],
            'views': views,
        }

    # ── Self-verification ───────────────────────────────────────────

    def self_verify(
        self,
        response: str,
        detected_objects: List[str],
        consistent_objects: List[str],
    ) -> Dict[str, Any]:
        """Verify model response against detections and consistency.

        Checks:
        1. Are mentioned objects in the detector output?
        2. Are mentioned objects multi-view consistent?
        3. Does the response express appropriate uncertainty?
        """
        mentioned = self.reward_fn.extract_objects(response)
        mentioned_lower = {o.lower() for o in mentioned}
        detected_lower = {o.lower() for o in detected_objects}
        consistent_lower = {o.lower() for o in consistent_objects}

        ungrounded = mentioned_lower - detected_lower
        inconsistent = mentioned_lower - consistent_lower

        return {
            'mentioned': list(mentioned),
            'grounded': list(mentioned_lower & detected_lower),
            'ungrounded': list(ungrounded),
            'consistent': list(mentioned_lower & consistent_lower),
            'inconsistent_mentions': list(inconsistent),
            'is_reliable': len(ungrounded) == 0 and len(inconsistent) == 0,
            'confidence': 1.0 - (len(ungrounded) + len(inconsistent)) / max(len(mentioned), 1),
        }

    # ── Main inference ──────────────────────────────────────────────

    def infer(
        self,
        question: str,
        scene_objects: Optional[List[Dict[str, Any]]] = None,
        use_multiview: bool = True,
        use_cot: bool = True,
    ) -> Dict[str, Any]:
        """Run the full inference pipeline.

        Args:
            question:       User question about the scene.
            scene_objects:  List of objects in the scene (ground truth or detected).
            use_multiview:  Whether to perform multi-view consistency check.
            use_cot:        Whether to use CoT prompting.

        Returns:
            Dict with answer, reasoning trace, verification results.
        """
        if scene_objects is None:
            scene_objects = []

        # Step 1: Detector output
        detector_prompt = self.simulate_detector(scene_objects)
        gt_names = [o['name'] for o in scene_objects]

        # Step 2: Multi-view consistency
        mv_result = None
        consistent_objects = gt_names  # Default: trust all
        if use_multiview and gt_names:
            mv_result = self.run_multiview_check(gt_names)
            consistent_objects = mv_result['consistent']

        # Step 3: Generate response with CoT
        if use_cot:
            prompt = (
                f"USER: "
                f"{SYSTEM_PROMPT}\n"
                f"{detector_prompt}\n"
                f"Question: {question}\n"
                f"ASSISTANT:"
            )
        else:
            prompt = f"USER: {question}\nASSISTANT:"

        inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.pad_token_id or
                             self.processor.tokenizer.eos_token_id,
            )

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        # Step 4: Self-verification
        verification = self.self_verify(response, gt_names, consistent_objects)

        # Step 5: If unreliable, re-generate with stricter prompt
        if not verification['is_reliable'] and use_multiview:
            strict_prompt = (
                f"USER: "
                f"{SYSTEM_PROMPT}\n"
                f"{detector_prompt}\n"
                f"IMPORTANT: Only mention these verified objects: {', '.join(consistent_objects)}\n"
                f"Question: {question}\n"
                f"ASSISTANT:"
            )

            inputs = self.processor.tokenizer(
                strict_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,   # Greedy for reliability
                    pad_token_id=self.processor.tokenizer.pad_token_id or
                                 self.processor.tokenizer.eos_token_id,
                )

            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()

            verification = self.self_verify(response, gt_names, consistent_objects)
            verification['was_regenerated'] = True
        else:
            verification['was_regenerated'] = False

        # Extract final answer from CoT
        final_answer = response
        if '[Answer]' in response:
            final_answer = response.split('[Answer]')[-1].strip()

        return {
            'question': question,
            'full_response': response,
            'final_answer': final_answer,
            'detector_output': detector_prompt,
            'multi_view': mv_result,
            'verification': verification,
            'consistent_objects': consistent_objects,
        }
