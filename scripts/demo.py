"""
Gradio demo for VLA Hallucination Mitigation.

Interactive demo showing:
- CoT reasoning trace
- Multi-view consistency checks
- Self-verification
- Detector grounding

Usage:
  python scripts/demo.py --model_dir ./outputs/final_model
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.objects import AI2THOR_OBJECTS
from src.scene_generator import generate_scene
from src.inference import VLAInferencePipeline
from src.multi_view import MultiViewConsistency
from src.reward import HallucinationReward


def create_demo(pipeline: VLAInferencePipeline):
    """Create Gradio interface."""
    import gradio as gr

    def process_query(question, room_type, complexity):
        """Handle user query."""
        # Generate scene
        scene = generate_scene(complexity, room_type)

        # Run inference
        result = pipeline.infer(
            question=question,
            scene_objects=scene.objects,
            use_multiview=True,
            use_cot=True,
        )

        # Format output
        answer = result['final_answer']
        full_cot = result['full_response']
        detector = result['detector_output']
        verification = result['verification']

        # Multi-view info
        mv_info = ""
        if result['multi_view']:
            mv = result['multi_view']
            mv_info = f"**Consistent objects**: {', '.join(mv['consistent'])}\n"
            mv_info += f"**Inconsistent objects**: {', '.join(mv['inconsistent']) or 'None'}\n"
            mv_info += f"**View scores**: {json.dumps(mv['scores'], indent=2)}"

        # Verification info
        verify_info = (
            f"**Grounded**: {', '.join(verification['grounded'])}\n"
            f"**Ungrounded**: {', '.join(verification['ungrounded']) or 'None'}\n"
            f"**Reliable**: {'Yes' if verification['is_reliable'] else 'No'}\n"
            f"**Confidence**: {verification['confidence']:.2%}\n"
            f"**Regenerated**: {'Yes' if verification.get('was_regenerated') else 'No'}"
        )

        # Scene info
        scene_info = (
            f"**Room**: {scene.room_type} ({scene.complexity})\n"
            f"**Objects**: {', '.join(scene.get_object_list())}\n"
            f"**Hazards**: {', '.join(scene.hazards) or 'None'}"
        )

        return answer, full_cot, detector, mv_info, verify_info, scene_info

    # Build interface
    with gr.Blocks(
        title="VLA Hallucination Mitigation Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# VLA Hallucination Mitigation\n"
            "**MSc Dissertation** — Vedhagiri Alagesan, Heriot-Watt University\n\n"
            "This demo shows the full inference pipeline with detector grounding, "
            "multi-view consistency, chain-of-thought reasoning, and self-verification."
        )

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(
                    label="Question",
                    value="What do you see in front of you?",
                    placeholder="Ask about the scene...",
                )
                room = gr.Dropdown(
                    choices=["Kitchen", "LivingRoom", "Bedroom", "Bathroom"],
                    value="Kitchen",
                    label="Room Type",
                )
                complexity = gr.Dropdown(
                    choices=["simple", "cluttered", "hazard"],
                    value="simple",
                    label="Scene Complexity",
                )
                btn = gr.Button("Ask", variant="primary")

            with gr.Column(scale=2):
                answer_box = gr.Textbox(label="Final Answer", lines=3)
                scene_box = gr.Markdown(label="Scene Info")

        with gr.Accordion("Chain-of-Thought Reasoning", open=False):
            cot_box = gr.Textbox(label="Full CoT Response", lines=10)

        with gr.Row():
            with gr.Column():
                detector_box = gr.Textbox(label="Detector Output", lines=2)
            with gr.Column():
                mv_box = gr.Markdown(label="Multi-View Consistency")

        with gr.Accordion("Self-Verification", open=False):
            verify_box = gr.Markdown(label="Verification Results")

        btn.click(
            process_query,
            inputs=[question, room, complexity],
            outputs=[answer_box, cot_box, detector_box, mv_box, verify_box, scene_box],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=None, help="Model directory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.model_dir:
        # Load real model
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
        from peft import PeftModel

        config_path = os.path.join(args.model_dir, "training_config.json")
        with open(config_path) as f:
            config = json.load(f)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = LlavaForConditionalGeneration.from_pretrained(
            config['base_model'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, args.model_dir)
        processor = AutoProcessor.from_pretrained(args.model_dir)

        pipeline = VLAInferencePipeline(model=model, processor=processor)
    else:
        print("No model directory specified. Demo will run with placeholder model.")
        print("Use --model_dir to specify trained model path.")
        # Create a minimal pipeline for demonstration
        pipeline = None

    if pipeline:
        demo = create_demo(pipeline)
        demo.launch(server_port=args.port, share=args.share)
    else:
        print("Cannot start demo without a model. Train first with: python scripts/train.py")


if __name__ == "__main__":
    main()
