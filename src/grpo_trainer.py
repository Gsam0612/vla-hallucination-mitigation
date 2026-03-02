"""
Group Relative Policy Optimization (GRPO) Trainer.

Dissertation Section 4.5 — GRPO Training Procedure:
1. Sampling:  For each prompt, generate K candidate responses
2. Scoring:   Each candidate scored with hallucination-aware reward
3. Grouping:  Compute group-relative advantages (normalize within group)
4. Update:    Policy gradient update — no critic network needed

GRPO simplifies PPO by using group-based outcome advantages:
  advantage_i = (reward_i - mean(rewards)) / std(rewards)
  loss = -E[ advantage_i * log π(y_i | x) ]

This avoids training a separate value/critic network while
preserving the fidelity of policy optimization.

References:
  [8]  Shi, 2025 — "A vision researcher's guide to PPO & GRPO"
  [11] Xiao & Gan, 2025 — "Fast-Slow Thinking GRPO for LVLMs"
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import json
import os
import time

from .config import GRPOConfig, TrainingConfig
from .reward import HallucinationReward
from .data_generator import SYSTEM_PROMPT


class GRPOTrainer:
    """GRPO trainer for hallucination mitigation in VLAs.

    Two-phase training:
      Phase 1 (SFT):  Supervised fine-tuning on CoT-formatted data
                       to teach the model the response structure.
      Phase 2 (GRPO): RL fine-tuning using hallucination-aware rewards
                       so the model learns from its own mistakes.
    """

    def __init__(
        self,
        model,
        processor,
        reward_fn: HallucinationReward,
        config: GRPOConfig,
        training_config: TrainingConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.reward_fn = reward_fn
        self.config = config
        self.training_config = training_config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Optimizer for GRPO phase
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Logging
        self.log: List[Dict[str, Any]] = []

    # ── Generation ──────────────────────────────────────────────────

    def generate_candidates(
        self,
        prompt: str,
        k: int = 4,
    ) -> List[str]:
        """Generate K candidate responses for a prompt.

        Corresponds to GRPO Step 1 (Sampling).
        Text-only: no images used (synthetic data).
        """
        candidates = []
        for _ in range(k):
            inputs = self.processor.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.training_config.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=0.9,
                    pad_token_id=self.processor.tokenizer.pad_token_id or
                                 self.processor.tokenizer.eos_token_id,
                )

            text = self.processor.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant part
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:")[-1].strip()
            candidates.append(text)

        return candidates

    # ── Scoring ─────────────────────────────────────────────────────

    def score_candidates(
        self,
        candidates: List[str],
        ground_truth: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Score each candidate with the hallucination reward (Step 2)."""
        results = []
        for c in candidates:
            r = self.reward_fn.compute_reward(
                response=c,
                gt_objects=ground_truth['objects'],
                gt_counts=ground_truth.get('counts'),
                gt_colors=ground_truth.get('colors'),
                gt_relations=ground_truth.get('relations'),
                gt_hazards=ground_truth.get('hazards'),
                multi_view_verified=ground_truth.get('multi_view_verified'),
            )
            results.append(r)
        return results

    # ── Advantage computation ───────────────────────────────────────

    @staticmethod
    def compute_advantages(rewards: List[float]) -> torch.Tensor:
        """Compute group-relative advantages (Step 3).

        advantage_i = (reward_i - mean) / (std + eps)

        This normalizes rewards within each group so the model
        learns RELATIVE quality, not absolute reward scale.
        """
        r = torch.tensor(rewards, dtype=torch.float32)
        mean = r.mean()
        std = r.std() + 1e-8
        return (r - mean) / std

    # ── Log probability computation ─────────────────────────────────

    def compute_log_prob(
        self,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Compute log-probability of response given prompt.

        Returns scalar tensor (mean log-prob over response tokens).
        Text-only: no images used (synthetic data).
        """
        full_text = f"{prompt} {response}"

        inputs = self.processor.tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_config.max_length,
        ).to(self.device)

        # Create labels: mask the prompt, only compute loss on response
        labels = inputs['input_ids'].clone()

        # Find where response starts
        prompt_ids = self.processor.tokenizer.encode(
            prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids) + 1  # +1 for any special tokens

        # Be safe: use min of prompt_len and total length
        prompt_len = min(prompt_len, labels.shape[1] - 1)
        labels[:, :prompt_len] = -100

        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels,
        )

        # outputs.loss is the mean NLL over response tokens
        # log_prob ≈ -loss (negative of NLL)
        return -outputs.loss

    # ── GRPO Training Step ──────────────────────────────────────────

    def grpo_step(
        self,
        sample: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one GRPO update step (Steps 1-4 combined).

        Args:
            sample: Training sample with question, ground_truth, etc.

        Returns:
            Step metrics dict.
        """
        # Build prompt (without the answer)
        prompt = (
            f"USER: "
            f"{SYSTEM_PROMPT}\n"
            f"{sample['detector_prompt']}\n"
            f"Question: {sample['question']}\n"
            f"ASSISTANT:"
        )

        gt = sample['ground_truth']
        if 'multi_view_verified' in sample:
            gt['multi_view_verified'] = sample['multi_view_verified']

        # Step 1: Generate K candidates
        candidates = self.generate_candidates(
            prompt, k=self.config.num_candidates)

        # Step 2: Score candidates
        reward_results = self.score_candidates(candidates, gt)
        rewards = [r['total_reward'] for r in reward_results]

        # Step 3: Compute group advantages
        advantages = self.compute_advantages(rewards)

        # Step 4: Policy gradient update
        # L = -E[ advantage_i * log π(response_i | prompt) ]
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for i, (candidate, advantage) in enumerate(zip(candidates, advantages)):
            if abs(advantage.item()) < 1e-6:
                continue  # Skip zero-advantage candidates

            log_prob = self.compute_log_prob(prompt, candidate)
            # Negative because we maximize the GRPO objective
            loss_i = -advantage.to(self.device) * log_prob
            total_loss = total_loss + loss_i

        total_loss = total_loss / max(len(candidates), 1)

        if total_loss.requires_grad:
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            self.optimizer.step()

        # Metrics
        step_metrics = {
            'loss': total_loss.item(),
            'mean_reward': sum(rewards) / len(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'mean_hallucination_rate': sum(
                r['hallucination_rate'] for r in reward_results
            ) / len(reward_results),
            'mean_recall': sum(
                r['recall'] for r in reward_results
            ) / len(reward_results),
            'advantages_std': advantages.std().item(),
        }

        return step_metrics

    # ── Full GRPO Training Loop ─────────────────────────────────────

    def train_grpo(
        self,
        training_data: List[Dict[str, Any]],
        num_epochs: int = None,
        save_dir: str = None,
    ) -> List[Dict[str, Any]]:
        """Run the full GRPO training loop.

        Args:
            training_data: List of training samples (from data_generator).
            num_epochs:    Number of GRPO epochs.
            save_dir:      Directory to save checkpoints.

        Returns:
            Training log (list of step metrics).
        """
        num_epochs = num_epochs or self.config.grpo_epochs
        save_dir = save_dir or self.training_config.output_dir

        self.model.train()
        all_metrics = []

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"GRPO Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            epoch_metrics = {
                'loss': [], 'reward': [], 'hallucination_rate': [], 'recall': [],
            }

            progress = tqdm(
                enumerate(training_data),
                total=len(training_data),
                desc=f"Epoch {epoch + 1}",
            )

            for step, sample in progress:
                try:
                    metrics = self.grpo_step(sample)

                    epoch_metrics['loss'].append(metrics['loss'])
                    epoch_metrics['reward'].append(metrics['mean_reward'])
                    epoch_metrics['hallucination_rate'].append(
                        metrics['mean_hallucination_rate'])
                    epoch_metrics['recall'].append(metrics['mean_recall'])

                    # Update progress bar
                    if (step + 1) % 10 == 0:
                        avg_loss = sum(epoch_metrics['loss'][-10:]) / 10
                        avg_reward = sum(epoch_metrics['reward'][-10:]) / 10
                        avg_hall = sum(epoch_metrics['hallucination_rate'][-10:]) / 10
                        progress.set_postfix({
                            'loss': f"{avg_loss:.4f}",
                            'reward': f"{avg_reward:.2f}",
                            'hall_rate': f"{avg_hall:.2%}",
                        })

                    all_metrics.append(metrics)

                except Exception as e:
                    print(f"\nStep {step} error: {e}")
                    continue

            # Epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Avg Loss:              {sum(epoch_metrics['loss'])/max(len(epoch_metrics['loss']),1):.4f}")
            print(f"  Avg Reward:            {sum(epoch_metrics['reward'])/max(len(epoch_metrics['reward']),1):.2f}")
            print(f"  Avg Hallucination Rate: {sum(epoch_metrics['hallucination_rate'])/max(len(epoch_metrics['hallucination_rate']),1):.2%}")
            print(f"  Avg Recall:            {sum(epoch_metrics['recall'])/max(len(epoch_metrics['recall']),1):.2%}")

            # Save checkpoint
            if save_dir:
                ckpt_dir = os.path.join(save_dir, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                self.model.save_pretrained(ckpt_dir)
                print(f"  Checkpoint: {ckpt_dir}")

        self.log = all_metrics
        return all_metrics

    # ── Save / Export ───────────────────────────────────────────────

    def save_training_log(self, path: str):
        """Save training metrics to JSON."""
        with open(path, 'w') as f:
            json.dump(self.log, f, indent=2)
        print(f"Training log saved: {path}")
