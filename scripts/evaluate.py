"""
Evaluation entry point.

Runs ablation study + comparison with existing methods.

Usage:
  python scripts/evaluate.py --model_dir ./outputs/final_model --num_scenes 100
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainingConfig, EvalConfig
from src.reward import HallucinationReward
from src.multi_view import MultiViewConsistency
from src.evaluation import (
    VLAEvaluator, format_ablation_table,
    format_comparison_table, save_evaluation_report,
)


def load_trained_model(model_dir: str, base_model: str = "llava-hf/llava-1.5-7b-hf"):
    """Load the GRPO-trained model."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
    from peft import PeftModel

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {base_model}")
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapters: {model_dir}")
    model = PeftModel.from_pretrained(model, model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    return model, processor


def main():
    parser = argparse.ArgumentParser(description="VLA Evaluation")
    parser.add_argument("--model_dir", required=True, help="Path to trained model")
    parser.add_argument("--num_scenes", type=int, default=100, help="Scenes per config")
    parser.add_argument("--output_dir", default="./outputs/eval", help="Eval output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    config_path = os.path.join(args.model_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            train_config = json.load(f)
        base_model = train_config.get('base_model', "llava-hf/llava-1.5-7b-hf")
    else:
        base_model = "llava-hf/llava-1.5-7b-hf"

    model, processor = load_trained_model(args.model_dir, base_model)

    # Setup evaluator
    evaluator = VLAEvaluator(
        model=model,
        processor=processor,
        reward_fn=HallucinationReward(),
        mv_checker=MultiViewConsistency(),
    )

    # Run ablation study
    print("\n" + "=" * 60)
    print("ABLATION STUDY (Dissertation Table 3)")
    print("=" * 60)

    ablation = evaluator.run_ablation(num_scenes_per_config=args.num_scenes)
    print("\n" + format_ablation_table(ablation))

    # Comparison with baselines
    print("\n" + "=" * 60)
    print("COMPARISON WITH EXISTING METHODS")
    print("=" * 60)

    our_full = ablation.get('full_with_grpo', ablation.get('full_no_grpo', {}))
    comparison = VLAEvaluator.compare_with_baselines(our_full)
    print("\n" + format_comparison_table(comparison))

    # Save report
    save_evaluation_report(
        ablation, comparison,
        os.path.join(args.output_dir, "evaluation_report.json"),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    if 'baseline' in ablation and 'full_with_grpo' in ablation:
        baseline_hr = ablation['baseline'].get('mean_hallucination_rate', 0)
        full_hr = ablation['full_with_grpo'].get('mean_hallucination_rate', 0)
        improvement = baseline_hr - full_hr
        print(f"\nHallucination Rate Reduction: {baseline_hr:.2%} -> {full_hr:.2%} ({improvement:.2%} improvement)")
        print(f"Baseline Recall: {ablation['baseline'].get('mean_recall', 0):.2%}")
        print(f"Full System Recall: {ablation['full_with_grpo'].get('mean_recall', 0):.2%}")


if __name__ == "__main__":
    main()
