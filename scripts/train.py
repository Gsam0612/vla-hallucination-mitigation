"""
Training entry point.

Two-phase training:
  Phase 1 (SFT):   Supervised fine-tuning on CoT-formatted data
  Phase 2 (GRPO):  RL fine-tuning with hallucination-aware rewards

Usage (Colab):
  %cd vla-hallucination-mitigation
  !python scripts/train.py --output_dir ./outputs --num_samples 5000
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainingConfig, RewardConfig, GRPOConfig, MultiViewConfig
from src.reward import HallucinationReward
from src.multi_view import MultiViewConsistency
from src.data_generator import generate_dataset
from src.dataset import VLADataset, collate_fn
from src.grpo_trainer import GRPOTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_processor(config: TrainingConfig):
    """Load LLaVA model with 4-bit quantization + LoRA."""
    from transformers import (
        AutoProcessor,
        LlavaForConditionalGeneration,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    print(f"Loading {config.model_name}...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(config.model_name)

    print(f"Model loaded! GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Apply LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


def phase1_sft(model, processor, training_data, config: TrainingConfig):
    """Phase 1: Supervised fine-tuning on CoT-formatted data."""
    from transformers import TrainingArguments, Trainer

    print("\n" + "=" * 60)
    print("PHASE 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)

    dataset = VLADataset(training_data, processor, max_length=config.max_length)
    print(f"Dataset: {len(dataset)} samples")

    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "sft"),
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=8,
        learning_rate=config.sft_learning_rate,
        weight_decay=0.01,
        warmup_steps=config.sft_warmup_steps,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
    )

    print("Starting SFT training...")
    trainer.train()
    print("SFT training complete!")

    return model


def phase2_grpo(model, processor, training_data, config: TrainingConfig):
    """Phase 2: GRPO fine-tuning with hallucination-aware rewards."""
    print("\n" + "=" * 60)
    print("PHASE 2: GRPO Training")
    print("=" * 60)

    reward_fn = HallucinationReward(config.reward)

    grpo_trainer = GRPOTrainer(
        model=model,
        processor=processor,
        reward_fn=reward_fn,
        config=config.grpo,
        training_config=config,
    )

    # Use a subset for GRPO (it's slow because of generation)
    grpo_subset_size = min(500, len(training_data))
    grpo_data = random.sample(training_data, grpo_subset_size)
    print(f"GRPO training samples: {grpo_subset_size}")

    metrics = grpo_trainer.train_grpo(
        training_data=grpo_data,
        num_epochs=config.grpo.grpo_epochs,
        save_dir=config.output_dir,
    )

    # Save training log
    grpo_trainer.save_training_log(
        os.path.join(config.output_dir, "grpo_training_log.json"))

    return model, metrics


def save_model(model, processor, config: TrainingConfig, training_data):
    """Save model, processor, and config."""
    output_dir = os.path.join(config.output_dir, "final_model")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # Save training config
    config_dict = {
        'base_model': config.model_name,
        'lora_r': config.lora_r,
        'lora_alpha': config.lora_alpha,
        'training_samples': len(training_data),
        'sft_epochs': config.sft_epochs,
        'grpo_epochs': config.grpo.grpo_epochs,
        'grpo_candidates_k': config.grpo.num_candidates,
        'hallucination_types': [
            'object_existence', 'misidentification',
            'attribute', 'spatial_relation',
        ],
        'reward_config': {
            'object_existence_penalty': config.reward.object_existence_penalty,
            'misidentification_penalty': config.reward.misidentification_penalty,
            'attribute_error_penalty': config.reward.attribute_error_penalty,
            'spatial_relation_penalty': config.reward.spatial_relation_penalty,
            'correct_object_bonus': config.reward.correct_object_bonus,
            'multi_view_consistency_bonus': config.reward.multi_view_consistency_bonus,
            'cot_quality_bonus': config.reward.cot_quality_bonus,
        },
        'pipeline_components': [
            'yolov8_detector_grounding',
            'multi_view_consistency_4_views',
            'chain_of_thought_reasoning',
            'self_verification',
            'grpo_rl_training',
        ],
    }

    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nModel saved: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="VLA Hallucination Mitigation Training")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=5000, help="Training samples")
    parser.add_argument("--sft_epochs", type=int, default=2, help="SFT epochs")
    parser.add_argument("--grpo_epochs", type=int, default=3, help="GRPO epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT phase")
    parser.add_argument("--skip_grpo", action="store_true", help="Skip GRPO phase")
    args = parser.parse_args()

    # Config
    config = TrainingConfig(
        output_dir=args.output_dir,
        num_training_samples=args.num_samples,
        seed=args.seed,
        sft_epochs=args.sft_epochs,
    )
    config.grpo.grpo_epochs = args.grpo_epochs

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Step 1: Generate data
    print("Generating training data with CoT format...")
    mv_checker = MultiViewConsistency(config.multi_view)
    training_data = generate_dataset(
        n=config.num_training_samples,
        mv_checker=mv_checker,
    )
    print(f"Generated {len(training_data)} samples")

    # Step 2: Load model
    model, processor = load_model_and_processor(config)

    # Step 3: Phase 1 - SFT
    if not args.skip_sft:
        model = phase1_sft(model, processor, training_data, config)

    # Step 4: Phase 2 - GRPO
    if not args.skip_grpo:
        model, grpo_metrics = phase2_grpo(model, processor, training_data, config)

    # Step 5: Save
    save_model(model, processor, config, training_data)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
