"""
Configuration classes for VLA Hallucination Mitigation Pipeline.

Maps to dissertation sections:
- RewardConfig -> Section 4.5 (GRPO Training Procedure)
- MultiViewConfig -> Section 4.4.2 (Multi-View Consistency)
- GRPOConfig -> Section 4.5 (GRPO Training Procedure)
- TrainingConfig -> Section 4 (Methodology)
- EvalConfig -> Section 4.6 (Evaluation Protocol)
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class RewardConfig:
    """Hallucination-aware reward weights (Section 4.5).

    Penalty/bonus structure:
    | Type                  | Weight | Description                    |
    |-----------------------|--------|--------------------------------|
    | Object Existence      | -1.0   | Claiming non-existent objects  |
    | Misidentification     | -0.8   | Wrong object classification    |
    | Attribute Error       | -0.5   | Wrong color/count/size         |
    | Spatial Relation      | -0.6   | Wrong spatial relations        |
    | Missing Object        | -0.3   | Not mentioning present objects |
    | Correct Object        | +0.2   | Correctly identifying objects  |
    | Correct Count         | +0.3   | Correct object counts          |
    | Grounded Response     | +0.5   | Using detection grounding      |
    | Safety Awareness      | +0.4   | Correctly identifying hazards  |
    | Multi-View Consistency| +0.3   | Multi-view verified objects    |
    | CoT Quality           | +0.2   | Proper reasoning chain         |
    """
    # Penalties
    object_existence_penalty: float = -1.0
    misidentification_penalty: float = -0.8
    attribute_error_penalty: float = -0.5
    spatial_relation_penalty: float = -0.6
    missing_object_penalty: float = -0.3

    # Bonuses
    correct_object_bonus: float = 0.2
    correct_count_bonus: float = 0.3
    correct_relation_bonus: float = 0.2
    grounded_response_bonus: float = 0.5
    safety_awareness_bonus: float = 0.4
    multi_view_consistency_bonus: float = 0.3
    cot_quality_bonus: float = 0.2

    # Bounds
    min_reward: float = -2.0
    max_reward: float = 2.0
    base_reward: float = 1.0


@dataclass
class MultiViewConfig:
    """Multi-view consistency configuration (Section 4.4.2).

    - 4 camera angles: front (0°), left (-45°), right (+45°), overhead (-90°)
    - Aggregation: intersection, majority, or union
    - Threshold: 50% view agreement for object validation
    """
    num_views: int = 4
    view_angles: Dict[str, int] = field(default_factory=lambda: {
        'front': 0,
        'left': -45,
        'right': 45,
        'overhead': -90,
    })
    consistency_threshold: float = 0.5
    aggregation: str = 'majority'


@dataclass
class GRPOConfig:
    """GRPO training configuration (Section 4.5).

    Group Relative Policy Optimization:
    1. Generate K candidate responses per prompt
    2. Score each with hallucination-aware reward
    3. Compute group-relative advantages (normalize within group)
    4. Policy gradient update (no critic network needed)
    """
    num_candidates: int = 4
    temperature: float = 0.8
    max_new_tokens: int = 300
    kl_coeff: float = 0.05
    clip_range: float = 0.2
    grpo_epochs: int = 3
    grpo_mini_batch_size: int = 2
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4


@dataclass
class TrainingConfig:
    """Overall training configuration."""
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    output_dir: str = "./outputs"
    num_training_samples: int = 5000
    num_eval_samples: int = 500
    seed: int = 42

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # SFT phase
    sft_epochs: int = 2
    sft_batch_size: int = 2
    sft_learning_rate: float = 2e-5
    sft_warmup_steps: int = 100
    max_length: int = 1024

    # Sub-configs
    reward: RewardConfig = field(default_factory=RewardConfig)
    multi_view: MultiViewConfig = field(default_factory=MultiViewConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)


@dataclass
class EvalConfig:
    """Evaluation configuration (Section 4.6)."""
    num_eval_scenes: int = 500
    complexities: List[str] = field(default_factory=lambda: [
        'simple', 'cluttered', 'hazard',
    ])
    rooms: List[str] = field(default_factory=lambda: [
        'Kitchen', 'LivingRoom', 'Bedroom', 'Bathroom',
    ])
    ablation_configs: List[str] = field(default_factory=lambda: [
        'baseline',
        'detection_only',
        'detection_multiview',
        'detection_cot',
        'full_no_grpo',
        'full_with_grpo',
    ])
