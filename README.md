# VLA Hallucination Mitigation

**Mitigating Hallucination and Perception Errors in Vision-Language Agents**

MSc Dissertation — Vedhagiri Alagesan, Heriot-Watt University  
Supervisor: Dr. Oliver Lemon

---

## Overview

Vision-Language Agents (VLAs) suffer from **hallucination** — generating descriptions of objects, attributes, or spatial relations that don't exist in the scene. This project implements a comprehensive pipeline to reduce hallucination in VLAs through five integrated techniques:

1. **Detector Grounding** — Simulated YOLOv8 object detection anchors the VLM to actually observed objects
2. **Multi-View Consistency** — 4 camera viewpoints (front, left, right, overhead) filter false detections
3. **Chain-of-Thought (CoT) Reasoning** — Structured `[Observation] → [Multi-View Check] → [Reasoning] → [Verification] → [Answer]` format
4. **GRPO Training** — Group Relative Policy Optimization with hallucination-aware rewards teaches the model to learn from its own mistakes
5. **Self-Verification** — Post-generation checks against grounded evidence with re-generation on low confidence

## Architecture

```
                    ┌──────────────┐
                    │  Input Image │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  YOLOv8      │
                    │  Detector    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌─────────────────┐
                    │  Multi-View  │◄────│ 4 Camera Views   │
                    │  Consistency │     │ (0°,-45°,45°,-90°)│
                    └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  LLaVA-1.5   │
                    │  + LoRA      │
                    │  (CoT format)│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Self-       │
                    │  Verification│
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Final Answer│
                    └──────────────┘

Training: SFT (Phase 1) → GRPO (Phase 2)
```

## Project Structure

```
vla-hallucination-mitigation/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py           # Package info
│   ├── config.py             # All configuration dataclasses
│   ├── objects.py            # AI2-THOR object definitions (38 objects)
│   ├── reward.py             # Hallucination-aware reward function
│   ├── multi_view.py         # Multi-view consistency checker
│   ├── scene_generator.py    # AI2-THOR scene generation
│   ├── data_generator.py     # Training data with CoT format
│   ├── dataset.py            # PyTorch Dataset with label masking
│   ├── grpo_trainer.py       # GRPO training loop
│   ├── evaluation.py         # Ablation study & baseline comparison
│   └── inference.py          # Full inference pipeline
├── scripts/
│   ├── train.py              # Training entry point
│   ├── evaluate.py           # Evaluation entry point
│   └── demo.py               # Gradio interactive demo
└── notebooks/
    └── VLA_Hallucination_Mitigation.ipynb  # Main Colab notebook
```

## Quick Start

### Google Colab (Recommended)

1. Open [the notebook](notebooks/VLA_Hallucination_Mitigation.ipynb) in Google Colab
2. Select **Runtime → Change runtime type → T4 GPU**
3. Run all cells

### Local Setup

```bash
git clone https://github.com/Gsam0612/vla-hallucination-mitigation.git
cd vla-hallucination-mitigation
pip install -r requirements.txt

# Train (SFT + GRPO)
python scripts/train.py --output_dir ./outputs --num_samples 5000

# Evaluate
python scripts/evaluate.py --model_dir ./outputs/final_model --num_scenes 100

# Demo
python scripts/demo.py --model_dir ./outputs/final_model
```

## Training Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)
- **Model**: LLaVA-1.5-7B with 4-bit NF4 quantization
- **LoRA**: r=16, alpha=32, targeting attention + MLP projections
- **Data**: 5000 AI2-THOR scenes with CoT-formatted answers
- **Label masking**: Loss computed only on ASSISTANT tokens
- **Optimizer**: Paged AdamW 8-bit

### Phase 2: GRPO (Group Relative Policy Optimization)
- **K=4** candidate responses per prompt
- **Hallucination-aware reward function** with typed penalties:
  - Object existence: -1.0
  - Misidentification: -0.8
  - Attribute error: -0.5
  - Spatial relation: -0.6
- **Group-relative advantages**: `A_i = (R_i - mean(R)) / std(R)`
- **No critic network** required (advantage computed from group statistics)

### CoT Format
```
[Observation] Detected objects: coffee machine (conf: 0.95), apple (conf: 0.91)
[Multi-View Check] coffee machine: 4/4 views ✓ | apple: 3/4 views ✓
[Reasoning] I observe a black coffee machine on the counter. Red apples are visible nearby.
[Verification] All mentioned objects confirmed by detector. Counts consistent across views.
[Answer] I can see a black coffee machine and red apples on the kitchen counter.
```

## Evaluation

### Ablation Study (Dissertation Table 3)

| Configuration | Components | Hallucination Rate ↓ | Recall ↑ | Precision ↑ |
|---|---|---|---|---|
| baseline | Raw LLaVA | High | Moderate | Low |
| detection_only | + YOLOv8 grounding | Reduced | Improved | Improved |
| detection_multiview | + Multi-view (4 views) | Lower | Improved | Higher |
| detection_cot | + CoT reasoning | Lower | Improved | Higher |
| full_no_grpo | All components, SFT only | Low | High | High |
| **full_with_grpo** | **All + GRPO** | **Lowest** | **Highest** | **Highest** |

*Actual numbers are generated during evaluation. Run the notebook to see results.*

### Comparison with Published Baselines
- **POPE** (Li et al., 2023): Polling-based Object Probing Evaluation
- **ICD** (Leng et al., 2024): Induced Contrastive Decoding
- **MIHBench** (Chen et al., 2025): Multi-modal Image Hallucination Benchmark

## Key Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Base model | LLaVA-1.5-7B | Strong vision-language capability, fits Colab T4 |
| Quantization | 4-bit NF4 | ~5GB VRAM, minimal quality loss |
| Fine-tuning | LoRA (r=16) | Memory efficient, trains only 0.5% of parameters |
| RL algorithm | GRPO | No critic network, group-relative baselines |
| Multi-view | 4 angles | Balances coverage with compute cost |
| Environment | AI2-THOR | Rich indoor scenes with ground truth |

## Limitations

- **Sim-to-real gap**: Trained on AI2-THOR simulations; visual grounding may need domain adaptation for real-world deployment
- **Detector simulation**: Uses simulated YOLOv8 output; real deployment should use actual detection model
- **CoT transfer**: Structured reasoning patterns transfer well across domains, but specific object vocabulary is AI2-THOR-specific

## Citation

```bibtex
@mastersthesis{alagesan2025vla,
  title={Mitigating Hallucination and Perception Errors in Vision-Language Agents},
  author={Alagesan, Vedhagiri},
  school={Heriot-Watt University},
  year={2025},
  supervisor={Lemon, Oliver}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
