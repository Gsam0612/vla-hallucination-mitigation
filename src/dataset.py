"""
PyTorch Dataset with proper label masking for VLA training.

Key fix: Only train on ASSISTANT response tokens.
The USER prompt tokens are masked with -100 so they don't
contribute to the loss. This prevents the model from "learning"
to predict the question — it only learns to generate answers.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional


class VLADataset(Dataset):
    """Dataset for VLA hallucination mitigation training.

    Features:
    - Proper label masking (only ASSISTANT tokens count)
    - Text-only: no image tokens (synthetic data, no real images)
    - Supports both SFT and GRPO data formats
    """

    ASSISTANT_TOKEN = "ASSISTANT:"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        processor,
        max_length: int = 1024,
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        conversation = item['conversation']

        # ── Tokenize full conversation (text-only, no images) ───────
        inputs = self.processor.tokenizer(
            conversation,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        # ── Mask labels: only train on ASSISTANT response ───────────
        labels = input_ids.clone()

        # Find where "ASSISTANT:" appears in the token sequence
        # Tokenize just the assistant marker to find its token ids
        assistant_ids = self.processor.tokenizer.encode(
            self.ASSISTANT_TOKEN, add_special_tokens=False
        )
        assistant_len = len(assistant_ids)

        # Search for the ASSISTANT marker in input_ids
        mask_end = 0  # Default: mask nothing (fallback)
        ids_list = input_ids.tolist()
        for i in range(len(ids_list) - assistant_len + 1):
            if ids_list[i:i + assistant_len] == assistant_ids:
                mask_end = i + assistant_len
                break

        # Mask everything before and including "ASSISTANT:"
        if mask_end > 0:
            labels[:mask_end] = -100

        # Also mask padding tokens
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collator that stacks batch tensors (text-only, no pixel_values)."""
    return {
        'input_ids':      torch.stack([x['input_ids'] for x in batch]),
        'attention_mask':  torch.stack([x['attention_mask'] for x in batch]),
        'labels':          torch.stack([x['labels'] for x in batch]),
    }
