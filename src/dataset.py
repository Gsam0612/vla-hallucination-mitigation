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
from PIL import Image


class VLADataset(Dataset):
    """Dataset for VLA hallucination mitigation training.

    Features:
    - Proper label masking (only ASSISTANT tokens count)
    - Handles image token alignment
    - Supports both SFT and GRPO data formats
    """

    ASSISTANT_TOKEN = "ASSISTANT:"

    def __init__(
        self,
        data: List[Dict[str, Any]],
        processor,
        max_length: int = 1024,
        image_size: tuple = (640, 480),
    ):
        self.data = data
        self.processor = processor
        self.max_length = max_length
        self.placeholder = Image.new('RGB', image_size, color='gray')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        conversation = item['conversation']

        # ── Tokenize full conversation ──────────────────────────────
        inputs = self.processor(
            text=conversation,
            images=self.placeholder,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
        )

        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)

        # Truncate if needed (preserve image tokens at start)
        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

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
            'pixel_values': pixel_values,
            'labels': labels,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collator that stacks batch tensors."""
    return {
        'input_ids':      torch.stack([x['input_ids'] for x in batch]),
        'attention_mask':  torch.stack([x['attention_mask'] for x in batch]),
        'pixel_values':    torch.stack([x['pixel_values'] for x in batch]),
        'labels':          torch.stack([x['labels'] for x in batch]),
    }
