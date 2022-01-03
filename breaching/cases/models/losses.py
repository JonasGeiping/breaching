"""Implement some additional loss functions."""

import torch
from typing import Optional


class CausalLoss(torch.nn.Module):
    """Cross Entropy variant for next-token prediction in causal language modeling."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """If no labels are given, then the same sequence is re-used."""
        # Based on https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1069
        # Shift so that tokens < n predict n
        shift_logits = outputs[..., :-1, :].contiguous()
        if labels is None:
            shift_labels = outputs[..., 1:].contiguous()
        elif labels.dtype == torch.long:
            shift_labels = labels[..., 1:].view(-1).contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        else:
            shift_labels = labels[..., 1:, :]
        # Flatten the tokens
        return self.loss_fct(shift_logits, shift_labels)


class MLMLoss(torch.nn.Module):
    def __init__(self, *args, vocab_size=50_000, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Not sure if this needs to be its own function."""
        # Flatten the tokens
        return self.loss_fct(outputs.view(-1, self.vocab_size), labels.view(-1))
