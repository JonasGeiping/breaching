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
        shift_logits = outputs[:, :-1, :].contiguous()
        if labels is None:
            shift_labels = outputs[:, 1:].contiguous()
        elif labels.dtype == torch.long:
            shift_labels = labels[:, 1:].contiguous().view(-1)
        else:
            shift_labels = labels[:, 1:, :].contiguous().view(-1, labels.shape[-1])
        # Flatten the tokens
        return self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels)


class MLMLoss(torch.nn.Module):
    def __init__(self, *args, vocab_size=50_000, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor):
        """Make sure to handle both soft labels and numeric targets."""
        # Flatten the tokens
        if labels.dtype == torch.long:
            labels = labels.view(-1)
        else:
            labels = labels.view(-1, self.vocab_size)
        return self.loss_fct(outputs.view(-1, self.vocab_size), labels)


class MostlyCausalLoss(torch.nn.Module):
    """Sanity check loss for last-token inconsistencies...
    Do not use this for anything resembling actual language model training."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """If no labels are given, then the same sequence is re-used."""
        # Based on https://github.com/huggingface/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py#L1069
        # Shift so that tokens < n predict n
        shift_logits = outputs[:, :, :].contiguous()
        if labels is None:
            shift_labels = outputs[:, 1:].contiguous()
        elif labels.dtype == torch.long:
            shift_labels = torch.cat([labels[:, 1:], labels[:, -1:]], dim=1).contiguous().view(-1)
        else:
            shift_labels = labels[:, 1:, :].contiguous().view(-1, labels.shape[-1])

        # Flatten the tokens
        return self.loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels)
