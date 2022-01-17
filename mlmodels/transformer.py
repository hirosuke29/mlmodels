from typing import Optional
import math

import torch
from torch import nn
from torch.nn import functional


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model: int) -> None:
        """
        Args:
            d_model: embedded vector length
        """
        super().__init__()
        self.d_k: int = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor [batch_size, sequence_length, d_model]
        Returns:
            out: torch.Tensor [batch_size, sequence_length, d_model]
        """
        query, key, value = self.w_q(x), self.w_k(x), self.w_v(x)
        query /= math.sqrt(self.d_k)
        attn_weight = torch.bmm(query, key.transpose(-2, -1))
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask, -1e9)
        attn_weight = functional.softmax(attn_weight, dim=-1)
        attn = torch.matmul(attn_weight, value)
        out = self.out(attn)
        return out
