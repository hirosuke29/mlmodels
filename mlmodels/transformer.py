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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int) -> None:
        """
        Args:
            d_model: embedded vector length
        """
        super().__init__()
        assert (
            d_model % num_head == 0
        ), f"d_model({d_model}) must be dividible by num_head({num_head})"
        self.d_k: int = d_model
        self.num_head: int = num_head
        self.dim_per_head: int = d_model // num_head
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
        batch_size: int = x.size()[0]
        query: torch.Tensor = self.w_q(x)
        key: torch.Tensor = self.w_k(x)
        value: torch.Tensor = self.w_v(x)
        # [batch_size, sequence_length, d_model]
        # -> [batch_size, sequence_length, num_head, dim_per_head]
        # -> [batch_size, num_head, sequence_length, dim_per_head]
        query = query.view(batch_size, -1, self.num_head, self.dim_per_head).transpose(
            1, 2
        )
        key = key.view(batch_size, -1, self.num_head, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_head, self.dim_per_head).transpose(
            1, 2
        )
        query /= math.sqrt(self.d_k)
        attn_weight = torch.matmul(query, key.transpose(2, 3))
        if mask is not None:
            # TODO make it works
            attn_weight = attn_weight.masked_fill(mask, -1e9)
        attn_weight = functional.softmax(attn_weight, dim=-1)
        attn = torch.matmul(attn_weight, value)
        # [batch_size, num_head, sequence_length, dim_per_head]
        # -> [batch_size, sequence_length, num_head, dim_per_head]
        # -> [batch_size, sequence_length, d_model]
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_k)
        out = self.out(attn)
        return out
