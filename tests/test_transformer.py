import torch
from torch import nn

from mlmodels import transformer

def test_scaled_dot_product_attention_shape() -> None:
    attn: nn.Module = transformer.ScaledDotProductAttention(d_model=200)
    size: torch.Size = torch.Size((10, 100, 200))
    input: torch.Tensor = torch.rand(size)
    out: torch.Tensor = attn(input)
    assert size == out.size()
