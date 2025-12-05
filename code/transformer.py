"""
Transformer model for language modeling. Uses RoPE for positional encoding,
PeriLN for normalization, and a pluggable MoE module.
"""



import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
Dimension Key:

B: batch size
S: sequence length
M: B * S
D: embedding dimension
H: hidden dimension in expert networks
N: number of experts
K: number of chosen experts per token
V: vocabulary size
"""


class Attention(nn.Module):
    def __init__(self, D, n_heads, max_seq_len=2048):
        super().__init__()
        assert D % n_heads == 0
        self.n_heads = n_heads
        self.d_head = D // n_heads
        assert self.d_head % 2 == 0, "RoPE requires even head dimension"

        # initialize linear layers
        self.qkv = nn.Linear(D, 3 * D)
        self.qkv.weight.data.normal_(0, 1/math.sqrt(D))
        self.qkv.bias.data.zero_()
        self.out = nn.Linear(D, D)
        self.out.weight.data.normal_(0, 1/math.sqrt(D))
        self.out.bias.data.zero_()
        
        # precompute RoPE cos/sin for max sequence length
        freqs = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        pos = torch.arange(max_seq_len).unsqueeze(1)
        angles = pos * freqs
        self.register_buffer('cos', angles.cos().repeat_interleave(2, dim=-1))
        self.register_buffer('sin', angles.sin().repeat_interleave(2, dim=-1))
        
        # precompute causal mask
        causal = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer('causal_mask', causal)

    def _apply_rope(self, x_BSHD):
        S = x_BSHD.size(1)
        cos, sin = self.cos[:S][None, :, None, :], self.sin[:S][None, :, None, :]
        x1, x2 = x_BSHD[..., ::2], x_BSHD[..., 1::2]
        out = torch.stack([x1 * cos[..., ::2] - x2 * sin[..., 1::2], x1 * sin[..., ::2] + x2 * cos[..., 1::2]], dim=-1)
        return out.flatten(-2)

    def forward(self, x_BSD, mask_BS=None):
        B = x_BSD.size(0)
        S = x_BSD.size(1)
        
        q, k, v = self.qkv(x_BSD).chunk(3, dim=-1)
        q = q.view(B, S, self.n_heads, self.d_head)
        k = k.view(B, S, self.n_heads, self.d_head)
        v = v.view(B, S, self.n_heads, self.d_head)

        q, k = self._apply_rope(q), self._apply_rope(k)
        
        # build attention mask: combine causal + padding
        if mask_BS is not None:
            # Combine padding mask with causal mask
            pad_mask = mask_BS[:, None, None, :].expand(B, 1, S, S)
            causal = self.causal_mask[:S, :S]
            attn_mask = pad_mask & causal
            is_causal = False
        else:
            attn_mask = None
            is_causal = True  # use efficient causal attention
        
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
            attn_mask=attn_mask, is_causal=is_causal
        )
        return self.out(y.transpose(1, 2).reshape(B, S, -1))


class TransformerBlock(nn.Module):
    def __init__(self, D, n_heads, moe, max_seq_len=2048):
        super().__init__()
        self.attn = Attention(D, n_heads, max_seq_len)
        self.moe = moe
        self.pre_attn_norm, self.post_attn_norm = nn.LayerNorm(D), nn.LayerNorm(D)
        self.pre_moe_norm, self.post_moe_norm = nn.LayerNorm(D), nn.LayerNorm(D)

    def forward(self, x_BSD, mask_BS=None):
        x_BSD = x_BSD + self.post_attn_norm(self.attn(self.pre_attn_norm(x_BSD), mask_BS))
        x_BSD = x_BSD + self.post_moe_norm(self.moe(self.pre_moe_norm(x_BSD)))
        return x_BSD


class Transformer(nn.Module):
    def __init__(self, V, D, n_heads, n_layers, moe_fn, max_seq_len=2048):
        super().__init__()
        self.embed = nn.Embedding(V, D)
        self.blocks = nn.ModuleList([
            TransformerBlock(D, n_heads, moe_fn(), max_seq_len) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(D)
        self.unembed = nn.Linear(D, V, bias=False)
        self.unembed.weight = self.embed.weight
        self.moe_fn = moe_fn
        
    def forward(self, tokens_BS, mask_BS=None):
        x_BSD = self.embed(tokens_BS)
        for block in self.blocks:
            x_BSD = block(x_BSD, mask_BS)
        return self.unembed(self.norm(x_BSD))

