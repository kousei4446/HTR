# =============================================================
# 📁 htr/models/blocks.py
# -------------------------------------------------------------
"""共有ブロック: FullGatedConv2d / SEBlock / MHAttention / PositionalEncoding"""
import torch, math
import torch.nn as nn

__all__ = ["FullGatedConv2d", "SEBlock", "MHAttention", "PositionalEncoding"]

class FullGatedConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv  = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.gate  = nn.Conv2d(in_ch, out_ch, k, s, p)

    def forward(self, x):
        return self.conv(x) * torch.sigmoid(self.gate(x))

class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, ch // reduction), nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, *_ = x.shape
        s = self.pool(x).view(b, c)
        w = self.fc(s).view(b, c, 1, 1)
        return x * w

class PositionalEncoding(nn.Module):
    """標準的な Sine/Cosine 位置エンコーディング"""
    def __init__(self, dim: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class MHAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return out