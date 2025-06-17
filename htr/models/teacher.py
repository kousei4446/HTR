# =============================================================
# 📁 htr/models/teacher.py
# -------------------------------------------------------------
from torch import nn
from .blocks import FullGatedConv2d, SEBlock, PositionalEncoding, MHAttention

class TeacherModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            FullGatedConv2d(1, 64), nn.ReLU(), SEBlock(64), nn.MaxPool2d(2, 2),
            FullGatedConv2d(64,128), nn.ReLU(), SEBlock(128), nn.MaxPool2d(2, 2)
        )
        self.proj  = nn.Linear(128 * 16, 256)
        self.pos   = PositionalEncoding(256)
        self.attn  = MHAttention(dim=256, heads=4)
        self.fc    = nn.Linear(256, vocab_size)

    def forward(self, x):
        f = self.encoder(x)
        b, c, h, w = f.size()
        f = f.permute(0,3,1,2).contiguous().view(b, w, c*h)
        y = self.fc(self.attn(self.pos(self.proj(f))))
        return y.log_softmax(2)
