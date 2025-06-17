from torch import nn
from .blocks import FullGatedConv2d, SEBlock, MHAttention

class StudentModel(nn.Module):
    """軽量 Student (チャネル・隠れ次元を半分、ヘッド数=1)"""
    def __init__(self, vocab_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            FullGatedConv2d(1, 32), nn.ReLU(), SEBlock(32, reduction=32), nn.MaxPool2d(2,2),
            FullGatedConv2d(32, 64), nn.ReLU(), SEBlock(64, reduction=32), nn.MaxPool2d(2,2)
        )
        self.proj = nn.Linear(64*8, 128)
        self.attn = MHAttention(dim=128, heads=1)
        self.fc   = nn.Linear(128, vocab_size)

    def forward(self, x):
        f = self.encoder(x)
        b, c, h, w = f.size()
        f = f.permute(0,3,1,2).contiguous().view(b, w, c*h)
        y = self.fc(self.attn(self.proj(f)))
        return y.log_softmax(2)
