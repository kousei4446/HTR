import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import editdistance
import numpy as np

from htr.datasets.reader import Dataset as RawDataset

# ---------- データ準備 ----------
raw = RawDataset(source="data")
raw.read_partitions()

# 認識文字とマッピング
chars = sorted({c for t in raw.dataset['train']['gt'] for c in t})
idx2c = ['<blank>'] + chars
c2i = {c: i for i, c in enumerate(idx2c)}

# 画像前処理：グレースケール化・リサイズ・正規化
tfm = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])







# ---------- データセット定義 ----------
class LineDS(Dataset):
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = tfm(Image.open(self.paths[i]))
        label = torch.tensor([c2i[c] for c in self.labels[i]], dtype=torch.long)
        return img, label







# CTC対応のcollate関数
def collate(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)
    labels_cat = torch.cat(labels)
    img_len = torch.full((len(imgs),), imgs.shape[-1] // 4, dtype=torch.long)
    label_len = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return imgs, labels_cat, img_len, label_len

train_loader = DataLoader(
    LineDS(raw.dataset['train']['dt'], raw.dataset['train']['gt']),
    batch_size=64,
    shuffle=True,
    collate_fn=collate
)




import torch
import torch.nn as nn
import torch.nn.functional as F

# FullGatedConv2d
class FullGatedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x) * torch.sigmoid(self.gate(x))

# Squeeze-and-Excitation block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Multi-Head Self-Attention（1D）
class MHAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.attn(x, x, x)
        return out

# 論文風TeacherModel
class TeacherModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        vocab_size = len(vocab)  # vocabのサイズを取得

        self.encoder = nn.Sequential(
            FullGatedConv2d(1, 64), nn.ReLU(), SEBlock(64),
            nn.MaxPool2d(2, 2),
            FullGatedConv2d(64, 128), nn.ReLU(), SEBlock(128),
            nn.MaxPool2d(2, 2)
        )

        self.flatten_proj = nn.Linear(128 * 8, 256)
        self.attn = MHAttention(dim=256, num_heads=4)

        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        # x: (B, 1, 32, 128)
        y = self.encoder(x)  # -> (B, 128, 8, 32)
        b, c, h, w = y.size()
        y = y.permute(0, 3, 1, 2).contiguous().view(b, w, c * h)  # (B, W, C×H)
        y = self.flatten_proj(y)  # (B, W, D)
        y = self.attn(y)  # (B, W, D)
        y = self.fc(y)  # (B, W, Vocab)
        return y.log_softmax(2)  # for CTC loss





# ---------- 学習処理 ----------
def train_teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = TeacherModel(idx2c).to(device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        total_loss = 0
        count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for imgs, labels, input_lens, label_lens in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)                   # (B, W, Vocab)
            outputs = outputs.permute(1, 0, 2)      # (W, B, Vocab) for CTC
            loss = ctc(outputs, labels, input_lens, label_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / count
        print(f"Epoch {epoch+1} 平均loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "teacher.pth")
    print("モデルを teacher.pth として保存しました。")

# ---------- 実行 ----------
if __name__ == "__main__":
    train_teacher()
