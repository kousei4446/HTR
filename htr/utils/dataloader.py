# dataloader_utils.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMG_SIZE = (32, 128)

# 画像前処理（Teacher / Student で共通）
tfm = transforms.Compose([
    transforms.Grayscale(),                    # 1ch
    transforms.Resize((32, 128)),              # H=32, W=128
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])         # [-1, 1] スケール
])

# 1️⃣ 文字テーブル作成 ----------------------------------------------------------
def build_char_tables(raw_reader):
    """
    raw_reader.dataset['train']['gt'] などから
    idx2c と c2i を作成して返す
    """
    chars = sorted({c for txt in raw_reader.dataset['train']['gt'] for c in txt})
    idx2c = ['<blank>'] + chars               # blank=0 を先頭に
    c2i   = {c: i for i, c in enumerate(idx2c)}
    return idx2c, c2i

# 2️⃣ Dataset ---------------------------------------------------------------
class LineDS(Dataset):
    """
    paths: 画像ファイルパスのリスト
    labels: 対応する文字列ラベルのリスト
    c2i: 文字 → インデックス辞書
    """
    def __init__(self, paths, labels, c2i):
        self.paths = paths
        self.labels = labels
        self.c2i = c2i

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = tfm(Image.open(self.paths[idx]))
        # 文字列ラベルを int テンソルへ
        lab = torch.tensor([self.c2i[c] for c in self.labels[idx]],
                           dtype=torch.long)
        return img, lab

# 3️⃣ collate_fn -----------------------------------------------------------
def collate(batch):
    """
    画像は固定サイズ、ラベルは可変長 → CTC が要求する形式にまとめる
    Returns:
        imgs:      (B, 1, 32, 128)
        labels:    (sum(len(lbl_i)),)
        img_lens:  (B,)  ここでは W/4 と仮定
        label_lens:(B,)
    """
    imgs, labels = zip(*batch)

    imgs = torch.stack(imgs)                    # (B, 1, 32, 128)
    labels_cat = torch.cat(labels)              # 1 次元に連結

    # Conv + Pooling 構成に応じて入力長を計算
    # ↓ 例：MaxPool(2,2)×2 → W が 4 分の 1
    img_lens = torch.full((len(imgs),),
                          imgs.shape[-1] // 4,
                          dtype=torch.long)

    label_lens = torch.tensor([len(l) for l in labels],
                              dtype=torch.long)
    return imgs, labels_cat, img_lens, label_lens
