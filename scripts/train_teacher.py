"""Teacher モデルを CTC Loss で事前学習し、teacher.pth に保存"""
import torch, pathlib
from torch.utils.data import DataLoader
from tqdm import tqdm

from htr.datasets.reader import Dataset as RawDataset  # 既存リーダをそのまま利用
from htr.utils.dataloader import LineDS, collate, build_char_tables
from htr.models.teacher import TeacherModel

# ---------- データ用意 ----------
raw = RawDataset(source="data")
raw.read_partitions()          # ★ これを忘れずに呼ぶ
idx2c, c2i = build_char_tables(raw)

train_loader = DataLoader(
    LineDS(raw.dataset['train']['dt'], raw.dataset['train']['gt'], c2i),
    batch_size=64, shuffle=True, collate_fn=collate, num_workers=0
)



# ---------- モデル & 損失 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = TeacherModel(vocab_size=len(idx2c)).to(device)
ctc    = torch.nn.CTCLoss(blank=0, zero_infinity=True)
optim  = torch.optim.Adam(model.parameters(), lr=1e-3)




# ---------- 学習ループ ----------
for epoch in range(50):
    model.train(); tot = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    
    for imgs, labels, in_lens, lab_lens in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs).permute(1,0,2)  # (T,B,V)
        loss = ctc(logits, labels, in_lens, lab_lens)
        optim.zero_grad(); loss.backward(); optim.step()
        tot += loss.item(); pbar.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1} Mean Loss: {tot/len(train_loader):.4f}")



# ---------- 保存 ----------
path = pathlib.Path("teacher.pth")
torch.save(model.state_dict(), path)          # ← これだけで OK
print(f"✨ Saved teacher weights to {path} ({path.stat().st_size/1e6:.2f} MB)")




