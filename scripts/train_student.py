
# =============================================================
# 📁 scripts/train_student.py
# -------------------------------------------------------------
"""Knowledge Distillation 付き Student 学習スクリプト"""
import torch, pathlib
from torch.utils.data import DataLoader
from tqdm import tqdm

from htr.datasets.reader  import Dataset as RawDataset
from htr.utils.dataloader import LineDS, collate, build_char_tables
from htr.models.teacher   import TeacherModel
from htr.models.student   import StudentModel

def main():
    raw = RawDataset(source="data"); raw.read_partitions()
    idx2c, c2i = build_char_tables(raw)
    train_loader = DataLoader(
        LineDS(raw.dataset['train']['dt'], raw.dataset['train']['gt'], c2i),
        batch_size=64, shuffle=True, collate_fn=collate, num_workers=0
    )

    # ----- モデル -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = TeacherModel(len(idx2c)).to(device)
    teacher.load_state_dict(torch.load("teacher.pth", map_location=device))
    teacher.eval();  # 勾配不要
    student = StudentModel(len(idx2c)).to(device)

    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    kd_loss  = torch.nn.KLDivLoss(reduction="batchmean")
    optim    = torch.optim.Adam(student.parameters(), lr=1e-3)
    alpha, beta = 0.5, 0.5  # 重み

    # ----- 学習 -----
    for epoch in range(30):
        student.train(); total = 0
        pbar = tqdm(train_loader, desc=f"Student Epoch {epoch+1}")
        for imgs, lbl, in_len, lab_len in pbar:
            imgs, lbl = imgs.to(device), lbl.to(device)
            with torch.no_grad():
                t_log = teacher(imgs)             # (B,W,V)
            s_log = student(imgs)                 # (B,W,V)
            l_ctc = ctc_loss(s_log.permute(1,0,2), lbl, in_len, lab_len)
            l_kd  = kd_loss(s_log, t_log.exp().detach())  # Student logP vs Teacher P
            loss  = alpha*l_ctc + beta*l_kd
            optim.zero_grad(); loss.backward(); optim.step()
            total += loss.item(); pbar.set_postfix(loss=loss.item())
        print(f"Epoch {epoch+1} Mean Loss: {total/len(train_loader):.4f}")

    # ----- 保存 -----
    path = pathlib.Path("student.pth")
    path.write_bytes(torch.save(student.state_dict(), path.as_posix()) or b"")
    print(f"✨ Saved distilled student to {path}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # 明示的に spawn
    main()