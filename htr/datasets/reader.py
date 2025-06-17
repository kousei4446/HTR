# =============================================================
# 📁 htr/datasets/reader.py
# -------------------------------------------------------------
"""BenthamDatasetR0-GT 用の簡易リーダー。
Images/Lines に PNG、Transcriptions に txt、Partitions に list ファイルという
公式配布レイアウトを想定している。"""
import os, html
class Dataset:
    def __init__(self, source: str = "data", name: str = "bentham"):
        self.source = source
        self.name   = name
        self.partitions = ["train", "valid", "test"]
        self.dataset = {p: {"dt": [], "gt": []} for p in self.partitions}

    def read_partitions(self):
        base   = os.path.join(self.source, "BenthamDatasetR0-GT")
        imgdir = os.path.join(base, "Images", "Lines")
        labdir = os.path.join(base, "Transcriptions")
        partdir= os.path.join(base, "Partitions")

        # 1) すべての transcriptions を辞書化
        labels = {}
        for fname in os.listdir(labdir):
            fid  = os.path.splitext(fname)[0]
            text = " ".join(open(os.path.join(labdir, fname), encoding="utf-8").read().split())
            labels[fid] = html.unescape(text)

        # 2) 各 split の .lst を読み込み
        maps = {
            "train": "TrainLines.lst",
            "valid": "ValidationLines.lst",
            "test" : "TestLines.lst"
        }
        for split, lstname in maps.items():
            with open(os.path.join(partdir, lstname), encoding="utf-8") as f:
                ids = [l.strip() for l in f if l.strip()]
            for fid in ids:
                img = os.path.join(imgdir, f"{fid}.png")
                if os.path.isfile(img) and fid in labels:
                    self.dataset[split]["dt"].append(img)
                    self.dataset[split]["gt"].append(labels[fid])
