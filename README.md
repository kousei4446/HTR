# Handwritten Text Recognition (HTR) – **Teacher × Student Distillation Pipeline**

> **Bentham / IAM 対応・PyTorch 実装**
>
>  HTR の　技術要素を学ぶための実装です。
---

## 🔍 What’s inside?

| モジュール              | 技術要素                                                                         |
| ------------------ | ---------------------------------------------------------------------------- |
| **Teacher**        | Full‑Gated Conv + SE Block + Multi‑Head Self‑Attention + Positional Encoding |
| **KD Training**    | CTC Loss + KL Distillation                                                   |
| **Dataset Loader** | BenthamDatasetR0‑GT 自動パース                                                    |

---

## 🚀 Quick Start

```bash
# 1) clone & create env
python -m venv .venv && source .venv/bin/activate  # Windows は .venv\Scripts\activate
pip install -r requirements.txt

# 2) install package (editable mode)
pip install -e .

# 3) place dataset
./data/BenthamDatasetR0-GT/Images/Lines/*.png
./data/BenthamDatasetR0-GT/Transcriptions/*.txt
./data/BenthamDatasetR0-GT/Partitions/TrainLines.lst ...

# 4) train teacher (~1h on RTX 3060)
python scripts/train_teacher.py

```

生成された `teacher.pth` を使って推論デモも可能です。

---

## 🛠️ Project Structure

```
├─ htr/                # library package
│  ├─ models/          #   ├ teacher.py / student.py / blocks.py
│  ├─ datasets/        #   └ reader.py (Bentham loader)
│  └─ utils/           #   └ dataloader.py
├─ scripts/            # training entrypoints
│  ├─ train_teacher.py
└─ data/               # dataset root (git‑ignored)
```

---

## 📊 Benchmarks (Bentham Valid)




## 🖋️ Author

**Kousei**  / CS B4 → Graduate School (2026) 予定
好きな領域：AI × 社会実装・モデル軽量化・MLOps

<a href="https://mysite2-38c23.web.app/">作品集</a>

---

<p align="center"><b>⭐ Feel free to fork / star / contact!</b></p>


## 実行コマンド
```
python scripts/infer_demo.py --img image/image.png --student
```

```
python scripts/infer_demo.py --img image/image.png
```