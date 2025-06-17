# Handwritten Text Recognition (HTR) – **Teacher × Student Distillation Pipeline**

> **Bentham / IAM 対応・PyTorch 実装**
>
> 軽量 Student モデルでも高精度を維持できる “Knowledge Distillation” パイプラインを **ゼロから設計 & コード実装** しました。
---

## 🔍 What’s inside?

| モジュール              | 技術要素                                                                         |
| ------------------ | ---------------------------------------------------------------------------- |
| **Teacher**        | Full‑Gated Conv + SE Block + Multi‑Head Self‑Attention + Positional Encoding |
| **Student**        | チャネル数 ½・ヘッド数 1 の軽量設計                                                         | 
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

# 5) distill student (~40 min)
python scripts/train_student.py
```

生成された `teacher.pth` / `student.pth` を使って推論デモも可能です。

---

## 🛠️ Project Structure

```
├─ htr/                # library package
│  ├─ models/          #   ├ teacher.py / student.py / blocks.py
│  ├─ datasets/        #   └ reader.py (Bentham loader)
│  └─ utils/           #   └ dataloader.py
├─ scripts/            # training entrypoints
│  ├─ train_teacher.py
│  └─ train_student.py
└─ data/               # dataset root (git‑ignored)
```

---

## 📊 Benchmarks (Bentham Valid)

| Model                                          | CER ↓     | Params    | FPS\* (RTX 3060) |
| ---------------------------------------------- | --------- | --------- | ---------------- |
| **Teacher**                                    | **3.2 %** | 7.6 M     | 430 img/s        |
| **Student**                                    | 4.1 %     | **2.9 M** | **830 img/s**    |
| <sub>\*greedy decode / FP16 / batch = 32</sub> |           |           |                  |

---

## ✨ Why it matters

* **論文実装力** – arXiv <2412.18524> のアーキテクチャを忠実に再現
* **最適化センス** – 蒸留で *モデルサイズ 62 % 削減*、*速度 1.9× UP*、精度低下 0.9 pp に抑制
* **クリーン設計** – ライブラリ化 & `pip install -e .` 対応で再利用容易
* **Windows / Linux 両対応** – `num_workers=0` fallback & spawn‑safe スクリプト

---

## 🗺️ Roadmap / TODO

* [ ] IAM Dataset サポート
* [ ] Transformer Decoder + BeamSearch 推論
* [ ] ONNX / TensorRT エクスポート
* [ ] CI (GitHub Actions) で単体テスト & 速度ベンチ

---

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