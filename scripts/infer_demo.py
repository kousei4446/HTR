"""CLI で画像を推論し、認識テキストと共に表示も可能

Usage examples
--------------
# 推論のみ
python scripts/infer_demo.py --img sample.png

# Student モデルでフォルダ一括 & 表示
python scripts/infer_demo.py --dir ./samples --student --show
"""
import argparse, pathlib, torch, matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from htr.models.teacher import TeacherModel
from htr.models.student import StudentModel
from htr.utils.dataloader import build_char_tables, IMG_SIZE
from htr.datasets.reader import Dataset as RawDataset






# ---------- CLI Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, help="単一画像パス")
parser.add_argument("--dir", type=str, help="フォルダ内 *.png を一括推論")
parser.add_argument("--student", action="store_true", help="StudentModel を使用")
parser.add_argument("--ckpt", type=str, default=None, help="weights ファイルを指定")
parser.add_argument("--show", action="store_true", help="読み込んだ画像を表示する")
args = parser.parse_args()
if not (args.img or args.dir):
    parser.error("--img か --dir のいずれかを指定してください")

# ---------- vocab & transform ----------
raw = RawDataset(source="data"); raw.read_partitions()
idx2c, _ = build_char_tables(raw)
_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])
BLANK = 0

# 文字→インデックスを確認
# for i, c in enumerate(idx2c[:120]):
#     print(i, c)

# ---------- model loader ----------

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.student:
        model = StudentModel(len(idx2c))
        ckpt  = args.ckpt or "student.pth"
    else:
        model = TeacherModel(len(idx2c))  
        ckpt  = args.ckpt or "teacher.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    model.eval().to(device)
    return model, device

model, device = load_model()

# ---------- helper: greedy CTC decode ----------

def greedy_decode(logits):
    seq = logits.argmax(1).tolist()
    out, prev = [], BLANK
    for s in seq:
        if s != prev and s != BLANK:
            out.append(idx2c[s])
        prev = s
    return "".join(out)



# ---------- gather paths ----------
paths = []
if args.img: paths.append(pathlib.Path(args.img))
if args.dir: paths.extend(sorted(pathlib.Path(args.dir).glob("*.png")))

# ---------- inference loop ----------
for p in paths:
    pil = Image.open(p).convert("L")
    img = _tf(pil).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        logit = model(img)[0]               # (W,V)
    text = greedy_decode(logit.cpu())
    print(f"{p.name}: {text}")

    if args.show:
        plt.figure(figsize=(6,2))
        plt.imshow(pil, cmap="gray")
        plt.title(text)
        plt.axis("off")
        plt.tight_layout()
plt.show() if args.show else None