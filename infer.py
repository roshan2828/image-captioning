import torch
from PIL import Image
from torchvision import transforms

from model import ImageCaptionModel
from utils import build_vocab


print("🔹 Starting inference...")


# -------- SETTINGS --------
IMAGE_PATH = "test_images/test3.jpg"
MODEL_PATH = "best_model.pth"   # ✅ use best model
MAX_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------- BUILD VOCAB --------
print("🔹 Building vocabulary...")

captions = []
with open("captions.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        _, caption = parts
        captions.append(caption)

vocab = build_vocab(captions)
inv_vocab = {v: k for k, v in vocab.items()}

print(f"✅ Vocabulary built (size: {len(vocab)})")


# -------- LOAD MODEL --------
print("🔹 Loading model...")

model = ImageCaptionModel(vocab_size=len(vocab)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model loaded")


# -------- IMAGE TRANSFORM --------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# -------- LOAD IMAGE --------
print("🔹 Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")
image = transform(image).unsqueeze(0)
image = image.to(DEVICE)   # ✅ important fix

print("✅ Image loaded")


# -------- GENERATE CAPTION --------
print("🔹 Generating caption...")

with torch.no_grad():
    caption_ids = model.generate_caption(
        image=image,
        vocab=vocab,
        max_len=MAX_LEN
    )

words = []
for idx in caption_ids:
    word = inv_vocab.get(idx, "<unk>")
    if word == "<end>":
        break
    if word not in ["<start>", "<pad>"]:
        words.append(word)

caption = " ".join(words)


# -------- OUTPUT --------
print("\n🖼 IMAGE:", IMAGE_PATH)
print("📝 GENERATED CAPTION:")
print(caption)
