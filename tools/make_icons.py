"""產生 PWA 圖示 (PNG 192/512)。"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "mobile"
for size in (192, 512):
    img = Image.new("RGB", (size, size), "#0f1115")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([size * 0.08, size * 0.08, size * 0.92, size * 0.92], radius=size * 0.18, fill="#d64545")
    # 簡單 K 線圖形
    for i, (lo, hi, up) in enumerate([(0.62, 0.42, True), (0.55, 0.35, True), (0.5, 0.3, False), (0.45, 0.22, True)]):
        x = size * (0.27 + i * 0.155)
        w = size * 0.07
        d.line([x + w / 2, size * (lo + 0.06), x + w / 2, size * (hi - 0.04)], fill="white", width=max(2, size // 64))
        d.rectangle([x, size * hi, x + w, size * lo], fill="white" if up else "#0f1115", outline="white", width=max(2, size // 96))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", int(size * 0.16))
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    d.text((size * 0.5, size * 0.80), "台股籌碼", fill="white", font=font, anchor="mm")
    img.save(OUT / f"icon-{size}.png")
    print("saved", OUT / f"icon-{size}.png")
