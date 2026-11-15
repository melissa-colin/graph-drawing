"""Sheet of the 500 x 500 renders written by benchmark/compare.py: one column per method, one row per graph.
Each render is pasted as it is, not resampled.
Run from the project root:  python benchmark/make_sheet.py   (benchmark/out/renders -> benchmark/out/sheet.png)"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RENDERS = HERE / "out" / "renders"
OUT = HERE / "out" / "sheet.png"
METHODS = ["Aléatoire", "Notre modèle", "Kamada-Kawai", "DeepGD", "CoRe-GD"]
SIZE, GAP, TOP, LEFT = 500, 14, 70, 190


def make_sheet(renders=RENDERS, out=OUT):
    names = sorted({re.search(r"grafo[0-9.]+$", p.stem).group(0) for p in renders.glob("*.png")},
                   key=lambda n: int(re.search(r"\.(\d+)$", n).group(1)))
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
    sheet = Image.new("RGB", (LEFT + len(METHODS) * (SIZE + GAP), TOP + len(names) * (SIZE + GAP)), "white")
    draw = ImageDraw.Draw(sheet)
    for c, method in enumerate(METHODS):
        draw.text((LEFT + c * (SIZE + GAP) + SIZE // 2, TOP // 2), method, fill="black", font=font, anchor="mm")
    for r, name in enumerate(names):
        y0 = TOP + r * (SIZE + GAP)
        draw.text((LEFT // 2, y0 + SIZE // 2), f"{name.split('.')[-1]}\nnœuds", fill="black", font=font,
                  anchor="mm", align="center")
        for c, method in enumerate(METHODS):
            x0 = LEFT + c * (SIZE + GAP)
            sheet.paste(Image.open(renders / f"{method.replace(' ', '_')}_{name}.png"), (x0, y0))
            draw.rectangle((x0 - 1, y0 - 1, x0 + SIZE, y0 + SIZE), outline="#9ca3af", width=2)
    sheet.save(out)
    print("written:", out)


if __name__ == "__main__":
    make_sheet()
