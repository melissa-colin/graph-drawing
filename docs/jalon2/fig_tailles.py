"""Figure of the milestone-2 slides (Yani's part): the same Kamada-Kawai drawing of a Rome test graph, rendered
at 500, 200 and 100 px with nodes of radius 4 px, the defects of each image in red, all shown at the same
display size. Run from the project root after benchmark/compare.py:

    python docs/jalon2/fig_tailles.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "demo")]

import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import core  # noqa: E402
from data.input import load_graphml  # noqa: E402
from resolution import rescale_to_frame, with_style  # noqa: E402

GRAPH = "grafo7140.41"  # a 41-node test graph (demo/rome_test_graphs.json)
SIZES, RADIUS, SHOWN = (500, 200, 100), 4.0, 400


def main():
    g = load_graphml(ROOT / "data" / "rome" / f"{GRAPH}.graphml")
    positions = torch.load(ROOT / "benchmark" / "out" / "positions.pt")["Kamada-Kawai"][g.name]
    tiles = []
    for size in SIZES:
        styled = with_style(g, RADIUS, 1.0)
        P = rescale_to_frame(positions, size, RADIUS)
        d = core.Drawing(styled, size, P, torch.zeros(g.edge_index.shape[1], 2))
        d.measures = core.measure(d)
        image = core.render(d, scale=1, defects=True).resize((SHOWN, SHOWN), Image.NEAREST)
        tiles.append((size, image, d.measures))
    pad, top, bottom = 16, 34, 58
    sheet = Image.new("RGB", (len(tiles) * (SHOWN + pad) + pad, SHOWN + top + bottom), "#f7f6f2")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    for i, (size, image, m) in enumerate(tiles):
        x = pad + i * (SHOWN + pad)
        sheet.paste(image, (x, top))
        draw.rectangle((x - 1, top - 1, x + SHOWN, top + SHOWN), outline="#ddd8cc")
        draw.text((x, 8), f"{size} × {size} px, nœuds de {RADIUS:g} px", fill="#14213d", font=font)
        draw.text((x, top + SHOWN + 8), f"nœuds fusionnés {100 * m['merged_nodes']:.0f} %, cachés "
                  f"{100 * m['hidden_nodes']:.0f} %", fill="#4a5568", font=small)
        draw.text((x, top + SHOWN + 30), f"arêtes ambiguës {m['ambiguous_edges']:.2f} par arête".replace(".", ","), fill="#4a5568",
                  font=small)
    out = Path(__file__).with_name("tailles.png")
    sheet.save(out)
    print(out, [(s, {k: round(v, 3) for k, v in m.items() if k in core.MEASURE_LABELS}) for s, _, m in tiles])


if __name__ == "__main__":
    main()
