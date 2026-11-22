"""Demo of the defense: draw a graph for a given screen (image size, node radius, edge width), see the defects of
the rendered image in red, watch the refinement in screen space remove them, bend the edges.

    python demo/app.py --checkpoint runs/model.pt        then open http://127.0.0.1:7860

Without --checkpoint, only Kamada-Kawai is available as a starting drawing. Runs on the CPU; the logic is in
demo/core.py (tested without the interface).
"""
import argparse
import json
import tempfile
from pathlib import Path

import gradio as gr

import core

DATA = core.ROOT / "data" / "rome"
# 40 test graphs of Rome (split of data/input.split_dataset, never seen in training), spread over the sizes:
# fixed by the split's seed, so listed once in this file instead of re-reading every graph at startup
TEST_GRAPHS = core.ROOT / "demo" / "rome_test_graphs.json"


def rome_test_graphs():
    if not DATA.exists():
        return []
    return [name for name in json.loads(TEST_GRAPHS.read_text()) if (DATA / f"{name}.graphml").exists()]


def get_graph(source, rome_name, kind, n, seed, upload):
    if source == "Rome (test)":
        if not rome_name:
            raise gr.Error("Choisir un graphe de Rome")
        from data.input import load_graphml
        return load_graphml(DATA / f"{rome_name}.graphml")
    if source == "Fichier .graphml":
        if upload is None:
            raise gr.Error("Déposer un fichier .graphml")
        import networkx as nx
        return core.graph_from_networkx(nx.read_graphml(upload), Path(upload).stem)
    return core.generate(kind, n, seed)


def build_app(model=None):
    methods = {"Kamada-Kawai": "kk"} | ({"Notre modèle": "ours"} if model is not None else {})
    names = rome_test_graphs()

    def graph_inputs():
        with gr.Group():
            source = gr.Radio(["Rome (test)", "Générateur", "Fichier .graphml"],
                              value="Rome (test)" if names else "Générateur", label="Graphe")
            rome = gr.Dropdown(names, value=names[len(names) // 2] if names else None,
                               label="Graphe de test de Rome (jamais vu à l'entraînement)")
            with gr.Row():
                kind = gr.Dropdown(list(core.GENERATORS), value="arbre", label="Générateur")
                n = gr.Slider(6, 120, value=40, step=1, label="Nœuds")
                seed = gr.Number(0, precision=0, label="Graine")
            upload = gr.File(file_types=[".graphml"], type="filepath", label="Fichier .graphml")
        return [source, rome, kind, n, seed, upload]

    def screen_inputs():
        with gr.Row():
            size = gr.Slider(80, 500, value=150, step=10, label="Côté de l'image (px)")
            radius = gr.Slider(1, 10, value=4, step=0.5, label="Rayon des nœuds (px)")
            width = gr.Slider(0.5, 4, value=1, step=0.5, label="Épaisseur des arêtes (px)")
        return [size, radius, width]

    def run_draw(source, rome, kind, n, seed, upload, size, radius, width, method, refine, steps, curves, defects):
        g = get_graph(source, rome, kind, n, seed, upload)
        d = core.draw(g, methods[method], int(size), float(radius), float(width), refine, curves, model, int(steps))
        start = core.Drawing(d.graph, d.size, d.frames[0], d.offsets * 0)
        gif = core.animation(d, Path(tempfile.mkdtemp()) / "affinage.gif", scale=2) if refine and steps > 0 else None
        timing = " · ".join(f"{k} {v:.1f} s" for k, v in d.seconds.items())
        table = [[label, round(float(core.measure(start)[key]), 3), round(float(d.measures[key]), 3)]
                 for key, label in core.MEASURE_LABELS.items()]
        return (core.render(start, 2, defects), core.render(d, 2, defects), gif, table,
                f"{g.num_nodes} nœuds, {g.edge_index.shape[1]} arêtes · {timing}")

    def run_compare(source, rome, kind, n, seed, upload, size, radius, width):
        g = get_graph(source, rome, kind, n, seed, upload)
        best = "ours" if model is not None else "kk"
        drawings = [core.draw(g, "kk", int(size), float(radius), float(width), refine=False),
                    core.draw(g, best, int(size), float(radius), float(width), refine=True, model=model),
                    core.draw(g, best, int(size), float(radius), float(width), refine=True, curves=True, model=model)]
        return [core.render(d, 2, True) for d in drawings] + [core.measures_table(drawings)]

    with gr.Blocks(title="Dessin de graphes en espace écran") as app:
        gr.Markdown("# Dessiner un graphe pour un écran donné\nEn rouge : les défauts de l'image rendue "
                    "(nœuds cachés ou fusionnés, arête qui touche un nœud qu'elle ne relie pas).")
        with gr.Tab("Dessiner"):
            with gr.Row():
                with gr.Column(scale=1):
                    g_in = graph_inputs()
                    s_in = screen_inputs()
                    method = gr.Radio(list(methods), value=list(methods)[-1], label="Dessin de départ")
                    with gr.Row():
                        refine = gr.Checkbox(True, label="Affinage en espace écran")
                        steps = gr.Slider(0, 100, value=25, step=5, label="Pas d'affinage")
                    with gr.Row():
                        curves = gr.Checkbox(False, label="Arêtes courbes")
                        defects = gr.Checkbox(True, label="Montrer les défauts")
                    go = gr.Button("Dessiner", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        before = gr.Image(type="pil", label="Départ, mis à l'échelle de l'écran")
                        after = gr.Image(type="pil", label="Résultat")
                    gif = gr.Image(type="filepath", label="L'affinage pas à pas")
                    table = gr.Dataframe(headers=["Mesure (sur l'image)", "Départ", "Résultat"], label="Mesures")
                    info = gr.Markdown()
            go.click(run_draw, g_in + s_in + [method, refine, steps, curves, defects], [before, after, gif, table, info])
        with gr.Tab("Comparer"):
            with gr.Row():
                with gr.Column(scale=1):
                    g_in2 = graph_inputs()
                    s_in2 = screen_inputs()
                    go2 = gr.Button("Comparer", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        img = [gr.Image(type="pil", label="Kamada-Kawai"),
                               gr.Image(type="pil", label="Affiné à l'écran"),
                               gr.Image(type="pil", label="Affiné + arêtes courbes")]
                    table2 = gr.Dataframe(headers=["Mesure", "Kamada-Kawai", "Affiné", "+ courbes"], label="Mesures")
            go2.click(run_compare, g_in2 + s_in2, img + [table2])
    app.callbacks = {"draw": run_draw, "compare": run_compare}  # for the tests
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, help="our model (runs/model.pt); Kamada-Kawai only without it")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(core.load_model(args.checkpoint)).launch(server_name="127.0.0.1", server_port=args.port)
