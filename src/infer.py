import argparse
from pathlib import Path

import torch

from data.input import GraphInput, load_graphml
from data.output import Layout, save_png, save_positions
from model import LayoutModel


def load_model(model_path: Path) -> LayoutModel:
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unknown checkpoint format for {model_path}: {type(checkpoint)!r}")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint for {model_path}: expected 'config' and 'state_dict'.")

    model = LayoutModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def infer(model_path: Path, graphml_path: Path) -> tuple[Layout, GraphInput]:
    graph = load_graphml(graphml_path, node_size=4.0)
    model = load_model(model_path)
    with torch.no_grad():
        return model(graph), graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a graph using a trained model.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("graphml_path", type=Path)
    parser.add_argument("out_png", type=Path)
    args = parser.parse_args()

    layout, graph = infer(args.model_path, args.graphml_path)
    save_png(layout, graph, args.out_png)
    save_positions(layout, graph, args.out_png.with_suffix(".json"))