import argparse
from pathlib import Path

import torch

from data.input import GraphInput, load_graphml
from data.output import Layout, save_png, save_positions
from loss import Loss
from model import LayoutModel
from refine import ROUNDS, STEPS, draw
from train import REFINE_WEIGHTS


def load_model(model_path: Path) -> LayoutModel:
    checkpoint = torch.load(model_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unknown checkpoint format for {model_path}: {type(checkpoint)!r}")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint for {model_path}: expected 'config' and 'state_dict'.")

    config = dict(checkpoint["config"])
    config.pop("conv", None)  # very old checkpoints: option removed since
    config.setdefault("head", "full")  # checkpoints saved before the separable head used the full one
    model = LayoutModel(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def infer(model_path: Path, graphml_path: Path, rounds: int = ROUNDS, steps: int = STEPS) -> tuple[Layout, GraphInput]:
    """Draw a graph: the model with `rounds` rounds, then `steps` refinement steps on the positions with the
    refinement loss of train.REFINE_WEIGHTS (see refine.py; steps = 0 for the model alone)."""
    graph = load_graphml(graphml_path, node_size=4.0)
    model = load_model(model_path)
    return draw(model, graph, Loss(weights=REFINE_WEIGHTS), rounds, steps), graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a graph using a trained model.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("graphml_path", type=Path)
    parser.add_argument("out_png", type=Path)
    parser.add_argument("--rounds", type=int, default=ROUNDS, help="model rounds")
    parser.add_argument("--steps", type=int, default=STEPS, help="refinement steps on the positions (0: model only)")
    args = parser.parse_args()

    layout, graph = infer(args.model_path, args.graphml_path, args.rounds, args.steps)
    save_png(layout, graph, args.out_png)
    save_positions(layout, graph, args.out_png.with_suffix(".json"))