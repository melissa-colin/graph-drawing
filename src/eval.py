import argparse
from pathlib import Path

from data.input import GraphInput, load_graphml, split_dataset
from data.output import Layout


def reference_layout(graph: GraphInput, width: int, height: int, margin: float,
                     method: str = "kamada_kawai") -> Layout:
    """Drawing from a classic networkx method, rescaled into [margin, width - margin] x [margin, height - margin].

    networkx methods return {node: (x, y)} in [-1, 1]: they must be put back in the order of
    graph.node_ids, then into the model's frame.
    """
    # TODO(human): 1. rebuild a networkx graph from graph.edge_index (and graph.node_ids)
    #              2. apply nx.kamada_kawai_layout (or spring_layout depending on method)
    #              3. (N, 2) tensor in node_ids order, then rescale into the frame
    #   Question: keep the width / height ratio of the drawing, or stretch it to fill the whole frame?
    raise NotImplementedError


def metrics(layout: Layout, graph: GraphInput) -> dict[str, float]:
    """All evaluation metrics of a drawing, as floats (no gradient here)."""
    # TODO(human): stress, node overlap, edge crossings (src/metrics/)
    raise NotImplementedError


def evaluate(model_path: Path, test_files: list[Path]) -> dict[str, dict[str, float]]:
    """Mean of each metric over the test graphs, for the model and for each reference.

    Example return value: {"model": {"stress": 41.2, ...}, "kamada_kawai": {"stress": 18.7, ...}}
    """
    # TODO(human): load the model (infer.load_model), and for each test graph: model drawing,
    #              reference drawing, metrics() for each; then the means
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on the test graphs.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rome"))
    parser.add_argument("--max-graphs", type=int, default=None, help="limit the number of graphs (quick runs)")
    args = parser.parse_args()
    # TODO(human): split_dataset (same split as training), evaluate, print a readable table
