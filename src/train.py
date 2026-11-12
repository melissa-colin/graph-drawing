import copy
import os
import random
from pathlib import Path

# Less GPU memory fragmentation (4 GB card); must be set before CUDA is initialized.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data.input import GraphInput, flip_pe_signs, load_dataset, redraw_noise, split_dataset
from data.output import Layout
from model import LayoutModel
from loss import Loss, LossFn



def batch_loss(model: LayoutModel, batch: Batch, loss_fn: LossFn, augment: bool = False,
               iterations: int | None = None) -> torch.Tensor:
    """Mean loss over the graphs of a batch.

    The model sees the whole batch at once (disjoint union of the graphs); the positions are then cut
    back per graph with batch.ptr, because loss_fn takes one graph at a time.
    augment: draw new noise and new positional-encoding signs for each graph (training only).
    iterations: number of model iterations, the model's default if None.
    The batch is moved to the model's device (the DataLoader yields CPU batches).
    """
    if augment:
        batch = Batch.from_data_list([flip_pe_signs(redraw_noise(graph)) for graph in batch.to_data_list()])
    batch = batch.to(next(model.parameters()).device)
    graphs = batch.to_data_list()
    layout = model(batch, iterations)
    ptr = batch.ptr.tolist()
    losses = [
        loss_fn(Layout(layout.positions[ptr[k] : ptr[k + 1]], layout.width, layout.height), graph)[0]  # (total, metric values)
        for k, graph in enumerate(graphs)
    ]
    return torch.stack(losses).mean()


def train_step(model: LayoutModel, batch: Batch, loss_fn: LossFn, optimizer: torch.optim.Optimizer) -> float:
    """One optimizer step. Two tricks from CoRe-GD, both kept after testing (docs/idees/melissa-archi.md):
    - a random number of iterations around model.iterations, so the model works for any number of rounds;
    - gradient clipping, so that one bad batch cannot make a huge update.
    """
    iterations = max(round(random.gauss(model.iterations, 1.0)), 1)
    optimizer.zero_grad()
    loss = batch_loss(model, batch, loss_fn, augment=True, iterations=iterations)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def validation_loss(model: LayoutModel, graphs: list[GraphInput], loss_fn: LossFn, batch_size: int = 32) -> float:
    """Mean loss per graph. Positional encodings are seen as they are (no sign flip)."""
    if not graphs:
        return 0.0

    model.eval()
    try:
        total = 0.0
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            total += float(batch_loss(model, batch, loss_fn).item()) * batch.num_graphs
        return total / len(graphs)
    finally:
        model.train()


def train(
    model: LayoutModel,
    graphs: list[GraphInput],
    loss_fn: LossFn,
    epochs: int,
    lr: float = 2e-4,
    log_every: int = 0,
    batch_size: int = 1,
    val_graphs: list[GraphInput] | None = None,
    patience: int | None = None,
) -> list[float]:
    """Train the model on a list of graphs for a number of epochs.

    The graphs are shuffled every epoch and grouped into batches of batch_size graphs (of mixed sizes);
    one optimizer step per batch, on the mean loss of its graphs.
    """
    if patience is not None and val_graphs is None:
        raise ValueError("patience requires val_graphs to be provided.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss: float | None = None
    epochs_without_improvement = 0

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        total_weight = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False, unit="batch"):
            epoch_loss += train_step(model, batch, loss_fn, optimizer) * batch.num_graphs
            total_weight += batch.num_graphs

        epoch_loss /= total_weight if total_weight > 0 else 1
        history.append(epoch_loss)

        if val_graphs is not None:
            val_loss = validation_loss(model, val_graphs, loss_fn, batch_size)
            if best_val_loss is None or val_loss < best_val_loss:
                best_state = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if log_every > 0 and epoch % log_every == 0:
                print(f"Epoch {epoch}: train_loss = {epoch_loss:.4f}, val_loss = {val_loss:.4f}")
        elif log_every > 0 and epoch % log_every == 0:
            print(f"Epoch {epoch}: loss = {epoch_loss:.4f}")

        if patience is not None and val_graphs is not None:
            if epochs_without_improvement >= patience:
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history

def main(
    data_dir: Path,
    out_path: Path,
    epochs: int,
    lr: float,
    loss_fn: LossFn,
    width: int = 500,
    height: int = 500,
    batch_size: int = 1,
    patience: int | None = None,
) -> None:
    train_files, val_files, test_files = split_dataset(data_dir)
    graphs = load_dataset(data_dir)
    train_graphs = [graphs[path.stem] for path in train_files]
    val_graphs = [graphs[path.stem] for path in val_files]
    test_graphs = [graphs[path.stem] for path in test_files]

    if not train_graphs:
        raise ValueError(f"No training graphs found in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    in_features = train_graphs[0].x.shape[1]
    model = LayoutModel(in_features, hidden=64, width=width, height=height, margin=4.0).to(device)
    train(
        model,
        train_graphs,
        loss_fn=loss_fn,
        epochs=epochs,
        lr=lr,
        log_every=1,
        batch_size=batch_size,
        val_graphs=val_graphs,
        patience=patience,
    )

    if val_graphs:
        val_loss = validation_loss(model, val_graphs, loss_fn, batch_size)
        print(f"Validation loss: {val_loss:.4f}")
    if test_graphs:
        test_loss = validation_loss(model, test_graphs, loss_fn, batch_size)
        print(f"Test loss: {test_loss:.4f}")

    checkpoint = {
        "state_dict": {name: tensor.cpu() for name, tensor in model.state_dict().items()},
        "config": model.config,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)


if __name__ == "__main__":
    loss_fn = Loss(weights={
        "screen_fill": 1,
        "stress": 0,
        "node_overlaps": 0,
        "edge_overlaps": 0,
        "edge_anlges": 0, # TODO
    })

    main(
        Path("data/rome"),
        Path("runs/model.pt"),
        epochs=30,
        lr=2e-4,
        loss_fn=loss_fn,
        batch_size=4,
        patience=5,
    )
