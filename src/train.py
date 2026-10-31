import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from data.input import GraphInput, load_graphml, split_dataset
from model import LayoutModel
from loss import Loss, LossFn



def train_step(model: LayoutModel, graph: GraphInput, loss_fn: LossFn, optimizer: torch.optim.Optimizer) -> float:
    optimizer.zero_grad()
    layout = model(graph)
    loss, metrics_values = loss_fn(layout, graph)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def random_batches(graphs: list[GraphInput], batch_size: int, seed: int) -> list[list[GraphInput]]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}.")

    shuffled = graphs.copy()
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def train_step_batch(model: LayoutModel, batch: list[GraphInput], loss_fn: LossFn, optimizer: torch.optim.Optimizer) -> float:
    if not batch:
        return 0.0

    optimizer.zero_grad()
    total_loss = 0.0
    for graph in batch:
        loss, metrics_values = loss_fn(model(graph), graph)
        total_loss = total_loss + loss
    mean_loss = total_loss / len(batch)
    mean_loss.backward()
    optimizer.step()
    return float(mean_loss.item())


@torch.no_grad()
def validation_loss(model: LayoutModel, graphs: list[GraphInput], loss_fn: LossFn) -> float:
    if not graphs:
        return 0.0

    model.eval()
    try:
        total = 0.0
        for graph in graphs:
            loss, metrics_values = loss_fn(model(graph), graph)
            total += float(loss.item())
        return total / len(graphs)
    finally:
        model.train()


def train(
    model: LayoutModel,
    graphs: list[GraphInput],
    loss_fn: LossFn,
    epochs: int,
    lr: float = 1e-3,
    log_every: int = 0,
    batch_size: int = 1,
    val_graphs: list[GraphInput] | None = None,
    patience: int | None = None,
) -> list[float]:
    """Train the model on a list of graphs for a number of epochs.

    Default behavior is unchanged: if batch_size = 1 and no validation/early-stopping
    options are provided, the code follows the original per-graph SGD loop.
    """
    if patience is not None and val_graphs is None:
        raise ValueError("patience requires val_graphs to be provided.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss: float | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        if batch_size <= 1:
            batches = [[graph] for graph in graphs]
        else:
            batches = random_batches(graphs, batch_size=batch_size, seed=epoch)

        epoch_loss = 0.0
        total_weight = 0
        for batch in tqdm(batches, desc=f"Epoch {epoch}/{epochs}", leave=False, unit="graph"):
            if len(batch) == 1:
                batch_loss = train_step(model, batch[0], loss_fn, optimizer)
            else:
                batch_loss = train_step_batch(model, batch, loss_fn, optimizer)
            epoch_loss += batch_loss * len(batch)
            total_weight += len(batch)

        epoch_loss /= total_weight if total_weight > 0 else 1
        history.append(epoch_loss)

        if val_graphs is not None:
            val_loss = validation_loss(model, val_graphs, loss_fn)
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
    train_graphs = [load_graphml(path) for path in train_files]
    val_graphs = [load_graphml(path) for path in val_files]
    test_graphs = [load_graphml(path) for path in test_files]

    if not train_graphs:
        raise ValueError(f"No training graphs found in {data_dir}")

    in_features = train_graphs[0].x.shape[1]
    model = LayoutModel(in_features, hidden=64, width=width, height=height, margin=4.0)
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
        val_loss = validation_loss(model, val_graphs, loss_fn)
        print(f"Validation loss: {val_loss:.4f}")
    if test_graphs:
        test_loss = validation_loss(model, test_graphs, loss_fn)
        print(f"Test loss: {test_loss:.4f}")

    checkpoint = {
        "state_dict": model.state_dict(),
        "config": {
            "in_features": model.encoder.net[0].in_features,
            "hidden": model.encoder.net[0].out_features,
            "width": model.width,
            "height": model.height,
            "margin": model.head.margin,
            "temperature": model.head.temperature,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)


if __name__ == "__main__":
    loss_fn = Loss(weights={
        "screen": 1,
        "stress": 0,
        "node_overlaps": 0,
        "edge_overlaps": 0,
        "edge_anlges": 0, # TODO
    })

    main(
        Path("data/rome"),
        Path("runs/model.pt"),
        epochs=10,
        lr=1e-3,
        loss_fn=loss_fn,
        batch_size=4,
        patience=3,
    )
