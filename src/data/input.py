"""Format d'entrée commun : un graphe prêt à être donné au modèle.

Tout le monde importe GraphInput. Personne ne manipule de tenseurs en vrac.
"""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class GraphInput:
    name: str                 # nom du fichier sans extension, ex. "grafo2774.26"
    edge_index: torch.Tensor  # (2, E) long : chaque colonne est une arête [source, cible], nœuds numérotés 0..N-1
    x: torch.Tensor           # (N, F) float : attributs des nœuds donnés au modèle
    node_size: torch.Tensor   # (N,) float : rayon de chaque nœud, en pixels
    dist: torch.Tensor        # (N, N) float : plus courts chemins dans le graphe (utilisé par le stress)

    @property
    def num_nodes(self) -> int:
        return self.node_size.shape[0]


def load_graphml(path: Path, node_size: float = 4.0) -> GraphInput:
    """Lit un fichier .graphml de la base Rome et le convertit en GraphInput.

    À décider en équipe : edge_index contient-il chaque arête une fois, ou dans les deux sens ?
    """
    # TODO(rôle B) : nx.read_graphml, renuméroter les nœuds 0..N-1, construire edge_index, dist, x
    raise NotImplementedError


def split_dataset(data_dir: Path, test_ratio: float = 0.2, seed: int = 0) -> tuple[list[Path], list[Path]]:
    """Découpe la base en fichiers d'entraînement et de test, de façon reproductible.

    Indice : le nombre de nœuds est écrit dans le nom du fichier (grafoXXXX.<N>.graphml).
    """
    # TODO(rôle B)
    raise NotImplementedError
