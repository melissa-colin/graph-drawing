import copy
import random
import zlib

import networkx as nx
import numpy
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

# -----------------------------------------------------------------------------------------------------------
# ------------------------------- PRIVATE TOOLS -------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------
def _pack_distances(graph):
    """Cache format: dist_index is dropped (16 bytes per pair, most of the file) and dist_value becomes the
    whole upper triangle i < j, in torch.triu_indices order, with inf for the pairs that were left out."""
    n = graph.num_nodes
    triangle = torch.full((n * (n - 1) // 2,), float("inf"))
    i, j = graph.dist_index
    triangle[i * n - i * (i + 1) // 2 + (j - i - 1)] = graph.dist_value  # position of (i, j) in triu order
    packed = copy.copy(graph)
    del packed.dist_index
    packed.dist_value = triangle
    return packed


def _unpack_distances(graph):
    """Inverse of _pack_distances: rebuild dist_index and keep only the finite pairs (0 is impossible for i < j)."""
    pairs = torch.triu_indices(graph.num_nodes, graph.num_nodes, offset=1)
    useful = torch.isfinite(graph.dist_value)
    graph.dist_index = pairs[:, useful]
    graph.dist_value = graph.dist_value[useful]
    return graph



# -----------------------------------------------------------------------------------------------------------
# ------------------------------- CLASS DEFINITION ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------

class GraphInput(Data):
    """A graph ready for the model, batchable by torch_geometric.loader.DataLoader.

    Every argument has a default: PyG rebuilds empty GraphInput objects when it collates or separates a batch.
    num_nodes comes from Data (inferred from x).
    """

    def __init__(self, name=None, node_ids=None, edge_index=None, x=None, node_size=None, dist=None,
                 edge_width=1.0, pe_dim=0, **kwargs):
        """dist: (N, N) float, graph distances, inf if no path exists. Only the useful pairs are kept."""
        super().__init__(x=x, edge_index=edge_index, **kwargs)
        self.name = name
        self.node_ids = node_ids
        self.node_size = node_size
        if dist is not None:
            useful = torch.ones_like(dist, dtype=torch.bool).triu(diagonal=1) & torch.isfinite(dist) & (dist > 0)
            self.dist_index = useful.nonzero().t()
            self.dist_value = dist[self.dist_index[0], self.dist_index[1]]
        self.edge_width = float(edge_width) 
        self.pe_dim = int(pe_dim)

    def __inc__(self, key, value, *args, **kwargs):
        # est appelé par PyG pendant le batching : colle les graphes en ajoutant num_nodes à dist_index pour que les indices restent corrects
        if key == "dist_index":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    @property
    def edge_index_bidirectional(self):
        """Edges in both directions: (u, v) and (v, u), for directed message-passing usage."""
        return to_undirected(self.edge_index, num_nodes=self.num_nodes)



# -----------------------------------------------------------------------------------------------------------
# ----------------------------------- PUBLIC TOOLS ----------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------


def laplacian_pe(graph, node_ids, k):
    """Laplacian positional encodings: the k smoothest NON-constant eigenvectors of the normalized
    Laplacian, one row per node in node_ids order. Returns a (N, k) float tensor.

    Smooth modes give each node a coordinate that respects the graph structure: nodes close in the
    graph get close values. Small graphs (N - 1 < k) are padded with zeros up to k columns.
    """
    L = nx.normalized_laplacian_matrix(graph, nodelist=node_ids).toarray()
    _, eigenvectors = numpy.linalg.eigh(L)  # sorted by eigenvalue, smoothest first
    pe = eigenvectors[:, 1 : k + 1]  # drop the first one, the trivial mode
    if pe.shape[1] < k:
        pe = numpy.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return torch.from_numpy(pe).float()


def redraw_noise(graph, generator=None):
    """Training-time augmentation, as in CoRe-GD: new uniform noise in the first column of x at every batch.

    The noise lets the model tell apart nodes that the other features cannot (symmetric nodes); since it
    changes every batch, the model cannot rely on any particular value.
    Returns a NEW GraphInput (the original graph is left untouched); only column 0 changes.
    """
    redrawn = copy.copy(graph)  # shallow: the other tensors are shared, not duplicated
    redrawn.x = graph.x.clone()
    redrawn.x[:, 0] = torch.rand(graph.num_nodes, generator=generator)
    return redrawn


def flip_pe_signs(graph, generator=None):
    """Training-time augmentation: multiplies each positional-encoding column of x by a random +1 or -1.

    An eigenvector v and -v are equally valid, so the model must not depend on the sign it receives.
    Returns a NEW GraphInput (the original graph is left untouched); only the last pe_dim columns change.
    """
    pe_dim = int(graph.pe_dim)  # a 1-element tensor once separated from a PyG batch
    if pe_dim == 0:
        return graph
    signs = torch.randint(0, 2, (pe_dim,), generator=generator).float() * 2 - 1
    flipped = copy.copy(graph)  # shallow: the other tensors are shared, not duplicated
    flipped.x = graph.x.clone()
    flipped.x[:, -pe_dim:] *= signs
    return flipped


def load_graphml(path, node_size=4.0, edge_width=1.0, pe_dim=8):
    """
    Charge un graphe au format GraphML et le convertit en un objet GraphInput une seule fois et le sauvegarde en cache.

    On numérote les noeds de 0 à N-1 et les arêtes dans un tableau de 2 lignes (haut = source, bas = target).
    Le x contient 3 colonnes : 
    - du bruit uniforme (pour différencier les noeuds symétriques)
    - le degré du noeud (nombre de voisins)
    - une constante 1 (pour que le modèle puisse apprendre un biais)
    Auquel on ajoute 8 encodage de position (les 8 vecteurs propres les plus lisses du Laplacien normalisé).

    On calcule aussi la distance entre chaque paire de noeuds (inf si pas de chemin) le nombre d'arêtes du plus court chemin entre chaque paire de nœuds. La loss s'en sert pour comparer avec les distances dessinées.

    L'objet GraphInput hérite de Data (l'objet graphe de PyG) et contient les informations suivantes :
    - name : le nom du graphe (le nom du fichier sans l'extension)
    - node_ids : la liste des identifiants des noeuds (dans l'ordre des lignes de x)
    - edge_index : les arêtes du graphe (2, E)
    - x : les features des noeuds (N, F)
    - node_size : le rayon de chaque noeud en pixels (N,)
    - dist_index : les paires de noeuds i < j avec 0 < d < inf (2, P)
    - dist_value : la distance entre les paires de noeuds (P,)
    - edge_width : l'épaisseur des arêtes en pixels
    - pe_dim : le nombre de colonnes d'encodage de position à la fin de x (0 si aucun)
    """
    graph = nx.read_graphml(path)
    node_ids = [str(node) for node in graph.nodes()]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    edges = [(node_index[u], node_index[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
    # Column 0 is random noise (see redraw_noise), seeded by the graph name so that evaluation is reproducible.
    generator = torch.Generator().manual_seed(zlib.crc32(path.stem.encode()))
    features = torch.column_stack((
        torch.rand(len(node_ids), generator=generator),
        torch.tensor([graph.degree(node) for node in node_ids], dtype=torch.float32),
        torch.ones(len(node_ids), dtype=torch.float32),
    ))
    distances = torch.from_numpy(nx.floyd_warshall_numpy(graph, nodelist=node_ids).astype("float32"))
    if pe_dim > 0:  # last, flip_pe_signs works on the last pe_dim columns
        features = torch.cat((features, laplacian_pe(graph, node_ids, pe_dim)), dim=1)
    return GraphInput(
        name=path.stem,
        node_ids=node_ids,
        edge_index=edge_index,
        x=features,
        node_size=torch.full((len(node_ids),), float(node_size), dtype=torch.float32),
        dist=distances,
        edge_width=float(edge_width),
        pe_dim=pe_dim,
    )


def load_dataset(data_dir, node_size=4.0, edge_width=1.0, pe_dim=8):
    """Load every .graphml of data_dir, keyed by file stem, through a cache file in data_dir.

    The first call parses all the files (slow: shortest paths and eigenvectors for each graph) and saves
    the result; the next calls just reload it. The cache name holds the parameters, and the cache is rebuilt
    if the set of .graphml files changed. Delete the cache file to force a rebuild after changing load_graphml.
    """
    files = sorted(data_dir.glob("*.graphml"))
    names = [path.stem for path in files]
    cache_path = data_dir / f"cache_size{node_size:g}_width{edge_width:g}_pe{pe_dim}_noise.pt"
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)  # our own file: unpickling GraphInput is safe
        if cache["names"] == names:
            return {name: _unpack_distances(graph) for name, graph in zip(names, cache["graphs"])}
        print(f"{cache_path.name}: the .graphml files changed, rebuilding the cache.")

    graphs = [load_graphml(path, node_size, edge_width, pe_dim) for path in tqdm(files, desc="Parsing graphs", unit="graph")]
    torch.save({"names": names, "graphs": [_pack_distances(graph) for graph in graphs]}, cache_path)
    return dict(zip(names, graphs))


def split_dataset(data_dir, val_ratio=0.1, test_ratio=0.2, seed=0):
    """Split into disjoint train/val/test sets with a reproducible shuffle.

    The ratios are applied on the whole dataset, with train = remaining files after
    reserving validation and test sets. The three sets are disjoint and deterministic.
    """
    files = sorted(data_dir.glob("*.graphml"))
    if not files:
        return [], [], []

    shuffled = files[:]
    random.Random(seed).shuffle(shuffled)

    n_val = int(round(len(shuffled) * val_ratio))
    n_test = int(round(len(shuffled) * test_ratio))
    n_val = max(0, min(len(shuffled), n_val))
    n_test = max(0, min(len(shuffled) - n_val, n_test))

    val_start = 0
    val_end = n_val
    test_start = n_val
    test_end = n_val + n_test
    train_end = len(shuffled)

    train_files = shuffled[test_end:train_end]
    val_files = shuffled[val_start:val_end]
    test_files = shuffled[test_start:test_end]
    return train_files, val_files, test_files
