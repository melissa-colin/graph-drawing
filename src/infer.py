"""Dessiner un nouveau graphe avec un modèle entraîné.
Lancer depuis la racine du dépôt : python src/infer.py <modele.pt> <graphe.graphml> <sortie.png>
"""

from pathlib import Path

from data.output import Layout


def infer(model_path: Path, graphml_path: Path) -> Layout:
    """Charge le modèle et le graphe, renvoie le dessin."""
    # TODO(rôle A)
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(rôle A) : lire les arguments, appeler infer, enregistrer avec data.output.save_png
    pass
