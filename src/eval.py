"""Évaluation d'un modèle entraîné. Lancer depuis la racine du dépôt : python src/eval.py

Règle : inclure des métriques absentes de la loss, et comparer aux méthodes de référence
(SGD, Kamada-Kawai, FORBID...) sur les mêmes graphes, mesurées sur le rendu final.
"""

from pathlib import Path


def evaluate(model_path: Path, test_files: list[Path]) -> dict[str, float]:
    """Renvoie la moyenne de chaque métrique sur les graphes de test, pour le modèle et les références."""
    # TODO(rôle C)
    raise NotImplementedError


if __name__ == "__main__":
    # TODO(rôle C)
    pass
