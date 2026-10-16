"""Métriques de qualité d'un dessin, utilisables par la loss, l'évaluation ou ailleurs.

Convention : une métrique est une fonction (layout, graph, ...) -> tenseur scalaire.
Une métrique utilisée dans la loss doit être différentiable.
Règle d'évaluation : eval.py utilise aussi des métriques absentes de la loss (loi de Goodhart).
"""
