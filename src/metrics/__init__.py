"""Quality metrics of a drawing, usable by the loss, the evaluation or elsewhere.

Convention: a metric is a function (layout, graph, ...) -> scalar tensor.
A metric used in the loss must be differentiable.
Evaluation rule: eval.py also uses metrics that are not in the loss (Goodhart's law).
"""
