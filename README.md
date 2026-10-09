# Graph drawing différentiable en espace écran

Projet semestriel ENSEIRB-MATMECA 3A (option IA) — 2026/2027.
Équipe : Melissa Colin, Yani, Quentin. Encadrant : Romain Giot (LaBRI, équipe BKB).

## Objectif

Optimiser la position des nœuds d'un graphe **directement en espace écran** :
rendu différentiable par fonctions de distance signées (SDF 2D) → fonctions de coût
calculées sur les pixels → descente de gradient (puis, en perspective, modèle profond).
Sujet complet : [`docs/sujet/`](docs/sujet/).

## Échéances

| Date       | Jalon                                                    |
|------------|----------------------------------------------------------|
| 06/11/2026 | Jalon 1 + slides d'avancement à M. Clément               |
| 18/12/2026 | Jalon 2 + slides d'avancement à M. Clément               |
| 27/01/2027 | Soutenance en anglais + démo + 1–2 slides impact environnemental |

## Où trouver quoi

| Besoin                                         | Emplacement                         |
|------------------------------------------------|-------------------------------------|
| Tâches, qui fait quoi                          | Tickets + tableau du projet GitHub  |
| Sujet, consignes                               | `docs/sujet/`                       |
| Mails / réunions avec l'encadrant              | `docs/echanges/`                    |
| Compte rendu de séance                         | `docs/journal/`                     |
| Fiches de lecture                              | `docs/etat-de-l-art/`               |
| Questions, idées, décisions, impact env.       | `docs/notes.md`                     |
| Code                                           | `src/graphdraw/`                    |
| Exploration                                    | `notebooks/`                        |

Règles de travail : [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Installation

```bash
git clone git@github.com:melissa-colin/graph-drawing.git && cd graph-drawing
uv sync        # installe les bibliothèques listées dans pyproject.toml
```

La base Rome va dans `data/` (non versionnée).
