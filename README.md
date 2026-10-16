# Graph drawing différentiable en espace écran

Projet semestriel ENSEIRB-MATMECA 3A (option IA) — 2026/2027.
Équipe : Melissa Colin, Yani, Quentin. Encadrant : Romain Giot (LaBRI, équipe BKB).

## Objectif

Optimiser la position des nœuds d'un graphe **directement en espace écran** :
rendu différentiable par fonctions de distance signées (SDF 2D) → fonctions de coût
calculées sur les pixels → descente de gradient (puis, en perspective, modèle profond).
Sujet complet : [`docs/sujet.pdf`](docs/sujet.pdf).

## Échéances

| Date       | Jalon                                                    |
|------------|----------------------------------------------------------|
| 06/11/2026 | Jalon 1 + slides d'avancement à M. Clément               |
| 18/12/2026 | Jalon 2 + slides d'avancement à M. Clément               |
| 27/01/2027 | Soutenance en anglais + démo + 1–2 slides impact environnemental |

## Où trouver quoi

| Besoin                                   | Fichier                                  |
|------------------------------------------|------------------------------------------|
| Tâches, qui fait quoi                    | [Tableau Kanban](https://github.com/users/melissa-colin/projects/4) + [tickets](https://github.com/melissa-colin/graph-drawing/issues) |
| Sujet du projet (M. Giot)                | [`docs/sujet.pdf`](docs/sujet.pdf)       |
| Consignes du module (M. Clément)         | [`docs/consignes.md`](docs/consignes.md) |
| Historique des échanges                  | [`docs/echanges.md`](docs/echanges.md)   |
| État de l'art                            | [`docs/etat-de-l-art.md`](docs/etat-de-l-art.md) |
| Journal des séances, questions, idées, décisions, impact env. | [`docs/notes.md`](docs/notes.md) |
| Propositions détaillées (PDF + source LaTeX) | [`docs/idees/`](docs/idees/) |
| Code                                     | `src/graphdraw/`                         |

## Installation

```bash
git clone git@github.com:melissa-colin/graph-drawing.git && cd graph-drawing
```

La base Rome va dans `data/` (non versionnée).
