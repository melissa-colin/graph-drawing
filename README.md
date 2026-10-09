# Graph drawing différentiable en espace écran

Projet semestriel ENSEIRB-MATMECA 3A (option IA) — 2026/2027.
Équipe : Melissa Colin, Yani, Quentin. Encadrant : Romain Giot (LaBRI, équipe BKB).

## Objectif

Optimiser la position des nœuds d'un graphe **directement en espace écran** :
rendu différentiable par fonctions de distance signées (SDF 2D) → fonctions de coût
calculées sur les pixels → descente de gradient (puis, en perspective, modèle profond).
Sujet complet : [`docs/sujet/`](docs/sujet/).

## Échéances

| Date       | Jalon                                         |
|------------|-----------------------------------------------|
| 06/11/2026 | Jalon 1 (+ slides d'avancement à M. Clément)  |
| 18/12/2026 | Jalon 2 (+ slides d'avancement)               |
| 27/01/2027 | Soutenance (anglais) + démo + impact environnemental |

## Où trouver quoi

| Besoin                                  | Emplacement                          |
|-----------------------------------------|--------------------------------------|
| Tâches, qui fait quoi, avancement       | Tickets + tableau du projet GitHub   |
| Sujet, consignes                        | `docs/sujet/`                        |
| Mails / réunions avec l'encadrant       | `docs/echanges/`                     |
| Compte rendu de chaque séance           | `docs/journal/`                      |
| Choix techniques et leur justification  | `docs/decisions/`                    |
| État de l'art (fiches de lecture)       | `docs/etat-de-l-art/`                |
| Idées, questions ouvertes               | `docs/idees/`                        |
| Slides de jalons                        | `docs/jalons/`                       |
| Impact environnemental                  | `docs/impact-env/`                   |
| Code                                    | `src/graphdraw/`                     |
| Exploration                             | `notebooks/`                         |
| Expériences (configs + résultats)       | `experiments/`                       |

Règles de travail : [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Installation

```bash
git clone <url-du-depot> && cd graph-drawing
uv sync              # crée .venv et installe les dépendances
uv run pytest        # vérifie que tout tourne
```

Les données (base Rome) vont dans `data/` et ne sont pas versionnées.

## Structure du code

```
src/graphdraw/
├── sdf/      # rendu différentiable : SDF des nœuds et arêtes → cartes d'activation
├── losses/   # fonctions de coût en espace écran (recouvrement, collisions, lisibilité…)
├── optim/    # plongement par descente de gradient
├── models/   # plongement par réseau de neurones (perspective)
└── eval/     # métriques classiques + comparaison aux méthodes de référence
```
