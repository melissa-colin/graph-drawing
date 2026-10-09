# Règles de travail

Objectif : peu de règles, mais tenues par les trois.

## 1. Tout passe par un ticket

- Une tâche = un ticket (modèles : *Tâche*, *Fiche de lecture*, *Bug*).
- On s'assigne le ticket **avant** de commencer, puis on le déplace sur le tableau :
  `Backlog → À faire → En cours → En relecture → Fait`.
- Chaque ticket a une étiquette d'axe et, si possible, un jalon.

| Étiquette        | Sens                                              |
|------------------|---------------------------------------------------|
| `rendu-sdf`      | rasterisation différentiable par SDF              |
| `cout-ecran`     | fonctions de coût en espace écran                 |
| `optimisation`   | plongement par descente de gradient               |
| `modele-profond` | plongement par réseau de neurones                 |
| `evaluation`     | métriques, comparaison aux méthodes classiques    |
| `biblio`         | lecture, état de l'art                            |
| `orga`           | gestion de projet, réunions, slides               |
| `impact-env`     | analyse d'impact environnemental                  |
| `question-encadrant` | question à poser à M. Giot                    |

## 2. Branches et merge requests

- `main` est toujours dans un état qui fonctionne.
- Une branche par ticket : `<num>-description-courte` (ex. `12-sdf-segment`).
- **Code** : merge request obligatoire, relue par au moins un des deux autres.
- **Docs** (journal, échanges, fiches) : commit direct sur `main` autorisé.
- Dans la description de la MR : `Closes #12` pour fermer le ticket automatiquement.

## 3. Messages de commit

Format : `type: description courte` — types : `feat`, `fix`, `docs`, `test`, `refactor`, `exp` (expérience), `chore`.

```
feat: SDF d'un segment avec épaisseur
docs: CR séance du 10/10
exp: optimisation recouvrement sur rome/grafo113
```

## 4. Garder la trace

| Événement                         | Ce qu'on fait                                                     |
|-----------------------------------|-------------------------------------------------------------------|
| Mail ou réunion avec l'encadrant  | fichier dans `docs/echanges/` (copier `_modele.md`)               |
| Fin de séance du vendredi         | fichier dans `docs/journal/` (copier `_modele.md`) — 10 min max  |
| Choix technique non évident       | ADR dans `docs/decisions/` (copier `_modele.md`)                  |
| Papier lu                         | fiche dans `docs/etat-de-l-art/fiches/` + entrée dans `biblio.bib` |
| Idée / question ouverte           | `docs/idees/` ou ticket `question-encadrant`                      |

Nommage des fichiers datés : `AAAA-MM-JJ-sujet.md` (tri chronologique automatique).

## 5. Expériences

- Une expérience = une config dans `experiments/configs/` + résultats dans `experiments/results/<date>-<nom>/`.
- On note les résultats marquants dans le journal de séance.
- Les notebooks servent à explorer ; ce qui est réutilisé migre dans `src/graphdraw/`.
