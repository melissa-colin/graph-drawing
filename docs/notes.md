# Notes du projet

Journal des séances, questions, idées, décisions, impact environnemental.
**Le plus récent en haut** de chaque section.

## Journal des séances

Copier ce bloc en haut de la section à chaque fin de séance (10 min max) :

```
### AAAA-MM-JJ — rédigé par …
- Fait : Melissa … / Yani … / Quentin …
- Résultats / observations :
- Blocages :
- Décisions (→ aussi dans « Décisions » si importante) :
- Pour la prochaine fois : … (qui, ticket #…)
```

## Questions pour M. Giot

### Ce qu'on comprend du sujet
- …

### Ce qu'on ne comprend pas (encore)
- …

### À poser (liste du 16/10, reprise dans la proposition `docs/idees/melissa-2026-10-09.pdf`)

Orientation
- [ ] Modèle GNN qui prédit directement les positions écran, sans supervision, avec coûts calculés sur l'image rendue : orientation OK, ou commencer par l'optimisation directe d'un graphe ?
- [ ] Entrées au départ : structure (nœuds, arêtes) + taille des nœuds seulement, ou aussi épaisseurs, couleurs, formes ?
- [ ] Taille des graphes visée (nombre de nœuds), résolution d'image, temps de calcul acceptable ?

Évaluation
- [ ] Métriques et protocole recommandés pour comparer un dessin optimisé en espace écran à un dessin géométrique ?
- [ ] Méthodes de référence à comparer en priorité ? Implémentations réutilisables ?

Organisation et rendus
- [ ] Attendus aux jalons du 06/11 et du 18/12 ? État de l'art écrit attendu, ou seulement à l'oral ?
- [ ] Livrables finaux : code seul, notebook de démo, rapport ?
- [ ] Accès aux exemples de code mentionnés dans le sujet ?
- [ ] GPU disponibles (LaBRI, école) ou machines perso ?
- [ ] GitLab obligatoire, ou dépôt GitHub privé avec accès pour vous ?

### Tranchées par l'équipe (pas besoin de lui demander)
- On reste en 2D.
- Lien SDF / graphes, convolution en sortie, force-repulsion vs stress : à creuser nous-mêmes dans l'état de l'art.
- Pas de supervision : entraînement directement sur la loss (préentraînement par le stress, puis coûts écran).
- Répartition : 2 sur le pipeline (dont Melissa sur l'architecture), 1 sur la loss et l'évaluation.

Quand les questions partent : noter la date ici, et la réponse dans `echanges.md`.

## Idées

| Date | Idée | Auteur | Statut (à creuser / ticket #… / abandonnée + pourquoi) |
|------|------|--------|--------------------------------------------------------|
|      |      |        |                                                        |

## Décisions

Format : date — décision — pourquoi — alternatives écartées.

- **2026-10-09 — GitHub plutôt que GitLab (provisoire)** — l'équipe démarre sur GitHub ; le sujet mentionne GitLab → question posée à M. Giot. Si GitLab est demandé : migration (`git remote set-url` + import des tickets), les docs suivent sans effort.

## Impact environnemental

Obligatoire : 1–2 slides à la soutenance. On mesure dès le début.

- Énergie / émissions des calculs (optimisations, entraînements) : `codecarbon` est dans les dépendances de dev.
- Matériel utilisé (CPU/GPU, machine perso vs serveur) et durée.
- Rappel du sujet : coût ∝ (nb nœuds + nb arêtes) × (largeur × hauteur de l'image) → effet de la résolution à quantifier.
- Cours sur l'impact environnemental de l'IA prévu nov./déc. → y reporter les éléments utiles.

| Date | Expérience | Machine | Durée | Énergie (kWh) | CO₂eq (g) |
|------|------------|---------|-------|---------------|-----------|
|      |            |         |       |               |           |
