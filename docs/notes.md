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

### Organisation
- [ ] GitHub ou GitLab ? Si GitLab : lequel (LaBRI, école) et comment vous donner accès ?
- [ ] Qu'attendez-vous pour le jalon du 06/11 ? et celui du 18/12 ?
- [ ] Livrables finaux attendus (code, rapport, notebook de démo…) ?
- [ ] Le sujet mentionne des « exemples de code » : comment y accéder ?
- [ ] Ressources de calcul : GPU disponibles (LaBRI, école) ou machines perso ?
- [ ] Format préféré pour les points d'avancement (mail hebdo, visio…) ?

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
