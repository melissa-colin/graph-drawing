# 0001 — Forge git : GitHub ou GitLab

**Date** : 2026-10-09
**Statut** : proposé (à confirmer avec M. Giot)

## Contexte
Le sujet indique « git et GitLab » dans la pile technique. L'équipe a démarré sur GitHub.
L'encadrant doit pouvoir accéder au dépôt.

## Options envisagées
1. **GitHub** (dépôt privé) — l'équipe le connaît, GitHub Projects pour le Kanban ; mais pas forcément l'outil attendu par le LaBRI.
2. **GitLab** (LaBRI ou école) — conforme au sujet ; tableaux et jalons équivalents.

## Décision
Démarrage sur GitHub. Question posée à M. Giot. Si GitLab est demandé : migration
(`git remote set-url` + import des tickets), sans impact sur l'arborescence ni les docs.

## Conséquences
- Tout le suivi (docs/) étant en Markdown dans le dépôt, il migre avec lui.
- Les tickets/tableau devront être réimportés en cas de migration.
