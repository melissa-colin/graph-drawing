# Règles de travail

1. **Une tâche = un ticket.** On s'assigne avant de commencer et on déplace la carte sur le tableau
   (Backlog → À faire → En cours → En relecture → Fait).
2. **Code** : une branche par ticket, puis une merge request relue par un des deux autres
   (`Closes #12` dans la description ferme le ticket).
   **Docs** : commit direct sur `main`.
3. **Commits** : `type: description` — `feat`, `fix`, `docs`, `exp`, `chore`.
4. **Garder la trace** :
   - mail ou réunion → `docs/echanges/AAAA-MM-JJ-sujet.md` (copier `_modele.md`)
   - fin de séance → `docs/journal/AAAA-MM-JJ.md` (copier `_modele.md`, 10 min max)
   - papier lu → `docs/etat-de-l-art/fiches/<clé>.md` + `biblio.bib`
   - question, idée, décision, mesure d'impact → `docs/notes.md`
5. **Pousser souvent** (`git push`) : GitHub est notre sauvegarde.
