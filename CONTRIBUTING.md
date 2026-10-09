# Règles de travail

1. **Une tâche = un ticket.** On s'assigne avant de commencer et on déplace la carte sur le tableau
   (Backlog → À faire → En cours → En relecture → Fait).
2. **Code** : une branche par ticket, puis une merge request relue par un des deux autres
   (`Closes #12` dans la description ferme le ticket).
   **Docs** : commit direct sur `main`.
3. **Commits** : `type: description` — `feat`, `fix`, `docs`, `exp`, `chore`.
4. **Garder la trace**, un fichier par sujet dans `docs/` :
   - mail ou réunion → `echanges.md` (en haut)
   - papier lu → sa section dans `etat-de-l-art.md`
   - fin de séance, question, idée, décision, mesure d'impact → `notes.md`
5. **Pousser souvent** (`git push`) : GitHub est notre sauvegarde.
