#!/usr/bin/env bash
# Configure le dépôt GitHub : étiquettes, jalons, tickets de démarrage, tableau Kanban.
# Usage : scripts/setup_github.sh <owner/repo>
# À lancer une seule fois, après création du dépôt distant.
set -euo pipefail

REPO="${1:?usage: $0 owner/repo}"
OWNER="${REPO%%/*}"

echo "== Étiquettes"
label() { gh label create "$1" --repo "$REPO" --color "$2" --description "$3" --force; }
label rendu-sdf          "1f77b4" "Rasterisation différentiable par SDF"
label cout-ecran         "ff7f0e" "Fonctions de coût en espace écran"
label optimisation       "2ca02c" "Plongement par descente de gradient"
label modele-profond     "9467bd" "Plongement par réseau de neurones"
label evaluation         "8c564b" "Métriques et comparaison aux méthodes classiques"
label biblio             "e377c2" "Lecture, état de l'art"
label orga               "7f7f7f" "Gestion de projet, réunions, slides"
label impact-env         "17becf" "Analyse d'impact environnemental"
label question-encadrant "d62728" "Question à poser à M. Giot"

echo "== Jalons"
milestone() {
  gh api "repos/$REPO/milestones" -f title="$1" -f due_on="$2" -f description="$3" >/dev/null \
    || echo "  (jalon '$1' déjà présent ?)"
}
milestone "Jalon 1"     "2026-11-06T23:59:59Z" "Jalon 1 + slides d'avancement à M. Clément"
milestone "Jalon 2"     "2026-12-18T23:59:59Z" "Jalon 2 + slides d'avancement à M. Clément"
milestone "Soutenance"  "2027-01-27T08:00:00Z" "Soutenance en anglais + démo + impact environnemental"

echo "== Tickets de démarrage"
issue() { gh issue create --repo "$REPO" --title "$1" --label "$2" --milestone "Jalon 1" --body "$3" >/dev/null; echo "  + $1"; }

issue "Survol des références + liste compris / pas compris" "biblio,orga" \
"Consigne de M. Giot (mail du 08/10) : regarder les références/bibliothèques du sujet sans tout comprendre, et lister ce qu'on comprend / ne comprend pas.
- [ ] Remplir docs/idees/questions-encadrant.md
- [ ] Décider : réunion la semaine prochaine ou envoi des questions par mail"

issue "Envoyer les questions à M. Giot" "orga,question-encadrant" \
"Partir de docs/idees/questions-encadrant.md. Consigner la réponse dans docs/echanges/."

for ref in \
  "DL4GD|giovannangeli2022dl4gd" \
  "FORBID|giovannangeli2022forbid" \
  "(GD)²|ahmed2020gd2" \
  "Graph drawing by SGD (Zheng et al.)|zheng2018sgd" \
  "SmartGD|wang2023smartgd" \
  "DeepGD|wang2021deepgd" \
  "DeepFD|zhang2024deepfd" \
  "SDF 2D d'Iñigo Quilez|quilez2019sdf"; do
  title="${ref%%|*}"; key="${ref##*|}"
  issue "Lecture : $title" "biblio" \
"Clé BibTeX : \`$key\`
- [ ] Fiche docs/etat-de-l-art/fiches/$key.md
- [ ] Ligne à jour dans docs/etat-de-l-art/README.md"
done

issue "Récupérer la base de graphes Rome" "evaluation" \
"Trouver la source, documenter le téléchargement dans le README, stocker dans data/ (non versionné)."

issue "Mettre en place l'environnement (uv sync, PyTorch, CPU/GPU)" "orga" \
"Chacun vérifie que \`uv sync && uv run pytest\` fonctionne sur sa machine."

issue "Mettre en place la mesure d'impact (codecarbon)" "impact-env" \
"Voir docs/impact-env/README.md. Objectif : chaque expérience logue sa consommation."

issue "Préparer les slides du jalon 1" "orga" \
"Attendus à confirmer avec M. Giot. Envoyer à M. Clément."

echo "== Tableau Kanban"
gh auth refresh -s project >/dev/null 2>&1 || true
PROJECT_NUM=$(gh project create --owner "$OWNER" --title "Graph drawing différentiable" --format json --jq .number)
gh project link "$PROJECT_NUM" --owner "$OWNER" --repo "$REPO"
for n in $(gh issue list --repo "$REPO" --limit 100 --json number --jq '.[].number'); do
  gh project item-add "$PROJECT_NUM" --owner "$OWNER" --url "https://github.com/$REPO/issues/$n" >/dev/null
done

echo
echo "Fait. Dernière étape manuelle (l'API ne gère pas encore les colonnes) :"
echo "  ouvrir le tableau → vue Board → colonnes : Backlog, À faire, En cours, En relecture, Fait"
gh project view "$PROJECT_NUM" --owner "$OWNER" --web >/dev/null 2>&1 || true
