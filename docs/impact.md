# Impact environnemental du projet

Analyse demandée pour la soutenance (1–2 slides). Ce document donne les mesures, la méthode et leurs limites ;
les slides en reprennent l'essentiel. Mesures du 22/11/2026.

## Méthode

- **Outil** : `codecarbon` 3.3.2, traceur hors ligne (aucun appel réseau), intensité carbone du réseau électrique
  français fixée par codecarbon : **56 g CO₂eq/kWh** d'après nos lignes (3,45 g pour 0,0616 kWh).
- **Comment** : chaque calcul est lancé sous `src/tools/run_tracked.py` (ou `tools.carbon.track`), sans modifier le
  script mesuré ; une ligne par calcul dans `runs/emissions.csv`. Tableau : `python src/tools/carbon.py`.
- **Périmètre** : l'électricité consommée pendant le calcul par le CPU (mode `process` : seule la part de notre
  processus sur le serveur partagé), la RAM et le GPU du calcul. **Hors périmètre** : la fabrication du matériel,
  le refroidissement de la salle (PUE pris à 1), les postes de travail, le stockage et le réseau.
- **Machine** : deepeirb (ENSEIRB), un GPU NVIDIA RTX 6000 Ada par calcul (la colonne « machine » de codecarbon
  affiche toujours les deux GPU du serveur, même quand un seul est mesuré).
- **Biais corrigé en cours de route** : par défaut, codecarbon mesure *tous* les GPU de la machine. Avant le
  correctif `fix(tools): measure only the GPUs of the job`, un calcul sur un GPU se voyait compter l'autre GPU,
  au repos (19 à 27 W mesurés). Les quatre premières lignes ci-dessous sont donc des **majorants** ; la correction
  est estimée à part.

## Mesures

| Calcul | Durée | Énergie (kWh) | CO₂eq (g) | Énergie corrigée (kWh) |
|---|---|---|---|---|
| Entraînement du modèle (phases 1 et 2) | 27 min 38 s | 0,0616 | 3,45 | 0,049 à 0,053 |
| Benchmark (6 méthodes, 1 000 graphes de test) | 5 min 55 s | 0,0191 | 1,07 | 0,016 à 0,017 |
| Balayage en résolution (16 réglages, écran et témoin) | 48 min 32 s | 0,1527 | 8,56 | 0,131 à 0,137 |
| Arêtes courbes (3 réglages) | 6 min 17 s | 0,0228 | 1,28 | 0,020 à 0,021 |
| Témoins géométriques avec marge (16 réglages) | 28 min 41 s | 0,0898 | 5,03 | (déjà corrigé) |
| **Total mesuré** | **1 h 57 min** | **0,346** | **19,4** | **0,306 à 0,318** |

Correction = durée × puissance au repos du GPU non utilisé (19 à 27 W). Ordre de grandeur : 19 g CO₂eq, c'est
environ ce qu'émet une voiture thermique sur une à deux centaines de mètres (en comptant 120 à 190 g CO₂/km).

**Non mesuré** (à garder en tête, sans pouvoir le chiffrer précisément) : les entraînements et essais antérieurs
de l'équipe sur une RTX 3060 personnelle (de nombreux entraînements d'après les slides du jalon 2 : ablations,
trois graines par réglage, variantes de CoRe-GD ; environ une heure chacun d'après `src/train.py`, donc
probablement plus que tout le tableau ci-dessus) ; un premier balayage bloqué une douzaine de minutes
(sans calcul utile) ; les lancements rapides de vérification ; les tests sur les portables.

## Énergie par dessin et seuil de rentabilité de l'entraînement

`benchmark/energy.py`, 200 graphes de test de Rome, un graphe à la fois comme un utilisateur, GPU du calcul seul
mesuré (le GPU reste alloué pendant Kamada-Kawai, qui tourne sur le CPU : sa puissance au repos est comptée).

| Méthode | Temps / dessin | Énergie / dessin (mWh) | CO₂eq / dessin (mg) |
|---|---|---|---|
| Kamada-Kawai (networkx, CPU) | 24 ms | 0,55 | 0,031 |
| Notre modèle (GPU) | 14 ms | 0,38 | 0,021 |
| Notre modèle + affinage de l'équipe (un graphe, CPU) | 174 ms | 8,47 | 0,475 |
| Affinage écran à 100 px (paquets de 64, GPU) | 14 ms | 0,51 | 0,028 |
| Affinage écran à 200 px | 10 ms | 0,31 | 0,017 |
| Affinage écran à 500 px | 10 ms | 0,29 | 0,016 |

- **Seuil de rentabilité** : le modèle économise 0,17 mWh par dessin par rapport à Kamada-Kawai. L'entraînement
  (0,049 à 0,062 kWh) n'est donc remboursé qu'après **environ 300 000 à 360 000 dessins** de graphes de la taille
  de Rome. Sur de si petits graphes, Kamada-Kawai est déjà bon marché : l'intérêt énergétique d'un modèle appris
  viendrait des grands graphes, où Kamada-Kawai coûte en O(N²) à O(N³).
- **L'affinage d'un graphe seul coûte cher** (8,5 mWh, 20 fois le modèle) : `refine.draw` le fait sur le CPU,
  où les centaines de petites opérations d'un pas sont lentes. Par paquets sur le GPU, l'affinage écran coûte
  autant qu'un dessin de Kamada-Kawai (0,3 à 0,5 mWh).

## Effet de la résolution

Le sujet annonce un coût de l'ordre de (nœuds + arêtes) × (largeur × hauteur). Notre implémentation n'évalue
les pixels que dans les boîtes où deux formes se rencontrent (`metrics/overlaps_vectorised.batched_overlap`) :
le coût suit le **nombre de défauts**, pas la surface de l'image. D'où le résultat contre-intuitif du tableau :
affiner à 100 px coûte plus (0,51 mWh) qu'à 500 px (0,29 mWh), parce qu'un petit écran a beaucoup plus de
nœuds et d'arêtes qui se touchent.

## Ce qu'on en retient

1. L'ensemble des calculs mesurés de cette partie du projet consomme environ **0,3 kWh et 19 g CO₂eq** : peu,
   grâce au mix électrique français et à des calculs courts.
2. **Mesurer correctement demande de la vigilance** : un outil standard comptait un GPU qui ne travaillait pas
   (jusqu'à 20 % d'erreur sur l'entraînement).
3. L'entraînement du modèle n'est pas rentable énergétiquement face à Kamada-Kawai sur des graphes de Rome
   (seuil autour de 300 000 dessins) ; l'affinage écran, lui, coûte autant qu'un dessin classique quand il est
   fait par paquets.
4. Le principal poste non mesuré est en amont : les heures d'essais sur machine personnelle et la fabrication
   du matériel.
