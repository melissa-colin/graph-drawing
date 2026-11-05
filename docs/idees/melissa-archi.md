# Archi : ce qu'on a testé, ce qui reste

Modèle actuel : GNN récurrent de CoRe-GD (GRU + recâblage kNN dans le dessin) avec notre tête soft-argmax.
Entrées : bruit aléatoire, degré, constante 1, 8 PE Laplaciennes.

## Résultats (novembre 2026)

Tests courts : 3 000 graphes d'entraînement, 8 à 10 epochs, validation sur 500 graphes, loss `tmp_loss.loss_v0`.
Une même config relancée varie d'environ 0.03 : seuls les gros écarts comptent.

| Config | Meilleure val (2 graines) |
|---|---|
| itérations aléatoires + bruit aléatoire + clipping | **0.189 / 0.246** |
| itérations aléatoires + clipping | 0.325 / 0.313 |
| itérations aléatoires, sans clipping | 0.237 / 0.328 |
| idem + beacons | 0.290 / 0.322 |
| base (avec beacons) | 0.391 / 0.363 |
| conv GIN au lieu de GRU | 0.467 |
| valider avec 10 itérations | 0.393 |
| sans les colonnes degré et constante | 0.369 |

Gardé : itérations aléatoires à l'entraînement, colonne de bruit retirée à chaque batch, clipping des gradients.
Retiré du code : beacons (aussi testés en entraînement complet : val 0.244 contre 0.186 sans), GIN.

## Reste à essayer

- [ ] Entraînement complet avec la config gardée (comparer à 0.186, modèle sans beacons).
- [ ] Les dessins n'occupent pas tout le cadre : regarder ce que pénalise `frame_stress`.
- [ ] Validation très bruitée d'une epoch à l'autre : scheduler ReduceLROnPlateau comme CoRe-GD.
- [ ] Replay buffer de CoRe-GD.
- [ ] Sauvegarder le meilleur checkpoint pendant l'entraînement, pas seulement à la fin.

## Réduire la mémoire de la tête soft-argmax

Aujourd'hui : 250 000 scores par nœud (carte 500×500), 1 Mo par nœud. Le batch est bloqué à 4 sur la carte de 4 Go.

- [ ] Tête séparable : 500 scores pour x + 500 pour y, avec deux softmax. La position ne dépend que des deux profils, donc même sortie, mémoire divisée par 250 et 16 M → ~64 k paramètres.
- [ ] Grille plus grossière (`grid_w`, `grid_h` existent déjà, ex. 100×100).
- [ ] Précision mixte (`torch.autocast`, bf16).
- [ ] Accumulation de gradient : simuler un batch de 16 avec 4 × 4.
