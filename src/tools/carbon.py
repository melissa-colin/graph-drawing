"""Énergie et CO2 de nos calculs, pour l'analyse d'impact environnemental demandée à la soutenance.

    from tools.carbon import track
    with track("entrainement"):
        ...

Chaque bloc ajoute une ligne à runs/emissions.csv (durée en s, énergie en kWh, CO2 en kg).
Mesure locale par codecarbon, sans réseau, avec l'intensité carbone du réseau français ; seul notre
processus est compté (serveur partagé). Sur deepeirb, lancer avec CUDA_VISIBLE_DEVICES=0 (ou 1) :
codecarbon ne compte alors que ce GPU, sinon il compte aussi l'autre, même au repos.
Pour mesurer un script entier sans le modifier :
    codecarbon monitor --offline --country-iso-code FRA --no-api -- python src/train.py
"""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def track(name, out="runs/emissions.csv"):
    from codecarbon import OfflineEmissionsTracker  # importé ici : seules les mesures en ont besoin
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tracker = OfflineEmissionsTracker(country_iso_code="FRA", project_name=name, output_dir=str(out.parent),
                                      output_file=out.name, tracking_mode="process", log_level="error")
    tracker.start()
    try:
        yield tracker
    finally:
        tracker.stop()
