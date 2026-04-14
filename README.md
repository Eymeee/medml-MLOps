# medml-pipeline

Projet MLOps de prédiction médicale, de l'EDA à la mise en production, prêt à être poursuivi.

## Ce que contient déjà ce starter

- structure complète du dépôt demandée dans l'atelier
- dataset synthétique médical de démarrage dans `data/raw/`
- pipeline de preprocessing `scikit-learn`
- comparaison de trois modèles obligatoires : **MLP**, **SVM**, **XGBoost**
- scripts Python modulaires
- tests unitaires et API
- squelette DVC, MLflow, Docker et GitHub Actions
- API FastAPI

## Gestion du projet avec `uv`

### Installation

```bash
uv sync --all-groups
```

### Lancer les tests

```bash
uv run pytest
uv run pytest --cov=src --cov=api --cov-report=term-missing
```

### Préprocessing, entraînement, évaluation

```bash
uv run python -m src.preprocessing
uv run python -m src.train
uv run python -m src.evaluate
```

### API locale

```bash
uv run uvicorn api.app:app --reload
```

### MLflow UI

```bash
uv run mlflow ui
```

### DVC

```bash
uv run dvc init
uv run dvc add data/raw/sample_medical_dataset.csv
uv run dvc repro
```

## Workflow recommandé

1. Remplacer le dataset synthétique par un vrai dataset médical.
2. Mettre à jour `params.yaml` avec la nouvelle cible et les nouvelles colonnes.
3. Réaliser l'EDA dans `notebooks/`.
4. Ajuster le preprocessing et les tests si le schéma change.
5. Lancer `uv run python -m src.train` pour comparer les trois modèles.
6. Démarrer l'API et valider `/health`, `/model-info` et `/predict`.
7. Versionner les données avec DVC.
8. Commiter régulièrement.

## Structure

```text
medml-pipeline/
├── data/
├── notebooks/
├── src/
├── api/
├── tests/
├── models/
├── docker/
├── dvc.yaml
├── params.yaml
├── pyproject.toml
└── README.md
```

## Dataset de démarrage

Le fichier `data/raw/sample_medical_dataset.csv` est **synthétique**. Il sert à faire tourner immédiatement le projet, mais il doit être remplacé par un dataset médical réel pour la présentation finale.

## Remarques

- `mlflow` et `dvc` sont déclarés dans `pyproject.toml`. Si l'environnement n'a pas encore été synchronisé, certaines commandes ne fonctionneront pas avant `uv sync`.
- Le code gère l'absence de modèle sauvegardé en retournant une erreur claire côté API.
