
# Prédiction du succès d’un entretien d’embauche

Ce projet Data Science en Python vise à prédire le succès d’un entretien d’embauche à partir de données de CV (âge, diplôme, test d’anglais, expérience, etc.).

## Structure du projet
- `data/` : jeux de données (données synthétiques générées)
- `notebooks/` : notebook Jupyter complet avec toutes les étapes du projet
- `scripts/` : scripts Python (génération de données, sauvegarde du modèle)
- `requirements.txt` : dépendances Python

## Fonctionnalités réalisées
- Génération de données synthétiques équilibrées (script `scripts/genere_candidats.py`)
- Exploration, visualisation et analyse des données (notebook)
- Prétraitement avancé (imputation, encodage, scaling, gestion des variables catégorielles et numériques)
- Modélisation avec plusieurs algorithmes : Logistic Regression, Random Forest, SVC, XGBoost
- Optimisation des hyperparamètres avec Optuna (gestion des conflits de noms de paramètres)
- Comparaison des modèles (accuracy, AUC, courbe ROC, importance des variables)
- Visualisation des résultats (barplots, boxplots, courbes ROC, importance des variables)
- Sauvegarde du pipeline complet (prétraitement + modèle) avec joblib (`scripts/model/pipeline_entretien.joblib`)
- Chargement du pipeline sauvegardé et prédiction sur de nouvelles données (inférence dans le notebook)

## Installation
Installez les dépendances avec :
```bash
pip install -r requirements.txt
```

## Utilisation
1. **Génération des données** : Exécutez `python scripts/genere_candidats.py` pour générer un jeu de données équilibré dans `data/`.
2. **Exploration et modélisation** : Ouvrez le notebook `notebooks/prediction_entretien_embauche.ipynb` et suivez les étapes : exploration, prétraitement, modélisation, validation, visualisation.
3. **Optimisation** : L’optimisation des hyperparamètres et la comparaison des modèles sont automatisées dans le notebook.
4. **Sauvegarde et inférence** : Le pipeline complet est sauvegardé et peut être rechargé pour faire des prédictions sur de nouveaux candidats (voir la dernière section du notebook).


## Exemple d’inférence dans le notebook
Dans le notebook, une cellule permet de charger le pipeline sauvegardé et de prédire le résultat pour un nouveau candidat ou un fichier CSV externe.


## API FastAPI professionnelle pour la prédiction
Une API FastAPI avancée est disponible dans `scripts/api_entretien.py` pour exposer le modèle de prédiction.

### Fonctionnalités de l’API
- Prédiction unique (`/predict`) et batch (`/predict_batch`)
- Validation stricte des entrées (types, bornes, exemples)
- Réponses structurées et typées
- Documentation Swagger enrichie et interactive (exemples, descriptions, modèles de réponse)

### Lancer l’API
```bash
uvicorn scripts.api_entretien:app --reload
```

Accédez à la documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)

### Exemple d’appel API (prédiction unique)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "age": 30,
           "diplome": "BTS",
           "note_anglais": 85,
           "experience": 5,
           "entreprises_precedentes": 2,
           "distance_km": 4.5,
           "score_entretien": 8.2,
           "score_competence": 7.5,
           "score_personnalite": 80,
           "sexe": "F"
         }'
```

Réponse :
```json
{
  "prediction": 1,
  "probabilite_retenu": 0.87
}
```

### Prédiction batch
Utilisez l’endpoint `/predict_batch` avec une liste de candidats (voir Swagger UI pour le format exact et les exemples interactifs).

### Swagger : documentation interactive
La documentation Swagger (OpenAPI) est enrichie avec :
- Des exemples de requêtes et réponses pour chaque endpoint
- Des descriptions détaillées des champs et des modèles
- La possibilité de tester l’API directement depuis l’interface web

> **Astuce :** Pour générer un client Python ou TypeScript à partir du Swagger, utilisez l’URL `/openapi.json`.

## Auteurs
Projet réalisé par mon github [Victoire Kasende](https://github.com/VictoryKasende/).
