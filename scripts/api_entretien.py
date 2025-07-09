
# =====================
# API Prédiction Entretien d'Embauche (FastAPI)
# =====================
"""
API professionnelle pour la prédiction du succès d’un entretien d’embauche à partir de données de CV.

Fonctionnalités :
- Prédiction unique ou batch
- Documentation Swagger enrichie
- Validation stricte des entrées (Pydantic)
- Réponses structurées et exemples interactifs
"""

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field, constr
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from typing import List, Optional
import os

app = FastAPI(
    title="API Prédiction Entretien d'Embauche",
    description="""
API professionnelle pour prédire le succès d’un entretien d’embauche à partir de données de CV.

**Endpoints :**
- `POST /predict` : Prédiction pour un candidat
- `POST /predict_batch` : Prédiction pour plusieurs candidats

Documentation interactive : `/docs`
""",
    version="1.0.0"
)

# Chargement du pipeline sauvegardé (joblib)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "pipeline_entretien.joblib")
pipeline = joblib.load(MODEL_PATH)

class Candidat(BaseModel):
    age: int = Field(..., example=30, ge=15, le=70, description="Âge du candidat")
    diplome: str = Field(..., example="BTS", description="Niveau de diplôme")
    note_anglais: float = Field(..., example=85, ge=0, le=100, description="Score au test d'anglais")
    experience: int = Field(..., example=5, ge=0, le=50, description="Années d'expérience")
    entreprises_precedentes: Optional[int] = Field(0, example=2, ge=0, le=20, description="Nombre d'entreprises précédentes")
    distance_km: Optional[float] = Field(0.0, example=4.5, ge=0, le=1000, description="Distance domicile-entreprise (km)")
    score_entretien: Optional[float] = Field(0.0, example=8.2, ge=0, le=10, description="Score d'entretien sur 10")
    score_competence: Optional[float] = Field(0.0, example=7.5, ge=0, le=10, description="Score de compétence sur 10")
    score_personnalite: Optional[float] = Field(0.0, example=80, ge=0, le=100, description="Score de personnalité")
    sexe: constr(strip_whitespace=True, min_length=1) = Field(..., example="F", description="Sexe du candidat (M/F)")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., example=1, description="1 = retenu, 0 = non retenu")
    probabilite_retenu: float = Field(..., example=0.87, description="Probabilité d'être retenu")

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédire le succès d'un entretien",
    tags=["Prédiction"],
    response_description="Résultat de la prédiction pour un candidat",
    description="Prédisez si un candidat sera retenu à partir de ses caractéristiques."
)
async def predict_candidat(
    candidat: Candidat = Body(..., example={
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
    })
):
    """Prédiction pour un candidat unique."""
    data = pd.DataFrame([candidat.dict()])
    prediction = pipeline.predict(data)[0]
    proba = pipeline.predict_proba(data)[0][1]
    return PredictionResponse(prediction=int(prediction), probabilite_retenu=round(float(proba), 4))

@app.post(
    "/predict_batch",
    response_model=List[PredictionResponse],
    summary="Prédire pour plusieurs candidats",
    tags=["Prédiction"],
    response_description="Résultat de la prédiction pour plusieurs candidats",
    description="Prédisez pour une liste de candidats (batch)."
)
async def predict_batch(
    candidats: List[Candidat] = Body(..., example=[{
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
    }])
):
    """Prédiction batch pour plusieurs candidats."""
    data = pd.DataFrame([c.dict() for c in candidats])
    predictions = pipeline.predict(data)
    probas = pipeline.predict_proba(data)[:, 1]
    return [PredictionResponse(prediction=int(pred), probabilite_retenu=round(float(proba), 4)) for pred, proba in zip(predictions, probas)]

@app.get("/", tags=["Accueil"], summary="Accueil de l'API", response_class=HTMLResponse)
def root():
    """Page d'accueil HTML simple pour l'API."""
    return """
    <html>
        <head>
            <title>API Prédiction Entretien d'Embauche</title>
            <style>
                body { font-family: Arial, sans-serif; background: #f7f7f7; margin: 0; padding: 0; }
                .container { max-width: 600px; margin: 60px auto; background: #fff; border-radius: 8px; box-shadow: 0 2px 8px #ddd; padding: 32px; }
                h1 { color: #2b4c7e; }
                p { font-size: 1.1em; }
                a { color: #2b4c7e; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>API Prédiction Entretien d'Embauche</h1>
                <p>Bienvenue sur l'API professionnelle de prédiction d'entretien d'embauche !</p>
                <p>Consultez la <a href='/docs'>documentation interactive Swagger</a> pour tester l'API.</p>
                <ul>
                    <li><b>POST</b> <code>/predict</code> : Prédiction pour un candidat</li>
                    <li><b>POST</b> <code>/predict_batch</code> : Prédiction pour plusieurs candidats</li>
                </ul>
                <p style="color: #888; font-size: 0.95em;">Projet Data Science &copy; 2025</p>
            </div>
        </body>
    </html>
    """

