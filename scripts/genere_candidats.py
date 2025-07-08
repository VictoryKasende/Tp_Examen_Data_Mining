import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000  # Par classe (donc 10000 au total ensuite)

def generate_data(n, label):
    """
    label=1: admis
    label=0: non-admis
    Génère des variables dont la distribution favorise ou défavorise l'admission.
    """
    # Diplôme fortement discriminant
    if label == 1:
        diplome = np.random.choice(['Licence', 'Master', 'Doctorat'], n, p=[0.3, 0.55, 0.15])
        note_anglais = np.random.normal(78, 10, n).clip(62, 100)
        experience = np.random.randint(3, 16, n)
        entreprises_precedentes = np.random.poisson(2.2, n)
        distance_km = np.abs(np.random.normal(5, 3, n)).clip(0, 20)
        score_entretien = np.random.normal(8.2, 0.8, n).clip(6.5, 10)
        score_competence = np.random.normal(8.1, 0.85, n).clip(6, 10)
        score_personnalite = np.random.normal(82, 8, n).clip(65, 100).astype(int)
    else:
        diplome = np.random.choice(['BTS', 'Licence', 'Master'], n, p=[0.55, 0.40, 0.05])
        note_anglais = np.random.normal(58, 15, n).clip(0, 75)
        experience = np.random.randint(0, 8, n)
        entreprises_precedentes = np.random.poisson(1.2, n)
        distance_km = np.abs(np.random.normal(13, 7, n)).clip(0, 30)
        score_entretien = np.random.normal(5.7, 1, n).clip(2, 7.5)
        score_competence = np.random.normal(5.4, 1.2, n).clip(2, 7.7)
        score_personnalite = np.random.normal(67, 10, n).clip(45, 85).astype(int)

    age = np.random.randint(21, 45, n)
    sexe = np.random.choice(['M', 'F'], n)
    # Ajout d'un peu de bruit : 10% des samples inversent certains critères pour casser la trop forte linéarité
    mask = np.random.rand(n) < 0.1
    if label == 1:
        note_anglais[mask] = np.random.normal(62, 8, mask.sum()).clip(45, 72)
        score_competence[mask] = np.random.normal(6, 1, mask.sum()).clip(3, 10)
    else:
        note_anglais[mask] = np.random.normal(78, 10, mask.sum()).clip(62, 100)
        score_competence[mask] = np.random.normal(8, 0.85, mask.sum()).clip(7, 10)
    # On assemble tout
    return pd.DataFrame({
        'age': age,
        'diplome': diplome,
        'note_anglais': note_anglais.astype(int),
        'experience': experience,
        'entreprises_precedentes': entreprises_precedentes,
        'distance_km': np.round(distance_km, 1),
        'score_entretien': np.round(score_entretien, 1),
        'score_competence': np.round(score_competence, 1),
        'score_personnalite': score_personnalite,
        'sexe': sexe,
        'retenu': label
    })

# Générer & mélanger
df1 = generate_data(n, 1)
df0 = generate_data(n, 0)
df = pd.concat([df1, df0], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Sauvegarde
df.to_csv("data/candidats_mlpro.csv", index=False)
print(df.head())

# Vérifie
print(df['retenu'].value_counts())