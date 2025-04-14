import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# === Configurations ===
INPUT_PATH = "cheat_features.csv"
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
N_SPLITS = 5
RANDOM_STATE = 42

# === Chargement des données ===
df = pd.read_csv(INPUT_PATH)
df = df.drop(columns=["game_id"], errors="ignore")
df = pd.get_dummies(df, columns=["player", "result"], drop_first=True)

X = df.drop(columns=["label", "cheat_ratio"])
y = df["label"]

# === Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Poids de classe pour XGBoost
pos_weight = (y == 0).sum() / (y == 1).sum()

# === Définir les modèles ===
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    verbosity=0
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('rf', rf)],
    voting='soft'
)

# === Validation croisée sur le modèle combiné
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
scores = cross_validate(
    ensemble,
    X_scaled,
    y,
    cv=cv,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=False
)

print("Résultats validation croisée (VotingClassifier - moyennes sur {} folds) :".format(N_SPLITS))
for metric in scores:
    if "test" in metric:
        print(f"{metric}: {np.mean(scores[metric]):.4f}")

# === Entraînement des modèles
xgb.fit(X_scaled, y)
rf.fit(X_scaled, y)
ensemble.fit(X_scaled, y)

# === Sauvegarde
os.makedirs(MODELS_DIR, exist_ok=True)

with open(os.path.join(MODELS_DIR, "xgb_model.pkl"), "wb") as f:
    pickle.dump((xgb, scaler, X.columns.tolist()), f)

with open(os.path.join(MODELS_DIR, "rf_model.pkl"), "wb") as f:
    pickle.dump((rf, scaler, X.columns.tolist()), f)

with open(os.path.join(MODELS_DIR, "ensemble_model.pkl"), "wb") as f:
    pickle.dump((ensemble, scaler, X.columns.tolist()), f)

print(f"Tous les modèles sauvegardés dans : {MODELS_DIR}")
