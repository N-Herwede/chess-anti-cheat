import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

DATA_PATH = "data/games_dataset.csv"
MODEL_DIR = "models"
RESULTS_DIR = "results"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Preprocessing dataset...")
feature_cols = [
    'avg_centipawn_loss', 'max_eval_swing', 'num_mates', 'elo', 'TotalMoves',
    'num_blunders', 'longest_good_streak', 'trend_flips', 'perfect_moves',
    'eval_std', 'blunder_ratio', 'termination_type', 'temps_moyen_par_coup',
    'moyenne_oscillations_eval', 'stddev_oscillations_eval', 'nb_swings_100cp',
    'nb_coups_parfaits', 'mean_time_per_move', 'variance_time_per_move'
]
feature_cols = [col for col in df.columns if col in feature_cols]

X = df[feature_cols]
y = df['label']

print("Splitting train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("Handling missing values before SMOTE...")
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Starting RandomizedSearchCV for XGBoost...")
xgb_param_grid = {
    'n_estimators': [300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train), random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_distributions=xgb_param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_

print("Starting RandomizedSearchCV for CatBoost...")
cat_param_grid = {
    'iterations': [500, 700, 900],
    'learning_rate': [0.01, 0.03, 0.1],
    'depth': [4, 6, 8]
}

cat_search = RandomizedSearchCV(
    CatBoostClassifier(eval_metric='AUC', random_seed=42, verbose=0),
    param_distributions=cat_param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

cat_search.fit(X_train, y_train)
cat_model = cat_search.best_estimator_

print("Saving models...")
os.makedirs(MODEL_DIR, exist_ok=True)
with open(f"{MODEL_DIR}/anticheat_model_xgboost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open(f"{MODEL_DIR}/anticheat_model_catboost.pkl", "wb") as f:
    pickle.dump(cat_model, f)

def threshold_analysis(model, X_test, y_test, model_name):
    y_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []

    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        precision = precision_score(y_test, y_pred_thresh)
        recall = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        results.append({
            "Threshold": round(thresh, 2),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-score": round(f1, 3)
        })

    results_df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(f"{RESULTS_DIR}/threshold_analysis_{model_name}.csv", index=False)

    plt.figure(figsize=(10,6))
    plt.plot(results_df['Threshold'], results_df['Precision'], label='Precision')
    plt.plot(results_df['Threshold'], results_df['Recall'], label='Recall')
    plt.plot(results_df['Threshold'], results_df['F1-score'], label='F1-score')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Analysis - {model_name}")
    plt.legend()
    plt.grid()
    plt.savefig(f"{RESULTS_DIR}/threshold_plot_{model_name}.png")
    plt.close()
    print(f"Threshold analysis for {model_name} saved.")

threshold_analysis(xgb_model, X_test, y_test, "xgboost")
threshold_analysis(cat_model, X_test, y_test, "catboost")

print("Training and threshold tuning completed for both models.")
