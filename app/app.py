import streamlit as st
import chess.pgn
import pandas as pd
import subprocess
import pickle
import io
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from api_fetcher import fetch_chessdotcom_games, fetch_lichess_games, pgn_to_game
from analyze_game import analyze_game_stockfish

STOCKFISH_PATH = "../engine/stockfish/stockfish.exe"
XGB_MODEL_PATH = "../models/anticheat_model_xgboost.pkl"
CAT_MODEL_PATH = "../models/anticheat_model_catboost.pkl"
THRESHOLD = 0.35

def load_games_from_pgnfile(uploaded_file):
    games = []
    pgn_text = uploaded_file.read().decode('utf-8')
    pgn_io = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(pgn_io)
        if game is None:
            break
        games.append(game)
    return games

def predict_with_models(features, xgb_model, cat_model, threshold):
    df = pd.DataFrame([features])

    # 👉 Correct Feature Alignment
    model_features = xgb_model.get_booster().feature_names
    df = df[model_features]  # Force correct order

    xgb_proba = xgb_model.predict_proba(df)[0, 1]
    cat_proba = cat_model.predict_proba(df)[0, 1]

    xgb_pred = int(xgb_proba >= threshold)
    cat_pred = int(cat_proba >= threshold)

    if xgb_pred == 1 and cat_pred == 1:
        global_verdict = "Flag fort - Deux modèles détectent triche"
    elif xgb_pred == 1 or cat_pred == 1:
        global_verdict = "Flag moyen - Un modèle détecte triche"
    else:
        global_verdict = "Clean - Aucun modèle ne détecte triche"

    return xgb_proba, xgb_pred, cat_proba, cat_pred, global_verdict

def get_game_name(game):
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    return f"{white} vs {black}"

def display_game_metadata(game):
    site = game.headers.get("Site", "Unknown")
    white = game.headers.get("White", "Unknown")
    black = game.headers.get("Black", "Unknown")
    date = game.headers.get("Date", "Unknown")
    result = game.headers.get("Result", "Unknown")
    opening = game.headers.get("Opening", "Unknown")
    time_control = game.headers.get("TimeControl", "Unknown")

    with st.container():
        st.markdown("### 🎨 Game Metadata")
        st.info(f"**White:** {white}\n\n**Black:** {black}")
        st.markdown(f"**Result:** {result}")
        st.markdown(f"**Opening:** {opening}")
        st.markdown(f"**Time Control:** {time_control}")
        st.markdown(f"**Date:** {date}")
        if site.startswith("http"):
            st.markdown(f"[View Game Online]({site})")

def display_game_stats(features, player_label):
    with st.container():
        st.markdown(f"### 📋 Game Statistics ({player_label})")
        df_stats = pd.DataFrame({
            "Feature": list(features.keys()),
            "Value": list(features.values())
        })
        st.dataframe(df_stats, height=350)

def plot_eval_chart(features, player_label):
    if 'moyenne_oscillations_eval' in features and 'stddev_oscillations_eval' in features:
        fig, ax = plt.subplots(figsize=(6, 3))
        x = np.arange(0, 20)
        y = features['moyenne_oscillations_eval'] + features['stddev_oscillations_eval'] * np.sin(x)
        ax.plot(x, y, label=f"{player_label} Evaluation Swings")
        ax.set_xlabel("Move Number")
        ax.set_ylabel("Evaluation Swing (centipawns)")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

def display_shap(xgb_model, features, player_label):
    explainer = shap.TreeExplainer(xgb_model)
    df = pd.DataFrame([features])
    
    # 👉 Force same feature order
    model_features = xgb_model.get_booster().feature_names
    df = df[model_features]
    
    shap_values = explainer(df)

    st.markdown(f"### 📊 SHAP Feature Importance ({player_label})")
    fig, ax = plt.subplots(figsize=(6, 3))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Chess Anti-Cheat Detection", layout="wide")
    st.title("♟️ Chess Anti-Cheat Detection")

    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    if 'games' not in st.session_state:
        st.session_state['games'] = []

    choice = st.sidebar.radio("Select Input Type", ("Lichess Username", "Chess.com Username", "Upload PGN"))

    if choice == "Lichess Username":
        username = st.sidebar.text_input("Enter Lichess Username:", value=st.session_state['username'])
        st.session_state['username'] = username
        if st.sidebar.button("Fetch Games"):
            fetched_games = fetch_lichess_games(username)
            st.session_state['games'] = [pgn_to_game(game['pgn']) for game in fetched_games]

    elif choice == "Chess.com Username":
        username = st.sidebar.text_input("Enter Chess.com Username:", value=st.session_state['username'])
        st.session_state['username'] = username
        if st.sidebar.button("Fetch Games"):
            fetched_games = fetch_chessdotcom_games(username)
            st.session_state['games'] = [pgn_to_game(game['pgn']) for game in fetched_games]

    elif choice == "Upload PGN":
        uploaded_file = st.sidebar.file_uploader("Upload a PGN file", type=["pgn"])
        if uploaded_file is not None:
            st.session_state['games'] = load_games_from_pgnfile(uploaded_file)

    if st.session_state['games']:
        games = st.session_state['games']
        game_options = [get_game_name(game) for game in games]
        game_idx = st.selectbox("Select a Game to Analyze", range(len(games)), format_func=lambda x: game_options[x])
        display_game_metadata(games[game_idx])

        if st.button("Analyze Selected Game"):
            with open(XGB_MODEL_PATH, "rb") as f:
                xgb_model = pickle.load(f)

            with open(CAT_MODEL_PATH, "rb") as f:
                cat_model = pickle.load(f)

            features_white, features_black = analyze_game_stockfish(games[game_idx], STOCKFISH_PATH)

            if features_white:
                white = games[game_idx].headers.get("White", "Unknown")
                st.header(f"♟️ White Player: {white}")

                xgb_proba_w, xgb_pred_w, cat_proba_w, cat_pred_w, verdict_w = predict_with_models(features_white, xgb_model, cat_model, THRESHOLD)
                st.metric(label="XGBoost Cheat Probability", value=f"{xgb_proba_w:.2f}")
                st.metric(label="CatBoost Cheat Probability", value=f"{cat_proba_w:.2f}")
                st.success(f"Verdict: {verdict_w}")

                display_game_stats(features_white, player_label="White")
                plot_eval_chart(features_white, player_label="White")
                display_shap(xgb_model, features_white, player_label="White")

            if features_black:
                black = games[game_idx].headers.get("Black", "Unknown")
                st.header(f"♟️ Black Player: {black}")

                xgb_proba_b, xgb_pred_b, cat_proba_b, cat_pred_b, verdict_b = predict_with_models(features_black, xgb_model, cat_model, THRESHOLD)
                st.metric(label="XGBoost Cheat Probability", value=f"{xgb_proba_b:.2f}")
                st.metric(label="CatBoost Cheat Probability", value=f"{cat_proba_b:.2f}")
                st.success(f"Verdict: {verdict_b}")

                display_game_stats(features_black, player_label="Black")
                plot_eval_chart(features_black, player_label="Black")
                display_shap(xgb_model, features_black, player_label="Black")

if __name__ == "__main__":
    main()
