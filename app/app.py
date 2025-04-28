import streamlit as st
import chess.pgn
import pandas as pd
import pickle
import io
import shap
import matplotlib.pyplot as plt
import numpy as np
from api_fetcher import fetch_chessdotcom_games, fetch_lichess_games, pgn_to_game
from analyze_game import analyze_game_stockfish

# --- Config
STOCKFISH_PATH = "../engine/stockfish/stockfish.exe"
XGB_MODEL_PATH = "../models/anticheat_model_xgboost.pkl"
CAT_MODEL_PATH = "../models/anticheat_model_catboost.pkl"

# --- Fonctions Utilitaires
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

def predict_with_models(features, xgb_model, cat_model, threshold_xgb, threshold_cat):
    df = pd.DataFrame([features])
    model_features = xgb_model.get_booster().feature_names
    df = df[model_features]

    xgb_proba = xgb_model.predict_proba(df)[0, 1]
    cat_proba = cat_model.predict_proba(df)[0, 1]

    xgb_pred = int(xgb_proba >= threshold_xgb)
    cat_pred = int(cat_proba >= threshold_cat)

    if xgb_pred == 1 and cat_pred == 1:
        global_verdict = "❌ Flag fort"
    elif xgb_pred == 1 or cat_pred == 1:
        global_verdict = "⚠️ Flag moyen"
    else:
        global_verdict = "✅ Clean"

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
        st.info(f"**White:** {white}\n\n**Black:** {black}\n\n**Result:** {result}\n\n**Opening:** {opening}\n\n**Time Control:** {time_control}\n\n**Date:** {date}")
        if site.startswith("http"):
            st.markdown(f"[View Game Online]({site})")

def display_shap(xgb_model, features, player_label):
    explainer = shap.TreeExplainer(xgb_model)
    df = pd.DataFrame([features])
    model_features = xgb_model.get_booster().feature_names
    df = df[model_features]

    shap_values = explainer(df)

    st.markdown(f"### 📊 SHAP Feature Importance ({player_label})")
    fig, ax = plt.subplots(figsize=(6, 3))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

# --- MAIN APP
def main():
    st.set_page_config(page_title="♟️ Chess Anti-Cheat Detection", layout="wide")
    st.title("♟️ Chess Anti-Cheat Detection")

    if 'username' not in st.session_state:
        st.session_state['username'] = ""
    if 'games' not in st.session_state:
        st.session_state['games'] = []

    choice = st.sidebar.radio("Select Input Type", ("Lichess Username", "Chess.com Username", "Upload PGN"))

    with st.sidebar.expander("⚙️ Threshold Settings (click to expand)", expanded=False):
        threshold_xgb = st.slider("XGBoost Threshold", 0.0, 1.0, 0.35, 0.01)
        threshold_cat = st.slider("CatBoost Threshold", 0.0, 1.0, 0.50, 0.01)

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

        analyze_all_checkbox = st.checkbox("✅ Analyser toutes les parties du joueur")

        if analyze_all_checkbox:
            if st.button("Analyze All Games"):
                with open(XGB_MODEL_PATH, "rb") as f:
                    xgb_model = pickle.load(f)
                with open(CAT_MODEL_PATH, "rb") as f:
                    cat_model = pickle.load(f)

                verdicts = {"✅ Clean": 0, "⚠️ Flag moyen": 0, "❌ Flag fort": 0}
                username = st.session_state['username'].strip().lower()

                progress_bar = st.progress(0)
                results_per_game = []

                for idx, game in enumerate(games):
                    try:
                        features_white, features_black = analyze_game_stockfish(game, STOCKFISH_PATH)

                        white_player = game.headers.get("White", "").strip().lower()
                        black_player = game.headers.get("Black", "").strip().lower()

                        if username == white_player and features_white:
                            xgb_proba, _, cat_proba, _, verdict = predict_with_models(features_white, xgb_model, cat_model, threshold_xgb, threshold_cat)
                            verdicts[verdict] += 1
                            results_per_game.append({
                                "Game": f"{white_player} (White) vs {black_player} (Black)",
                                "XGBoost Proba": f"{xgb_proba:.2%}",
                                "CatBoost Proba": f"{cat_proba:.2%}",
                                "Verdict": verdict
                            })

                        elif username == black_player and features_black:
                            xgb_proba, _, cat_proba, _, verdict = predict_with_models(features_black, xgb_model, cat_model, threshold_xgb, threshold_cat)
                            verdicts[verdict] += 1
                            results_per_game.append({
                                "Game": f"{white_player} (White) vs {black_player} (Black)",
                                "XGBoost Proba": f"{xgb_proba:.2%}",
                                "CatBoost Proba": f"{cat_proba:.2%}",
                                "Verdict": verdict
                            })

                    except Exception as e:
                        st.warning(f"Erreur lors de l'analyse d'une partie : {e}")
                        continue

                    progress_bar.progress((idx + 1) / len(games))

                st.header("📊 Résultat Global")
                st.success(f"✅ Parties Clean : {verdicts['✅ Clean']}")
                st.warning(f"⚠️ Flags Moyen : {verdicts['⚠️ Flag moyen']}")
                st.error(f"❌ Flags Fort : {verdicts['❌ Flag fort']}")

                if sum(verdicts.values()) > 0:
                    fig, ax = plt.subplots()
                    labels = list(verdicts.keys())
                    sizes = list(verdicts.values())
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)

                    st.subheader("📋 Détail par Partie")
                    df_results = pd.DataFrame(results_per_game)
                    st.dataframe(df_results, use_container_width=True)
                else:
                    st.info("Aucune partie trouvée pour ce joueur.")

        else:
            if st.button("Analyze Selected Game"):
                with open(XGB_MODEL_PATH, "rb") as f:
                    xgb_model = pickle.load(f)
                with open(CAT_MODEL_PATH, "rb") as f:
                    cat_model = pickle.load(f)

                selected_game = games[game_idx]
                display_game_metadata(selected_game)

                features_white, features_black = analyze_game_stockfish(selected_game, STOCKFISH_PATH)

                if features_white:
                    st.subheader("♟️ White Player")
                    xgb_proba_w, _, cat_proba_w, _, verdict_w = predict_with_models(features_white, xgb_model, cat_model, threshold_xgb, threshold_cat)
                    st.metric("XGBoost Probability", f"{xgb_proba_w:.2f}")
                    st.metric("CatBoost Probability", f"{cat_proba_w:.2f}")

                    if "Clean" in verdict_w:
                        st.success(f"Verdict: {verdict_w}")
                    elif "moyen" in verdict_w:
                        st.warning(f"Verdict: {verdict_w}")
                    else:
                        st.error(f"Verdict: {verdict_w}")

                    display_shap(xgb_model, features_white, "White")

                if features_black:
                    st.subheader("♟️ Black Player")
                    xgb_proba_b, _, cat_proba_b, _, verdict_b = predict_with_models(features_black, xgb_model, cat_model, threshold_xgb, threshold_cat)
                    st.metric("XGBoost Probability", f"{xgb_proba_b:.2f}")
                    st.metric("CatBoost Probability", f"{cat_proba_b:.2f}")

                    if "Clean" in verdict_b:
                        st.success(f"Verdict: {verdict_b}")
                    elif "moyen" in verdict_b:
                        st.warning(f"Verdict: {verdict_b}")
                    else:
                        st.error(f"Verdict: {verdict_b}")

                    display_shap(xgb_model, features_black, "Black")

if __name__ == "__main__":
    main()
