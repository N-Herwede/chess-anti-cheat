import streamlit as st
st.set_page_config(page_title="Chess Anti-Cheat Detector", layout="centered")

import chess.pgn
import tempfile
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import shap
import pandas as pd

import feature_extractor
from api_fetcher import fetch_chessdotcom_games, fetch_lichess_games, pgn_to_game

@st.cache_resource
def load_ensemble_model():
    with open("models/ensemble_model.pkl", "rb") as f:
        model, scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names

@st.cache_resource
def load_xgb_model():
    with open("models/xgb_model.pkl", "rb") as f:
        model, scaler, feature_names = pickle.load(f)
    return model, scaler, feature_names

ensemble_model, scaler, feature_names = load_ensemble_model()
xgb_model, xgb_scaler, xgb_features = load_xgb_model()

st.title("♟️ Chess Anti-Cheat Detector")
st.write("Upload a PGN file or fetch a game from Chess.com or Lichess to detect potential engine assistance.")

st.subheader("🔍 Check a Chess.com or Lichess User")
username = st.text_input("Enter username")
platform = st.radio("Platform", ["Chess.com", "Lichess"])

if "fetched_games" not in st.session_state:
    st.session_state.fetched_games = []
if "game_options" not in st.session_state:
    st.session_state.game_options = []

if st.button("Fetch recent games"):
    try:
        raw_games = fetch_chessdotcom_games(username) if platform == "Chess.com" else fetch_lichess_games(username)
        st.session_state.fetched_games = []
        st.session_state.game_options = []

        for g in raw_games:
            try:
                game_obj = pgn_to_game(g["pgn"])
                game_obj.headers["Site"] = g.get("site", "")
                game_obj.headers["URL"] = g.get("url", "")
                game_obj.headers["Result"] = g.get("result", "?")
                game_obj.headers["White"] = g.get("white", "?")
                game_obj.headers["Black"] = g.get("black", "?")
                st.session_state.fetched_games.append(game_obj)

                label = f"{game_obj.headers['White']} vs {game_obj.headers['Black']} ({game_obj.headers['Result']}) - {game_obj.headers['Site']}"
                st.session_state.game_options.append(label)
            except:
                continue
    except Exception as e:
        st.error(f"Error fetching games: {e}")

selected_game = None
if st.session_state.fetched_games:
    selected_index = st.selectbox(
        "Choose a game to analyze",
        options=list(range(len(st.session_state.fetched_games))),
        format_func=lambda i: st.session_state.game_options[i]
    )
    selected_game = st.session_state.fetched_games[selected_index]

uploaded_game = None
uploaded_file = st.file_uploader("Or upload a PGN file", type=["pgn"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pgn") as temp_pgn:
        temp_pgn.write(uploaded_file.read())
        temp_pgn_path = temp_pgn.name
    with open(temp_pgn_path, "r", encoding="utf-8") as f:
        uploaded_game = chess.pgn.read_game(f)

game = selected_game if selected_game else uploaded_game
if not game:
    st.stop()

if "last_game_idx" not in st.session_state:
    st.session_state.last_game_idx = None
if selected_game and st.session_state.last_game_idx != selected_index:
    st.session_state.move_index = 0
    st.session_state.last_game_idx = selected_index

st.subheader("PGN Preview")
st.code(str(game), language="text")

st.subheader("Lancer l'analyse")
if st.button("🧠 Analyser cette partie"):
    with st.spinner("Analyzing with Stockfish, please wait..."):
        try:
            features_df = feature_extractor.extract_features(game, ignore_first_n_moves=10)
        except Exception as e:
            st.error(f"Erreur pendant l’analyse : {e}")
            st.stop()

    if features_df.empty:
        st.error("⚠️ Impossible d'extraire les features de cette partie.")
        st.stop()

    for col in feature_names:
        if col not in features_df.columns:
            features_df[col] = 0
    features_df = features_df[feature_names]
    scaled_features = scaler.transform(features_df)

    ensemble_proba = ensemble_model.predict_proba(scaled_features)

    def explain_side(label, proba_cheat):
        if proba_cheat > 0.70:
            st.write(f"**{label} Player:** 🚨 Suspect ({proba_cheat:.2%} probability)")
        else:
            st.write(f"**{label} Player:** ✅ Clean ({(1 - proba_cheat):.2%} probability)")

    st.markdown("### Ensemble Model Prediction")
    explain_side("White", ensemble_proba[0][1])
    if len(ensemble_proba) > 1:
        explain_side("Black", ensemble_proba[1][1])

    st.markdown("### 🔍 Features extraites :")
    st.dataframe(features_df)

    st.markdown("---")
    st.subheader("SHAP Explanation (XGBoost only)")

    xgb_model = ensemble_model.named_estimators_["xgb"]
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(scaled_features)
    shap.summary_plot(shap_values, features_df, feature_names=feature_names, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
