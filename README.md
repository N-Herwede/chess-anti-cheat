# Chess Anti-Cheat Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)

A full-stack web application that detects potential cheating in chess games using machine learning and Stockfish evaluation. Built with Streamlit and integrated with real-time data from Chess.com and Lichess.

---

##  Features

-  Upload PGN files or analyze games directly from Chess.com & Lichess
-  Interactive replay board with autoplay and navigation controls
-  Cheating prediction for each player using:
  - Random Forest
  - XGBoost
-  SHAP-based explainability of model predictions
-  Player metadata (Elo, result, etc.)
-  Auto-detect suspicious streaks and engine matches

---

## Machine Learning Models

Models trained on features extracted using Stockfish evaluations:

- Total moves by each player
- Engine-matching streaks
- Max streaks of best moves
- Eval swings and ratio of best moves
- Elo comparison and unexpected results
- And more

---

## Future Plans

- Support for analyzing complete Chess.com and Lichess history
- Integrate Maia (behavioral engine) for playstyle comparison
- Add per-move engine agreement visualizations
- Export results and visualizations to PDF/PNG

 ---

##  Run Locally

```bash
git clone https://github.com/N-Herwede/chess-anti-cheat
cd chess-anti-cheat
pip install -r requirements.txt
streamlit run app/web_app.py
