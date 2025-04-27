## Noah HERWEDE - Laïfa AHMED-YAHIA M1-DS2E

# Chess Anti-Cheat Detector  
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Stockfish](https://img.shields.io/badge/Stockfish-Engine-orange)
A full-stack web application that detects potential cheating in chess games using machine learning and Stockfish evaluation. Built with Streamlit and integrated with real-time data from Chess.com and Lichess.

---

##  Features

-  Upload PGN files or analyze games directly from Chess.com & Lichess
-  Cheating prediction for each player using:
  - Catboost
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
## Credit

This project uses Stockfish, a powerful open-source chess engine developed by the Stockfish community.
Stockfish is licensed under the GNU General Public License v3 (GPLv3).
Copyright (C) 2004–2025 The Stockfish developers
You can find the official source code here: https://github.com/official-stockfish/Stockfish 

This project utilizes data from the Lichess Open Database, made available by Lichess under the Creative Commons CC0 license.
You can find the official data here: https://database.lichess.org/

---

## How to Replicate

- Download a dataset from https://database.lichess.org/ (*Note: Files range from 8GB to 150+GB*). Alternatively, you can use a pre-processed dataset from Kaggle.
- Extract the .zst file using 7-Zip (or another compatible tool).
- Run the Python script xxxxx to filter games with existing analysis. Alternatively, process the games through Stockfish to generate position evaluations using xxxxx.
- Execute the script xxxx to train the machine learning model using the game dataset.
- Use Apifeatcher to retrieve data for the specific games you want to analyze.
- Lauch the main script

---

## Launch on your computer 

