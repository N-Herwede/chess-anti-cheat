import pickle
import os
import numpy as np

class CheatModel:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_cheating(self, white_ratio: float, black_ratio: float, white_top_moves: int, black_top_moves: int):
        # These are basic placeholder features. You can expand them later.
        total_white = white_top_moves if white_top_moves > 0 else 1
        total_black = black_top_moves if black_top_moves > 0 else 1

        features = np.array([[white_top_moves, black_top_moves, total_white, total_black,
                              white_ratio, black_ratio, 1500, 1500]])  # Elo default = 1500
        return self.model.predict(features)[0]
