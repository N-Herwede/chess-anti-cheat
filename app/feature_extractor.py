import chess.pgn
import io
import pandas as pd
import os
from move_quality_eval import StockfishAnalyzer

STOCKFISH_PATH = os.path.join("engine", "stockfish", "stockfish.exe")

analyzer = StockfishAnalyzer(depth=15, multipv=3)

def extract_features(game, ignore_first_n_moves=10):
    board = game.board()
    moves = list(game.mainline_moves())
    headers = game.headers

    if len(moves) <= ignore_first_n_moves + 1:
        return pd.DataFrame()

    stats = {
        "white": {"accurate": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0, "cp_losses": []},
        "black": {"accurate": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0, "cp_losses": []}
    }

    for idx, move in enumerate(moves):
        fen_before = board.fen()
        move_uci = move.uci()

        if idx < ignore_first_n_moves:
            board.push(move)
            continue

        # Updated classification logic
        top_moves, best_score = analyzer.get_top_moves(fen_before)
        top_move_list = [mv for mv, _ in top_moves]
        board.push(move)
        side = "white" if not board.turn else "black"

        if move_uci in top_move_list:
            stats[side]["accurate"] += 1
            stats[side]["cp_losses"].append(0)
            continue

        score_played = analyzer.get_score_for_move(fen_before, move_uci)
        if score_played is None or best_score is None:
            continue

        cp_loss = abs(best_score - score_played)
        cp_loss = min(cp_loss, 500)

        if cp_loss < 50:
            stats[side]["accurate"] += 1
        elif cp_loss < 150:
            stats[side]["inaccuracy"] += 1
        elif cp_loss < 400:
            stats[side]["mistake"] += 1
        else:
            stats[side]["blunder"] += 1

        stats[side]["cp_losses"].append(cp_loss)

    def build_row(color, is_white, elo, result):
        losses = stats[color]["cp_losses"]
        avg_cp = sum(losses) / len(losses) if losses else 0
        acc = max(0, 100 - avg_cp * 0.1)
        return {
            "accurate_moves": stats[color]["accurate"],
            "inaccurate_moves": stats[color]["inaccuracy"],
            "mistake_moves": stats[color]["mistake"],
            "blunder_moves": stats[color]["blunder"],
            "average_cp_loss": avg_cp,
            "accuracy_score": acc,
            "elo": elo,
            "player_white": int(is_white),
            "result_loss": int(result == "loss"),
            "result_win": int(result == "win")
        }

    white_elo = int(headers.get("WhiteElo", headers.get("Elo White", -1)))
    black_elo = int(headers.get("BlackElo", headers.get("Elo Black", -1)))
    result = headers.get("Result", headers.get("Score", "*"))

    if result == "1-0":
        white_result, black_result = "win", "loss"
    elif result == "0-1":
        white_result, black_result = "loss", "win"
    else:
        white_result = black_result = "draw"

    white_row = build_row("white", True, white_elo, white_result)
    black_row = build_row("black", False, black_elo, black_result)

    return pd.DataFrame([white_row, black_row])

def close_analyzer():
    analyzer.close()
