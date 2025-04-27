import chess.pgn
import subprocess
import numpy as np
import io

STOCKFISH_PATH = "../engine/stockfish/stockfish.exe"

def open_stockfish(stockfish_path):
    return subprocess.Popen(
        [stockfish_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
        bufsize=1
    )

def send_command(engine, command):
    engine.stdin.write(command + '\n')
    engine.stdin.flush()

def read_until(engine, target):
    lines = []
    while True:
        line = engine.stdout.readline().strip()
        if target in line:
            break
        lines.append(line)
    return lines

def analyze_game_stockfish(game, stockfish_path):
    engine = open_stockfish(stockfish_path)
    board = game.board()

    evals = []
    moves_info = []

    send_command(engine, "uci")
    read_until(engine, "uciok")
    send_command(engine, "isready")
    read_until(engine, "readyok")
    send_command(engine, "ucinewgame")
    send_command(engine, "isready")
    read_until(engine, "readyok")

    moves = list(game.mainline_moves())

    mate_count_white = 0
    mate_count_black = 0

    for i, move in enumerate(moves):
        board.push(move)
        moves_uci = [m.uci() for m in board.move_stack]
        send_command(engine, "position startpos moves " + " ".join(moves_uci))
        send_command(engine, "go depth 12")

        current_turn = board.turn  # False = Black to move (means White just moved)

        while True:
            line = engine.stdout.readline().strip()
            if line.startswith('info') and 'score cp' in line:
                try:
                    parts = line.split('score cp ')[1]
                    score = int(parts.split(' ')[0])
                    evals.append(score / 100.0)
                except:
                    pass
            if line.startswith('info') and 'score mate' in line:
                if not current_turn:
                    mate_count_white += 1
                else:
                    mate_count_black += 1
                break
            if line.startswith('bestmove'):
                break

        moves_info.append({
            "move": move,
            "turn": not current_turn,  # True = White moved, False = Black moved
            "eval": evals[-1] if evals else 0
        })

    engine.kill()

    evals_white = [info["eval"] for info in moves_info if info["turn"] == True]
    evals_black = [info["eval"] for info in moves_info if info["turn"] == False]

    # --- Calcul Features par couleur ---

    def calculate_features(evals_subset, mate_count):
        if len(evals_subset) < 2:
            return None

        swings = [abs(evals_subset[i+1] - evals_subset[i]) for i in range(len(evals_subset)-1)]
        nb_swings_100cp = sum(1 for s in swings if s > 1.0)
        num_blunders = sum(1 for s in swings if s > 2.0)
        nb_coups_parfaits = sum(1 for s in swings if s <= 0.2)

        longest_good_streak = 0
        current_streak = 0
        for s in swings:
            if s <= 0.5:
                current_streak += 1
                longest_good_streak = max(longest_good_streak, current_streak)
            else:
                current_streak = 0

        trend_flips = 0
        for i in range(1, len(evals_subset)-1):
            if (evals_subset[i] - evals_subset[i-1]) * (evals_subset[i+1] - evals_subset[i]) < 0:
                trend_flips += 1

        eval_std = np.std(evals_subset)
        moyenne_oscillations_eval = np.mean(swings)
        stddev_oscillations_eval = np.std(swings)

        mean_time_per_move = 3.0  # simulé
        variance_time_per_move = 0.5

        blunder_ratio = num_blunders / len(swings) if swings else 0

        return {
            'TotalMoves': len(evals_subset),
            'avg_centipawn_loss': np.mean([abs(e) for e in evals_subset]),
            'max_eval_swing': max(swings),
            'num_mates': mate_count,
            'num_blunders': num_blunders,
            'longest_good_streak': longest_good_streak,
            'trend_flips': trend_flips,
            'perfect_moves': nb_coups_parfaits,
            'eval_std': eval_std,
            'blunder_ratio': blunder_ratio,
            'temps_moyen_par_coup': mean_time_per_move,
            'moyenne_oscillations_eval': moyenne_oscillations_eval,
            'stddev_oscillations_eval': stddev_oscillations_eval,
            'nb_swings_100cp': nb_swings_100cp,
            'nb_coups_parfaits': nb_coups_parfaits,
            'mean_time_per_move': mean_time_per_move,
            'variance_time_per_move': variance_time_per_move
        }

    features_white = calculate_features(evals_white, mate_count_white)
    features_black = calculate_features(evals_black, mate_count_black)

    # --- Elo / Termination type (identiques pour les deux) ---

    try:
        white_elo = int(game.headers.get("WhiteElo", 1500))
        black_elo = int(game.headers.get("BlackElo", 1500))
    except:
        white_elo = black_elo = 1500

    termination = game.headers.get("Termination", "Normal")
    termination_mapping = {"Normal": 0, "Time forfeit": 1, "Abandoned": 2}
    termination_type = termination_mapping.get(termination, 0)

    if features_white:
        features_white['elo'] = white_elo
        features_white['termination_type'] = termination_type
    if features_black:
        features_black['elo'] = black_elo
        features_black['termination_type'] = termination_type

    return features_white, features_black
