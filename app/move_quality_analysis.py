import chess.pgn
import subprocess
import pandas as pd
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

def evaluate_position(engine, moves, depth=12):
    send_command(engine, "position startpos moves " + " ".join(moves))
    send_command(engine, f"go depth {depth}")
    eval_score = None

    while True:
        line = engine.stdout.readline().strip()
        if line.startswith('info') and 'score cp' in line:
            try:
                parts = line.split('score cp ')[1]
                score = int(parts.split(' ')[0])
                eval_score = score / 100.0
            except:
                pass
        if line.startswith('bestmove'):
            break
    return eval_score

def analyze_move_quality(game, stockfish_path):
    engine = open_stockfish(stockfish_path)
    board = game.board()
    moves = list(game.mainline_moves())

    move_list = []
    evaluations = []

    send_command(engine, "uci")
    read_until(engine, "uciok")
    send_command(engine, "isready")
    read_until(engine, "readyok")
    send_command(engine, "ucinewgame")
    send_command(engine, "isready")
    read_until(engine, "readyok")

    for move in moves:
        move_list.append(board.san(move))
        board.push(move)
        uci_moves = [m.uci() for m in board.move_stack]
        eval_score = evaluate_position(engine, uci_moves)
        evaluations.append(eval_score)

    engine.kill()

    if len(evaluations) < 2:
        return None

    swings = [abs(evaluations[i+1] - evaluations[i]) for i in range(len(evaluations)-1)]
    features = {
        'avg_centipawn_loss': sum(abs(e) for e in evaluations) / len(evaluations),
        'max_eval_swing': max(swings),
        'TotalMoves': len(evaluations),
        'swings': swings,
        'evaluations': evaluations
    }
    return features

def load_pgn(filepath):
    games = []
    with open(filepath, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games

if __name__ == "__main__":
    pgn_path = input("Enter PGN file path: ")
    games = load_pgn(pgn_path)

    for idx, game in enumerate(games):
        print(f"Analyzing game {idx+1}...")
        features = analyze_move_quality(game, STOCKFISH_PATH)
        if features:
            print(f"Move Quality Features for game {idx+1}: {features}")
        else:
            print(f"Not enough data for game {idx+1}.")
