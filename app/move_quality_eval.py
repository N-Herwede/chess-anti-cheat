import subprocess
import os

STOCKFISH_PATH = os.path.join("engine", "stockfish", "stockfish.exe")

class StockfishAnalyzer:
    def __init__(self, depth=15, multipv=3):
        self.depth = depth
        self.multipv = multipv
        self.engine = subprocess.Popen(
            [STOCKFISH_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            bufsize=1
        )
        self._set_multipv(multipv)

    def _set_multipv(self, n):
        self.engine.stdin.write(f"setoption name MultiPV value {n}\n")
        self.engine.stdin.flush()

    def get_top_moves(self, fen):
        self.engine.stdin.write(f"position fen {fen}\n")
        self.engine.stdin.write(f"go depth {self.depth}\n")
        self.engine.stdin.flush()

        top_moves = []
        best_score = None
        while True:
            line = self.engine.stdout.readline()
            if line.startswith("info") and " pv " in line and " score cp " in line:
                try:
                    move = line.split(" pv ")[1].split()[0]
                    score = int(line.split("score cp ")[1].split()[0])
                    top_moves.append((move, score))
                    if best_score is None:
                        best_score = score
                except:
                    continue
            if line.startswith("bestmove"):
                break
        return top_moves, best_score

    def get_score_for_move(self, fen, move):
        self.engine.stdin.write(f"position fen {fen}\n")
        self.engine.stdin.write(f"go depth {self.depth} searchmoves {move}\n")
        self.engine.stdin.flush()

        while True:
            line = self.engine.stdout.readline()
            if "score cp" in line:
                try:
                    score = int(line.split("score cp ")[1].split()[0])
                    return score
                except:
                    return None
            if line.startswith("bestmove"):
                break
        return None

    def classify_move(self, fen, played_move):
        top_moves, best_score = self.get_top_moves(fen)
        top_move_list = [mv for mv, _ in top_moves]

        if played_move in top_move_list:
            return "accurate", 0

        score_played = self.get_score_for_move(fen, played_move)
        if score_played is None or best_score is None:
            return "unknown", None

        cp_loss = abs(best_score - score_played)
        cp_loss = min(cp_loss, 500)

        if cp_loss < 30:
            return "accurate", cp_loss
        elif cp_loss < 100:
            return "inaccuracy", cp_loss
        elif cp_loss < 300:
            return "mistake", cp_loss
        else:
            return "blunder", cp_loss

    def close(self):
        if self.engine.poll() is None:
            self.engine.stdin.write("quit\n")
            self.engine.stdin.flush()
            self.engine.terminate()



