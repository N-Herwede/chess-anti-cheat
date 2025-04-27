import requests
import io
import chess.pgn
import json

USER_AGENT = {
    "User-Agent": "ChessAntiCheatBot/1.0 (+https://github.com/N-Herwede/chess-anti-cheat)"
}


def fetch_chessdotcom_games(username, max_games=10):
    username = username.lower()
    archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
    try:
        res = requests.get(archives_url, headers=USER_AGENT, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"[Chess.com] Failed to fetch archives for '{username}': {e}")

    archive_urls = res.json().get("archives", [])[-6:]  # Last 6 months
    all_games = []

    for url in reversed(archive_urls):
        try:
            res = requests.get(url, headers=USER_AGENT, timeout=10)
            res.raise_for_status()
            games_data = res.json().get("games", [])
            for game in games_data:
                if "pgn" not in game:
                    continue
                pgn = game["pgn"]
                game_obj = chess.pgn.read_game(io.StringIO(pgn))
                if not game_obj:
                    continue
                all_games.append({
                    "pgn": pgn,
                    "white": game_obj.headers.get("White", "?"),
                    "black": game_obj.headers.get("Black", "?"),
                    "result": game_obj.headers.get("Result", "?"),
                    "url": game.get("url", ""),
                    "site": "Chess.com",
                    "headers": game_obj.headers
                })
                if len(all_games) >= max_games:
                    break
        except requests.RequestException:
            continue
        if len(all_games) >= max_games:
            break

    return all_games


def fetch_lichess_games(username, max_games=10):
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "opening": "true",
        "analysed": "true",
        "pgnInJson": "true"
    }
    headers = {
        "Accept": "application/x-ndjson",
        **USER_AGENT
    }

    try:
        res = requests.get(url, params=params, headers=headers, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"[Lichess] Failed to fetch games for '{username}': {e}")

    games = []
    for line in res.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                pgn = data.get("pgn", "")
                game_obj = chess.pgn.read_game(io.StringIO(pgn))
                if not game_obj:
                    continue
                games.append({
                    "pgn": pgn,
                    "white": game_obj.headers.get("White", "?"),
                    "black": game_obj.headers.get("Black", "?"),
                    "result": game_obj.headers.get("Result", "?"),
                    "url": "",
                    "site": "Lichess",
                    "headers": game_obj.headers
                })
                if len(games) >= max_games:
                    break
            except Exception:
                continue

    return games


def pgn_to_game(pgn_text):
    return chess.pgn.read_game(io.StringIO(pgn_text))
