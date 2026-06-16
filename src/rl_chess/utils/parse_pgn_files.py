from io import StringIO

import chess
import chess.pgn


def filter_draw_games(filename):
    with open(filename) as file:
        pgn_string = ""
        line = file.readline()

        number_of_draw_games = 0
        while line:
            if line[:6] == "[Event":
                token_list = pgn_string.strip().split()
                string = '"1/2-1/2"]'
                if string in token_list:
                    with open("draw_games.pgn", "a") as draw_file:
                        draw_file.write(pgn_string)
                    number_of_draw_games += 1
                pgn_string = ""
            pgn_string += line
            line = file.readline()


def fetch_games_from_file(filename: str) -> list[chess.pgn.Game]:
    game_list = []
    with open(filename) as file:
        pgn_string = ""
        line = file.readline()

        while line:
            if line[:6] == "[Event":
                pgn = StringIO(pgn_string)
                game = chess.pgn.read_game(pgn)
                if game is not None:
                    game_list.append(game)
                pgn_string = ""
            pgn_string += line
            line = file.readline()

    return game_list


def fetch_moves_from_game(game: chess.pgn.Game) -> list[chess.Move]:
    moves = []
    for move in game.mainline_moves():
        moves.append(move)
    return moves
