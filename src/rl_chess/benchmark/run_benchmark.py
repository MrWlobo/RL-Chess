import argparse
import inspect
import json
import random
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn
from chess.pgn import Game

from rl_chess.agents.base import ChessAgent
from rl_chess.agents.baseline.maia_agent import MaiaAgent
from rl_chess.agents.baseline.random_agent import RandomAgent
from rl_chess.agents.custom import AGENT_REGISTRY
from rl_chess.benchmark.config.load_config import BenchmarkConfig


class ChessBenchmark:
    def __init__(
        self,
        n_games: int,
        agent: ChessAgent,
        output_file: Path,
        pass_threshold: float = 0.5,
        deterministic: bool = True,
    ) -> None:
        config = BenchmarkConfig()  # ty:ignore[missing-argument]

        self.n_games: int = n_games
        self.agent: ChessAgent = agent
        self.output_file: Path = (
            Path(__file__) / config.results_dir / output_file
        )
        self.pass_threshold = pass_threshold
        self.deterministic: bool = deterministic

        positions_file = config.positions.path
        self.position_fens = []
        with positions_file.open("r") as f:
            for fen in f:
                self.position_fens.append(fen)

        self.lc0_exe: Path = config.maia.exe.lc0_path
        self.maia_weights: dict[str, Path] = config.maia.weights

        if not self.deterministic:
            random.shuffle(self.position_fens)

        self.results: dict[str, dict] = {
            "RandomAgent": {
                "wins": 0,
                "losses": 0,
                "draws": 0,
            },
            "Maia1100": {
                "wins": 0,
                "losses": 0,
                "draws": 0,
            },
            "Maia1400": {
                "wins": 0,
                "losses": 0,
                "draws": 0,
            },
            "Maia1600": {
                "wins": 0,
                "losses": 0,
                "draws": 0,
            },
            "Maia1900": {
                "wins": 0,
                "losses": 0,
                "draws": 0,
            },
            "games": {},
        }

    def run(self):
        games_per_color = max(self.n_games // 2, 1)

        # vs. RandomAgent
        curr_opponent = "RandomAgent"
        white, black = self.agent, RandomAgent()
        self.results["games"][curr_opponent] = []
        for i in range(games_per_color):
            board = chess.Board(self.position_fens[i])
            game = self._play_game(board, white, black)
            self._add_result(game, curr_opponent, is_opponent_black=True)

            board = chess.Board(self.position_fens[i])
            game = self._play_game(board, black, white)
            self._add_result(game, curr_opponent, is_opponent_black=False)

        if not self._is_threshold_met(curr_opponent):
            return

        # vs. Maia 1100 Elo
        curr_opponent = "Maia1100"
        self.results["games"][curr_opponent] = []
        white = self.agent
        with MaiaAgent(self.maia_weights["maia_1100"], self.lc0_exe) as black:
            for i in range(games_per_color):
                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, white, black)
                self._add_result(game, curr_opponent, is_opponent_black=True)

                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, black, white)
                self._add_result(game, curr_opponent, is_opponent_black=False)

        if not self._is_threshold_met(curr_opponent):
            return

        # vs. Maia 1400 Elo
        curr_opponent = "Maia1400"
        self.results["games"][curr_opponent] = []
        white = self.agent
        with MaiaAgent(self.maia_weights["maia_1400"], self.lc0_exe) as black:
            for i in range(games_per_color):
                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, white, black)
                self._add_result(game, curr_opponent, is_opponent_black=True)

                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, black, white)
                self._add_result(game, curr_opponent, is_opponent_black=False)

        if not self._is_threshold_met(curr_opponent):
            return

        # vs. Maia 1600 Elo
        curr_opponent = "Maia1600"
        self.results["games"][curr_opponent] = []
        white = self.agent
        with MaiaAgent(self.maia_weights["maia_1600"], self.lc0_exe) as black:
            for i in range(games_per_color):
                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, white, black)
                self._add_result(game, curr_opponent, is_opponent_black=True)

                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, black, white)
                self._add_result(game, curr_opponent, is_opponent_black=False)

        if not self._is_threshold_met(curr_opponent):
            return

        # vs. Maia 1900 Elo
        curr_opponent = "Maia1900"
        self.results["games"][curr_opponent] = []
        white = self.agent
        with MaiaAgent(self.maia_weights["maia_1900"], self.lc0_exe) as black:
            for i in range(games_per_color):
                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, white, black)
                self._add_result(game, curr_opponent, is_opponent_black=True)

                board = chess.Board(self.position_fens[i])
                game = self._play_game(board, black, white)
                self._add_result(game, curr_opponent, is_opponent_black=False)

    def save_results(self) -> None:
        with self.output_file.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

    def _play_game(
        self, board: chess.Board, white: ChessAgent, black: ChessAgent
    ) -> Game:
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = white.predict(board)
                board.push(move)
            else:
                move = black.predict(board)
                board.push(move)

        reason = "Unknown"
        if board.is_checkmate():
            reason = "Normal (Checkmate)"
        elif board.is_stalemate():
            reason = "Draw by Stalemate"
        elif board.is_insufficient_material():
            reason = "Draw by Insufficient Material"
        elif board.is_seventyfive_moves():
            reason = "Draw by 75-move rule"
        elif (
            board.is_fivefold_repetition()
            or board.can_claim_threefold_repetition()
        ):
            reason = "Draw by Repetition"

        game = chess.pgn.Game.from_board(board)
        game.headers["Result"] = board.result()
        game.headers["Termination"] = reason
        game.headers["White"] = white.__class__.__name__
        game.headers["Black"] = black.__class__.__name__
        game.headers["Variant"] = "From Position"

        return game

    def _add_result(
        self, game: chess.pgn.Game, opponent_name: str, is_opponent_black: bool
    ) -> None:
        if game.headers["Result"] == "1/2-1/2":
            self.results[opponent_name]["draws"] += 1
        elif game.headers["Result"] == "1-0":
            if is_opponent_black:
                self.results[opponent_name]["wins"] += 1
            else:
                self.results[opponent_name]["losses"] += 1
        elif game.headers["Result"] == "0-1":
            if is_opponent_black:
                self.results[opponent_name]["losses"] += 1
            else:
                self.results[opponent_name]["wins"] += 1

        self.results["games"][opponent_name].append(str(game))

    def _is_threshold_met(self, opponent_name: str) -> bool:
        return (
            self.results[opponent_name]["wins"]
            / (
                self.results[opponent_name]["wins"]
                + self.results[opponent_name]["losses"]
                + self.results[opponent_name]["draws"]
            )
            >= self.pass_threshold
        )


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chess Agent Benchmark")
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play against each opponent",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="File to save the benchmark results. Defaults to '<agent_name>_<YYYYMMDD_HHMM>.json'",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.5,
        help="Minimum win rate required to move on to next, more diffictult, opponent (default: 0.5)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Run the benchmark deterministically. Defaults to False.",
    )

    subparsers = parser.add_subparsers(
        dest="agent_name", required=True, help="Target agent to benchmark"
    )
    for name, agent_cls in AGENT_REGISTRY.items():
        agent_parser = subparsers.add_parser(
            name, help=f"Benchmark the {name} agent"
        )
        sig = inspect.signature(agent_cls.__init__)
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = (
                param.annotation
                if param.annotation != inspect.Parameter.empty
                else str
            )
            if param.default == inspect.Parameter.empty:
                agent_parser.add_argument(
                    f"--{param_name}", type=param_type, required=True
                )
            else:
                if isinstance(param_type, bool):
                    action = "store_false" if param.default else "store_true"
                    agent_parser.add_argument(f"--{param_name}", action=action)
                else:
                    agent_parser.add_argument(
                        f"--{param_name}",
                        type=param_type,
                        default=param.default,
                    )
    return parser


def main():
    parser = _build_cli()
    args = parser.parse_args()
    all_args = vars(args)

    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        m_name = getattr(args, "agent_name", "unknown")
        args.output_file = f"{m_name}_benchmark_{timestamp}.json"

    # benchmark-level options
    agent_name = all_args.pop("agent_name")
    games = all_args.pop("games")
    pass_threshold = all_args.pop("pass_threshold")
    deterministic = all_args.pop("deterministic")
    output_file = all_args.pop("output_file")

    # agent-level options
    target_class = AGENT_REGISTRY[agent_name]
    agent = target_class(**all_args)

    bench = ChessBenchmark(
        n_games=games,
        agent=agent,
        output_file=output_file,
        pass_threshold=pass_threshold,
        deterministic=deterministic,
    )
    bench.run()
    bench.save_results()


if __name__ == "__main__":
    main()
