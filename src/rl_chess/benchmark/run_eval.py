from rl_chess.benchmark.config import BenchmarkConfig


class ChessBenchmark:
    def __init__(self, config: BenchmarkConfig) -> None:
        positions_file = config.positions.path
        self.position_fens = []
        with positions_file.open("r") as f:
            for fen in f:
                self.position_fens.append(fen)


config = BenchmarkConfig()  # ty:ignore[missing-argument]
bench = ChessBenchmark(config)
