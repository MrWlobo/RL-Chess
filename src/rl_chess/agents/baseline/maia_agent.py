from pathlib import Path

import chess
import chess.engine


class MaiaAgent:
    def __init__(self, weight_path: Path, exe_path: Path):
        self.weight_path = weight_path
        self.exe_path = exe_path
        self.engine: chess.engine.SimpleEngine | None = None
        self.limit = chess.engine.Limit(nodes=1)

    def __enter__(self):
        """Starts the engine process."""
        self.engine = chess.engine.SimpleEngine.popen_uci(
            [str(self.exe_path), f"--weights={self.weight_path}"]
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Kils the engine process when leaving the 'with' block."""
        if self.engine:
            self.engine.quit()
            self.engine = None

    def predict(self, board: chess.Board) -> chess.Move:
        if not self.engine:
            raise RuntimeError(
                "Engine not started. Use 'with MaiaAgent(...) as agent:'"
            )
        result = self.engine.play(board, self.limit)
        return result.move  # ty:ignore[invalid-return-type]
