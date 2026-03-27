import abc

import chess


class ChessAgent(abc.ABC):
    """Base class for chess agents. Wraps models, meant to handle loading saved etc."""

    @abc.abstractmethod
    def predict(self, board: chess.Board) -> chess.Move:
        pass
