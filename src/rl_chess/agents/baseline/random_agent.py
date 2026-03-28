import random
from typing import override

import chess

from rl_chess.agents.base import ChessAgent


class RandomAgent(ChessAgent):
    @override
    def predict(self, board: chess.Board) -> chess.Move:
        legal_moves = board.legal_moves

        return random.choice(list(legal_moves))
