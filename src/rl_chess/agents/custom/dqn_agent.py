from pathlib import Path
from typing import override

import chess
import torch

from rl_chess.agents.base import ChessAgent
from rl_chess.models.dqn.chess_dqn import (
    ChessCNN,
    board_to_tensor,
    get_best_legal_move,
)


class DQNAgent(ChessAgent):
    def __init__(self, model_path: Path) -> None:
        num_actions = 64 * 64
        self.model = ChessCNN(out_actions=num_actions)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    @override
    def predict(self, board: chess.Board) -> chess.Move:
        with torch.no_grad():
            input_tensor = board_to_tensor(board)
            raw_output = self.model(input_tensor)

            return get_best_legal_move(raw_output[0], board)
