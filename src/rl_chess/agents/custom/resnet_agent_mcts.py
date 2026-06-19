from pathlib import Path
from typing import override

import chess
import torch

from rl_chess.agents.base import ChessAgent
from rl_chess.models.resnet.chess_res import ChessResNet
from rl_chess.utils.MonteCarloTreeSearch import MonteCarloTreeSearch
from rl_chess.utils.train_utils import (
    get_next_moves,
)


class ResNetMCTSAgent(ChessAgent):
    def __init__(self, model_path: Path) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = ChessResNet()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.MCTS = MonteCarloTreeSearch(num_searches=20)

    @override
    def predict(self, board: chess.Board) -> chess.Move:
        with torch.no_grad():
            return get_next_moves(
                boards=[board],
                neural_network=self.model,
                device=self.device,
                move_search=self.MCTS,
            )[0][0]
