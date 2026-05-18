import chess
import numpy as np
import torch
from torch import nn

from rl_chess.utils.train_utils import (
    board_to_tensor,
    boards_to_tensor,
    move_to_index,
)


class MCTSNode:
    def __init__(self, fen: str, parent: "MCTSNode" = None, prior_p: float = 0):
        self.fen: str = fen  # Stan planszy (np. fen)
        self.parent: MCTSNode = parent
        self.children: dict[chess.Move, MCTSNode] = {}
        self.n = 0  # Liczba odwiedzin (N)
        self.q = 0  # Średnia wartość (średnie value z sieci)
        self.p = prior_p  # Prawdopodobieństwo z sieci (policy z sieci)
        self.active_board = chess.Board()

    def value(self, c_puct: float) -> float:
        # Implementacja wzoru PUCT: Q + U
        u = c_puct * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return self.q + u


class MonteCarloTreeSearch:
    def __init__(
        self,
        c_puct: float = 1.4,
        num_simulations: int = 100,
        max_boards: int = 100,
    ):
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.active_board = chess.Board()
        self.active_boards = [chess.Board() for _ in range(max_boards)]

    def multi_search(
        self,
        initial_fens: list[str],
        neural_network: nn.module,
        device: torch.device,
    ):
        if len(self.active_boards) < len(initial_fens):
            self.active_boards += [
                chess.Board()
                for _ in range(len(initial_fens) - len(self.active_boards))
            ]
        roots = [MCTSNode(fen) for fen in initial_fens]
        print("Starting  MCTS multi search...")

        with torch.no_grad():
            for _ in range(self.num_simulations):
                nodes: list[MCTSNode] = roots
                search_paths: list[list[MCTSNode]] = [[n] for n in nodes]

                for idx, node in enumerate(nodes):
                    while node.children:
                        node = max(
                            node.children.values(),
                            key=lambda x: x.value(self.c_puct),
                        )
                        search_paths[idx].append(node)
                    self.active_boards[idx].set_fen(node.fen)
                    nodes[idx] = node

                policy_list, value_list = neural_network(
                    boards_to_tensor(boards=self.active_boards, device=device)
                )
                policy_list = [p.cpu for p in policy_list]

                for idx, node in enumerate(nodes):
                    for move in self.active_boards[idx].legal_moves:
                        p_move = policy_list[idx][move_to_index(move)]
                        self.active_boards[idx].push(move)
                        node.children[move] = MCTSNode(
                            self.active_boards[idx].fen(),
                            parent=node,
                            prior_p=p_move,
                        )
                        self.active_boards[idx].pop()

                    self.backpropagate(search_paths[idx], value_list[idx])

    def search(
        self,
        initial_fen: str,
        neural_network: nn.Module,
        device: torch.device,
    ) -> chess.Move:
        root = MCTSNode(initial_fen)
        print("Starting MCTS search...")

        with torch.no_grad():
            for _ in range(self.num_simulations):
                node = root
                search_path = [node]

                while node.children:
                    node = max(
                        node.children.values(),
                        key=lambda x: x.value(self.c_puct),
                    )
                    search_path.append(node)

                self.active_board.set_fen(node.fen)
                policy, value = neural_network(
                    board_to_tensor(board=self.active_board, device=device)
                )
                policy = policy.cpu()

                legal_moves = self.active_board.legal_moves

                for move in legal_moves:
                    p_move = policy[0][move_to_index(move)]
                    self.active_board.push(move)
                    node.children[move] = MCTSNode(
                        self.active_board.fen(), parent=node, prior_p=p_move
                    )
                    self.active_board.pop()

                self.backpropagate(search_path, value)

        return max(root.children.items(), key=lambda x: x[1].n)[0]

    def backpropagate(self, path: list[MCTSNode], value: float):
        for node in reversed(path):
            node.q = (node.n * node.q + value) / (node.n + 1)
            node.n += 1
            value = -value
