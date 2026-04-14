import time

import chess
import numpy as np
import torch
from torch import nn

from rl_chess.utils.training import board_to_tensor, move_to_index


class MCTSNode:
    def __init__(
        self, state: str, parent: "MCTSNode" = None, prior_p: float = 0
    ):
        self.state = state  # Stan planszy (np. fen)
        self.parent = parent
        self.children = {}  # Słownik: {move: MCTSNode}
        self.n = 0  # Liczba odwiedzin (N)
        self.q = 0  # Średnia wartość (Q)
        self.p = prior_p  # Prawdopodobieństwo z sieci (P)
        self.temp_board = chess.Board()

    def value(self, c_puct: float) -> float:
        # Implementacja wzoru PUCT: Q + U
        u = c_puct * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return self.q + u


class MonteCarloTreeSearch:
    def __init__(self, c_puct: float = 1.4, num_simulations: int = 100):
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temp_board = chess.Board()

    def search(
        self,
        initial_state: str,
        neural_network: nn.Module,
        device: torch.device,
    ) -> chess.Move:
        root = MCTSNode(initial_state)

        tic = time.time()
        print("Starting MCTS search...")

        with torch.no_grad():
            for i in range(self.num_simulations):
                node = root
                search_path = [node]

                # 1. SELEKCJA
                while node.children:
                    move, node = max(
                        node.children.items(),
                        key=lambda x: x[1].value(self.c_puct),
                    )
                    search_path.append(node)

                # 2. EKSPANSJA I EWALUACJA (Twój ResNet)
                # Przygotuj tensor dla sieci
                self.temp_board.set_fen(node.state)
                policy, value = neural_network(
                    board_to_tensor(board=self.temp_board, device=device)
                )
                policy = policy.cpu()

                # Stwórz dzieci dla wszystkich legalnych ruchów
                legal_moves = self.temp_board.legal_moves

                for move in legal_moves:
                    # Pobierz p dla tego ruchu z wyjścia sieci (głowa Policy)
                    p_move = policy[0][move_to_index(move)]
                    self.temp_board.push(move)
                    node.children[move] = MCTSNode(
                        self.temp_board.fen(), parent=node, prior_p=p_move
                    )
                    self.temp_board.pop()

                print(
                    f"MCTS cycle {i + 1} before backpropagation, time: {time.time() - tic:.5f}"
                )

                # 3. BACKPROPAGATION
                self.backpropagate(search_path, value)
                print(
                    f"MCTS cycle {i + 1} completed, time: {time.time() - tic:.5f}"
                )
        toc = time.time()
        print(f"MCTS Time: {toc - tic}")

        return max(root.children.items(), key=lambda x: x[1].n)[0]

    def backpropagate(self, path: list[MCTSNode], value: float):
        # Pamiętaj o naprzemienności: wartość dla przeciwnika jest ujemna
        for node in reversed(path):
            node.q = (node.n * node.q + value) / (node.n + 1)
            node.n += 1
            value = -value  # Zmiana perspektywy gracza
