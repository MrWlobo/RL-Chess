import chess
import numpy as np
import torch
from torch import nn

from rl_chess.utils.train_utils import (
    boards_to_tensor,
    index_to_move,
    move_to_index,
)


class MCTSNode:
    __slots__ = ["fen", "parent", "children", "n", "q", "p"]

    def __init__(self, parent: "MCTSNode" = None, prior_p: float = 0):
        self.parent: MCTSNode = parent
        self.children: dict[chess.Move, MCTSNode] = {}
        self.n = 0  # Liczba odwiedzin (N)
        self.q = 0  # Średnia wartość (średnie value z sieci)
        self.p = prior_p  # Prawdopodobieństwo z sieci (policy z sieci)

    def value(self, c_puct: float) -> float:
        # Implementacja wzoru PUCT: Q (avg network value) + U (combined policy and n_visits metric)
        u = c_puct * self.p * np.sqrt(self.parent.n) / (1 + self.n)
        return self.q + u


class MonteCarloTreeSearch:
    def __init__(
        self,
        c_puct: float = 1.4,
        num_searches: int = 100,
        max_boards: int = 100,
        alpha: float = 0,
        epsilon: float = 0,
        training_mode: bool = False,
    ):
        self.c_puct = c_puct
        self.num_searches = num_searches
        self.stored_boards = [chess.Board() for _ in range(max_boards)]
        self.alpha = alpha
        self.epsilon = epsilon
        self.training_mode = training_mode

    def get_pi(self, root, num_moves, temperature=0.01):
        visits = np.array([node.n for node in root.children.values()])
        moves = list(root.children.keys())

        if len(root.children) == 0:
            raise ValueError("Root node has no children")

        if temperature <= 1e-3:
            pi = np.zeros(num_moves)
            best_move = max(root.children.items(), key=lambda x: x[1].n)[0]
            pi[move_to_index(best_move)] = 1.0
            return pi

        logits = np.log(visits + 1e-8) / temperature
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)

        pi = np.zeros(num_moves)
        indices = [move_to_index(m) for m in moves]
        pi[indices] = probs

        return pi

    def sample_from_pi(self, pi):
        pi = np.asarray(pi, dtype=np.float64)
        s = pi.sum()
        if s <= 0:
            raise ValueError("pi sum is zero")
        pi = pi / s
        index = np.random.choice(len(pi), p=pi)
        return index_to_move(index)

    def _create_childs_with_noise(
        self,
        move_count: int,
        nodes: list[MCTSNode],
        initial_fens: list[str],
        alpha: float,
        epsilon: float,
    ):
        with torch.no_grad():
            policy_list, _ = self.neural_network(
                boards_to_tensor(boards=self.active_boards, device=self.device)
            )
            policy_list = [p.cpu().numpy() for p in policy_list]
            for idx, node in enumerate(nodes):
                self.active_boards[idx].set_fen(initial_fens[idx])
                board = self.active_boards[idx]
                legal_moves = list(board.legal_moves)
                move_indices = [move_to_index(m) for m in legal_moves]
                logits = policy_list[idx][move_indices]
                probs = self._normalize_logits(logits)

                for i, move in enumerate(legal_moves):
                    node.children[move] = MCTSNode(
                        parent=node, prior_p=probs[i]
                    )

                if self.training_mode:
                    # alpha is distribution of noise, lower alpha -> more noise
                    # if epsilon = 0.25  # 25% random, 75% policy
                    children = list(node.children.values())
                    noise = np.random.dirichlet([alpha] * len(children))
                    for i, child in enumerate(children):
                        child.p = (1 - epsilon) * child.p + epsilon * noise[i]

    def _normalize_logits(self, logits: np.ndarray) -> np.ndarray:
        if len(logits) > 0:
            exp_logits = np.exp(logits - np.max(logits))
            move_probs = exp_logits / np.sum(exp_logits)
        else:
            move_probs = []
        return move_probs

    def batch_search(
        self,
        move_count: int,
        initial_fens: list[str],
        neural_network: nn.Module,
        device: torch.device,
    ) -> tuple[list[chess.Move], list[np.ndarray]]:

        n = len(initial_fens)
        if len(self.stored_boards) < n:
            self.stored_boards += [
                chess.Board() for _ in range(n - len(self.stored_boards))
            ]
        self.active_boards = self.stored_boards[:n]
        self.neural_network = neural_network
        self.device = device
        searches = self.num_searches

        roots = [MCTSNode() for _ in range(n)]
        self._create_childs_with_noise(
            move_count=move_count,
            nodes=roots,
            initial_fens=initial_fens,
            alpha=self.alpha,
            epsilon=self.epsilon,
        )

        # print("Starting  MCTS batch search...")

        with torch.no_grad():
            for _ in range(searches):
                nodes: list[MCTSNode] = list(roots)
                search_paths: list[list[MCTSNode]] = [[n] for n in nodes]

                for idx, node in enumerate(nodes):
                    self.active_boards[idx].set_fen(initial_fens[idx])
                    while node.children:
                        move, node = max(
                            node.children.items(),
                            key=lambda x: x[1].value(self.c_puct),
                        )
                        search_paths[idx].append(node)
                        self.active_boards[idx].push(move)
                    nodes[idx] = node
                policy_list, value_list = self.neural_network(
                    boards_to_tensor(
                        boards=self.active_boards, device=self.device
                    )
                )
                policy_tensor = policy_list.cpu().numpy()
                value_list = value_list.cpu().numpy()

                for idx, node in enumerate(nodes):
                    board = self.active_boards[idx]
                    if board.is_game_over():
                        if board.is_checkmate():
                            value_list[idx] = -1.0
                        else:
                            value_list[idx] = 0
                        self.backpropagate(search_paths[idx], value_list[idx])
                        continue

                    legal_moves = list(board.legal_moves)
                    move_indices = [move_to_index(move) for move in legal_moves]
                    logits = policy_tensor[idx, move_indices]

                    move_probs = self._normalize_logits(logits)

                    for i, move in enumerate(legal_moves):
                        node.children[move] = MCTSNode(
                            parent=node, prior_p=move_probs[i]
                        )

                    self.backpropagate(search_paths[idx], value_list[idx])

        temperature = 1.0 if self.training_mode and move_count < 30 else 0.0

        pi_targets_list = [
            self.get_pi(root=r, num_moves=4096, temperature=temperature)
            for r in roots
        ]
        moves = []
        for i in range(n):
            if temperature > 0:
                chosen_move = self.sample_from_pi(pi_targets_list[i])
            else:
                chosen_move = max(
                    roots[i].children.items(), key=lambda x: x[1].n
                )[0]

            moves.append(chosen_move)
        return moves, pi_targets_list

    def backpropagate(self, path: list[MCTSNode], value: float):
        for node in reversed(path):
            node.q = (node.n * node.q + value) / (node.n + 1)
            node.n += 1
            value = -value
