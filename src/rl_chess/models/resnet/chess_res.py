import random
import time
from collections import deque
from pathlib import Path

import chess
import gym
import gym_chess  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_chess.utils.MonteCarloTreeSearch import MonteCarloTreeSearch
from rl_chess.utils.training import (
    board_to_tensor,
    boards_to_tensor,
    get_next_moves,
    move_to_index,
)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection!
        return F.leaky_relu(out, negative_slope=0.01)


class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=10, channels=128):
        super().__init__()
        # Warstwa wejściowa
        self.start_conv = nn.Conv2d(17, channels, kernel_size=3, padding=1)
        self.bn_start = nn.BatchNorm2d(channels)

        # Wieża rezydualna (Tu 3060 Ti pokaże moc)
        self.res_blocks = nn.ModuleList(
            [ResBlock(channels) for _ in range(num_res_blocks)]
        )

        # Policy Head (Ruchy)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)

        # Value Head (Ocena pozycji)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn_start(self.start_conv(x)), negative_slope=0.01)

        for block in self.res_blocks:
            x = block(x)

        # Policy
        p = F.leaky_relu(self.policy_conv(x), negative_slope=0.01)
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)

        # Value
        v = F.leaky_relu(self.value_conv(x), negative_slope=0.01)
        v = v.view(v.size(0), -1)
        v = F.leaky_relu(self.value_fc1(v), negative_slope=0.01)
        value = torch.tanh(self.value_fc2(v))

        return policy, value


class ReplayMemory:
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def execute_move_with_reward(board: chess.Board, move: chess.Move) -> float:
    custom_reward = 0.0
    captured_piece = board.piece_at(move.to_square)
    piece_value_scale = 0.1

    if captured_piece:
        val = PIECE_VALUES[captured_piece.piece_type] * piece_value_scale
        custom_reward += val
    custom_reward -= 0.005

    if move.promotion:  # pawn promotion -> always hetman
        custom_reward += (
            PIECE_VALUES[chess.QUEEN] - PIECE_VALUES[chess.PAWN]
        ) * piece_value_scale

    board.push(move)

    if board.is_repetition(3):
        custom_reward = -0.2

    if board.is_game_over():
        custom_reward = 1.0 if board.is_checkmate() else 0.0

    return np.clip(custom_reward, -1, 1)


class ChessRES:
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.99  # discount rate (gamma)
    replay_memory_size = 200_000  # size of replay memory
    mini_batch_size = (
        1024  # size of the training data set sampled from the replay memory
    )
    batches_per_cycle = 30  # number of batches to train on per cycle

    # Neural Network
    loss_fn = nn.SmoothL1Loss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None  # NN Optimizer. Initialize later.

    def __init__(self, device_type="cpu"):
        if device_type in ["cuda", "cpu"]:
            ChessRES.device = torch.device(device_type)
        else:
            raise ValueError(
                f"Device type {device_type} is not supported. Must be one of ['cuda', 'cpu']"
            )
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if ChessRES.device.type == "cuda"
            else None
        )

    def _simulate_games(self, memory: ReplayMemory, episodes: int):
        boards_wrapper = {i: chess.Board() for i in range(episodes)}
        active_boards = boards_wrapper.copy()
        while active_boards:
            current_keys = list(active_boards.keys())
            num_boards = len(active_boards)
            if self.verbose:
                print(f"\rGames left: {num_boards}  ", end="")
            random_mask = [
                random.random() < self.epsilon for _ in range(num_boards)
            ]
            rand_indices = {
                current_keys[i] for i, x in enumerate(random_mask) if x == 1
            }

            random_boards_dict = {i: active_boards[i] for i in rand_indices}
            if random_boards_dict:
                random_boards = list(random_boards_dict.values())
                random_moves = [
                    random.choice(list(board.legal_moves))
                    for board in random_boards
                ]

                for (i, board), move in zip(
                    random_boards_dict.items(), random_moves, strict=True
                ):
                    fen = board.fen()
                    reward = execute_move_with_reward(board, move)
                    memory.append((fen, move, reward))
                    if board.is_game_over():
                        active_boards.pop(i)

            res_boards_dict = {
                i: board
                for i, board in active_boards.items()
                if i not in rand_indices
            }
            if res_boards_dict:
                res_boards = list(res_boards_dict.values())

                res_moves = get_next_moves(
                    boards=res_boards,
                    neural_network=self.policy_res,
                    device=ChessRES.device,
                    move_search=self.move_search,
                )

                for (i, board), move in zip(
                    res_boards_dict.items(), res_moves, strict=True
                ):
                    fen = board.fen()
                    reward = execute_move_with_reward(board, move)
                    memory.append((fen, move, reward))
                    if board.is_game_over():
                        active_boards.pop(i)

    def train(
        self,
        episodes: int,
        cycles: int,
        render: bool = False,
        verbose: bool = False,
        epsilon: float = 1.0,
        epsilon_decrease=None,
        file=None,
        keep_training=False,
        move_search=None,
    ):
        """
        args:
            episodes (int): Number of games to play
            cycles (int): Number of game cycles
            render (bool): Whether to render the game or not
            verbose (bool): Whether to print information about the game
            epsilon_decrease (float): Decrease in epsilon after each episode
            file (str): File to save the network to
            keep_training (bool): Whether to load and keep training or start from scratch
        """

        self.epsilon = epsilon
        self.episodes = episodes
        self.cycles = cycles
        self.render = render
        self.verbose = verbose
        self.epsilon_decrease = (
            1 / cycles if epsilon_decrease is None else epsilon_decrease
        )
        self.move_search = move_search
        memory = ReplayMemory(maxlen=self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        self.policy_res = ChessResNet().to(ChessRES.device)
        self.target_res = ChessResNet().to(ChessRES.device)

        if file is None:
            file = "chess_dqn.pt"

        file_path = Path(__file__).parent.resolve() / "trained" / file

        # Load network if file is given
        if keep_training and file_path.exists():
            self.policy_res.load_state_dict(
                torch.load(
                    file_path,
                    map_location=ChessRES.device,
                    weights_only=True,
                )
            )
            print("Network loaded from file: " + str(file_path))

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_res.load_state_dict(self.policy_res.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(
            self.policy_res.parameters(), lr=self.learning_rate_a
        )

        for cycle in range(cycles):
            if verbose:
                print(f"cycle {cycle}, epsilon: {self.epsilon:.4f}")
                print(f"Simulating {episodes} games...")
            tic = time.time()

            self._simulate_games(memory, episodes)

            toc = time.time()
            if self.verbose:
                print(f"\rSimulation time: {(toc - tic):.4f}")

            tic = time.time()
            if self.verbose:
                print("Training policy network...")
            moves_optimized = 0
            for _ in range(
                self.batches_per_cycle
            ):  # Zrób 10 kroków nauki na cykl
                if len(memory) > self.mini_batch_size:
                    batch = memory.sample(self.mini_batch_size)
                    moves_optimized += len(batch)
                    self.optimize(batch)

            # Decay epsilon
            self.epsilon = max(self.epsilon - self.epsilon_decrease, 0.05)

            # Copy policy network to target network
            self.target_res.load_state_dict(self.policy_res.state_dict())
            current_file = (
                file[: file.find(".")] + "_" + str(cycle // 100) + ".pt"
            )
            current_file_path = (
                Path(__file__).parent.resolve() / "trained" / (current_file)
            )
            torch.save(self.policy_res.state_dict(), current_file_path)
            toc = time.time()
            if self.verbose:
                if moves_optimized > 0:
                    print(
                        f"Training time: {(toc - tic):.4f}, {((toc - tic) / moves_optimized):.4f} per move",
                        end="\n",
                    )
                else:
                    print("No optimization performed.")

        # Save policy
        torch.save(self.policy_res.state_dict(), file_path)
        print("\nTraining complete.")

    # Optimize policy network
    def optimize(self, batch):
        fens, moves, rewards = zip(*batch, strict=True)
        boards = [chess.Board(fen=f) for f in fens]

        tensor_input_curr = boards_to_tensor(
            boards=boards, device=ChessRES.device
        )
        logits, values = self.policy_res(tensor_input_curr)

        # --- POLICY LOSS (Głowica Ruchów) ---
        move_indices = torch.tensor(
            [move_to_index(m) for m in moves], device=ChessRES.device
        )
        policy_loss = F.cross_entropy(logits, move_indices)

        # make a move
        for i, board in enumerate(boards):
            board.push(moves[i])

        new_boards = boards

        # --- VALUE LOSS (Głowica Oceny Pozycji) ---
        with torch.no_grad():
            active_indices = [
                i for i, b in enumerate(new_boards) if not b.is_game_over()
            ]
            target_values = (
                torch.tensor(rewards, device=ChessRES.device)
                .float()
                .unsqueeze(1)
            )

            if active_indices:
                active_boards = [new_boards[i] for i in active_indices]
                tensor_input_next = boards_to_tensor(
                    boards=active_boards, device=ChessRES.device
                )
                _, next_values = self.target_res(tensor_input_next)
                target_values[active_indices] = (
                    target_values[active_indices]
                    - (self.discount_factor_g * next_values)
                ).clamp(-1.0, 1.0)

        value_loss = F.mse_loss(values, target_values.detach())

        # --- TOTAL LOSS ---
        # Łączymy obie straty. Możesz dodać mnożnik (np. 0.5) dla value_loss,
        # aby zbalansować naukę ruchów i oceny.
        total_loss = policy_loss + (value_loss * 0.1)

        self.optimizer.zero_grad()

        if ChessRES.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                # Ponowne obliczenie jeśli używasz autocast dla stabilności
                loss = total_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        return total_loss.item()

    # Run the Chess environment with the learned policy
    def test(
        self,
        episodes: int = 1,
        verbose: bool = True,
        file: str = "chess_res.pt",
        move_search: MonteCarloTreeSearch | None = None,
    ):
        # Create Chess instance
        env = gym.make("Chess-v0")

        file_path = Path(__file__).parent.resolve() / "trained" / file

        # Load learned policy
        policy_res = ChessResNet().to(ChessRES.device)
        policy_res.load_state_dict(
            torch.load(
                file_path, map_location=ChessRES.device, weights_only=True
            )
        )
        policy_res.eval()  # switch model to evaluation mode

        for i in range(episodes):
            board = env.reset()  # Initialize to state 0
            with torch.no_grad():
                _, value = policy_res(
                    board_to_tensor(board=board, device=ChessRES.device)
                )
            if verbose:
                print(env.render())
                print(f"Ocena pozycji (Value): {value.item():.4f}", end="\n\n")
            done = False  # True when game is over
            turn = i
            while not done:
                # Select best move
                with torch.no_grad():
                    _, value = policy_res(
                        board_to_tensor(board=board, device=ChessRES.device)
                    )

                if turn % 2 == 0:
                    # Ruch sieci
                    move = get_next_moves(
                        boards=[board],
                        neural_network=policy_res,
                        device=ChessRES.device,
                        move_search=move_search,
                    )[0]
                else:
                    # Losowy ruch
                    move = random.choice(list(board.legal_moves))

                # Execute move
                new_board, reward, done, info = env.step(move)
                if verbose:
                    print(env.render())
                    print(f"Ocena pozycji (Value): {value.item():.4f}")
                    if turn % 2 == 0:
                        print("^^^ Ruch sieci")
                    else:
                        print("^^^ Ruch losowy")
                    print()
                board = new_board

                turn += 1

            # print(f"end: {board.is_game_over()}")
            # print(f"checkmate: {board.is_checkmate()}")

            print("\n" + "=" * 30)
            if i % 2 == 0:
                print("Białe - Sieć, Czarne - Losowy ruch")
            else:
                print("Białe - Losowy ruch, Czarne - Sieć")
            res = board.result()  # Zwraca "1-0", "0-1", "1/2-1/2" lub "*"

            if res == "1-0":
                print("WYNIK: 1-0 (Wygrana Białych)")
            elif res == "0-1":
                print("WYNIK: 0-1 (Wygrana Czarnych)")
            elif res == "1/2-1/2":
                print("WYNIK: 1/2-1/2 (Remis)")
            else:
                print("WYNIK: Gra nie została rozstrzygnięta (*)")

            # Szczegółowy powód na podstawie flag, które mi wysłałeś:
            if board.is_checkmate():
                print("POWÓD: Mat")
            elif board.is_stalemate():
                print("POWÓD: Pat")
            elif board.is_insufficient_material():
                print("POWÓD: Niewystarczający materiał")
            elif board.is_seventyfive_moves():
                print("POWÓD: Zasada 75 ruchów")
            elif board.is_fivefold_repetition():
                print("POWÓD: Pięciokrotne powtórzenie")

            print("=" * 30)
        env.close()


if __name__ == "__main__":
    # 1. Czy CUDA jest dostępna?
    print(f"Czy CUDA jest dostępna? {torch.cuda.is_available()}")

    # 2. Ile urządzeń CUDA widzi system?
    print(f"Liczba dostępnych GPU: {torch.cuda.device_count()}")

    # 3. Nazwa aktualnie używanej karty
    if torch.cuda.is_available():
        print(f"Nazwa karty: {torch.cuda.get_device_name(0)}")
    chess_res = ChessRES(
        device_type="cuda"
    )  # "cpu" or "cuda", cpu works better using small models

    MCTS = MonteCarloTreeSearch(c_puct=1.4, num_simulations=10)

    train_params = {
        "episodes": 100,  # episodes per cycle
        "cycles": 2000,
        "epsilon": 0.4,
        "epsilon_decrease": (
            1 / 1600
        ),  # decault decay (epsilon_decrease = 1/cycles)
        "file": "chess_res2.pt",
        "verbose": True,
        "keep_training": True,
        "move_search": MCTS,
    }
    chess_res.train(**train_params)

    test_params = {
        "episodes": 1,
        # "file": "chess_res1.pt",
        "file": "chess_res2_0.pt",
        "verbose": True,
        "move_search": MCTS,
    }
    chess_res.test(**test_params)
