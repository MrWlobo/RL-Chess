import gym
import gym_chess
import random
import time
import chess
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import os


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def get_custom_reward(board: chess.Board, move: chess.Move) -> float:
    custom_reward = 0.0
    captured_piece = board.piece_at(move.to_square)
    
    if captured_piece:
        val = PIECE_VALUES[captured_piece.piece_type] * 0.1
        # custom_reward += val 
    if board.is_game_over():
        if board.is_checkmate():
            custom_reward = 10.0
        else:
            custom_reward = 0.0
    # custom_reward -= board.fullmove_number * 0.001
    return custom_reward

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection!
        return F.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_res_blocks=10, channels=128):
        super().__init__()
        # Warstwa wejściowa
        self.start_conv = nn.Conv2d(17, channels, kernel_size=3, padding=1)
        self.bn_start = nn.BatchNorm2d(channels)
        
        # Wieża rezydualna (Tu 3060 Ti pokaże moc)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        
        # Policy Head (Ruchy)
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 8 * 8, 4096)
        
        # Value Head (Ocena pozycji)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn_start(self.start_conv(x)))
        
        for block in self.res_blocks:
            x = block(x)
            
        # Policy
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)
        
        # Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        # return policy, value
        return policy
    
class ReplayMemory():
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

def board_to_array(board: chess.Board) -> np.ndarray:
    matrix = np.zeros((17, 8, 8), dtype=np.float32)
    is_black_turn = (board.turn == chess.BLACK)
    piece_to_layer = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:    # chess.SQUARES idą od 0 do 63 (A1, B1... H8)
        piece = board.piece_at(square)
        if piece:
            
            row = chess.square_rank(square) # 0-7
            col = chess.square_file(square) # 0-7
            
            # change perspective logic
            # own pieces on layers 0-5, enemy pieces are on layers 6-11
            layer = piece_to_layer[piece.piece_type]
            if is_black_turn:
                if piece.color == chess.WHITE:
                    layer += 6
            else:
                if piece.color == chess.BLACK:
                    layer += 6
            
            matrix[layer, row, col] = 1.0
 
    if is_black_turn:
        if board.has_kingside_castling_rights(chess.BLACK): matrix[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): matrix[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE): matrix[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): matrix[15, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.WHITE): matrix[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): matrix[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): matrix[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): matrix[15, :, :] = 1.0

    # 18. Zasada 50 ruchów (znormalizowana do zakresu 0-1)
    matrix[16, :, :] = board.halfmove_clock / 100.0
    return matrix

def boards_to_tensor(boards: list[chess.Board]):
    array_list = [board_to_array(board=board) for board in boards]
    single_ndarray = np.array(array_list)
    return torch.from_numpy(single_ndarray).to(ChessRES.device).float()

def ensure_queen_promotion(board: chess.Board, move: chess.Move) -> chess.Move:
    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(move.to_square)
        if rank in [0, 7]:
            move.promotion = chess.QUEEN
                
    return move

def move_from_output(nn_output: int) -> chess.Move:
    move_from_idx = nn_output // 64
    move_to_idx = nn_output % 64
    
    # chess.square_name(0) -> "a1", chess.square_name(63) -> "h8"
    from_square = chess.square_name(move_from_idx)
    to_square = chess.square_name(move_to_idx)
    if from_square == to_square:
        return None
    return chess.Move.from_uci(from_square + to_square)

def get_best_legal_move(output: torch.Tensor, board: chess.Board) -> chess.Move:
    sorted_indices = output.flatten().argsort(descending=True)
    legal_moves = set(board.legal_moves)
    for output in sorted_indices:
        move = move_from_output(output)
        if move == None:
            continue
        move = ensure_queen_promotion(board=board, move=move)
        if move in legal_moves:
            return move

def rewards_of_next_move(original_boards: list[chess.Board], neural_network: nn.Module) -> tuple[list[float], list[chess.Board]]:
    # Returns sum rewards of next move.
    # Also modifies original_boards IN PLACE.
    batch_size = len(original_boards)
    rewards_sum = batch_size * [0.0]
    
    active_indices = [i for i, b in enumerate(original_boards) if not b.is_game_over()]
    
    if not active_indices:
        return rewards_sum

    active_boards = [original_boards[i] for i in active_indices]
    tensor_input = boards_to_tensor(boards=active_boards)
    
    with torch.no_grad():
        outputs = neural_network(tensor_input)
    
    for idx_in_batch, original_idx in enumerate(active_indices):
        output = outputs[idx_in_batch]
        board = active_boards[idx_in_batch]
        
        move = get_best_legal_move(output=output, board=board)
        reward = get_custom_reward(board=board, move=move)
        board.push(move)
        rewards_sum[original_idx] += reward
        
    return rewards_sum

def move_to_index(move: chess.Move) -> int:
    # move.from_square and move.to_square are numbers 0-63
    return (move.from_square * 64) + move.to_square

def get_next_moves(boards: list[chess.Board], neural_network: nn.Module) -> list[chess.Move]:
    tensor_input = boards_to_tensor(boards=boards)
    outputs = neural_network(tensor_input)
    moves = [get_best_legal_move(output=output, board=board) for output, board in zip(outputs, boards)]
    return moves

class ChessRES():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9        # discount rate (gamma)    
    replay_memory_size = 50000      # size of replay memory
    mini_batch_size = 128           # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.SmoothL1Loss()     # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    def __init__(self, device_type="cpu"):
        if device_type in ["cuda", "cpu"]:
            ChessRES.device = torch.device(device_type)
        else:
            raise ValueError(f"Device type {device_type} is not supported. Must be one of ['cuda', 'cpu']")
        self.scaler = torch.amp.GradScaler("cuda") if ChessRES.device.type == "cuda" else None

    def _simulate_games(self, memory: ReplayMemory, episodes: int):
        boards = [chess.Board() for _ in range(episodes)]
        while boards:
            boards = [board for board in boards if not board.is_game_over()]
            num_boards = len(boards)
            if self.verbose:
                print(f"\rGames left: {num_boards}  ", end="")
            random_mask = [random.random() < self.epsilon for _ in range(num_boards)]
            rand_indices = set([i for i,x in enumerate(random_mask) if x == 1])
            
            new_random_boards = []
            new_res_boards = []
            
            random_boards = [boards[i] for i in rand_indices]
            if random_boards:
                random_moves = [random.choice(list(board.legal_moves)) for board in random_boards]
                new_random_boards = [board.copy() for board in random_boards]
                for i, board in enumerate(new_random_boards):
                    board.push(random_moves[i])
                    
                for board, move, new_board in zip(random_boards, random_moves, new_random_boards):
                    reward = get_custom_reward(board, move)
                    memory.append((board, move, new_board, reward, new_board.is_game_over())) 
            
            res_boards = [boards[i] for i in range(num_boards) if i not in rand_indices]
            if res_boards:
                res_moves = get_next_moves(boards=res_boards, neural_network=self.policy_res)
                new_res_boards = [board.copy() for board in res_boards]
                for i, board in enumerate(new_res_boards):
                    board.push(res_moves[i])
                    
                for board, move, new_board in zip(res_boards, res_moves, new_res_boards):
                    reward = get_custom_reward(board, move)
                    memory.append((board, move, new_board, reward, new_board.is_game_over())) 

            boards = new_random_boards + new_res_boards

    def train(self, episodes: int,
              cycles: int,
              render: bool=False,
              verbose: bool=False,
              epsilon_decrease=None,
              file=None,
              keep_training=False,
              device="cpu"):
        """
        args:
            episodes (int): Number of games to play
            cycles (int): Number of game cycles
            render (bool): Whether to render the game or not
            verbose (bool): Whether to print information about the game
            epsilon_decrease (float): Decrease in epsilon after each episode
            file (str): File to save the network to
            keep_training (bool): Whether to keep training or start from scratch
        """
        
        self.epsilon = 1
        self.episodes = episodes
        self.cycles = cycles
        self.render = render
        self.verbose = verbose
        self.epsilon_decrease = 1/cycles if epsilon_decrease == None else epsilon_decrease
        memory = ReplayMemory(maxlen=self.replay_memory_size)
        
        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        self.policy_res = ChessResNet().to(ChessRES.device)
        self.target_res = ChessResNet().to(ChessRES.device)
        
        # Load network if file is given
        if file != None and keep_training == True:
            if os.path.exists(file):
                self.policy_res.load_state_dict(torch.load(file, map_location=ChessRES.device, weights_only=True))
                print("Network loaded from file: " + file)
        if file == None:
            file = "chess_res.pt"

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_res.load_state_dict(self.policy_res.state_dict())
        
        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(self.policy_res.parameters(), lr=self.learning_rate_a)

        for cycle in range(cycles):
            
            if verbose:
                print(f"cycle {cycle}, epsilon: {self.epsilon:.4f}")
                print(f"Simulating {episodes} games...")
            tic = time.time()
            
            self._simulate_games(memory, episodes)

            toc = time.time()
            if self.verbose:
                print(f"\rSimulation time: {(toc-tic):.4f}")
                


            tic = time.time()
            if self.verbose:
                print(f"Training policy network...")
            for _ in range(10): # Zrób 10 kroków nauki na cykl
                if len(memory) > self.mini_batch_size:
                    batch = memory.sample(self.mini_batch_size)
                    self.optimize(batch)

            # Decay epsilon
            self.epsilon = max(self.epsilon - self.epsilon_decrease, 0.05)

            # Copy policy network to target network
            self.target_res.load_state_dict(self.policy_res.state_dict())

            torch.save(self.policy_res.state_dict(), file)
            toc = time.time()
            if self.verbose:
                print(f"Training time: {(toc-tic):.4f}, {((toc-tic)/len(batch)):.4f} per move", end='\n')

        # Save policy
        torch.save(self.policy_res.state_dict(), file)
        print("\nTraining complete.")

    # Optimize policy network
    def optimize(self, batch):   
        boards, moves, new_boards, rewards, dones = zip(*batch)
        
        tensor_input_curr = boards_to_tensor(boards=boards)
        all_q_values = self.policy_res(tensor_input_curr)

        move_indices = torch.tensor([move_to_index(m) for m in moves], device=ChessRES.device).reshape(-1, 1)
        current_q_values = all_q_values.gather(1, move_indices).squeeze(-1)

        enemy_rewards = rewards_of_next_move(new_boards, self.target_res)
        combined_rewards = torch.tensor([float(r - e_r) for r, e_r in zip(rewards, enemy_rewards)], device=ChessRES.device)
        
        target_q_values = torch.zeros(len(boards), device=ChessRES.device)
        active_indices = [i for i, b in enumerate(new_boards) if not b.is_game_over()]
        
        if active_indices:
            active_boards = [new_boards[i] for i in active_indices]
            tensor_input_next = boards_to_tensor(boards=active_boards)
            with torch.no_grad():
                next_max_q = self.target_res(tensor_input_next).max(1)[0]
                
            target_q_values[active_indices] = combined_rewards[active_indices] + self.discount_factor_g * next_max_q
        
        terminal_indices = [i for i, b in enumerate(new_boards) if b.is_game_over()]
        if terminal_indices:
            target_q_values[terminal_indices] = combined_rewards[terminal_indices]

        
        self.optimizer.zero_grad()

        if ChessRES.device.type == "cuda":
            with torch.amp.autocast("cuda"):
                loss = self.loss_fn(current_q_values, target_q_values.detach())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.loss_fn(current_q_values, target_q_values.detach())
            loss.backward()
            self.optimizer.step()
        
        return loss.item()

    # Run the Chess environment with the learned policy
    def test(self, episodes, verbose=True, file="chess_res.pt"):
        # Create Chess instance
        env = gym.make('Chess-v0')

        # Load learned policy
        policy_res = ChessResNet().to(ChessRES.device)
        policy_res.load_state_dict(torch.load(file, map_location=ChessRES.device, weights_only=True))
        policy_res.eval()    # switch model to evaluation mode

        for i in range(episodes):
            board = env.reset()  # Initialize to state 0
            if verbose:
                print(env.render(), end='\n\n')
            done = False         # True when game is over

            while(not done):  
                # Select best move   
                with torch.no_grad():
                    output = policy_res(boards_to_tensor(board=board))
                    move = get_best_legal_move(output=output[0], board=board)

                # Execute move
                new_board, reward, done, info = env.step(move)
                if verbose:
                    print(env.render(), end='\n\n')
                board = new_board
            print(f"end: {board.is_game_over()}")
            print(f"checkmate: {board.is_checkmate()}")

            print("\n" + "="*30)
            res = board.result() # Zwraca "1-0", "0-1", "1/2-1/2" lub "*"
            
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
            
            print("="*30)
        env.close()

if __name__ == '__main__':
    chess_res = ChessRES(device_type="cuda")     # "cpu" or "cuda", cpu works better using small models
    chess_res.train(1, 1000, epsilon_decrease=None,
                    file="chess_res1.pt",
                    verbose=True,
                    keep_training=True)  
    chess_res.test(1, file="chess_res1.pt")