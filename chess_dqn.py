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
        custom_reward += val 
    if board.is_game_over():
        if board.is_checkmate():
            custom_reward = 10.0
        else:
            custom_reward = 0.0
    # custom_reward -= board.fullmove_number * 0.001
    return custom_reward


class ChessCNN(nn.Module):
    def __init__(self, out_actions=4096):
        super().__init__()

        # 1. Pierwsza warstwa konwolucyjna
        # In_channels = 17 (Twoje warstwy planszy)
        # Out_channels = 64 (Liczba filtrów, które uczą się cech)
        self.conv1 = nn.Conv2d(17, 64, kernel_size=3, padding=1)
        
        # 2. Druga warstwa konwolucyjna
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 3. Warstwa w pełni połączona (Linear)
        # Po konwolucjach z padding=1, rozmiar to nadal 8x8.
        # Więc wejście do Linear to: 128 filtrów * 8 * 8 px
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        
        # 4. Wyjście (4096 akcji)
        self.out = nn.Linear(512, out_actions)

    def forward(self, x):
        # x shape: [batch, 17, 8, 8]
        
        # Przepuszczamy przez konwolucje z aktywacją ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # "Spłaszczamy" tensor z (8, 8, 128) na jeden długi wektor
        x = x.view(x.size(0), -1) 
        
        # Warstwy klasyczne
        x = F.relu(self.fc1(x))
        x = self.out(x)
        
        return x
    
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

def board_to_tensor(board: chess.Board) -> torch.Tensor:
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
    tensor = torch.from_numpy(matrix)
    return tensor.unsqueeze(0)

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
    tensor_list = [board_to_tensor(board=board) for board in active_boards]
    tensor_input = torch.vstack(tensor_list)
    
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
    tensor_list = [board_to_tensor(board=board) for board in boards]
    tensor_input = torch.vstack(tensor_list)
    outputs = neural_network(tensor_input)
    moves = [get_best_legal_move(output=output, board=board) for output, board in zip(outputs, boards)]
    return moves

class ChessDQN():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.9        # discount rate (gamma)    
    # network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10000      # size of replay memory
    # mini_batch_size = 128           # size of the training data set sampled from the replay memory
    # moves_to_evaluate = 1           # number of moves to evaluate when calculating reward
    # time_verbose = True

    # Neural Network
    loss_fn = nn.SmoothL1Loss()     # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    # num_states = 17*64*64
    num_actions = 64*64
    
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
            new_dqn_boards = []
            
            random_boards = [boards[i] for i in rand_indices]
            if random_boards:
                random_moves = [random.choice(list(board.legal_moves)) for board in random_boards]
                new_random_boards = [board.copy() for board in random_boards]
                for i, board in enumerate(new_random_boards):
                    board.push(random_moves[i])
                    
                for board, move, new_board in zip(random_boards, random_moves, new_random_boards):
                    reward = get_custom_reward(board, move)
                    memory.append((board, move, new_board, reward, new_board.is_game_over()))
            
            dqn_boards = [boards[i] for i in range(num_boards) if i not in rand_indices]
            if dqn_boards:
                dqn_moves = get_next_moves(boards=dqn_boards, neural_network=self.policy_dqn)
                new_dqn_boards = [board.copy() for board in dqn_boards]
                for i, board in enumerate(new_dqn_boards):
                    board.push(dqn_moves[i])
                    
                for board, move, new_board in zip(dqn_boards, dqn_moves, new_dqn_boards):
                    reward = get_custom_reward(board, move)
                    memory.append((board, move, new_board, reward, new_board.is_game_over())) 

            boards = new_random_boards + new_dqn_boards

    def train(self, episodes: int,
              cycles: int,
              render: bool=False,
              verbose: bool=False,
              epsilon_decrease=None,
              file=None,
              keep_training=False):
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
        self.epsilon_decrease = 1/episodes if epsilon_decrease == None else epsilon_decrease
        memory = ReplayMemory(maxlen=self.replay_memory_size)
        
        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        self.policy_dqn = ChessCNN(out_actions=self.num_actions)
        self.target_dqn = ChessCNN(out_actions=self.num_actions)
        
        # Load network if file is given
        if file != None and keep_training == True:
            if os.path.exists(file):
                self.policy_dqn.load_state_dict(torch.load(file))
                print("Network loaded from file: " + file)
                

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())
        

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(self.policy_dqn.parameters(), lr=self.learning_rate_a)

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
                if len(memory) > 128:
                    batch = memory.sample(128)
                    self.optimize(batch)

            # Decay epsilon
            self.epsilon = max(self.epsilon - self.epsilon_decrease, 0)

            # Copy policy network to target network
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            toc = time.time()
            if self.verbose:
                print(f"Training time: {(toc-tic):.4f}, {((toc-tic)/len(batch)):.4f} per move", end='\n')

        # Save policy
        if file == None:
            file = "chess_dqn.pt"
        torch.save(self.policy_dqn.state_dict(), file)
        print("\nTraining complete.")

    # Optimize policy network
    def optimize(self, batch):   
        boards, moves, new_boards, rewards, dones = zip(*batch)
        
        tensor_input_curr = torch.vstack([board_to_tensor(board=b) for b in boards])
        all_q_values = self.policy_dqn(tensor_input_curr)

        move_indices = torch.tensor([move_to_index(m) for m in moves]).reshape(-1, 1)
        current_q_values = all_q_values.gather(1, move_indices).squeeze(-1)

        enemy_rewards = rewards_of_next_move(new_boards, self.target_dqn)
        combined_rewards = torch.tensor([float(r - e_r) for r, e_r in zip(rewards, enemy_rewards)])
        
        target_q_values = torch.zeros(len(boards))
        active_indices = [i for i, b in enumerate(new_boards) if not b.is_game_over()]
        
        if active_indices:
            active_boards = [new_boards[i] for i in active_indices]
            tensor_input_next = torch.vstack([board_to_tensor(board=b) for b in active_boards])
            with torch.no_grad():
                next_max_q = self.target_dqn(tensor_input_next).max(1)[0]
                
            target_q_values[active_indices] = combined_rewards[active_indices] + self.discount_factor_g * next_max_q
        
        terminal_indices = [i for i, b in enumerate(new_boards) if b.is_game_over()]
        if terminal_indices:
            target_q_values[terminal_indices] = combined_rewards[terminal_indices]

        loss = self.loss_fn(current_q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # Run the Chess environment with the learned policy
    def test(self, episodes, verbose=True, file="chess_dqn.pt"):
        # Create Chess instance
        env = gym.make('Chess-v0')

        # Load learned policy
        policy_dqn = ChessCNN(out_actions=self.num_actions) 
        policy_dqn.load_state_dict(torch.load(file))
        policy_dqn.eval()    # switch model to evaluation mode

        for i in range(episodes):
            board = env.reset()  # Initialize to state 0
            if verbose:
                print(env.render(), end='\n\n')
            done = False         # True when game is over

            while(not done):  
                # Select best move   
                with torch.no_grad():
                    output = policy_dqn(board_to_tensor(board=board))
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

    chess_dqn = ChessDQN()
    chess_dqn.train(10, 5, epsilon_decrease=0, file="chess_dqn1.pt", verbose=True, keep_training=True)
    chess_dqn.test(1, file="chess_dqn1.pt")