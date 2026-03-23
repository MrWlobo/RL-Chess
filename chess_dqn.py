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
    custom_reward -= board.fullmove_number * 0.001
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

def get_best_legal_move(outputs: torch.Tensor, board: chess.Board) -> chess.Move:
    sorted_indices = outputs.flatten().argsort(descending=True)
    legal_moves = set(board.legal_moves)
    for output in sorted_indices:
        move = move_from_output(output)
        if move == None:
            continue
        move = ensure_queen_promotion(board=board, move=move)
        if move in legal_moves:
            return move

def reward_of_next_moves(original_board: chess.Board, n_moves: int, target_dqn: nn.Module) -> float:
    # returns sum rewards of next n moves
    board = original_board.copy()
    reward_sum = 0.0
    
    signs = [-1, 1]
    for i in range(n_moves):
        current_sign = signs[i % 2]
        outputs = target_dqn(board_to_tensor(board=board))
        move = get_best_legal_move(outputs=outputs, board=board)
        reward_sum += get_custom_reward(board=board, move=move) * current_sign
        board.push(move)
        if board.is_game_over():
            break
    return reward_sum

def move_to_index(move: chess.Move) -> int:
    # move.from_square and move.to_square are numbers 0-63
    return (move.from_square * 64) + move.to_square

class ChessDQN():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001         # learning rate (alpha)
    discount_factor_g = 0.99        # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10000      # size of replay memory
    mini_batch_size = 512           # size of the training data set sampled from the replay memory
    moves_to_evaluate = 1           # number of moves to evaluate when calculating reward
    time_verbose = True

    # Neural Network
    loss_fn = nn.SmoothL1Loss()     # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    # num_states = 17*64*64
    num_actions = 64*64

    # Train the FrozeLake environment
    def train(self, episodes: int, render: bool=False, verbose: bool=True):
        # Create FrozenLake instance
        env = gym.make('Chess-v0')
        
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = ChessCNN(out_actions=self.num_actions)
        target_dqn = ChessCNN(out_actions=self.num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        end_by_checkmate = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        if verbose:
            print("Training...")
            
        for i in range(episodes):
            if verbose:
                print(f"\r{i+1}/{episodes}", end="")
            board = env.reset()  # Initialize state 
            done = False 

            tic = time.time()

            while(not done):
                
                legal_moves = env.legal_moves
                # Select move based on epsilon-greedy
                if random.random() < epsilon:
                    # select random move
                    move = random.choice(legal_moves)
                    
                else:
                    # select best move            
                    with torch.no_grad():
                        outputs = policy_dqn(board_to_tensor(board=board))
                        move = get_best_legal_move(outputs=outputs, board=board)
                
                # Promotions only to queen
                move = ensure_queen_promotion(board=board, move=move)

                # Execute action
                new_board, reward, done, info = env.step(move)
                reward = get_custom_reward(board=board, move=move)

                # Save experience into memory
                memory.append((board, move, new_board, reward, done)) 

                # Move to the next state
                board = new_board

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if board.is_checkmate():
                end_by_checkmate[i] = 1

            toc = time.time()
            if self.time_verbose:
                print(f"\nSimulation time: {toc-tic}")

            # Check if enough experience has been collected
            if len(memory)>self.mini_batch_size:
                tic = time.time()
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0
                toc = time.time()
                if self.time_verbose:
                    print(f"Training time: {toc-tic}, {(toc-tic)/self.mini_batch_size} per move", end='\n\n')

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "chess_dqn.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average checkmates (Y-axis) vs episodes (X-axis)
        sum_checkmates = np.zeros(episodes)
        for x in range(episodes):
            sum_checkmates[x] = np.sum(end_by_checkmate[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_checkmates)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('chess_dqn.png')
        print("\nTraining complete.")

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for board, move, new_board, reward, done in mini_batch:

            if done: 
                # The agent achieves checkmate (reward=10) or a draw (reward=0)
                target = torch.FloatTensor([reward])

            else:
                # Evaluate the opponent's best move and penalize the agent for enabling it.
                # Similar to Negamax
                with torch.no_grad():
                    next_moves_reward = reward_of_next_moves(original_board=new_board, n_moves=self.moves_to_evaluate, target_dqn=target_dqn)
                    target = torch.FloatTensor(
                        [reward + (self.discount_factor_g * next_moves_reward)]
                    )
                    
            # Get the current set of Q values
            current_q = policy_dqn(board_to_tensor(board=board)).flatten()
            current_q_list.append(current_q)
            
            # Get the target set of Q values
            with torch.no_grad():
                target_q = target_dqn(board_to_tensor(board=board)).flatten().detach()
            move_idx = move_to_index(move=move)
            target_q[move_idx] = target.item()
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Run the Chess environment with the learned policy
    def test(self, episodes, verbose=True):
        # Create Chess instance
        env = gym.make('Chess-v0')

        # Load learned policy
        policy_dqn = ChessCNN(out_actions=self.num_actions) 
        policy_dqn.load_state_dict(torch.load("chess_dqn.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        # print('Policy (trained):')
        # self.print_dqn(policy_dqn)

        for i in range(episodes):
            board = env.reset()  # Initialize to state 0
            if verbose:
                print(env.render(), end='\n\n')
            done = False         # True when game is over

            while(not done):  
                # Select best move   
                with torch.no_grad():
                    outputs = policy_dqn(board_to_tensor(board=board))
                    move = get_best_legal_move(outputs=outputs, board=board)

                # Execute move
                new_board, reward, done, info = env.step(move)
                if verbose:
                    print(env.render(), end='\n\n')
                board = new_board

        env.close()

    # def print_dqn(self, dqn):
    #     # Get number of input nodes
    #     num_states = dqn.fc1.in_features

    #     # Loop each state and print policy to console
    #     for s in range(num_states):
    #         #  Format q values for printing
    #         q_values = ''
    #         for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
    #             q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
    #         q_values=q_values.rstrip()              # Remove space at the end

    #         # Map the best action to L D R U
    #         best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

    #         # Print policy in the format of: state, action, q values
    #         # The printed layout matches the FrozenLake map.
    #         print(f'{s:02},{best_action},[{q_values}]', end=' ')         
    #         if (s+1)%4==0:
    #             print() # Print a newline every 4 states

if __name__ == '__main__':

    chess_dqn = ChessDQN()
    chess_dqn.train(50)
    chess_dqn.test(1)