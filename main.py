import gym
import gym_chess
import random
import time
import chess
import numpy as np

def board_to_tensor(board: chess.Board):
    tensor = np.zeros((17, 8, 8), dtype=np.float32)
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
            if is_black_turn:
                if piece.color == chess.WHITE:
                    layer += 6
            else:
                if piece.color == chess.BLACK:
                    layer += 6
            
            tensor[layer, row, col] = 1.0
 
    if is_black_turn:
        if board.has_kingside_castling_rights(chess.BLACK): tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[15, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.WHITE): tensor[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): tensor[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0

    # 18. Zasada 50 ruchów (znormalizowana do zakresu 0-1)
    tensor[16, :, :] = board.halfmove_clock / 100.0
    
    return tensor



def main():

    env = gym.make('Chess-v0')
    board = env.reset()
    
    done = False
    
    start_time = time.perf_counter()

    while not done:

        legal_moves = env.legal_moves
        board_legal_moves = board.legal_moves
        action = random.choice(legal_moves)
        move_text = action.uci()
        move_text_new = move_text[:4]
        move = chess.Move.from_uci(move_text_new)
        # if move not in legal_moves:
        #     print(move_text, move_text_new)
        #     print("NOT IN LEGAL MOVES")
        print(len(legal_moves), len(set(board_legal_moves)))
        
        board, reward, done, info = env.step(action)
        
        # tensor = board_to_tensor(board)
        if reward > 0:
            print(reward)

        
    end_time = time.perf_counter()
    print(env.render())
    env.close()
    print(f"Czas wykonania: {end_time - start_time:.6f} sekund")

if __name__ == "__main__":
    main()
