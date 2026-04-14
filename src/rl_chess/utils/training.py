from __future__ import annotations

import time
from typing import TYPE_CHECKING

import chess
import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from rl_chess.utils.MonteCarloTreeSearch import (
        MonteCarloTreeSearch,  # noqa: TC004
    )


def board_to_array(board: chess.Board) -> np.ndarray:
    matrix = np.zeros((17, 8, 8), dtype=np.float32)
    is_black_turn = board.turn == chess.BLACK
    piece_to_layer = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    for square in chess.SQUARES:  # chess.SQUARES idą od 0 do 63 (A1, B1... H8)
        piece = board.piece_at(square)
        if piece:
            row = chess.square_rank(square)  # 0-7
            col = chess.square_file(square)  # 0-7

            if is_black_turn:
                row = 7 - row
                col = 7 - col

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
        if board.has_kingside_castling_rights(chess.BLACK):
            matrix[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            matrix[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            matrix[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            matrix[15, :, :] = 1.0
    else:
        if board.has_kingside_castling_rights(chess.WHITE):
            matrix[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            matrix[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            matrix[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            matrix[15, :, :] = 1.0

    # 18. Zasada 50 ruchów (znormalizowana do zakresu 0-1)
    matrix[16, :, :] = board.halfmove_clock / 100.0
    return matrix


def boards_to_tensor(boards: list[chess.Board], device: torch.device):
    array_list = [board_to_array(board=board) for board in boards]
    single_ndarray = np.array(array_list)
    return torch.from_numpy(single_ndarray).to(device).float()


def board_to_tensor(board: chess.Board, device: torch.device):
    return boards_to_tensor(boards=[board], device=device)


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
    mask = np.full(4096, -1e9, dtype=np.float32)
    output_flat = output.flatten().numpy()
    legal_moves = set(board.legal_moves)
    legal_indices = [move_to_index(m) for m in legal_moves]
    mask[legal_indices] = output_flat[legal_indices]
    best_move_idx = np.argmax(mask)
    move = move_from_output(best_move_idx)
    return ensure_queen_promotion(board, move)


def rewards_of_next_move(
    original_boards: list[chess.Board],
    neural_network: nn.Module,
    reward_function: Callable,
) -> tuple[list[float], list[chess.Board]]:
    # Returns sum rewards of next move.
    # Also modifies original_boards IN PLACE.
    batch_size = len(original_boards)
    rewards_sum = batch_size * [0.0]

    active_indices = [
        i for i, b in enumerate(original_boards) if not b.is_game_over()
    ]

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
        reward = reward_function(board=board, move=move)
        rewards_sum[original_idx] += reward

    return rewards_sum


def move_to_index(move: chess.Move) -> int:
    # move.from_square and move.to_square are numbers 0-63
    return (move.from_square * 64) + move.to_square


def get_next_moves(
    boards: list[chess.Board],
    neural_network: nn.Module,
    device: torch.device,
    move_search: MonteCarloTreeSearch | None = None,
) -> list[chess.Move]:
    if move_search is None:
        tensor_input = boards_to_tensor(boards=boards, device=device)
        with torch.no_grad():
            policy, _ = neural_network(tensor_input)
        return [
            get_best_legal_move(output=output, board=board)
            for output, board in zip(policy.cpu(), boards, strict=True)
        ]
    print("Using MCTS")
    tic = time.time()
    moves = [
        move_search.search(
            initial_state=board.fen(),
            neural_network=neural_network,
            device=device,
        )
        for board in boards
    ]
    toc = time.time()
    print(f"MCTS Time: {toc - tic} ##############################")
    return [
        ensure_queen_promotion(move=move, board=board)
        for move, board in zip(moves, boards, strict=True)
    ]
