from pathlib import Path

import chess
import pygame
import torch

from src.rl_chess.models.resnet.chess_res import ChessResNet
from src.rl_chess.utils.MonteCarloTreeSearch import MonteCarloTreeSearch
from src.rl_chess.utils.train_utils import (
    boards_to_tensor,
    get_best_legal_move,
    get_next_moves,
)

# Ustawienia
WIDTH, HEIGHT = 600, 600
SQUARE_SIZE = WIDTH // 8
ASSETS_DIR = Path(__file__).parent / "assets"
ANIMATION_SPEED = 15


def load_piece_images():
    pieces = ["pawn", "rook", "knight", "bishop", "queen", "king"]
    colors = ["white", "black"]
    images = {}
    for color in colors:
        for piece in pieces:
            name = f"{color}_{piece}"
            path = ASSETS_DIR / f"{name}.png"
            if path.exists():
                images[name] = pygame.transform.smoothscale(
                    pygame.image.load(str(path)), (SQUARE_SIZE, SQUARE_SIZE)
                )
    return images


def draw_end_message(screen, board):
    font = pygame.font.SysFont("Arial", 60, bold=True)
    if board.is_checkmate():
        winner = "Black wins!" if board.turn == chess.WHITE else "White wins!"
        text = font.render(winner, True, (255, 0, 0))
    elif board.is_stalemate() or board.is_insufficient_material():
        text = font.render("Draw!", True, (0, 0, 255))
    else:
        text = font.render("Game Over", True, (0, 0, 0))

    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    overlay = pygame.Surface((WIDTH, HEIGHT))
    overlay.set_alpha(128)
    overlay.fill((255, 255, 255))
    screen.blit(overlay, (0, 0))
    screen.blit(text, text_rect)


def draw_board(
    screen,
    board,
    selected_sq,
    possible_moves,
    piece_images,
    moving_piece=None,
    anim_pos=None,
):
    colors = [pygame.Color("#f0d9b5"), pygame.Color("#b58863")]
    type_to_name = {
        chess.PAWN: "pawn",
        chess.ROOK: "rook",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }

    for r in range(8):
        for c in range(8):
            sq = chess.square(c, 7 - r)
            color = colors[((r + c) % 2)]
            if sq == selected_sq:
                color = pygame.Color("#7b61ff")
            elif sq in possible_moves:
                color = pygame.Color("#90ee90")
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(
                    c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE
                ),
            )

            piece = board.piece_at(sq)
            if piece and (
                moving_piece is None or sq != moving_piece["start_sq"]
            ):
                key = f"{'white' if piece.color == chess.WHITE else 'black'}_{type_to_name[piece.piece_type]}"
                if key in piece_images:
                    screen.blit(
                        piece_images[key], (c * SQUARE_SIZE, r * SQUARE_SIZE)
                    )

    if moving_piece:
        screen.blit(piece_images[moving_piece["key"]], anim_pos)


def choose_move(board, model):
    use_mcts = False
    if use_mcts:
        MCTS = MonteCarloTreeSearch(num_searches=150)
        return get_next_moves(
            boards=[board],
            neural_network=model,
            device=torch.device("cpu"),
            move_search=MCTS,
        )[0][0]
    output = model(boards_to_tensor([board], torch.device("cpu")))
    return get_best_legal_move(output[0], board)


def play_gui():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Play against model")
    piece_images = load_piece_images()
    board = chess.Board()
    model = ChessResNet()

    # only pretrained: chess_res_pretrained292.pt

    model_path = Path(
        # "src/rl_chess/models/resnet/trained/chess_res_pretrained186.pt"
        # "src/rl_chess/models/resnet/trained/chess_res_finetuned_8.pt"
        "src/rl_chess/models/resnet/trained/chess_res_pretrained468.pt"
    )
    if model_path.exists():
        model.load_state_dict(
            torch.load(
                model_path, map_location=torch.device("cpu"), weights_only=True
            )
        )
    model.eval()

    type_to_name = {
        chess.PAWN: "pawn",
        chess.ROOK: "rook",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }

    selected_sq, possible_moves = None, []
    moving_piece = None
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and board.turn == chess.WHITE
                and not moving_piece
            ):
                mx, my = pygame.mouse.get_pos()
                sq = chess.square(mx // SQUARE_SIZE, 7 - my // SQUARE_SIZE)
                if selected_sq is None:
                    if board.piece_at(sq):
                        selected_sq = sq
                        possible_moves = [
                            m.to_square
                            for m in board.legal_moves
                            if m.from_square == selected_sq
                        ]
                else:
                    move = chess.Move(selected_sq, sq)
                    piece = board.piece_at(selected_sq)
                    if (
                        piece
                        and piece.piece_type == chess.PAWN
                        and (chess.square_rank(sq) in [0, 7])
                    ):
                        move.promotion = chess.QUEEN

                    if move in board.legal_moves:
                        board.push(move)
                        if not board.is_game_over():
                            with torch.no_grad():
                                ai_move = choose_move(board, model)
                                if ai_move:
                                    is_pawn = (
                                        board.piece_at(
                                            ai_move.from_square
                                        ).piece_type
                                        == chess.PAWN
                                    )
                                    is_last_rank = chess.square_rank(
                                        ai_move.to_square
                                    ) in [0, 7]

                                    if (
                                        ai_move.promotion is None
                                        and is_pawn
                                        and is_last_rank
                                    ):
                                        ai_move.promotion = chess.QUEEN

                                    piece = board.piece_at(ai_move.from_square)
                                    color_str = (
                                        "white"
                                        if piece.color == chess.WHITE
                                        else "black"
                                    )
                                    name_str = type_to_name[piece.piece_type]
                                    key = f"{color_str}_{name_str}"

                                    start_sq = ai_move.from_square
                                    target_sq = ai_move.to_square

                                    start_c, start_r = (
                                        chess.square_file(start_sq),
                                        7 - chess.square_rank(start_sq),
                                    )
                                    target_c, target_r = (
                                        chess.square_file(target_sq),
                                        7 - chess.square_rank(target_sq),
                                    )

                                    moving_piece = {
                                        "key": key,
                                        "start_sq": start_sq,
                                        "pos": [
                                            start_c * SQUARE_SIZE,
                                            start_r * SQUARE_SIZE,
                                        ],
                                        "target": [
                                            target_c * SQUARE_SIZE,
                                            target_r * SQUARE_SIZE,
                                        ],
                                        "move": ai_move,
                                    }
                    selected_sq, possible_moves = None, []

        if moving_piece:
            dx, dy = (
                moving_piece["target"][0] - moving_piece["pos"][0],
                moving_piece["target"][1] - moving_piece["pos"][1],
            )
            dist = (dx**2 + dy**2) ** 0.5
            if dist < ANIMATION_SPEED:
                board.push(moving_piece["move"])
                moving_piece = None
            else:
                moving_piece["pos"][0] += (dx / dist) * ANIMATION_SPEED
                moving_piece["pos"][1] += (dy / dist) * ANIMATION_SPEED

        screen.fill(pygame.Color("white"))
        draw_board(
            screen,
            board,
            selected_sq,
            possible_moves,
            piece_images,
            moving_piece,
            moving_piece["pos"] if moving_piece else None,
        )

        if board.is_game_over():
            draw_end_message(screen, board)

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


if __name__ == "__main__":
    play_gui()
