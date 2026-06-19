# RL-Chess
## Setup
1. Run `uv sync`
2. Run `pre-commit install`

## Training
comment / uncoment functions in
`src/rl_chess/models/resnet/chess_res.py`
then
`uv run python -m src.rl_chess.models.resnet.chess_res`

## Playing against AI
change model location in code
`src/rl_chess/gameplay/play_against_model.py`
then
`uv run python -m src.rl_chess.gameplay.play_against_model`