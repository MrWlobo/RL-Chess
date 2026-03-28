# Model Benchmarking
## Overview
This tool can be used to benchmark models against opponents of increasing skill levels. It uses a library of common starting positions reached after the first 3 moves,
as testing the agent repeatedly against the same, deterministic opponent would always lead to the same game being played.
## Step-by-step
1. Create a class in `rl_chess.agents.custom` that inherits from `rl_chess.agents.base` and implements the `predict(self, board: chess.Board) -> chess.Move` method.
This class can be used to load saved model weights from a file in the `__init__` method and can take custom `__init__` arguments
2. Register your class in the `rl_chess.agents.custom` `__init__.py` file, in the `AGENT_REGISTRY` dictionary.
3. Run the benchmark as described in the next section.
## Usage
```bash
uv run benchmark [GLOBAL_OPTIONS] <AGENT_NAME> [AGENT_OPTIONS]
```
The CLI is structured into **Global Options** (for the benchmark itself) and **Agent Options** (specific to the model you are testing). Global options must come *before* the agent name.
### Global Options
| Flag | Description | Default |
| :--- | :--- | :--- |
| `--games` | Total number of games per opponent (split evenly between playing as White and Black). | `100` |
| `--pass-threshold` | Win rate (0.0 to 1.0) required to advance to the next opponent. | `0.5` |
| `--deterministic` | Flag to disable shuffling of starting FEN positions. | `False` |
| `--output-file` | Custom filename for the YAML results. | `<agent>_<timestamp>.yaml` |

*Example: Run a deterministic benchmark, 50 games per opponent, with a 75% win rate required to keep testing against even stronger opponents:*
```bash
uv run benchmark --games 50 --pass-threshold 0.75 --deterministic DQNAgent --model_path "path/to/weights.pt"
```

### Agent Options
The specific agent's `__init__` arguments can be passed after the agent's name, as the script dynamically inspects the `__init__` method of the target agent class. You can view the specific arguments for your agent by using the `-h` flag *after* the agent name.

```bash
uv run benchmark DQNAgent -h
```
