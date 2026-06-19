from rl_chess.agents.custom.resnet_agent import ResNetAgent
from rl_chess.agents.custom.resnet_agent_mcts import ResNetMCTSAgent

AGENT_REGISTRY = {
    "ResNetAgent": ResNetAgent,
    "ResNetMCTSAgent": ResNetMCTSAgent,
}
