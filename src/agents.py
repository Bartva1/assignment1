
from src.configs import BASE_PPO_CONFIG
from src.rewards import own_reward, welfare_reward


def build_ppo_agent(agent_cls, config, reward_fn):
    """Creates a factory function make_agent(n_inputs, seed) compatible with play_match."""

    def make_agent(n_inputs, seed):
        agent = agent_cls(n_inputs, 4, config=config, seed=seed)
        agent.training_reward = reward_fn
        return agent

    return make_agent


def get_phase1_agents(agent_cls):
    """Phase 1: Canonical baselines (Selfish PPO vs Welfare PPO)."""
    return {
        'PPO (Own Reward)': build_ppo_agent(
            agent_cls, BASE_PPO_CONFIG, own_reward
        ),
        'PPO (Welfare)': build_ppo_agent(
            agent_cls, BASE_PPO_CONFIG, welfare_reward
        ),
    }