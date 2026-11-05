"""src/agents.py - Agent factory builders and sweep registries."""

from src.configs import A2C_CONFIG, BASE_PPO_CONFIG, REINFORCE_CONFIG
from src.rewards import (
    alpha_welfare_reward,
    conditional_reciprocity_reward,
    own_reward,
    welfare_reward,
)


def build_ppo_agent(agent_cls, config, reward_fn):
    """Creates a factory function make_agent(n_inputs, seed) compatible with play_match."""

    def make_agent(n_inputs, seed):
        agent = agent_cls(n_inputs, 4, config=config, seed=seed)
        agent.training_reward = reward_fn
        return agent

    return make_agent


def get_algorithm_comparison_agents(agent_cls, reward_fn=own_reward):
    """Compares the 3 RL algorithms (PPO vs A2C vs REINFORCE) on the same reward function."""
    return {
        'PPO (Baseline)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, reward_fn),
        'A2C (Baseline)': build_ppo_agent(agent_cls, A2C_CONFIG, reward_fn),
        'REINFORCE (Baseline)': build_ppo_agent(
            agent_cls, REINFORCE_CONFIG, reward_fn
        ),
    }


def get_sweep_agents(agent_cls):
  
    return {
        'PPO (Own)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, own_reward),
        'A2C (Own)': build_ppo_agent(agent_cls, A2C_CONFIG, own_reward),
        'REINFORCE (Own)': build_ppo_agent(agent_cls, REINFORCE_CONFIG, own_reward),

        'PPO (Welfare)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, welfare_reward),
        'A2C (Welfare)': build_ppo_agent(agent_cls, A2C_CONFIG, welfare_reward),
        'REINFORCE (Welfare)': build_ppo_agent(agent_cls, REINFORCE_CONFIG, welfare_reward),

        # Currently only do sweeps for other algorithms on PPO as I expect this is the best algorithm, but might need to switch
        'PPO (Alpha=0.25)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, alpha_welfare_reward(0.25)),
        'PPO (Alpha=0.50)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, alpha_welfare_reward(0.50)),
        'PPO (Alpha=0.75)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, alpha_welfare_reward(0.75)),

        
        'Reciprocal (TFT, Alpha=0.5)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, conditional_reciprocity_reward(
                alpha_welfare_reward(0.5), own_reward
            ),
        ),
        'Reciprocal (TFT, Pure Welfare)': build_ppo_agent(agent_cls, BASE_PPO_CONFIG, conditional_reciprocity_reward(
                welfare_reward, own_reward
            ),
        ),
    }