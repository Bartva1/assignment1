from dataclasses import dataclass

BASE_PPO_CONFIG = dict(
    lr=3e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    n_epochs=4,
    n_minibatches=4,
    entropy_coef=0.01,
    value_coef=0.5,
    normalize_advantages=True,
    use_critic=True,
    hidden=64,
    max_grad_norm=0.5,
)

A2C_CONFIG = dict(BASE_PPO_CONFIG, clip=None, n_epochs=1, n_minibatches=1)
REINFORCE_CONFIG = dict(A2C_CONFIG, use_critic=False)

ALGORITHM_CONFIGS = {
    'PPO': BASE_PPO_CONFIG,
    'A2C': A2C_CONFIG,
    'REINFORCE': REINFORCE_CONFIG,
}


@dataclass(frozen=True, init=False)
class RunConfig:
    red_lamp_indices = slice(108, 112)
    seeds = tuple(i for i in range(1,6))


@dataclass(frozen=True, init=False)
class PathConfig:
    checkpoint_path = "results/sweep_matches.pkl"