
BASE_PPO_CONFIG = dict(
    lr=3e-3,
    gamma=0.99,
    gae_lambda=0.95,
    clip=0.2,
    n_epochs=4,
    n_minibatches=4,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5,
)


BASELINE_CONFIGS = {
    'default': BASE_PPO_CONFIG,
}