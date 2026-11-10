from configs.configs import RunConfig
import numpy as np


def own_reward(my_reward, other_reward, observations):
    return my_reward


def welfare_reward(my_reward, other_reward, observations):
    return my_reward + other_reward


def alpha_welfare_reward(alpha=0.5):
    """Convex combination between own reward and social welfare."""

    # We need an extra function to make sure we only have 3 arguments in the signature
    def reward_fn(my_reward, other_reward, observations):
        return (1.0 - alpha) * my_reward + alpha * (my_reward + other_reward)

    return reward_fn


def conditional_reciprocity_reward(coop_reward_fn, defect_reward_fn):
    """Tit-for-Tat reward: checks Red's last 4 pickups.

    Switches to defect_reward_fn if Red took any blue coin (-1).
    """

    def reward_fn(my_reward, other_reward, observations):
        red_lamps = observations[:, RunConfig.red_lamp_indices]
        red_defected = (red_lamps == -1).any(axis=1)

        reward_coop = coop_reward_fn(my_reward, other_reward, observations)
        reward_defect = defect_reward_fn(my_reward, other_reward, observations)
        return np.where(red_defected, reward_defect, reward_coop)

    return reward_fn