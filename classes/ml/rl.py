from typing import List

import numpy


class Rl:
    """ Reinforcement Learning """

    # Ref: http://karpathy.github.io/2016/05/31/rl/

    @staticmethod
    def discount_rewards_episode(rewards_episode: List[float], gamma):
        # Take 1D float array of rewards and gamma (discount factor for reward), and compute discounted rewards
        discounted_rewards_episode = numpy.zeros_like(rewards_episode, dtype=numpy.float32)
        running_add = 0
        for t in reversed(range(0, rewards_episode.size)):
            running_add = running_add * gamma + rewards_episode[t]
            discounted_rewards_episode[t] = running_add
        return discounted_rewards_episode
