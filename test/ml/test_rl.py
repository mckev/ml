import unittest

import numpy

from classes.ml.rl import Rl


class TestRl(unittest.TestCase):

    def test_discount_rewards_episode(self):
        gamma = 0.99
        rewards = [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        rewards_episode = numpy.vstack(rewards)
        discounted_rewards_episode = Rl.discount_rewards_episode(rewards_episode, gamma)
        self.assertEqual(discounted_rewards_episode.tolist(),
                         [[2.74202823638916], [2.7697255611419678], [2.7977025508880615], [2.8259623050689697],
                          [2.8545072078704834], [2.88334059715271], [1.9023643732070923], [1.921580195426941],
                          [1.9409900903701782], [1.960595965385437], [0.9702990055084229], [0.9800999760627747],
                          [0.9900000095367432], [1.0], [0.0], [0.0]])
