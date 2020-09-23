import unittest
import numpy as np
from collections import defaultdict
from sarsa import td_update, sarsa, make_epsilon_greedy_policy
from env import FallEnv

class TestSARSA(unittest.TestCase):

    def test_td_update(self):
        Q = defaultdict(lambda: np.zeros(2))
        env = FallEnv()
        state = env.reset()
        next_state, reward, done, _ = env.step(0)
        target = td_update(Q, state, 0, reward, next_state, gamma=0.9, alpha=0.1,
                      done=done, action_=1)
        self.assertTrue(target == 0)
        for i in range(10):
            state = next_state
            next_state, reward, done, _ = env.step(2)
        target = td_update(Q, state, 1, reward, next_state, gamma=0.9, alpha=0.1,
                      done=done, action_=1)
        self.assertTrue(target != 0)

    def test_sarsa(self):
        rewards, lengths = sarsa(10)
        self.assertTrue(len(lengths)==len(rewards))
        self.assertTrue(sum(rewards) > 0)

    def test_exploration(self):
        Q = defaultdict(lambda: np.zeros(2))
        env = FallEnv()
        state = env.reset()
        policy = make_epsilon_greedy_policy(Q, 0.5, 4)
        actions = []
        for _ in range(10):
            actions.append(policy(state))
        equalities = [np.array_equal(actions[0], actions[k]) for k in range(len(actions))]
        self.assertFalse(all(equalities))

if __name__ == '__main__':
    unittest.main()
