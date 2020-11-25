import numpy as np
from collections import defaultdict

from env import FallEnv

#Most of this code is Code provided by Fabio Ferreira & Andre Biedenkapp
def sarsa(num_episodes: int,
        discount_factor: float = 1.0,
        alpha: float = 0.5,
        epsilon: float = 0.1):
    """
    Vanilla tabular SARSA algorithm
    :param num_episodes: number of episodes to train
    :param discount_factor: discount factor used in TD updates
    :param alpha: learning rate used in TD updates
    :param epsilon: exploration fraction
    :return: training and evaluation statistics (i.e. rewards and episode lengths)
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    env = FallEnv()
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []

    train_steps_list = []
    num_performed_steps = 0

    for i_episode in range(num_episodes + 1):
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
        state = env.reset()
        done = False
        episode_length, cummulative_reward = 0, 0
        while not done:  # roll out episode
            num_performed_steps += 1

            Q[state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha, policy_done,
                                                       action_=action)
        rewards.append(cummulative_reward)
        lens.append(episode_length)

        print(f'Done {i_episode:4d}/{num_episodes:4d} episodes')

    return rewards, lens

def make_epsilon_greedy_policy(Q: defaultdict, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.
    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """


    def policy_fn(observation):
        """ Returns an probability distribution for each action depending on an observation """
        policy = np.ones(nA) * epsilon / nA
        return policy

    return policy_fn



def td_update(q: defaultdict, state: int, action: int, reward: float, next_state: int, gamma: float, alpha: float,
              done: bool, action_: int):
    """ Simple TD update rule """

    return
