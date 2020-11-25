import numpy as np
from collections import defaultdict
from env import FallEnv

#Most of this code is provided by Fabio Ferreira & Andre Biedenkapp
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
    environment = FallEnv()
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(environment.action_space.n))

    # Keeps track of episode lengths and rewards
    rewards = []
    lens = []
    train_steps_list = []
    num_performed_steps = 0

    for i_episode in range(num_episodes + 1):
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        policy_state = environment.reset()
        done = False
        episode_length, cummulative_reward = 0, 0
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        while not done:  # roll out episode
            s_, policy_reward, done, _ = environment.step(policy_action)
            num_performed_steps += 1
            a_ = np.random.choice(list(range(environment.action_space.n)), p=policy(s_))
            cummulative_reward += policy_reward
            episode_length += 1

            Q[policy_state][policy_action] = td_update(Q, policy_state, policy_action,
                                                       policy_reward, s_, discount_factor, alpha, done,
                                                       action_=a_)
            policy_state = s_
            policy_action = a_

        rewards.append(cummulative_reward)
        lens.append(episode_length)
        train_steps_list.append(environment.total_steps)

    print('Done %4d/%4d %s' % (i_episode, num_episodes, 'episodes'))
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
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(  # random choice for tie-breaking only
            Q[observation] == Q[observation].max()
        ))
        policy[best_action] += (1 - epsilon)
        return policy

    return policy_fn



def td_update(q: defaultdict, state: int, action: int, reward: float, next_state: int, gamma: float, alpha: float,
              done: bool, action_: int):
    """ Simple TD update rule """

    td_target = reward + gamma * q[next_state][action_]
    if not done:
        td_delta = td_target - q[state][action]
    else:
        td_delta = td_target  # - 0
    return q[state][action] + alpha * td_delta

if __name__ == "__main__":
    sarsa(1)
