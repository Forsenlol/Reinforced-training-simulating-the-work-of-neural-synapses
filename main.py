import gym
import numpy as np
import operator

env = gym.make('MountainCar-v0')
possible_actions = env.action_space.n
print 'Possible actions are {}'.format(possible_actions)


class EpsilonGreedy():

    def __init__(self, episodes=1000, epsilon=0.2):
        self.episodes = episodes
        self.epsilon = epsilon
        self.values = {0: 0.0, 1: 0.0, 2: 0.0}
        self.counts = {0: 0, 1: 0, 2: 0}

        def explore(self):
            return np.random.choice(self.counts.keys())

            def exploit(self):

                return max(self.values.items(), \
                           key=operator.itemgetter(1))[0]

            def select_action(self, observation):

                if np.random.uniform(0, 1) < self.epsilon:
                    return self.explore()
                else:
                    return self.exploit()

        def update_counts(self, action):
            self.counts[action] = self.counts[action] + 1

        def update_values(self, action, reward):
            current_value = self.values[action]

        n = self.counts[action]
        self.values[action] = ((n - 1) / float(n)) * \
                              current_value + (1 / float(n)) * reward

        def update_all(self, action, reward):
            self.update_counts(action)

        self.update_values(action, reward)


epsilonlearn = EpsilonGreedy()
for episode in xrange(epsilonlearn.episodes):
    observation = env.reset()
    while True:
        env.render()
        action = epsilonlearn.select_action(observation)
        next_observation, reward, done, _ = env.step(action)
        epsilonlearn.update_all(action, reward)
        observation = next_observation
        if done:
            break
        env.destroy()
