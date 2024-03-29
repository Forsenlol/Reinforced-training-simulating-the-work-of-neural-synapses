import os
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
env.reset()


class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.num_actions = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


restore = True

if restore and os.path.isfile("polict.pt"):
    policy = torch.load("policy.pt")
else:
    policy = Policy()

optimizer = optim.Adam(policy.parameters(), lr=0.001)


def update_policy(states, actions, rewards, log_probs, gamma=0.99):
    """
    Расчет потерь, вычисление градиентов, обратное распространение и обновление параметров нейронной сети.
    """
    loss = []
    dis_rewards = rewards[:]
    for i in range(len(dis_rewards) - 2, -1, -1):
        dis_rewards[i] = dis_rewards[i] + gamma * dis_rewards[i + 1]

    dis_rewards = torch.tensor(dis_rewards)
    for log_prob, reward in zip(log_probs, dis_rewards):
        loss.append(-log_prob * reward)

    loss = torch.cat(loss).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_policy_values(state):
    """
    Рассчет ненормализованных значений policy по входному массиву значений
    """
    state = Variable(torch.from_numpy(state)).type(torch.FloatTensor).unsqueeze(0)
    policy_values = policy(state)
    return policy_values


def generate_episode(t_max=1000):
    """
    Создание эпизода. Сохранение состояний, действий, наград и регистрация вероятностей. Обновление policy
    """
    states, actions, rewards, log_probs = [], [], [], []
    s = env.reset()

    for t in range(t_max):
        action_probs = F.softmax(get_policy_values(s), dim=-1)
        sampler = Categorical(action_probs)
        a = sampler.sample()
        log_prob = sampler.log_prob(a)
        new_s, r, done, _ = env.step(a.item())

        states.append(s)
        actions.append(a)
        rewards.append(r)
        log_probs.append(log_prob)

        s = new_s
        if done:
            break
    update_policy(states, actions, rewards, log_probs)
    return sum(rewards)


def play_episodes(num_episodes=10, render=False):
    """
    Запускаем игру используя обученную policy
    """
    for i in range(num_episodes):
        rewards = []
        s = env.reset()
        for _ in range(1000):
            if render:
                env.render()
            action_probs = F.softmax(get_policy_values(s), dim=-1)
            sampler = Categorical(action_probs)
            a = sampler.sample()
            log_prob = sampler.log_prob(a)
            new_s, r, done, _ = env.step(a.item())

            rewards.append(r)
            s = new_s
            if done:
                print("Episode {} finished with reward {}".format(i + 1, np.sum(rewards)))
                break


def plot_rewards(rewards, running_rewards):
    """
    Графики
    """
    plt.style.use('seaborn-darkgrid')
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    plt.subplots_adjust(hspace=.5)

    ax1.set_title('Episodic rewards')
    ax1.plot(rewards, label='Episodic rewards')
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Rewards")

    ax2.set_title('Running rewards')
    ax2.plot(running_rewards, label='Running rewards')
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Average rewards")

    fig.savefig("backpropagatw_test.png")


if __name__ == "__main__":
    num_episodes = 1500
    verbose = True
    print_every = 50
    target_avg_reward_100ep = 195
    running_reward = None
    rewards = []
    running_rewards = []
    restore_model = True

    if restore_model and os.path.isfile("polict.pt"):
        policy = torch.load("policy.pt")
    else:
        policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    for i in range(num_episodes):
        reward = generate_episode()
        rewards.append(reward)
        running_reward = np.mean(rewards[-100:])
        running_rewards.append(running_reward)

        if verbose:
            if not i % print_every:
                print("Episode: {}. Running reward: {}".format(i + 1, running_reward))

        if i >= 99 and running_reward >= target_avg_reward_100ep:
            print("Episode: {}. Running reward: {}".format(i + 1, running_reward))
            print("Ran {} episodes. Solved after {} episodes.".format(i + 1, i - 100 + 1))
            break
        elif i == num_episodes - 1:
            print("Couldn't solve after {} episodes".format(num_episodes))
    plot_rewards(rewards, running_rewards)
    torch.save(policy, "cartpole_policy_reinforce.pt")
