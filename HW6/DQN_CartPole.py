import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import imageio

# Hyperparameters
EPISODES = 60
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000

# Displaying animation
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = env.unwrapped
state = env.reset()[0]
pre_frames = []
pre_path = "D://下载//pre_DQN.gif"
done = False
while not done:
    pre_frames.append(env.render())
    action = env.action_space.sample()
    state, reward, done, info, _ = env.step(action)
env.close()
imageio.mimsave(pre_path, pre_frames, duration=50)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += GAMMA * torch.max(self.target_model(torch.FloatTensor(next_state)).detach())
            target_f = self.model(torch.FloatTensor(state))
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(torch.FloatTensor(state)))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

# Training DQN
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
rewards = []

for e in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}")
            break
        agent.replay()
    rewards.append(total_reward)
    if e % TARGET_UPDATE == 0:
        agent.update_target_model()

# Plotting results
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN CartPole')
plt.show()

# Save the model
torch.save(agent.model.state_dict(), 'dqn_cartpole.pth')

# Displaying animation
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = env.unwrapped
state = env.reset()[0]
post_frames = []
post_path = "D://下载//post_DQN.gif"
done = False
while not done:
    post_frames.append(env.render())
    action = agent.act(state)
    state, reward, done, info, _ = env.step(action)
env.close()
imageio.mimsave(post_path, post_frames, duration=50)