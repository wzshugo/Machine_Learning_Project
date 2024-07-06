import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Hyperparameters
EPISODES = 600
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Displaying animation
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = env.unwrapped
state = env.reset()[0]
pre_frames = []
pre_path = "D://下载//pre_AC.gif"
done = False
while not done:
    pre_frames.append(env.render())
    action = env.action_space.sample()
    state, reward, done, info, _ = env.step(action)
env.close()
imageio.mimsave(pre_path, pre_frames, duration=50)

# Actor-Critic Networks
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# A2C Agent
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=LEARNING_RATE)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probabilities = self.actor(state).detach().numpy()[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Calculate advantage
        value = self.critic(state)
        next_value = self.critic(next_state)
        advantage = reward + (1 - done) * GAMMA * next_value - value

        # Actor loss
        probabilities = self.actor(state)
        log_prob = torch.log(probabilities.squeeze(0)[action])
        actor_loss = -log_prob * advantage.detach()

        # Critic loss
        critic_loss = advantage.pow(2)

        # Total loss
        loss = actor_loss + critic_loss

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Training A2C
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = A2CAgent(state_size, action_size)
rewards = []

for e in range(EPISODES):
    state = env.reset()[0]
    total_reward = 0
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, info, _ = env.step(action)
        total_reward += reward
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e+1}/{EPISODES}, Score: {total_reward}")
            break
    rewards.append(total_reward)

# Plotting results
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('A2C CartPole')
plt.show()

# Save the models
torch.save(agent.actor.state_dict(), 'a2c_actor_cartpole.pth')
torch.save(agent.critic.state_dict(), 'a2c_critic_cartpole.pth')

# Displaying animation
env = gym.make('CartPole-v1', render_mode="rgb_array")
env = env.unwrapped
state = env.reset()[0]
post_frames = []
post_path = "D://下载//post_AC.gif"
done = False
while not done:
    post_frames.append(env.render())
    action = agent.act(state)
    state, reward, done, info, _ = env.step(action)
env.close()
imageio.mimsave(post_path, post_frames, duration=50)