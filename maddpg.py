# Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import gym
import random
from collections import deque
from pettingzoo.mpe import simple_adversary_v2

class ReplayBuffer():
    def __init__(self, max_size=100000):
        super(ReplayBuffer, self).__init__()
        self.max_size = max_size
        self.memory = deque(maxlen=self.max_size)
        
    # Add the replay memory
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Sample the replay memory
    def sample(self, batch_size):
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

class ActorNet(nn.Module):
    def __init__(self, state_num, action_num, min_action, max_action):
        super(ActorNet, self).__init__()
        self.input = nn.Linear(state_num, 256)
        self.fc = nn.Linear(256, 512)
        self.output = nn.Linear(512, action_num)

        # Get the action interval for clipping
        self.min_action = min_action
        self.max_action = max_action
    
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        action = self.output(x)
        action = torch.clamp(action, self.min_action, self.max_action)
        return action
    
    def output_layer_size(self, size, kernel_size, stride):
        return (size - kernel_size) // stride + 1

class CriticNet(nn.Module):
    def __init__(self, state_num, action_num):
        super(CriticNet, self).__init__()
        self.input = nn.Linear(state_num, 256)
        self.fc = nn.Linear(256, 512)
        self.output = nn.Linear(512, action_num)
    
    def forward(self, x, u):
        x = torch.cat([x, u], 1)        
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        value = self.output(x)
        return value
    
    def output_layer_size(self, size, kernel_size, stride):
        return (size - kernel_size) // stride + 1
    
class MADDPG():
    def __init__(self, env, n_agents, memory_size=10000000, batch_size=64, tau=0.01, gamma=0.95, learning_rate=1e-3, eps_min=0.05, eps_period=10000):
        super(MADDPG, self).__init__()
        self.env = env
        self.state_num = self.env.observation_space[0].shape[0]
        self.action_num = self.env.action_space.shape[0]
        self.action_max = float(env.action_space.high[0])
        self.action_min = float(env.action_space.low[0])
        self.n_agents = n_agents
                
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actors
        self.actors_net = [ActorNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device) for _ in range(self.n_agents)]
        self.actors_opt = [optim.Adam(n.parameters(), lr=learning_rate) for n in self.actors_net]
        
        # Target Actors
        self.actors_target_net = [ActorNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device) for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            self.actors_target_net[i].load_state_dict(self.actors_net[i].state_dict())
        
        # Critics
        self.critics_net = [CriticNet(self.state_num, self.action_num).to(self.device) for _ in range(self.n_agents)]
        self.critics_opt = [optim.Adam(n.parameters(), lr=learning_rate) for n in self.critics_net]
        
        # Target Critics
        self.critics_target_net = [CriticNet(self.state_num, self.action_num).to(self.device) for _ in range(self.n_agents)]
        for i in range(self.n_agents):
            self.critics_target_net[i].load_state_dict(self.critics_net[i].state_dict())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(memory_size)
        self.batch_size = batch_size

        # Learning setting
        self.gamma = gamma
        self.tau = tau
        
        # Noise setting
        self.epsilon = 1
        self.eps_min = eps_min
        self.eps_period = eps_period

    # Get the action
    def get_action(self, states, agent_num, exploration=True):
        state = torch.FloatTensor(states).unsqueeze(0).to(self.device)
        action = self.actors_net[agent_num](state).cpu().detach().numpy().flatten()
        
        if exploration:
            # Get noise (gaussian distribution with epsilon greedy)
            action_mean = (self.action_max + self.action_min) / 2
            action_std = (self.action_max - self.action_min) / 2
            action_noise = np.random.normal(action_mean, action_std, 1)[0]
            action_noise *= self.epsilon
            self.epsilon = self.epsilon - (1 - self.eps_min) / self.eps_period if self.epsilon > self.eps_min else self.eps_min
            
            # Final action
            action = action + action_noise
            action = np.clip(action, self.action_min, self.action_max)
        
        return action

    # Soft update a target network
    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    # Learn the policy
    def learn(self):
        # Replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        next_actions = [self.actors_target_net[i](next_states) for i in range(self.n_agents)]
        
        for i in range(self.n_agents):
            # Target Q values
            target_q = self.critics_target_net[i](next_states, next_actions).view(1, -1)
            target_q = (rewards[:, i] + self.gamma * target_q * (1-dones))

            # Current Q values
            values = self.critics_net[i](states, actions).view(1, -1)
            
            # Calculate the critic loss and optimize the critic network
            critic_loss = F.mse_loss(values, target_q)
            self.critics_opt[i].zero_grad()
            critic_loss.backward()
            self.critics_opt[i].step()
        
            # Calculate the actor loss and optimize the actor network
            actor_loss = -self.critics_net[i](states, self.actors_net[i](states)).mean()
            self.actors_opt[i].zero_grad()
            actor_loss.backward()
            self.actors_opt[i].step()

            # Soft update the target networks
            self.soft_update(self.critics_net[i], self.critics_target_net[i])
            self.soft_update(self.actors_net[i], self.actors_target_net[i])


def main():
    env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
    agent = MADDPG(env, n_agents=3, memory_size=100000, batch_size=64, tau=0.01, gamma=0.95, learning_rate=1e-3, eps_min=0.00001, eps_period=100000)
    ep_rewards = deque(maxlen=1)
    total_episode = 10000
    
    for i in range(total_episode):
        state = env.reset()
        ep_reward = 0
        while True:
            actions = []
            rewards = []
            
            # Get and supply actions.
            for j in range(agent.n_agents):
                action = agent.get_action(state, j)
                env.step(action)
                
                next_state, reward, done, _ = env.last()
                
                actions.append(action)
                rewards.append(reward)

            agent.replay_buffer.add(state, actions, rewards, next_state, done)

            if i > 2:
                agent.learn()
            
            if done:
                ep_rewards.append(ep_reward)
                if i % 1 == 0:
                    print("episode: {}\treward: {}".format(i, round(np.mean(ep_rewards), 3)))
                break

            state = next_state

if __name__ == '__main__':
    main()