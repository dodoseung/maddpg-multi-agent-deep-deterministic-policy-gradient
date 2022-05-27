# Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).
from audioop import avg
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

class CriticNet(nn.Module):
    def __init__(self, states_num, action_num, actions_num):
        super(CriticNet, self).__init__()
        self.input = nn.Linear(states_num + actions_num, 256)
        self.fc = nn.Linear(256, 512)
        self.output = nn.Linear(512, action_num)
    
    def forward(self, xs, us):
        x = torch.cat([xs, us], 1)        
        x = F.relu(self.input(x))
        x = F.relu(self.fc(x))
        value = self.output(x)
        return value
    
class MADDPG():
    def __init__(self, env, n_agents, memory_size=10000000, batch_size=64, tau=0.01, gamma=0.95, learning_rate=1e-3, eps_min=0.05, eps_period=10000):
        super(MADDPG, self).__init__()
        self.env = env
        self.n_agents = n_agents
                
        # Torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks and optimizers
        self.actors_net = []
        self.actors_opt = []
        self.actors_target_net = []
        self.critics_net = []
        self.critics_opt = []
        self.critics_target_net = []
        
        # Total number of states and actions
        self.states_num = sum([self.env.observation_spaces[agent].shape[0] for agent in self.env.agents])
        self.actions_num = sum([self.env.action_spaces[agent].shape[0] for agent in self.env.agents])
        
        for i, agent in enumerate(self.env.agents):
            self.state_num = self.env.observation_spaces[agent].shape[0]
            self.action_num = self.env.action_spaces[agent].shape[0]
            self.action_max = float(env.action_spaces[agent].high[0])
            self.action_min = float(env.action_spaces[agent].low[0])
            
            # Actors
            self.actors_net.append(ActorNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device))
            self.actors_opt = optim.Adam(self.actors_net[i].parameters(), lr=learning_rate)
            
            # Target Actors
            self.actors_target_net.append(ActorNet(self.state_num, self.action_num, self.action_min, self.action_max).to(self.device))
            self.actors_target_net[i].load_state_dict(self.actors_net[i].state_dict())
            
            # Critics
            self.critics_net.append(CriticNet(self.states_num, self.action_num, self.actions_num).to(self.device))
            self.critics_opt = optim.Adam(self.critics_net[i].parameters(), lr=learning_rate)
            
            # Target Critics
            self.critics_target_net.append(CriticNet(self.states_num, self.action_num, self.actions_num).to(self.device))
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
    def get_action(self, states, agent, exploration=True):
        state = torch.FloatTensor(states).unsqueeze(0).to(self.device)
        action = self.actors_net[agent](state).cpu().detach().numpy().flatten()

        if exploration:
            # Get noise (gaussian distribution with epsilon greedy)
            action_mean = (self.action_max + self.action_min) / 2
            action_std = (self.action_max - self.action_min) / 2
            action_num = self.env.action_spaces[self.env.agents[agent]].shape[0]
            action_noise = np.random.normal(action_mean, action_std, action_num)
            action_noise *= self.epsilon
            self.epsilon = self.epsilon - (1 - self.eps_min) / self.eps_period if self.epsilon > self.eps_min else self.eps_min
            
            # Final action
            action += action_noise
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
        print(states[0])
        states_all = [torch.FloatTensor(s).to(self.device) for s in states]
        actions_all = torch.FloatTensor(actions).to(self.device)
        rewards_all = torch.FloatTensor(rewards).to(self.device)
        next_states_all = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get next actions of all agents
        next_actions_all = [self.actors_target_net[i](next_states_all[i]) for i in range(self.n_agents)]
        
        for i in range(self.n_agents):
            # Actions and rewards for a single agent
            observations = states_all[i]
            actions = actions_all[i]
            rewards = rewards_all[i]
            
            # Target Q values
            target_q = self.critics_target_net[i](next_states_all, next_actions_all).view(1, -1)
            target_q = (rewards[:, i] + self.gamma * target_q * (1-dones))

            # Current Q values
            values = self.critics_net[i](states_all, actions_all).view(1, -1)
            
            # Calculate the critic loss and optimize the critic network
            critic_loss = F.mse_loss(values, target_q)
            self.critics_opt[i].zero_grad()
            critic_loss.backward()
            self.critics_opt[i].step()

            # Calculate new actions of the current agent
            new_actions = self.actors_net[i](observations)
            new_actions_all = actions_all[:i] + new_actions + actions_all[i+1:]

            # Calculate the actor loss and optimize the actor network
            actor_loss = -self.critics_net[i](states_all, new_actions_all).mean()
            self.actors_opt[i].zero_grad()
            actor_loss.backward()
            self.actors_opt[i].step()

            # Soft update the target networks
            self.soft_update(self.critics_net[i], self.critics_target_net[i])
            self.soft_update(self.actors_net[i], self.actors_target_net[i])


def main():
    env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
    env.reset()

    agent = MADDPG(env, n_agents=3, memory_size=100000, batch_size=64, tau=0.01, gamma=0.95, learning_rate=1e-3, eps_min=0.00001, eps_period=100000)
    ep_rewards = deque(maxlen=1)
    total_episode = 10000
    
    for i in range(total_episode):
        env.reset()
        ep_reward = [0] * agent.n_agents
        while True:
            states = []
            actions = []
            rewards = []
            next_states = []
            
            # Get and supply actions.
            for j, agent_name in enumerate(agent.env.agents):
                obs = env.observe(agent_name)
                action = agent.get_action(obs, j)
                env.step(action)
                
                next_state, reward, done, _ = env.last()
                states.extend(obs)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                
            ep_reward = [x + y for x, y in zip(ep_reward, rewards)]
            agent.replay_buffer.add(states, actions, rewards, next_states, done)

            if i > 2:
                agent.learn()
            
            if done:
                ep_rewards.append(ep_reward)
                if i % 1 == 0:
                    print("episode: {}\treward: {}".format(i, [round(sum(x) / len(x), 3) for x in zip(*ep_rewards)]))
                break

if __name__ == '__main__':
    main()