import torch
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
from network import Actor, Critic
from torch import optim
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ACTOR_LR = 0.0001
CRITIC_LR = 0.0001
BUFFER_SIZE = 100000
BATCH_SIZE = 128
TAU = 0.0001
GAMMA = 0.99


class DDPGAgent:
    def __init__(self, state_size=33, action_size=4, n_agents=1, seed=1):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        ## DDPG specific configuration
        hidden_size = 128
        self.n_agents = n_agents
        self.CHECKPOINT_FOLDER = './'

        self.actor = Actor(state_size, hidden_size, action_size)
        self.actor_target = Actor(state_size, hidden_size, action_size)

        self.critic = Critic(state_size, hidden_size, 1)
        self.critic_target = Critic(state_size, hidden_size, 1)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        #Noise
        self.noise = OUNoise((self.n_agents, action_size), seed)

        #Initialize replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Update Critic
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + GAMMA*Q_targets_next*(1-dones)

        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        #Update Actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        #Updating
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

        self.actor_loss = actor_loss.data
        self.critic_loss = critic_loss.data


    def soft_update(self, model, model_target):
        tau = TAU
        for target_param, local_param in zip(model_target.parameters(), model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


    def step(self, state, action, reward, next_state, done):
        for i in range(self.n_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def checkpoint(self):
        torch.save(self.actor.state_dict(), self.CHECKPOINT_FOLDER + 'checkpoint_actor.pth')
        torch.save(self.critic.state_dict(), self.CHECKPOINT_FOLDER + 'checkpoint_critic.pth')


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
