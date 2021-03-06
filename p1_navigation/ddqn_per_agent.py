import numpy as np
import random
from collections import namedtuple

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from sum_tree import SumTree

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Prioritized Experience Replay memory
        self.memory = PrioritizedExperienceReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                i_s_weights, indexes, experiences = self.memory.sample()
                self.learn(i_s_weights, indexes, experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, i_s_weights, indexes, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            i_s_weights: Importance sampling weights
            indexes: Indices of chosen samples
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # DDQN: Compute Q targets for current states
        best_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets = rewards + (gamma * self.qnetwork_target(next_states).detach().gather(1, best_actions) * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        td_error = q_targets - q_expected

        # Compute loss
        i_s_weights = torch.tensor(i_s_weights, dtype=torch.float).cuda()
        loss = (torch.abs(td_error) * i_s_weights).mean()
        # loss = F.mse_loss(q_expected * i_s_weights, q_targets * i_s_weights)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        abs_errors = torch.squeeze(torch.abs(td_error)).cpu().data.numpy()
        for index, abs_error in zip(indexes, abs_errors):
            self.memory.update(index, abs_error)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class PrioritizedExperienceReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    alpha = 0.6
    beta = 0.4
    beta_increment_per_sample = 0.001
    epsilon = 1e-6

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def compute_priority(self, td_error):
        return (td_error + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = self.experience(state, action, reward, next_state, done)
        max_priority = np.max(self.memory.tree[-self.memory.capacity:])
        if max_priority == 0:
            max_priority = 1.

        self.memory.add(max_priority, experience)

    def update(self, index, td_error):
        priority = self.compute_priority(td_error)
        self.memory.update(index, priority)

    def sample(self):
        """

        :return: importance weights, indices of sampled experiences, and sampled batch of experiences
        """
        self.beta = np.minimum(1., self.beta + self.beta_increment_per_sample)
        segment = self.memory.total() / self.batch_size
        indexes = []
        priorities = []
        experiences = []

        for i in range(self.batch_size):
            # pick a segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            index, priority, experience = self.memory.get(s)
            indexes.append(index)
            priorities.append(priority)
            experiences.append(experience)

        sampling_probs = np.divide(priorities, self.memory.total())
        # importance sampling
        i_s_weights = (self.batch_size * sampling_probs) ** -self.beta
        i_s_weights /= np.max(i_s_weights)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return i_s_weights, indexes, (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.memory.count
