import gym
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from collections import deque

# For visualization
from IPython.display import clear_output, display

# Variables for model 
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_layer1 = nn.Linear(state_size, 128)
        self.fc_layer2 = nn.Linear(128, 128)
        self.fc_layer3 = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc_layer1(state))
        x = F.relu(self.fc_layer2(x))
        return self.fc_layer3(x)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

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
        self.buffer_size = buffer_size
        self.memory_states = torch.zeros(self.buffer_size, 8).to(device)
        self.memory_actions = torch.zeros(self.buffer_size, 1).long().to(device)
        self.memory_rewards = torch.zeros(self.buffer_size, 1).to(device)
        self.memory_next_states = torch.zeros(self.buffer_size, 8).to(device)
        self.memory_dones = torch.zeros(self.buffer_size, 1).int().to(device)
        self.batch_size = batch_size
        self.count = 0
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        memory_idx = self.count % self.buffer_size

        self.memory_states[memory_idx] = torch.from_numpy(state).float()
        self.memory_actions[memory_idx] = torch.Tensor([action]).long()
        self.memory_rewards[memory_idx] = float(reward)
        self.memory_next_states[memory_idx] = torch.from_numpy(next_state).float()
        self.memory_dones[memory_idx] = int(done)

        self.count += 1

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        random_idxs = np.array(random.sample(range(len(self)), k=self.batch_size))

        states = self.memory_states[random_idxs]
        actions = self.memory_actions[random_idxs]
        rewards = self.memory_rewards[random_idxs]
        next_states = self.memory_next_states[random_idxs]
        dones = self.memory_dones[random_idxs]

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return min(self.count, self.buffer_size)


class Agent():
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

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

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

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Obtain random minibatch of tuples from D
        states, actions, rewards, next_states, dones = experiences

        # Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        #####
        # COMPLETE CODE HERE
        ### Calculate target value from bellman equation
        q_targets = rewards + gamma*(1-dones)*q_targets_next

        # Calculate expected value from local network
        q_expected = self.qnetwork_local(states).gather(1, actions)

        #####
        # COMPLETE CODE HERE
        ### Loss calculation (use Mean squared error)
        loss = (q_expected - q_targets)**2
        loss = F.mse_loss(q_expected, q_targets)
        #####

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


def dqn(agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        agent (Agent): RL agent
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # self.agent = agent
    start_time = time.time()
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        state = env.reset()[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        elapsed_time_s = time.time() - start_time
        end_char = "\n" if i_episode % 100 == 0 else ""
        print('\rEpisode {}\tAverage Score: {:.2f}\tCount: {:.2f}\tElapsed time: {:.2f} sec'.format(i_episode, np.mean(scores_window), len(agent.memory), elapsed_time_s), end=end_char)

    print('\nEnvironment trained for {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores


if __name__ == "__main__": 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    # Creating LunarLander environment
    env = gym.make('LunarLander-v2', render_mode="human")

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    state = env.reset()
    total_reward = 0
    render = False
    if render:
        img = plt.imshow(env.render())

    while True:
        if render:
            img.set_data(env.render())
            plt.axis('off')
            display(plt.gcf())
            clear_output(wait=True)
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Total reward: {total_reward}")

    agent = Agent(state_size=8, action_size=4, seed=0)
    scores = dqn(agent, n_episodes=1000)