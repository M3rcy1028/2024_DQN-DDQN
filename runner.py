import math
import numpy as np
import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable
from collections import namedtuple
import seaborn as sns
from memory import ExperienceReplay
from torch.utils.tensorboard import SummaryWriter
from environment import init_env

''' Hyperparameters
  nn            : Neural Network for Q-values
  loss          : MSE loss function
  device        : cpu or gpu
  logs          : path for storing log file
  env           : CartPole-v1
  algorithm     : used for training (dqn13, dqn15, ddqn)

  lr            : learning rate (0 < lr < 1)
  eps_start     : epsilon-decay initial value   (0 <= eps_start <= 1)
  eps_end       : epsilon-decay boundary value  (0 <= eps_end <= 1)
  eps_decay     : epsilon reduction amount for each iteration
  batch_size    : mini-batch size used in Experience Replay
  target_update : target network update period
  gamma         : discount factor (0 < gamma < 1)
  maxreward     : maximunm reward for training
'''

class Runner(): 
  def __init__(self, nn, loss, device, env, algorithm='dqn13', lr = 0.001, eps_start = 0.9, eps_end = 0.1, eps_decay = 200,
               batch_size = 128, target_update = 40, logs = "runs", gamma = 0.999, maxreward = 500):
    self.writer = SummaryWriter(logs)       # Visualize data with TensorBoard
    self.learner = nn                       # Main network
    self.loss = loss
    self.device = device
    self.env = env
    self.logs = logs
    self.algo = algorithm

    self.optimizer = optim.Adam(self.learner.parameters(), lr = lr)
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.batch_size = batch_size 
    self.target_update = target_update
    self.gamma = gamma
    self.maxreward = maxreward

    self.memory = ExperienceReplay(100000)  # capacity = 100,000
    self.steps = 0

    #### dqn & ddqn ####
    self.target = nn                        # Target network
    self.target.load_state_dict(self.learner.state_dict())
    self.target.eval() 
    
    # loss/reward/mean reward value for TensorBoard
    self.plots = {"Loss": torch.tensor([]).to(self.device),
                  "Reward": torch.tensor([]).to(self.device), 
                  "Mean Reward": torch.tensor([]).to(self.device)}
  
  # Select an action
  def select_action(self, state):
    self.steps = self.steps + 1
    sample = random.random()
    # epsilon-decaying
    eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps / self.eps_decay)

    if sample > eps_thresh: # Optimal action
      with torch.no_grad(): 
        action = torch.argmax(self.learner(state)).view(1, 1)
        return action
    else:                   # Random action
      return torch.tensor([[random.randrange(self.env.action_space.n)]], device = self.device, dtype=torch.long)

  # Train NN with data in Experience Reokay
  def train_inner(self): 
    # the size of memory is too small, do not train
    if len(self.memory) < self.batch_size: 
      return 0
    
    sample_transitions = self.memory.sample(self.batch_size)
    batch = self.memory.Transition(*zip(*sample_transitions))

    # Check if next state is None(=done) or not
    has_next_state = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype=torch.bool)
    next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Predicted Q(s,a)
    pred_values = self.learner(state_batch).gather(1, action_batch)

    # Initialize array
    next_state_values = torch.zeros(self.batch_size, device = self.device) # Initialize
    # Compute Q-value
    if (self.algo == 'dqn13'):      # argmax(a') Q_main(next_state, a', theta)
      next_state_values[has_next_state] = self.learner(next_states).max(1)[0].detach()
    elif (self.algo == 'dqn15'):    # argmax(a') Q_target(next_state, a', theta)
      next_state_values[has_next_state] = self.target(next_states).max(1)[0].detach()
    else: # ddqn                    # a^ = argmax Q_main(next_state, a', theta) -> Q_target(next_state, a^, theta)
      best_next_action = self.learner(next_states).argmax(1).unsqueeze(1) 
      next_state_values[has_next_state] = self.target(next_states).gather(1, best_next_action).squeeze().detach()

    # Target value = reward + gamma * Q_value
    target_values = reward_batch + next_state_values * self.gamma

    # loss value L(s, a) = (y_hat - y)**2
    loss = self.loss(pred_values, target_values.unsqueeze(1))

    # Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.learner.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    return loss

  # Perform the given action
  def env_step(self, action):
    state, reward, done, log, _ = self.env.step(action)
    return torch.FloatTensor([state]).to(self.device), torch.FloatTensor([reward]).to(self.device), done, log

  # Perform training
  def train(self, episodes=100, smooth=10): 
    steps = 0
    smoothed_reward = []
    mean_reward = 0
    for episode in range(episodes):
      c_loss = 0
      c_samples = 0
      rewards = 0

      state, _ = self.env.reset()  # Reset environment
      state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)

      # Perform learning
      for i in range(self.maxreward):
        action = self.select_action(state)
        next_state, reward, done, _ = self.env_step(action.item())

        if done:  # if game ends
          next_state = None
          
        self.memory.push(state, action, next_state, reward) # Save experience
        state = next_state  # Move to next state

        loss = self.train_inner() 
        rewards += reward.detach().item()

        if done:
          break
        
        steps += 1
        c_samples += self.batch_size
        c_loss += loss

      # Smoothing reward
      smoothed_reward.append(rewards)
      if len(smoothed_reward) > smooth: 
        smoothed_reward = smoothed_reward[-1*smooth: -1]
      
      # Write at TensorBoard
      self.writer.add_scalar("Loss", c_loss/c_samples, steps) 
      self.writer.add_scalar("Reward", rewards, episode)  
      self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)

      # Update loss and reward
      self.plots["Loss"] = torch.cat((self.plots["Loss"], torch.tensor([loss]).to(self.device)))
      self.plots["Reward"] = torch.cat((self.plots["Reward"], torch.tensor([rewards]).to(self.device)))
      self.plots["Mean Reward"] = torch.cat((self.plots["Mean Reward"], torch.tensor([np.mean(smoothed_reward)]).to(self.device)))

      if episode % 20 == 0: 
        print("\tEpisode {} \t Final Reward {:.2f} \t Average Reward: {:.2f}".format(episode, rewards, np.mean(smoothed_reward)))

      #### dqn15 & ddqn ####
      if (self.algo != 'dqn13'): # Update Target Network
        if i % self.target_update == 0:
          self.target.load_state_dict(self.learner.state_dict())

    self.env.close()

  # Run simulation and get result animation
  def run(self):
    sns.set_style("dark")
    sns.set_context("poster")

    fig = plt.figure() 
    ims = []
    rewards = 0
    state, _ = self.env.reset()
    state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(self.device)

    for time in range(self.maxreward):
      action = self.select_action(state) 
      state, reward, done, _ = self.env_step(action.data[0].item())
      rewards += reward

      if done:
        break
    
      im = plt.imshow(self.env.render(), animated=True)
      plt.axis('off')
      if (self.algo == 'dqn13'):
        plt.title("DQN(2013) Agent")
      elif (self.algo == 'dqn15'):
        plt.title("DQN(2015) Agent")
      else: # ddqn
        plt.title("DDQN Agent")
      ims.append([im])

    print("\tTotal Reward: ", rewards)
    self.env.close()
    print("\tSaving Animation ...")
    ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=1000)
    ani.save('%s-movie.gif'%self.logs, dpi = 300)
    # animation.save('animation.gif', writer='PillowWriter', fps=2)

  def plot(self):
    sns.set()
    sns.set_context("poster")

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(self.plots["Loss"])), self.plots["Loss"].detach().cpu().numpy())
    if (self.algo == 'dqn13'):
      plt.title("DQN(2013) Gradient Loss")
    elif (self.algo == 'dqn15'):
      plt.title("DQN(2015) Gradient Loss")
    else: # ddqn
      plt.title("DDQN Gradient Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig("%s/plot_%s.png"%(self.logs, "loss"))

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(self.plots["Reward"])), self.plots["Reward"].detach().cpu().numpy(), label="Reward")
    plt.plot(np.arange(len(self.plots["Mean Reward"])), self.plots["Mean Reward"].detach().cpu().numpy(), label="Mean Reward")
    plt.legend()
    if (self.algo == 'dqn13'):
      plt.title("DQN(2013) Gradient Rewards")
    elif (self.algo == 'dqn15'):
      plt.title("DQN(2015) Gradient Rewards")
    else: # ddqn
      plt.title("DDQN Gradient Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("%s/plot_%s.png"%(self.logs, "rewards"))

  def save(self): 
    torch.save(self.learner.state_dict(),'%s/model.pt'%self.logs)
