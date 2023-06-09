import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
import random
import base64, io
from tqdm import tqdm

import gymnasium as gym
import pickle

import numpy as np
from collections import deque, namedtuple



seed=1
class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed=seed):
        super(QNetwork, self).__init__()
        torch.manual_seed(seed)
        self.l1 = nn.Linear(state_size, 64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,action_size)

        self.action_size = action_size
        self.state_size = state_size


    def forward(self, s):
        s = self.l1(s)
        s = F.relu(s)
        s= self.l2(s)
        s = F.relu(s)
        s = self.l3(s)
        return s

Experience = namedtuple("Experience",field_names=['state','action','reward','next_state','done'])        

class ReplayBuffer():

    

    def __init__(self, action_size, buffer_size, batch_size,seed=seed):


        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
        self.seed = random.seed(seed)
    def add(self, *args):
        self.memory.append(Experience(*args))
    
    def index(self, idx):
        return self.memory.index(idx)

    def sample(self):

        experiences = random.sample(self.memory, self.batch_size)
        return experiences
    
    def __len__(self):
        return len(self.memory)
    

class Agent():

    def __init__(self, action_size, state_size, buffer_size, batch_size,gamma,update_freq=4, tau=0.001,eps_start=1,eps_decay=0.995,eps_end=0.01, LR=0.0005,seed=seed):
        self.action_size=action_size
        self.state_size=state_size
        
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        
        self.tau=tau
        self.update_freq = update_freq

        self.gamma= gamma

        self.seed=seed

        self.eps_start=eps_start
        self.eps_decay= eps_decay
        self.epd_end= eps_end
        self.eps = eps_start

        self.q_local = QNetwork(state_size, action_size)
        self.q_target = QNetwork(state_size, action_size)

        self.optimizer = optim.Adam(self.q_local.parameters(), lr=LR)

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size)

        self.time = 0

    def act(self, state, epsilon):

        if np.random.random() < epsilon:
            return np.random.randint(0,self.action_size)
        else:

            self.q_local.eval()
            with torch.no_grad():
                q_values = self.q_local(Tensor(state))
            self.q_local.train()
            return torch.argmax(q_values).item()



    def step(self, state, action, reward, next_state, done):

        if state is None or action is None or reward is None or next_state is None or done is None:
            print('please help, one of the values in the transition tuple is None')
        self.memory.add(state,action,reward, next_state, done)

        self.time = (self.time + 1)%self.update_freq

        #self.eps = max(self.eps*self.eps_decay, self.epd_end)

        if self.time==0:
            if len(self.memory) >= self.batch_size:

                sample = self.memory.sample()
                self.learn(sample)

    def learn(self, experiences):

        states = Tensor([e.state for e in experiences])
        actions = Tensor([e.action for e in experiences]).unsqueeze(1).to(torch.int64)
        next_state = Tensor([e.next_state for e in experiences])
        reward = Tensor([e.reward for e in experiences]).unsqueeze(1)
        done = Tensor([e.done for e in experiences])


        expected_q = self.q_local(states).gather(1, actions)

        target_q = reward
        target_q = target_q + (1-done)*self.gamma* self.q_target(next_state).detach().max(1)[0].unsqueeze(1)


        criterion = nn.MSELoss()
        loss = F.mse_loss(expected_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for local_par, target_par in zip(self.q_local.parameters(), self.q_target.parameters()):
            target_par.data.copy_(self.tau*local_par.data + (1.0-self.tau)*(target_par.data))
    def save_networks(self, PATH):

        torch.save(self.q_local, PATH)
        
        


env = gym.make("CartPole-v1")


batch_size=32
buffer_size=int(1e5)
gamma = 0.99
tau = 1e-4
lr=1e-4
update_every=4


agent = Agent(
    action_size=2,
    state_size=2,
    buffer_size=buffer_size,
    batch_size=batch_size,
    gamma=gamma,
    update_freq=update_every,
    tau=tau,
    eps_start=1,
    eps_decay=0.999,
    eps_end=0.01,
    LR=5e-4
)




E = int(1e5)
ep_returns = [0]

eps=1
eps_decay=0.995
eps_end = 0.01




with tqdm(range(E)) as tp:
    for i in tp:
        tp.set_postfix(episode=i, epsilon=eps,last_return = np.mean(ep_returns[-50:]))


        state, info = env.reset()
        state = state[0],state[2]
        
        ep_reward= 0

        done=False
        while not done:
            action = agent.act(state, eps)

            state_prime, reward, terminate, truncate, _ = env.step(action)
            state_prime = state_prime[0], state_prime[2]

            done = terminate or truncate
          
            
            agent.step(state, action, reward, state_prime,done)
            state=state_prime
            
            ep_reward += reward
        ep_returns.append(ep_reward)
        eps = max(eps_end, eps*eps_decay)


epr = np.array(ep_returns)


#maybe augment this so it also saves episodes and can be used with d3rlpy algorithms

expert_agent = agent

new_data_buffer = ReplayBuffer(action_size=2, buffer_size=10000000, batch_size=64)
cql_data_buffer = ReplayBuffer(action_size=2, buffer_size=10000000, batch_size=64)


episodes = 50000

env = gym.make('CartPole-v1')

history_lookback=16

with tqdm(range(episodes)) as tp:
    for i in tp:
        tp.set_postfix(episode=i)


        




        S, info = env.reset()
        S = S[0],S[2]
        terminate=False
        truncate=False

        state_hist = [S]
        action_hist = []
        reward_hist = []

        score=0


        while not terminate and not truncate:

            A = expert_agent.act(S,0.0)

            S_prime, reward, terminate, truncate, info = env.step(A)
            S_prime = S_prime[0], S_prime[2]
            done = terminate or truncate

            score+=reward
            #new_data_buffer.add(state_hist, action_hist, reward_hist, A, done)
            cql_data_buffer.add(state_hist[-1*history_lookback:], action_hist[-1*history_lookback:] + [A], reward_hist[-1*history_lookback:]+[reward], S_prime, done)
            state_hist.append(S_prime)
            action_hist.append(A)
            reward_hist.append(reward)
            #state_hist = state_hist[:]
            #action_hist = action_hist[:]
            #reward_hist = reward_hist[:]
            S = S_prime

        reward_hist = np.cumsum(reward_hist)
        reward_hist = score-reward_hist
        for i in range(1,len(action_hist)-1):
            

            end =i
            start = max(0, i-history_lookback)




            new_data_buffer.add(state_hist[start:end+1], action_hist[start:end], reward_hist[start:end+1], action_hist[i], False)
            

#pickle.dump(expert_agent, open('expert_agent.pkl','wb'))


pickle.dump(new_data_buffer,open('expert_data_buffer.pkl','wb'))
pickle.dump(cql_data_buffer,open('cql_data_buffer.pkl','wb'))

np.save("dqn_learning_curve_stats.npz",epr)

expert_agent.save_networks('expert_agent.pt')
