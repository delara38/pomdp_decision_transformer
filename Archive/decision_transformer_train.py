import os
import sys
from pathlib import Path




sys.path.append((Path(os.getcwd()).parent/"bruh").absolute())
sys.path.append(Path(os.getcwd()).parent.parent.absolute())


import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import ZeroPad2d
from tqdm import tqdm

import _pickle as pickle
import matplotlib.pyplot as plt

from models.attention_mechanisms import scaled_dot_product_attention
from models.general_transformer import TransformerDecoder, TransformerDecoderLayer, MultiHeadAttention, PaddingMask

from decision_transformer import DecisionTransformer


import gymnasium as gym
import random
from collections import namedtuple, deque



seed=1967
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
        return self.memory[idx]

    def sample_indexes(self, indexes):
        l = []
        for i in indexes:
              l.append(self.index(i))
        
        l = Experience(*zip(*l))
        return l

    def sample(self):

        experiences = random.sample(self.memory, self.batch_size)
        return experiences
    
    def __len__(self):
        return len(self.memory)
    
    def get_mem(self):
         return self.memory

    def append(self, x):
         self.memory += x.get_mem()


    def cut(self, k):
        self.memory = random.sample( self.memory, k)




def train(d1 ,d2, mix = 0, state_space_dim =2,n_actions=2,saveto = 'decision_transformer.pt'):

    def onehot( action):
         x = np.zeros(n_actions)
         x[action]=1
         return x
    
    if mix ==1:
         data=d2
    elif mix == 0.5:
         d1.append(d2)
         data=d1
    elif mix ==0:
        data=d1
    else:
         raise AttributeError
    
         


         


    num_layers=4
    dim_models = [32 for i in range(num_layers+1)]
    
    K=16
    num_heads=3
    attention_mechanism=scaled_dot_product_attention
    dim_feedforward=32
    dropout=0.0
    lr=0.00001



    dt = DecisionTransformer(num_layers=num_layers,
                            n_actions=n_actions,
                            state_space=state_space_dim,
                            max_length=K,
                            reward_dim=1,
                            dim_models=dim_models,
                            K=K,
                            num_heads=num_heads,
                            attention_mechanism=attention_mechanism,
                            dim_feedforward=dim_feedforward,
                            dropout=dropout,
                            seed=1967)

    optimizer = optim.Adam(dt.parameters(), lr=lr)
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    #training loop

    epochs = 1#5
    batch_size=32



    loss_graph = []

    mix_d = 2 if mix == 0.5 else 1
    for epoch in range(epochs):
        loss_1=[]
        with tqdm(range(int(len(data)/batch_size/mix_d))) as tp:
            for i in tp:
                    

                start = i*batch_size
                end = (i+1)*batch_size



                indexes = np.random.randint(0,len(data),size=(batch_size))

                batch = data.sample_indexes(indexes)


                s_h, a_h, r_h, a_chosen, done = batch

                s_h = list(map(lambda x: list(reversed(x)), s_h))
                a_h = list(map(lambda x: list(reversed(x)), a_h))
                r_h = list(map(lambda x: list(reversed(x)), r_h))

                a_h = list(map(lambda x: list(map(onehot, x)), a_h))
                a_pred = dt(r_h, s_h, a_h,0)

                #print(a_pred)

                optimizer.zero_grad()
                loss = criterion(a_pred, Tensor(a_chosen).long())
                loss.backward()
                
                #nn.utils.clip_grad_norm_(dt.parameters(), 100)
                optimizer.step()
                loss_1.append(loss.sum().item())
          
                if i % 10 == 0:
                    tp.set_postfix(batch = f"{i}/{int(len(data)/64)}", loss=np.mean(loss_1[-50:]),epoch=epoch)
            loss_graph.append(np.mean(loss_1))
    


    torch.save(dt,saveto)



print("loading one")
data = pickle.load(open('expert_data_buffer.pkl','rb'))
print("dine loading first, loading second")
d2 = pickle.load(open('expert_data_fully_observing_buffer.pkl','rb'))

d2.cut(len(data))
print('training modelsl now')
train(data,d2,0,'decision_transformer.pt')
train(data, d2, 0.5, 'decision_transformer_1_1.pt')
train(data, d2, 1, 'decision_transformer_0_1.pt')
