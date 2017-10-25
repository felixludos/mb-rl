# -*- coding: utf-8 -*-

import sys
#%pdb
import gym
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from collections import namedtuple
import gc
gc.enable()
#np.core.arrayprint._line_width = 120

EPISODES = 100
PRINT_STEP = 10

LAYER1_SIZE = 50
LAYER2_SIZE = 50
LEARNING_RATE = 1e-3

BUFFER_SIZE = 10000
BATCH_SIZE = 64

class TransitionModel(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(TransitionModel, self).__init__()

        # make sure all params are initizialized randomly [-1/np.sqrt(dim),1/np.sqrt(dim)]
        self.layer1 = nn.Linear(state_dim,LAYER1_SIZE)
        self.action_layer = nn.Linear(action_dim,LAYER2_SIZE,bias=False)
        self.layer2 = nn.Linear(LAYER1_SIZE,LAYER2_SIZE)
        self.output_layer = nn.Linear(LAYER2_SIZE,state_dim)

    def forward(self, state, action):
        x = F.relu(self.layer1(state))
        x = F.relu(self.action_layer(action) + self.layer2(x))
        ns = self.output_layer(x)
        return ns # predicted q value of this state-action pair

total_epochs = 0

def collect(max_samples=BUFFER_SIZE):
    global memory
    
    env = gym.make('Pendulum-v0')
    
    memory = []
    
    
    while len(memory) < max_samples:
        
        state = env.reset()
        state = torch.from_numpy(state).float().view(1,-1)
        
        for step in range(env.spec.timestep_limit):
            
            action = env.action_space.sample()
            
            new_state, _, done, _ = env.step(action)
            
            action = torch.from_numpy(action).float().view(1,-1)
            new_state = torch.from_numpy(new_state).float().view(1,-1)
            
            memory.append((state, action, new_state))
            
            if done:
                break
    
    memory = np.array(memory)

def reset():
    global model, criterion, optimizer
    model = TransitionModel(env.observation_space.shape[0], env.action_space.shape[0])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

def train(epochs=1000):
    global env, model, memory, criterion, optimizer, total_epochs
    
    for _ in range(epochs):
        
        total_loss = 0
        
        batches = np.array_split(memory, len(memory) / BATCH_SIZE)
        
        for minibatch in batches:
            
            state_batch = torch.cat([data[0] for data in minibatch],dim=0)
            action_batch = torch.cat([data[1] for data in minibatch],dim=0)
            next_state_batch = torch.cat([data[2] for data in minibatch],dim=0)
            
            loss = criterion(model(Variable(state_batch),Variable(action_batch)), 
                             Variable(next_state_batch))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if total_epochs and total_epochs % PRINT_STEP == 0:
                total_loss += loss.data[0]
            
        if total_epochs and total_epochs % PRINT_STEP == 0:
            print 'Epoch', total_epochs, 'Loss', total_loss / len(batches)
            
        total_epochs += 1


reset()
collect()
train()
















