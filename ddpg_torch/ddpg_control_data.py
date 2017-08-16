
# coding: utf-8

# In[1]:

import sys
import gym
import copy
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import filter_env
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer # <- has setting to either get constant batches (most recent exp), or random batches
import tensorflow as tf
import math
import ddpg_tf # tf net
import gc
gc.enable()
np.core.arrayprint._line_width = 120

# Hyperparameters/Settings
SEED = 5

VERBOSE = True

ACTOR_BN = False
LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-3
L2 = 0.01
TAU = 0.001
BATCH_SIZE = 50

REPLAY_BUFFER_SIZE = 200
REPLAY_START_SIZE = 50
GAMMA = 0.99

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ignore
# functions for replacing torch params with tf params
param_translation = {#'actor.layer0_bn.bias':'batch_norm_0/beta:0',
                    #'actor.layer0_bn.weight':'batch_norm_0/gamma:0',
                    'actor.layer1.bias':'actorb1:0',
                    'actor.layer1.weight':'actorW1:0',
                    #'actor.layer1_bn.bias':'batch_norm_1/beta:0',
                    #'actor.layer1_bn.weight':'batch_norm_1/gamma:0',
                    'actor.layer2.bias':'actorb2:0',
                    'actor.layer2.weight':'actorW2:0',
                    #'actor.layer2_bn.bias':'batch_norm_2/beta:0',
                    #'actor.layer2_bn.weight':'batch_norm_2/gamma:0',
                    'actor.output_layer.bias':'actorb3:0',
                    'actor.output_layer.weight':'actorW3:0',
                    'critic.action_layer.weight':'critic_W2a:0',
                    'critic.layer1.bias':'critic_b1:0',
                    'critic.layer1.weight':'critic_W1:0',
                    'critic.layer2.bias':'critic_b2:0',
                    'critic.layer2.weight':'critic_W2:0',
                    'critic.output_layer.bias':'critic_b3:0',
                    'critic.output_layer.weight':'critic_W3:0'}

def replaceNetParams(tf_net, torch_net, torch_target_net=None):
    
    #torch_vars = dict(torch_net.state_dict())
    
    tf_vars = dict([(str(v.name),v) for v in tf_net.all_vars])
    for name, param in torch_net.named_parameters():
        param.data.copy_(torch.Tensor(tf_vars[param_translation[str(name)]].eval().T).float())
        
    if torch_target_net is not None:
        for name, param in torch_target_net.named_parameters():
            param.data.copy_(torch.Tensor(tf_vars[param_translation[str(name)][:-2]+'/ExponentialMovingAverage:0'].eval().T).float())

# torch net definitions
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNet, self).__init__()

        self.actor = ActorNet(state_dim, action_dim)
        self.critic = CriticNet(state_dim, action_dim)

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state, action)
        return value

    def getAction(self, state):
        return self.actor(state)

    def getValue(self, state, action=None):
        if action is None:
            return self.critic(state, self.actor(state))
        return self.critic(state, action)

    #def train(self): # might not be necessary
    #    self.critic.train()
    #    self.actor.train()
    
    #def eval(self): # might not be necessary
    #    self.critic.eval()
    #    self.actor.eval()

class CriticNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()

        # make sure all params are initizialized randomly [-1/np.sqrt(dim),1/np.sqrt(dim)]
        self.layer1 = nn.Linear(state_dim,LAYER1_SIZE)
        self.action_layer = nn.Linear(action_dim,LAYER2_SIZE,bias=False)
        self.layer2 = nn.Linear(LAYER1_SIZE,LAYER2_SIZE)
        self.output_layer = nn.Linear(LAYER2_SIZE,1)

    def forward(self, state, action):
        x = F.relu(self.layer1(state))
        x = F.relu(self.action_layer(action) + self.layer2(x))
        q = self.output_layer(x)
        return q # predicted q value of this state-action pair

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()

        # make sure all params are initizialized randomly [-1/np.sqrt(dim),1/np.sqrt(dim)]
        if ACTOR_BN: self.layer0_bn = nn.BatchNorm1d(state_dim,affine=False)
        self.layer1 = nn.Linear(state_dim,LAYER1_SIZE)
        if ACTOR_BN: self.layer1_bn = nn.BatchNorm1d(LAYER1_SIZE,affine=False)
        self.layer2 = nn.Linear(LAYER1_SIZE,LAYER2_SIZE)
        if ACTOR_BN: self.layer2_bn = nn.BatchNorm1d(LAYER2_SIZE,affine=False)
        self.output_layer = nn.Linear(LAYER2_SIZE,action_dim)

    def forward(self, state):
        if ACTOR_BN: state = F.relu(self.layer0_bn(state))
        x = self.layer1(state)
        if ACTOR_BN: x = self.layer1_bn(x)
        x = F.relu(x)
        x = self.layer2(x)
        if ACTOR_BN: x = self.layer2_bn(x)
        x = F.relu(x)
        action = F.tanh(self.output_layer(x))
        return action # predicted best actions

def calc_error(a,b):
    return np.sqrt(np.sum((a-b)**2))

def main(args):
    if VERBOSE: print '***The Replay Buffer currently always returns the most recent experiences (instead of random), so the batches are constant between the tf and torch nets.'
    
    state_dim = 3
    action_dim =  1

    net = ActorCriticNet(state_dim,action_dim)

    target_net = copy.deepcopy(net)
    memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
    noise = OUNoise(action_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=L2)
    target_optim = optim.Optimizer(target_net.parameters(), {}) # to iterate over target params

    if VERBOSE: print '***Making gym env (only used to setup TF net).'
    
    # load tf net (restoring saved parameters)
    dtf = ddpg_tf.DDPG_TF(filter_env.makeFilteredEnv(gym.make('Pendulum-v0')),loadfilename='tf_params-0',printVars=False)

    
    if VERBOSE: print '***TF net restore complete.'

    # load control data (only using a every fourth data), and tf net results
    control_states = np.load('control_states.npy')[::4]
    control_rewards = np.load('control_rewards.npy')[::4]
    tf_record = np.load('tf_control_record.npy')
    
    # replace torch params with tf params, and run control data, collecting torch net results
    # first optimization step will occur at i == 50, upon which extra data is recorded to compare tf and torch
    # using: no bn, REPLAY_BUFFER_SIZE=200, REPLAY_START_SIZE=50, BATCH_SIZE=50, constant replay_buffer_batches (always the most recent experiences)
    replaceNetParams(dtf, net, target_net)

    if VERBOSE: print '***Torch net params initialized to TF net params.'

    original_net = copy.deepcopy(net) # save original net
    original_target_net = copy.deepcopy(target_net)

    torch_record = []

    loss = -1
    first_step = True

    for i in xrange(len(control_rewards)-1):
        state = torch.from_numpy(control_states[i].reshape(1,state_dim)).float()
        action = net.getAction(Variable(state)).data
        target_action = target_net.getAction(Variable(state)).data

        reward = torch.FloatTensor([[control_rewards[i]]]).float()

        new_state = torch.from_numpy(control_states[i+1].reshape(1,state_dim)).float()

        memory.add(state,action,reward,new_state,True)
        if memory.count() > REPLAY_START_SIZE:
            minibatch = memory.get_batch(BATCH_SIZE)
            state_batch = torch.cat([data[0] for data in minibatch],dim=0)
            action_batch = torch.cat([data[1] for data in minibatch],dim=0)
            reward_batch = torch.cat([data[2] for data in minibatch])
            next_state_batch = torch.cat([data[3] for data in minibatch],dim=0)
            done_batch = Tensor([data[4] for data in minibatch])

            # calculate y_batch from targets
            #next_action_batch = target_net.getAction(Variable(next_state_batch))
            value_batch = target_net.getValue(Variable(next_state_batch)).data
            y_batch = reward_batch + GAMMA * value_batch * done_batch

            if first_step:
                if VERBOSE: print '***First Optimization Step complete.'
                torch_ys = y_batch
                torch_batch = minibatch
                torch_outs = net.getValue(Variable(state_batch)).data

            # optimize net 1 step
            loss = criterion(net.getValue(Variable(state_batch)), Variable(y_batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.data[0]

            # update targets - using exponential moving averages
            for group, target_group in zip(optimizer.param_groups, target_optim.param_groups):
                for param, target_param in zip(group['params'], target_group['params']):
                    target_param.data.mul_(1 - TAU)
                    target_param.data.add_(TAU, param.data)

            if first_step:
                first_step_net = copy.deepcopy(net)
                first_step_target_net = copy.deepcopy(target_net)
                first_step = False

        torch_record.append([action.numpy()[0][0], target_action.numpy()[0][0],loss])
        loss = -1

    torch_record = np.array(torch_record)
    torch_outs = torch_outs.numpy().T[0]
    torch_ys = torch_ys.numpy().T[0]

    if VERBOSE: print '***Control Data run complete.'

    # compare torch and tf results
    # results for each net have 3 columns: [net action prediction, target net action prediction, loss (-1 if there was no training)]
    sel = np.arange(45,55)
    #print calc_error(tf_record[sel,:], torch_record[sel,:])
    print 'Result comparison:'
    print 'control_data_index | tf_net_action | tf_target_net_action | tf_loss | torch_net_action | torch_target_net_action | torch_loss'
    print np.hstack([sel[:,np.newaxis],tf_record[sel,:], torch_record[sel,:]])
    print '\t(a loss of -1 means no training occured in that step)'


    # load all tf results from before taking first optimization step
    tf_ys = np.load('tf_first_step_y_batch.npy')
    tf_rs = np.load('tf_first_step_reward_batch.npy')
    tf_ds = np.load('tf_first_step_done_batch.npy')
    tf_vs = np.load('tf_first_step_value_batch.npy')
    tf_outs = np.load('tf_first_step_output_values.npy')
    torch_wd = 1.36607 # weight decay loss of tf net at first optimization step - recorded directly from terminal output of tf net

    if VERBOSE:
        print '***Comparing first step stats'

        # compare tf and torch data from before taking first optimization step
        # including calculation of manual loss
        print '\terror in ys (between tf and torch)', calc_error(torch_ys, tf_ys)
        print '\terror in predictions (between tf and torch)', calc_error(torch_outs, tf_outs)
        print '\ttorch loss (manually calculated)', np.mean((torch_ys - torch_outs)**2)
        print '\ttf loss (manually calculated)', np.mean((tf_ys - tf_outs)**2)
        print '\ttorch loss', torch_record[50,2], '(not including weight decay)'
        print '\ttf loss', tf_record[50,2] - torch_wd, '(not including weight decay)'

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))

