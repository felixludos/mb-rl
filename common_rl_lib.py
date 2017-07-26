import sys
import numpy as np
get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
sys.path.append('p3')
import gridworld as gw

"""Useful Dictionary variants for table lookup algorithms, and RL"""
class SafeDict(dict):
    def __init__(self, default=0.):
        self.default = default 
        super(SafeDict, self).__init__()
    def __getitem__(self, idx):
        if idx not in self:
            return self.default
        return self.setdefault(idx, self.default)

class Policy:
    def __init__(self, decision=lambda x: None, default=None): # default chooses from action space if decision returns invalid action
        self.decision = decision # lambda with state as input
        self.default = default # lambda with action_space as input
        if self.default is None:
            self.default = lambda actions: np.random.choice(actions)
    def decide(self, state, action_space): # ideally action space is a set of possible actions
        action = self.decision(state) if state is not None else None
        if action not in action_space: # invalid decision
            return self.default(action_space)
        return action
    
class Const_Policy(dict): # policy where action space is independent of state
    def __init__(self, action_space, default=None): # works if action_space is constant
        self.default = default # must be a lambda that takes no args - to randomize action each time (only when action space is independent of state)
        if self.default is None:
            self.default = lambda: np.random.choice(action_space)
        super(Policy, self).__init__()
    def __getitem__(self, idx):
        if idx not in self:
            return self.default()
        return self.__getitem__(self, idx)

"""Small MDP Generators with discrete state and action spaces
With either deterministic or stochastic actions"""
# stochastic generator doesn't work yet
def generate_stochastic_dynamics(num_states=5, num_terminal=1, sparsity=0.5, reward_mag=2): # returns M(s,a,s') and R(s,a) and T(s) (state is terminal)
    dynamics = np.random.rand(num_states, num_states, num_states)
    rewards = np.random.randn(num_states, num_states) * reward_mag
    remove = np.random.choice(np.arange(0,dynamics.size,1,dtype=int), size=int(sparsity*dynamics.size),replace=False)
    for cell in remove:
        dynamics[cell/num_states**2,(cell/num_states)%num_states,cell%num_states] = 0
    terminal = np.zeros(num_states, dtype=bool)
    # l1 normalization of probabilities
    for i in range(num_states):
        for j in range(num_states):
            total = np.sum(dynamics[i,j,:])
            dynamics[i,j,:] /= total if total > 0 else 1
        terminal[i] = not np.sum(dynamics[i,:,:])
    
    # rewards
    rewards = np.random.rand()
    
    return dynamics

def generate_deterministic_dynamics(num_states=5, num_terminal=1, sparsity=0.5, reward_mag=2): # returns M(s,a)=s' and R(s,a) and list of terminal states
    dynamics = np.random.randint(num_states, size=(num_states, num_states))
    rewards = np.random.randn(num_states, num_states) * reward_mag
    remove = np.random.choice(np.arange(0,dynamics.size,dtype=int), size=int(sparsity*dynamics.size),replace=False)
    for cell in remove:
        dynamics[cell/num_states,cell%num_states] = -1 # means this transition is impossible
        rewards[cell/num_states,cell%num_states] = 0
    terminal = []
    # l1 normalization of probabilities
    for i in range(num_states):
        if np.sum(dynamics[i,:]) == -num_states: # terminal state
            terminal.append(i)
    if num_terminal is None:
        return dynamics, rewards, terminal # don't worry about number of terminal states
    # fixing the number of terminal states could change the density of transitions
    while len(terminal) > num_terminal: # too many terminal states
        s = terminal.pop()
        a = np.random.randint(num_states)
        dynamics[s,] = np.random.randint(num_states)
        rewards[s,cell%num_states] = 0
    while len(terminal) < num_terminal:
        s = np.random.randint(num_states)
        dynamics[s,:] = -np.ones(num_states)
        rewards[s,:] = np.zeros(num_states)
        terminal.append(s)
    return dynamics, rewards, terminal

class MDP_det:
    def __init__(self, num_states=5, num_terminal=1, sparsity=0.5):
        self.dynamics, self.rewards, self.terminal = generate_deterministic_dynamics(num_states, num_terminal, sparsity)
        self.num_states = num_states
        self.deterministic = True
        
    def reset(self):
        self.state = np.random.randint(self.num_states)
    
    def action_space(self, state=None):
        if state is None:
            state = self.state
        return np.arange(self.num_states)[self.dynamics[state,:]>=0]
    
    def step(self, action):
        reward = self.rewards[self.state, action]
        self.state = self.dynamics[self.state, action]
        if self.state == -1:
            raise Exception('Invalid state')
        return self.state, reward, self.state in self.terminal
    
    def isTerminal(self, state):
        return state in self.terminal
    
    def state_space(self):
        return np.arange(self.num_states)
    
    def getReward(self, action, newstate=None, state=None):
        if state is None:
            state = self.state
        return self.rewards[state, action]
    
    def getTransitions(self, action, state=None):
        if state is None:
            state = self.state
        return [(self.dynamics[state, action], 1.0)] # in a det MDP there is only 1 possible next state

"""Wrapper for Gridworld MDPs"""
gridWorlds = {'cliff':gw.getCliffGrid, 'cliff2':gw.getCliffGrid2, 'discount':gw.getDiscountGrid, 'bridge':gw.getBridgeGrid, 'book':gw.getBookGrid, 'maze':gw.getMazeGrid}
class GridMDP:
    def __init__(self, gridName=None, noise=0.2):
        if gridName is None:
            gridName = np.random.choice(gridWorlds.keys())
        self.gridName = gridName
        self.gridworld = gridWorlds[gridName]()
        self.gridworld.setNoise(noise)
        self.reset()
    
    def reset(self):
        self.state = self.gridworld.getStartState()
    
    def action_space(self, state=None):
        if state is None:
            state = self.state
        #print 's', state, self.gridworld.getPossibleActions(state)
        return self.gridworld.getPossibleActions(state)
    
    def sample_probs(self, probs): # returns next state
        sample = np.random.random()
        state, cumulative = probs[0]
        i = 0
        while sample > cumulative:
            i += 1
            cumulative += probs[i][1]
            state = probs[i][0]
        return state
    
    def step(self, action):
        newstate = self.sample_probs(self.gridworld.getTransitionStatesAndProbs(self.state, action))
        reward = self.gridworld.getReward(self.state, action, newstate)
        self.state = newstate
        return self.state, reward, self.gridworld.isTerminal(self.state)
    
    def isTerminal(self, state):
        return self.gridworld.isTerminal(state)
    
    def state_space(self):
        return self.gridworld.getStates()
    
    def getReward(self, action, newstate, state=None):
        if state is None:
            state = self.state
        return self.gridworld.getReward(state, action, newstate)
    
    def getTransitions(self, action, state=None):
        if state is None:
            state = self.state
        #print 't', state, action, '{', self.action_space(state) ,'}',
        trans = self.gridworld.getTransitionStatesAndProbs(state, action)
        #print 'leads to', trans
        return trans

"""Discretize states and actions"""
def getDiscreteRange(low, high, graining):
    inc = (high-low)/(graining-1.)
    return np.arange(low,high+inc/2,inc)

graining = 19
possible_state_values = getDiscreteRange(-1., 1., graining)
possible_action_values = getDiscreteRange(-2, 2, graining)

def state_maker(observation):
    return tuple([possible_state_values[possible_state_values>=o][0] for o in observation[:-1]] + [int(observation[2]*graining)])
def action_maker():
    return possible_action_values
