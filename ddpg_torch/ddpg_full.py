
import sys
import gym
import copy
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
from replay_buffer import ReplayBuffer
import gc
gc.enable()

# Hyperparameters/Settings
ENV_NAME = 'Pendulum-v0' #'InvertedPendulum-v1'
EPISODES = 100000
TEST = 10

LAYER1_SIZE = 400
LAYER2_SIZE = 300
LEARNING_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64

REPLAY_BUFFER_SIZE = 1000000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99

def main(argv):
	env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
	state_dim = env.observation_space.shape[0]
	action_dim =  env.action_space.shape[0]

	net = ActorCriticNet(state_dim,action_dim)
	target_net = copy.deepcopy(net)
	memory = ReplayBuffer(REPLAY_BUFFER_SIZE)
	noise = OUNoise(action_dim)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
	target_optim = optim.Optimizer(target_net.parameters(), {}) # to iterate over target params

	for episode in xrange(EPISODES):

		print 'Episode', episode,

		net.train()
		target_net.train() # not really necessary?

		state = Variable(torch.from_numpy(env.reset().reshape(1,state_dim)))
		noise.reset()

		# Train
		for step in xrange(env.spec.timestep_limit):

			# Take noisy action - for exploration
			action = net.getAction(state)[0] + noise.noise()
			new_state, reward, done, _ = env.step(action.data)

			memory.add(state,action,reward,new_state,done)
			if memory.count() > REPLAY_START_SIZE:
				minibatch = memory.get_batch(BATCH_SIZE)
				state_batch = np.asarray([data[0] for data in minibatch])
				action_batch = np.asarray([data[1] for data in minibatch])
				reward_batch = np.asarray([data[2] for data in minibatch])
				next_state_batch = np.asarray([data[3] for data in minibatch])
				done_batch = np.asarray([data[4] for data in minibatch])

				# resize action_batch (?)
				action_batch = np.resize(action_batch,[BATCH_SIZE,action_dim])

				# calculate y_batch - using targets
				next_action_batch, value_batch = zip(*[target_net(next_state) for next_state in next_state_batch])
				y_batch = [reward + GAMMA * value if not done else reward
							for reward, value, done in zip(reward_batch,value_batch,done_batch)]
				y_batch = np.resize(y_batch,[BATCH_SIZE,1])

				# optimize net 1 step
				loss = criterion(net(state_batch), y_batch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				# update targets - using exponential moving averages
				for group, target_group in zip(optimizer.param_groups, target_optim.param_groups):
					for param, target_param in zip(group['params'], target_group['params']):
						target_param.data.mul_(1 - TAU)
						target_param.data.add_(TAU, param.data)

			state = Variable(torch.from_numpy(new_state.reshape(1,state_dim)))
			if done: break
		
		print '- training complete'

		# Test
		if episode % 100 == 0 and episode > 100:
			net.eval() # set to eval - important for batch normalization
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = net.getAction(state) # direct action for test
					state, reward, done, _ = env.step(action)
					total_reward += reward
					if done: break
			ave_reward = total_reward / TEST
			print '\tTesting: Evaluation Average Reward:', ave_reward

class ActorCriticNet(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(ActorCriticNet, self).__init__()

		self.actor = ActorNet(state_dim, action_dim)
		self.critic = CriticNet(state_dim, action_dim)

	def forward(self, state):
		action = self.self.actor(state)
		value = self.critic(state, action)
		return value, action

	def getAction(self, state):
		return self.actor(state)

	def getValue(self, state, action=None):
		if action is None:
			return self.critic(state, self.actor(state))
		return self.critic(state, action)

	def train(self): # might not be necessary
		self.critic.train()
		self.actor.train()
		super(ActorCriticNet, self).train()
	
	def eval(self): # might not be necessary
		self.critic.eval()
		self.actor.eval()
		super(ActorCriticNet, self).eval()

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
		self.layer0_bn = nn.BatchNorm1d(state_dim)
		self.layer1 = nn.Linear(state_dim,LAYER1_SIZE)
		self.layer1_bn = nn.BatchNorm1d(LAYER1_SIZE)
		self.layer2 = nn.Linear(LAYER1_SIZE,LAYER2_SIZE)
		self.layer2_bn = nn.BatchNorm1d(LAYER2_SIZE)
		self.output_layer = nn.Linear(LAYER2_SIZE,action_dim)

	def forward(self, state):
		x = F.relu(self.layer0_bn(state))
		x = self.layer1(x)
		x = F.relu(self.layer1_bn(x))
		x = self.layer2(x)
		x = F.relu(self.layer2_bn(x))
		action = F.tanh(self.output_layer(x))
		return action # predicted best actions



if __name__ == "__main__":
	sys.exit(main(sys.argv))

