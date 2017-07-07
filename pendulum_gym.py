
import numpy as np
import gym

env = gym.make('Pendulum-v0')
env.reset()
for _ in range(1000):
    env.render()
    print env.action_space.high
    env.step(env.action_space.high)#np.array([0])) # take a random action


