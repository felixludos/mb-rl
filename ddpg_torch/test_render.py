import gym
import numpy as np

env = gym.make('Pendulum-v0')
for e in range(10):
    print 'episode', e
    env.reset()
    for _ in range(1000):
        env.render()
        _,_,done,_ = env.step(env.action_space.sample()) # take a random action
        if done: break

