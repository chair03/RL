import gym
from qagent import QAgent
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('FrozenLake-v1',is_slippery=False)
env.seed = 42
num_games = 500000
env.reset()
rewards = []
win_pers = []
agent = QAgent(env.observation_space.n,env.action_space.n)

for i in range(1,num_games+1):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.chooseAction(state)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        agent.updateQtable(state,action,reward,observation,done)
        state = observation
        if done:
            rewards.append(total_reward)
            
    if i%100 == 0:
        mean_reward = np.mean(rewards[-100:])
        win_pers.append(mean_reward)
        
    

plt.plot(win_pers)
plt.show()
