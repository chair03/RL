import gym
from qagent import QAgent
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v1',is_slippery = False)
env.seed = 42
num_games = 1000
env.reset()
rewards = []
win_pers = []
agent = QAgent(env.env)

for _ in range(1,num_games+1):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.chooseAction(state)
        observation, reward, done, info = env.step(state)
        total_reward += reward
        agent.updateQtable(state,action,reward,observation,done)
        if done:
            rewards.append(total_reward)
            agent.updateEpsilon()
    
    if num_games%10:
        mean_reward = sum(rewards[-10:])/10
        win_pers.append(mean_reward)

plt.plot(win_pers)
plt.show()
