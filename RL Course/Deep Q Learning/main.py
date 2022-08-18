from agent import Agent
from plotting_utils import plot_learning_curve
import gym
import numpy as np


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000

    scores = []
    eps_history = []
    

    agent = Agent(num_states=env.observation_space.shape,num_actions=env.action_space.n)
    
    for i in range(n_games):
        score = 0
        done = False
        state = env.reset()

        while not done:
            action = agent.chooseAction(state = state)
            new_state,reward,done,info = env.step(action)
            score += reward
            agent.updateNetwork(state,action,reward,new_state)
            state = new_state

        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 100 == 0:
            mean_score = np.mean(scores[-100:])
            print(f'episode {i}: score {mean_score:.1f} ')

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x,scores,eps_history,filename)