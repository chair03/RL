import random


from sympy import Q
from network import Network
import torch as T

class Agent:
    def __init__(self,num_states:int,num_actions:int) -> None:
        self.gamma = 0.99
        self.min_eps = 0.01
        self.learning_rate = 0.0001
        self.dec_eps = 5e-6
        self.epsilon = 0.99
        self.actions = range(num_actions)
        self.q = Network(learning_rate=self.learning_rate,num_actions=num_actions,input_dims=num_states)
        
    def updateNetwork(self,old_state,action,reward,new_state):
        
        self.q.optimizer.zero_grad()
        old_state = T.tensor(old_state,dtype=T.float).to(self.q.device)
        action = T.tensor(action).to(self.q.device)
        reward = T.tensor(reward).to(self.q.device)
        new_state = T.tensor(new_state,dtype=T.float).to(self.q.device)
        
        q_pred = self.q.forward(old_state)[action]
        q_next = self.q.forward(new_state).max()

        q_target = reward + self.gamma*q_next

        loss = self.q.loss(q_target,q_pred).to(self.q.device)
        loss.backward()
        self.q.optimizer.step()
        self.updateEpsilon()
    
    def updateEpsilon(self): 
        self.epsilon -= self.dec_eps
        if self.epsilon < self.min_eps:
            self.epsilon = self.min_eps
    
    def chooseAction(self,state):
        
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            state = T.tensor(state,dtype=T.float).to(self.q.device)
            
            actions = self.q.forward(state)
            best_action = T.argmax(actions).item()
            return best_action

