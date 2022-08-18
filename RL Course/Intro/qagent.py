import random

class QAgent:
    def __init__(self,num_states,num_actions) -> None:
        self.q = {}
        self.gamma = 0.9
        q_init = 0
        self.learning_rate = 0.01
        self.epsilon = 0.99
        self.actions = range(num_actions)
        for state in range(num_states):
            action_values = {}
            for action in range(num_actions):
                action_values[action] = q_init
            self.q[state] = action_values
        
    def updateQtable(self,old_state,action,reward,new_state,done):
        if done:
            self.q[old_state][action] += self.learning_rate*(reward-self.q[old_state][action])
        else:
            self.q[old_state][action] += self.learning_rate*(reward+self.gamma*max(self.q[new_state])-self.q[old_state][action])
        self.updateEpsilon()
    
    def updateEpsilon(self): 
        self.epsilon *= 0.999999
        if self.epsilon < 0.01:
            self.epsilon = 0.01 
    
    def chooseAction(self,state):
        
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q[state],key=self.q[state].get)


