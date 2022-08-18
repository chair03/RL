import torch.nn as nn
import torch.optim as optim
import torch as T
import torch.functional as F

class Network(nn.Module):
    def __init__(self,learning_rate:float,num_actions:int,input_dims:int) -> None:
        super(Network,self).__init__()

        self.fc1 = nn.Linear(*input_dims,128)
        self.fc2 = nn.Linear(128,num_actions)

        self.optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self,data):
        layer1 = nn.functional.relu(self.fc1(data))
        layer2 = self.fc2(layer1)
        return layer2
    '''
    def learn(self,data,q_value):
        self.optimizer.zero_grad()
        data = T.tensor(data).to(self.device)
        q_value = T.tensor(q_value).to(self.device)

        predictions = self.forward(data)

        cost = self.loss(predictions,q_value)

        cost.backward()
        self.optimizer.step()
    '''