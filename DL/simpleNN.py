import torch
import torch.nn as nn
import torch.nn.functional as F



class simpleNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(simpleNN,self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        print(out)
        return out
    

simple =simpleNN(10,20,2)
print(simple)
simple.forward(torch.randn(10))