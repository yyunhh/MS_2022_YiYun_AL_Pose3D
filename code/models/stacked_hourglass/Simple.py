import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3D favor

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 3)
        self.fc2 = nn.Linear(51, 32) #(51, 32) #48: 16*3
        self.fc3 = nn.Linear(32, 16)
        self.mu = nn.Linear(16, 3)
        self.log_var = nn.Linear(16, 3)
    
    def forward(self, x, c):
        output = torch.cat((x,c),2)
        output = torch.split(output, 1)
        output = tuple(map(lambda out: F.relu(self.fc1(out)), output))
        output = torch.flatten(torch.stack(output), start_dim=1)
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        mu, log_var = self.mu(output), self.log_var(output)

        return mu, log_var 

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        #input的shape->[batch_size, 35]
        
        self.fc1 = nn.Linear(37, 42) # latent dimension + 34 = input dimension 
        self.fc2 = nn.Linear(42, 46)
        self.fc3 = nn.Linear(46, 51) #(46, 51)
    
    def forward(self, x, c):
        output = torch.cat((x,c),1)
        
        #print(output.shape) # (12, 35)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        
        return output