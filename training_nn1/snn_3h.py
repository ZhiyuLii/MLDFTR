import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SNN(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden=[4,4,4],  #set a default value if there is not a hidden variable
            lamda=1e-3,
            beta=1.5,
            use_cuda=False):
        super(SNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.lamda = lamda
        self.beta = beta
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden[0], bias=True),
            nn.Sigmoid(),
            nn.Linear(self.hidden[0], self.hidden[1], bias=True),
            nn.Sigmoid(),
            nn.Linear(self.hidden[1], self.hidden[2], bias=True),
            nn.Sigmoid(),
            nn.Linear(self.hidden[2], self.output_dim, bias=False),
        )
        
    def forward(self, ml_in, is_training_data=False):
        inputs = torch.zeros((ml_in.shape[0],3))
        inputs[:,0] = torch.log(torch.pow((ml_in[:,0] + ml_in[:,1] + 1e-15), 1/3))
        inputs[:,1] = torch.log(torch.div((ml_in[:,0] - ml_in[:,1]),(ml_in[:,1] + ml_in[:,0] + 1e-15)) + 1 + 1e-15)
        inputs[:,2] = torch.log(torch.div(torch.pow((ml_in[:,2] + ml_in[:,3] + 2*ml_in[:,4] + 1e-15), 0.5), torch.pow((ml_in[:,0] + ml_in[:,1] + 1e-15), 4/3)))
        
        y_pred = self.model(inputs)
        exc = y_pred 
        
        return exc
        
