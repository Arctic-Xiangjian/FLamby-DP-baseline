import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 3)
        self.linear2 = torch.nn.Linear(3, output_dim)
    def forward(self, x):
        x=self.linear(x)
        x=F.leaky_relu(x)
        x=self.linear2(x)
        return torch.sigmoid(x)
