import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, in_size, out_size): 
        super(NN, self).__init__()
        self.layer1 = nn.Linear(in_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, out_size)
        self.dropout = nn.Dropout(0.7)
  
    def forward(self, x): 
        x = F.relu(self.layer1(x))
        x = self.dropout(F.relu(self.layer2(x)))
        x = F.relu(self.layer3(x)) 
        return x
