import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The teacher-student scaling with dataset size (figure S2) was performed with fully-connected teacher and student networks with two hidden layers and widths 96 and 192, respectively, using PyTorch [65]. The inputs were random vectors sampled uniformly from a hypercube of dimension d = 2, 3, · · · , 9.
"""

#@title MLP Classifier
class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, num_class, hiddens=[96, 192]):
        super(MLP_Classifier, self).__init__()
        assert len(hiddens)==2, "Exactly two hidden layers are required"

        self.layer1 = nn.Linear(input_dim, hiddens[0])
        self.layer2 = nn.Linear(hiddens[0], hiddens[1])
        self.layer3 = nn.Linear(hiddens[1], num_class)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return (self.layer3(x))


class MLP_Regressor(nn.Module):
    def __init__(self, input_dim, hiddens=[96, 192]):
        super(MLP_Regressor, self).__init__()
        assert len(hiddens)==2, "Exactly two hidden layers are required"

        self.layer1 = nn.Linear(input_dim, hiddens[0])
        self.layer2 = nn.Linear(hiddens[0], hiddens[1])
        self.layer3 = nn.Linear(hiddens[1], 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.tanh(self.layer3(x))

