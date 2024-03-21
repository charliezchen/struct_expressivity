from graph_tool import *

import sys
sys.path.append('./splinecam')

import splinecam as splinecam
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch

import time
import copy
import tqdm

#@title training
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from models import MLP_Classifier, MLP_Regressor
from utils import plot_domain_for_model

device='cuda'

task = "regression"

input_dim = 2
num_class = 10
n = 100
hiddens=[n, n]
if task == "regression":
    teacher_model = MLP_Regressor(input_dim, hiddens)
else:
    teacher_model = MLP_Classifier(input_dim, num_class, hiddens)

teacher_model = teacher_model.to(device)
teacher_model.eval()

plot_domain_for_model(teacher_model, f"figures/{task}", f"teacher")

# Generate data
sample_size = int(1e4)
# Randomly initialize from -1 to 1
data_x = torch.rand(sample_size, input_dim) * 2 - 1
data_x = data_x.to(device)

with torch.no_grad():
    data_y = teacher_model(data_x)

def student_learn(student_n):
    hiddens = [student_n, student_n]
    if task == "regression":
        dense_model = MLP_Regressor(input_dim, hiddens=hiddens).to(device)
        criterion = torch.nn.MSELoss(reduce='mean')
    else:
        dense_model = MLP_Classifier(input_dim, num_class, hiddens=hiddens).to(device)
        criterion = torch.nn.KLDivLoss(reduce='batchmean')
    optimizer = torch.optim.Adam(dense_model.parameters(), lr=1e-5)

    # Training parameters
    epochs = int(5e3)
    batch_size = 128

    losses = []
    # Training loop
    for epoch in tqdm(range(epochs)):
        permutation = torch.randperm(data_x.size()[0])

        for i in range(0, data_x.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = data_x[indices], data_y[indices]

            # Forward pass
            outputs = dense_model(batch_x)
            loss = criterion(F.log_softmax(outputs, dim=-1), F.softmax(batch_y, dim=-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach().cpu().numpy())

        # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}')
    loss = loss.detach().cpu().numpy()
    dense_model.eval()
    plot_domain_for_model(dense_model, f"figures/{task}", f"student_n_{student_n}_loss_{loss:.12f}")

for student_n in range(10, 110, 10):
    student_learn(student_n)

