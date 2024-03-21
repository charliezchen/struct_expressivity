import torch
import numpy as np
from models import MLP_Classifier
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

# Config a teacher model
input_dim = 20
num_class = 2
n = 600
hiddens=[n, n]
teacher_model = MLP_Classifier(input_dim, num_class, hiddens)

# Generate data
sample_size = int(1e4)
# Randomly initialize from -1 to 1
data_x = torch.rand(sample_size, input_dim) * 2 - 1
with torch.no_grad():
    data_y = teacher_model(data_x)

dense_model = MLP_Classifier(input_dim, num_class, hiddens=[5])


# Define loss function and optimizer
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

        losses.append(loss.detach().numpy())

    # print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.10f}')
plt.loglog(losses)
plt.title("Losses vs Epoch")
plt.show()

