import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import Dataset
from model import MLP, Flatten
from data_utils import *


def run_epoch(generator, epoch_size, optimizer=None):
    losses = []
    batch_count = 0
    for x_batch, y_batch in generator:
        logits = mlp(x_batch)
        loss = loss_function(logits, y_batch)
        print(f"loss: {loss}")
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.cpu().detach().numpy())
        batch_count += 1
        if batch_count >= epoch_size:
            break
    return np.array(losses).mean()


def score(generator, model):
    accuracy_count = 0
    for x_batch, y_batch in generator:
        probabilities = model(x_batch)
        indices = torch.argmax(probabilities, dim=1)
        pred = probabilities.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        accuracy_count += pred.eq(y_batch.view_as(pred)).sum().item()
    return float(accuracy_count) / len(generator.dataset)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else None)

data_params = {
    'batch_size': 64,
    'shuffle': True,
    'device': device}

print('Loading data...')
train_generator, val_generator, test_generator = load_data('data/mnist_full_set.csv', **data_params)
print(f"Train: {len(train_generator.dataset)} Validation: {len(val_generator.dataset)} Test: {len(test_generator.dataset)}")

num_epochs = 10
train_epoch_size = len(train_generator.dataset)
validate_epoch_size = len(val_generator.dataset)
learning_rate = .01
hidden_channels = 32

# basic_layers = nn.ModuleList([
#         nn.Conv2d(1, hidden_channels, 5),
#         # nn.BatchNorm2d(hidden_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(hidden_channels, hidden_channels, 5),
#         # nn.BatchNorm2d(hidden_channels),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(hidden_channels, hidden_channels, 3),
#         # nn.BatchNorm2d(hidden_channels),
#         nn.ReLU(),
#         # nn.MaxPool2d(2),
#         Flatten(),
#         nn.Linear(4 * hidden_channels, 512),
#         nn.ReLU(),
#         # nn.Linear(hidden_channels, 10),
#         nn.Linear(512, 10),
#     ])

basic_layers = nn.ModuleList([
    nn.Conv2d(1, 20, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20, 50, 5, 1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    Flatten(),
    nn.Linear(4*4*50, 500),
    nn.ReLU(),
    nn.Linear(500, 10),
    nn.LogSoftmax(dim=1)
])

mlp = MLP(basic_layers, device=device)

loss_function = nn.modules.loss.NLLLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate, momentum=.5)
# optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

train_accuracies = []
val_accuracies = []
best_val_accuracy = 0
print("Training model...")
print(f"lr={learning_rate}, batch_size={data_params['batch_size']}, n_epochs={num_epochs},"
      f" loss_function={loss_function}, opt={optimizer}")
print(f"model: {mlp}")
for epoch in range(num_epochs):
    print(f'Epoch {epoch}:')
    loss = run_epoch(train_generator, train_epoch_size, optimizer)
    # if epoch % 5 == 0:
    train_accuracies.append(score(train_generator, mlp))
    val_accuracies.append(score(val_generator, mlp))
    print(f'Train accuracy: {train_accuracies[-1]}')
    print(f'Validation accuracy: {val_accuracies[-1]}')
    if val_accuracies[-1] > best_val_accuracy:
        torch.save(mlp, 'weights/mlp_best_weights.pth')
mlp = torch.load('weights/mlp_best_weights.pth')

plt.title(f"lr={learning_rate}, hidden={hidden_channels}, loss={loss_function}")
plt.plot(train_accuracies, '-r', label='train')
plt.plot(val_accuracies, '-b', label='validation')
plt.legend(loc='lower right')
plt.show()

print("Testing model...")
print(f"Test accuracy: {score(test_generator, mlp)}")
