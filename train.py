import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import Dataset
from model import MLP
from data_utils import *


def run_epoch(generator, epoch_size, optimizer=None):
    losses = []
    batch_count = 0
    for x_batch, y_batch in generator:
        logits = mlp(x_batch)
        loss = loss_function(logits, y_batch)
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
        probabilities = model(x_batch, return_logits=False)
        indices = torch.multinomial(probabilities, 1)
        accuracy_count += (y_batch.view(-1) == indices.view(-1)).sum()
    return float(accuracy_count) / len(generator.dataset)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else None)
# device = None

data_params = {
    'batch_size': 128,
    'shuffle': True,
    'device': device}

print('Loading data...')
# train_generator, val_generator, test_generator = load_data('data/train_tasks.csv', **data_params)
train_generator, val_generator, test_generator = get_task('data/train_tasks.csv', **data_params)
print(f"Train: {len(train_generator.dataset)} Validation: {len(val_generator.dataset)} Test: {len(test_generator.dataset)}")

num_epochs = 60
train_epoch_size = len(train_generator.dataset)
validate_epoch_size = len(val_generator.dataset)
learning_rate = .0001

mlp = MLP(
    input_dimension=train_generator.dataset.instances.shape[1],
    output_dimension=train_generator.dataset.labels.unique().shape[0],
    hidden_dimension=512,
    num_layers=2,
    device=device)

loss_function = nn.modules.loss.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

train_accuracies = []
val_accuracies = []
best_val_accuracy = 0
print("Training model...")
for epoch in range(num_epochs):
    run_epoch(train_generator, train_epoch_size, optimizer)
    if epoch % 5 == 0:
        train_accuracies.append(score(train_generator, mlp))
        val_accuracies.append(score(val_generator, mlp))
        print(f'Epoch {epoch}:')
        print(f'Train accuracy: {train_accuracies[-1]}')
        print(f'Validation accuracy: {val_accuracies[-1]}')
        if val_accuracies[-1] > best_val_accuracy:
            torch.save(mlp, 'weights/mlp_best_weights.pth')
mlp = torch.load('weights/mlp_best_weights.pth')

plt.title('Losses')
plt.plot(train_accuracies, '-r', label='train')
plt.plot(val_accuracies, '-b', label='validation')
plt.legend(loc='lower right')
plt.savefig('accuracies.png')
plt.show()

print("Testing model...")
print(f"Test accuracy: {score(test_generator, mlp)}")
