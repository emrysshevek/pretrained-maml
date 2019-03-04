import os
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from dataset import Dataset
from model import MLP, Flatten
from data_utils import *


def sample_tasks(df, n_tasks, classes, num_instances, n_way, device=None):
    # TODO: is it ok if there is some overlap between train and test instances?
    tasks = []
    for x in range(n_tasks):
        task_classes = np.random.choice(classes, n_way, replace=False)
        # task_df = pd.concat([df.loc[df[df.columns[-1]] == x].sample(n=num_instances) for x in task_classes])
        # tasks.append(DataLoader(Dataset(task_df, device), batch_size=n_way*num_instances, shuffle=True))
        samples = [df.loc[df[df.columns[-1]] == x].sample(n=n_way * num_instances) for x in task_classes]
        task_train_df = pd.concat([samples[x][:num_instances] for x in range(len(task_classes))])
        task_test_df = pd.concat([samples[x][num_instances:] for x in range(len(task_classes))])
        tasks.append(
            (DataLoader(Dataset(task_train_df, device), batch_size=n_way * num_instances, shuffle=True),
             DataLoader(Dataset(task_test_df, device), batch_size=n_way * num_instances, shuffle=True))
        )
    return tasks


def run_base_epoch(data, model, loss_function, opt, layers=None):
    if not layers:
        layers = model.layers

    losses = []
    for x, y in data:
        loss = loss_function(model(x, layers=layers), y)
        losses.append(loss.cpu().detach().numpy())
        gradients = torch.autograd.grad(loss, layers.parameters())
        opt.step()

    return np.array(losses).mean()


def run_meta_epoch(tasks, model, base_lr, meta_opt, loss_function):
    meta_loss = 0

    for train, test in tasks:
        updated_layers = deepcopy(model.hidden_layers)
        opt = torch.optim.SGD(updated_layers.parameters(), base_lr)
        run_base_epoch(train, model, loss_function, opt, updated_layers)

        for x, y in test:
            meta_loss += loss_function(model(x, updated_layers), y)

    meta_opt.zero_grad()
    meta_loss.backward()
    meta_opt.step()

    return meta_loss / len(tasks[0])


def split_data(data, target, n_way):
    classes = data[target].unique()
    test_classes = np.random.choice(classes, n_way, replace=False)
    train_classes = [x for x in classes if x not in test_classes]

    test_df = data.loc[data[target].isin(test_classes)]
    train_df = data.loc[data[target].isin(train_classes)]

    return train_classes, train_df, test_classes, test_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--task', type=str, default='mnist')
    parser.add_argument('--base_lr', type=int, default=1e-4)
    parser.add_argument('--meta_lr', type=int, default=1e-2)
    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--hidden_layers', type=int, default=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else None)

    args = parser.parse_args()

    print("Loading Data")
    if args.task == "mnist":
        data_df = pd.read_csv("data/mnist_full_set.csv")
    else:
        raise NotImplementedError
    target = data_df.columns[-1]
    train_classes, train_df, test_classes, test_df = split_data(data_df, target, args.n_way+1)
    final_tasks = sample_tasks(test_df, 30, test_classes, args.k_shot, args.n_way, device)
    print(f"Training Classes: {train_classes}")
    print(f"Test Classes: {test_classes}")
    print()

    hidden_channels = 32
    basic_layers = nn.ModuleList([
        nn.Conv2d(1, hidden_channels, 5),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(hidden_channels, hidden_channels, 5),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(hidden_channels, hidden_channels, 3),
        nn.BatchNorm2d(hidden_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(hidden_channels, 2)
    ])

    hyperparams = dict(output_activation=nn.Softmax, device=device)

    loss_function = nn.modules.loss.CrossEntropyLoss()

    print("Training MAML Model")
    maml_model = MLP(hidden_layers=basic_layers, **hyperparams)
    meta_opt = torch.optim.Adam(maml_model.parameters(), lr=args.meta_lr)
    losses = []
    for epoch in range(args.num_epochs):
        tasks = sample_tasks(train_df, args.batch_size, train_classes, args.k_shot, args.n_way, device)
        meta_loss = run_meta_epoch(tasks, maml_model, args.base_lr, meta_opt, loss_function)
        losses.append(meta_loss.cpu().detach().numpy())
        print(f"Epoch {epoch} Loss: {losses[-1]}")
    plt.plot(losses)
    plt.show()
    print()


    print("Training Pretrain Model")
    pretrain_layers = basic_layers[:-1]
    pretrain_layers.append(nn.Linear(32, len(train_classes)))
    pretrain_model = MLP(hidden_layers=pretrain_layers, **hyperparams)
    pretrain_opt = torch.optim.Adam(pretrain_model.parameters(), lr=.001)
    pretrain_losses = []
    for epoch in range(10):
        data = DataLoader(Dataset(train_df, device), batch_size=16, shuffle=True)
        losses = []
        for x, y in data:
            loss = loss_function(pretrain_model(x), y)
            losses.append(loss.cpu().detach().numpy())
            pretrain_opt.zero_grad()
            loss.backward()
            pretrain_opt.step()
        pretrain_losses.append(np.mean(losses))
        print(f"Epoch {epoch} Loss: {pretrain_losses[-1]}")
    pretrain_model.hidden_layers[-1] = nn.Linear(32, args.n_way, bias=True)
    if use_cuda:
        pretrain_model.cuda(device)
    plt.plot(pretrain_losses)
    plt.show()
    print()


    print(f"Testing Final Models")

    models = {
        'baseline': MLP(hidden_layers=basic_layers, **hyperparams),
        'maml': maml_model,
        'pretrain': pretrain_model
    }

    loss_dict = {}
    for name, model in models.items():
        accuracies = []
        losses = []
        for train, test in final_tasks:

            opt = torch.optim.SGD(model.parameters(), lr=args.base_lr)
            for x, y in train:
                loss = loss_function(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss)

            accuracy_counts = 0
            for x, y in test:
                probabilities = model(x, return_logits=False)
                indices = torch.multinomial(probabilities, 1)
                accuracy_counts += (y.view(-1) == indices.view(-1)).sum()
            accuracies.append(float(accuracy_counts) / len(test.dataset))

        print(f"{name} Avg Loss: {sum(losses)/len(losses)} Avg Accuracy: {sum(accuracies)/len(accuracies)}")
    print()


