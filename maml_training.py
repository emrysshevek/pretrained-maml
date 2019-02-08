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
from model import MLP
from data_utils import *


def sample_tasks(df, batch_size, classes, num_instances, n_way, device=None):
    # TODO: is it ok if there is some overlap between train and test instances?
    tasks = []
    for x in range(batch_size):
        task_classes = np.random.choice(classes, n_way, replace=False)
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
        updated_layers = deepcopy(model.layers)
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
    parser.add_argument('--meta_lr', type=int, default=1e-4)
    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--k_shot', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hidden_dimension', type=int, default=512)
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
    train_classes, train_df, test_classes, test_df = split_data(data_df, target, args.n_way)
    final_train_df = pd.concat([test_df.loc[test_df[target] == x].sample(args.k_shot) for x in test_classes])
    final_test_df = test_df.drop(final_train_df.index)
    print(f"Training Classes: {train_classes}")
    print(f"Test Classes: {test_classes}")
    print()

    hyperparams = dict(input_dimension=train_df.shape[1]-1,
                       hidden_dimension=args.hidden_dimension,
                       num_layers=args.hidden_layers,
                       hidden_activation=nn.ReLU,
                       output_activation=nn.Softmax,
                       device=device)

    loss_function = nn.modules.loss.CrossEntropyLoss()

    print("Training MAML Model")
    maml_model = MLP(**hyperparams, output_dimension=args.n_way)
    meta_opt = torch.optim.SGD(maml_model.parameters(), lr=args.meta_lr)
    losses = []
    for epoch in range(args.num_epochs):
        tasks = sample_tasks(train_df, args.batch_size, train_classes, args.k_shot, args.n_way, device)
        meta_loss = run_meta_epoch(tasks, maml_model, args.base_lr, meta_opt, loss_function)
        losses.append(meta_loss.cpu().detach().numpy())
        print(f"Epoch {epoch+1} Loss: {losses[-1]}")
    plt.plot(losses)
    plt.show()
    print()


    print("Training Pretrain Model")
    pretrain_model = MLP(**hyperparams, output_dimension=len(train_classes))
    pretrain_opt = torch.optim.SGD(pretrain_model.parameters(), lr=.0001)
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
        print(f"Epoch {epoch+1} Loss: {pretrain_losses[-1]}")
    pretrain_model.layers[-1] = nn.Linear(args.hidden_dimension, args.n_way, bias=True)
    if use_cuda:
        pretrain_model.cuda(device)
    plt.plot(pretrain_losses)
    plt.show()
    print()


    print(f"Training Final Models")
    baseline_model = MLP(**hyperparams, output_dimension=args.n_way)

    models = {
        'baseline': {'model': baseline_model, 'opt': torch.optim.SGD(baseline_model.parameters(), lr=args.base_lr)},
        'maml': {'model': maml_model, 'opt': torch.optim.SGD(maml_model.parameters(), lr=args.base_lr)},
        'pretrain': {'model': pretrain_model, 'opt': torch.optim.SGD(pretrain_model.parameters(), lr=args.base_lr)}
    }

    opt = torch.optim.SGD(maml_model.parameters(), lr=.1)
    for name, model in models.items():
        for x, y in DataLoader(Dataset(final_train_df, device), batch_size=2*args.k_shot, shuffle=True):
            loss = loss_function(model['model'](x), y)
            model['opt'].zero_grad()
            loss.backward()
            print(f"{name} Loss: {loss}")
            model['opt'].step()
    print()

    print("Testing Final Models")
    generator = DataLoader(Dataset(final_test_df, device), shuffle=True)
    accuracy_counts = {name: 0 for name in models.keys()}
    for x, y in generator:
        for name, model in models.items():
            probabilities = model['model'](x, return_logits=False)
            indices = torch.multinomial(probabilities, 1)
            accuracy_counts[name] += (y.view(-1) == indices.view(-1)).sum()
    for name, accuracy in accuracy_counts.items():
        print(f"{name} Accuracy: {float(accuracy) / len(generator.dataset)}")

