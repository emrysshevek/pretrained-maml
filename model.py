from copy import deepcopy

import torch
from torch import nn


class Flatten(nn.Module):

    def forward(self, input):
        n, c, h, w = input.size()
        return input.view(n, -1)


class MLP(nn.Module):

    def __init__(self, hidden_layers, output_layer, output_activation=nn.LogSoftmax, device=None):
        super(MLP, self).__init__()
        self.output_activation = output_activation
        self.output_layer = output_layer
        self.hidden_layers = hidden_layers.extend(output_layer)
        self.device = device
        if device:
            self.cuda(device)

    def swap_output_layer(self, output_layer):
        layer_len = len(self.output_layer)
        self.output_layer = deepcopy(output_layer)
        del self.hidden_layers[-layer_len:]
        self.hidden_layers.extend(self.output_layer)
        if self.device:
            self.cuda(self.device)

    def forward(self, x, return_logits=True, layers=None):
        if not layers:
            layers = self.hidden_layers

        for layer in layers:
            x = layer(x)

        x = self.output_activation(x)

        return x

    def clone(self):
        output = deepcopy(self.output_layer)
        hidden = deepcopy(self.hidden_layers[:-len(output)])
        act = deepcopy(self.output_activation)
        device = self.device

        clone = MLP(hidden, output, act, device)

        for self_param, clone_param in zip(self.parameters(), clone.parameters()):
            clone_param = self_param.clone()

        return clone
