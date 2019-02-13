import torch
from torch import nn


class Flatten(nn.Module):

    def forward(self, input):
        n, c, h, w = input.size()
        return input.view(n, -1)


class MLP(nn.Module):

    def __init__(self, layers, output_activation=nn.Softmax, device=None):
        super(MLP, self).__init__()
        self.output_activation = output_activation(dim=1)
        self.layers = layers
        if device:
            self.cuda(device)

    def forward(self, x, return_logits=True, layers=None):
        if not layers:
            layers = self.layers

        for layer in layers:
            x = layer(x)

        if not return_logits:
            x = self.output_activation(x)

        return x
