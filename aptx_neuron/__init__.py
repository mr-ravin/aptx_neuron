import torch
import torch.nn as nn

__version__ = '0.0.4'

# -----------------------------------
# APTx Neuron
# -----------------------------------
class aptx_neuron(nn.Module):
    def __init__(self, input_dim, is_alpha_trainable=True, use_delta=True):
        super(aptx_neuron, self).__init__()
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(input_dim))
        else:
            self.register_buffer("alpha", torch.ones(input_dim))
        self.beta  = nn.Parameter(torch.randn(input_dim))
        self.gamma = nn.Parameter(torch.randn(input_dim))
        if use_delta: # like bias
            self.delta = nn.Parameter(torch.zeros(1))  # trainable
        else:
            self.delta = None  # not used, not created

    def forward(self, x):
        nonlinear = (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x
        y = nonlinear.sum(dim=1, keepdim=True)
        if self.delta is not None:  # only executed when usage of delta is enabled; default set to True.
            y = y + self.delta
        return y

# -----------------------------------
# APTx Layer (Multiple Neurons)
# -----------------------------------
class aptx_layer(nn.Module):
    def __init__(self, input_dim, output_dim, is_alpha_trainable=True, use_delta=True):
        super(aptx_layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(output_dim, input_dim))
        else:
            self.register_buffer("alpha", torch.ones(output_dim, input_dim))
        self.beta  = nn.Parameter(torch.randn(output_dim, input_dim))
        self.gamma = nn.Parameter(torch.randn(output_dim, input_dim))
        if use_delta: # like bias
            self.delta = nn.Parameter(torch.zeros(output_dim))
        else:
            self.delta = None # not used, not created

    def forward(self, x):  # x: [batch_size, input_dim]
        # x -> [B, 1, D]
        x_expanded = x.unsqueeze(1)  # [B, 1, D]
        # params -> [1, O, D] to broadcast with [B, 1, D]
        alpha = self.alpha.unsqueeze(0)  # [1, O, D]
        beta  = self.beta.unsqueeze(0)   # [1, O, D]
        gamma = self.gamma.unsqueeze(0)  # [1, O, D]
        # nonlinear: [B, O, D]
        nonlinear = (alpha + torch.tanh(beta * x_expanded)) * gamma * x_expanded
        # sum over input_dim → [B, O]
        y = nonlinear.sum(dim=2)
        if self.delta is not None:
            y = y + self.delta  # [O] → broadcast to [B, O]
        return y
