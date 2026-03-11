import torch
import torch.nn as nn

__version__ = '0.0.9'

# -----------------------------------
# APTx Neuron
# -----------------------------------
class aptx_neuron(nn.Module):
    def __init__(self, input_dim, is_alpha_trainable=True, use_delta=True):
        super().__init__()
        self.use_delta = use_delta
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(input_dim))
        else:
            self.register_buffer('alpha', torch.ones(input_dim))
        self.beta = nn.Parameter(torch.randn(input_dim))
        self.gamma = nn.Parameter(torch.randn(input_dim))
        if self.use_delta:
            self.delta = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("delta", None)

    def forward(self, x):  # x: [batch_size, input_dim]
        nonlinear = (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x
        # [batch_size, output_dim]
        y = nonlinear.sum(dim=1, keepdim=True)
        if self.use_delta:
            y = y + self.delta
        return y


# -----------------------------------
# APTx Layer (Vectorized Multiple Neurons)
# -----------------------------------
class aptx_layer(nn.Module):
    def __init__(self, input_dim, output_dim, is_alpha_trainable=True, use_delta=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_delta = use_delta
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(output_dim, input_dim))
        else:
            self.register_buffer('alpha', torch.ones(output_dim, input_dim))
        self.beta = nn.Parameter(torch.randn(output_dim, input_dim))
        self.gamma = nn.Parameter(torch.randn(output_dim, input_dim))
        if self.use_delta:
            self.delta = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter("delta", None)

    def forward(self, x):  # x: [batch_size, input_dim]
        # x -> [batch_size, 1, input_dim]
        x_exp = x.unsqueeze(1)
        nonlinear = (
            self.alpha + torch.tanh(self.beta.unsqueeze(0) * x_exp)
        ) * self.gamma.unsqueeze(0) * x_exp
        # [batch_size, output_dim]
        y = nonlinear.sum(dim=2)
        if self.use_delta:
            y = y + self.delta
        return y
