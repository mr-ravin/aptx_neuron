import torch
import torch.nn as nn

__version__ = '0.0.1'

# -----------------------------------
# APTx Neuron
# -----------------------------------
class aptx_neuron(nn.Module):
    def __init__(self, input_dim, is_alpha_trainable=True):
        super(aptx_neuron, self).__init__()
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(input_dim)) # To reduce trainable parameters from 3n + 1 to 2n + 1 (where n is the input dimension), replace with: self.alpha = torch.ones(input_dim)  # (fix α_i = 1 to make it non-trainable)
        else:
            self.alpha = torch.ones(input_dim)
        self.beta  = nn.Parameter(torch.randn(input_dim))
        self.gamma = nn.Parameter(torch.randn(input_dim))
        self.delta = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: [batch_size, input_dim]
        nonlinear = (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x
        y = nonlinear.sum(dim=1, keepdim=True) + self.delta
        return y

# -----------------------------------
# APTx Layer (Multiple Neurons)
# -----------------------------------
class aptx_layer(nn.Module):
    def __init__(self, input_dim, output_dim, is_alpha_trainable=True):
        super(aptx_layer, self).__init__()
        self.neurons = nn.ModuleList([aptx_neuron(input_dim, is_alpha_trainable) for _ in range(output_dim)])

    def forward(self, x):  # x: [batch_size, input_dim]
        outputs = [neuron(x) for neuron in self.neurons]  # list of [batch_size, 1]
        return torch.cat(outputs, dim=1)  # [batch_size, output_dim]