## APTx Neuron 
This repository offers a Python code for the PyTorch implementation of the APTx Neuron, as introduced in the paper "APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation".

**Paper Title**: APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation

**Author**: [Ravin Kumar](https://mr-ravin.github.io)

**Sources**:
- [Arxiv.org](https://arxiv.org/abs/2507.14270)
- [Research Gate](https://www.researchgate.net/publication/393889376_APTx_Neuron_A_Unified_Trainable_Neuron_Architecture_Integrating_Activation_and_Computation)
  
#### Github Repositories: 
- **APTx Neuron** (Pytorch + PyPI Package): [APTx Neuron](https://github.com/mr-ravin/aptx_neuron)
- **APTx Activation Function** (Pytorch + PyPI Package): [APTx Activation Function](https://github.com/mr-ravin/aptx_activation)
- **Experimentation Results with MNIST** (APTx Neuron): [MNIST Experimentation Code](https://github.com/mr-ravin/APTxNeuron)

#### Cite Paper as:
```
Kumar, Ravin. "APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation." arXiv preprint arXiv:2507.14270 (2025).
```
Or,
```
@article{kumar2025aptx,
  title={APTx Neuron: A Unified Trainable Neuron Architecture Integrating Activation and Computation},
  author={Kumar, Ravin},
  journal={arXiv preprint arXiv:2507.14270},
  year={2025}
}
```
 
---
### APTx Neuron
<b>Abstract</b>: We propose the APTx Neuron, a novel, unified neural computation unit that integrates non-linear activation and linear transformation into a single trainable expression. The APTx Neuron is derived from the [APTx activation function](https://arxiv.org/abs/2209.06119), thereby eliminating the need for separate activation layers and making the architecture both computationally efficient and elegant. The proposed neuron follows the functional form $y = \sum_{i=1}^{n} ((\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i) + \delta$, where all parameters $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ are trainable. We validate our APTx Neuron-based architecture on the MNIST dataset, achieving up to 96.69\% test accuracy within 11 epochs using approximately 332K trainable parameters. The results highlight the superior expressiveness and computational efficiency of the APTx Neuron compared to traditional neurons, pointing toward a new paradigm in unified neuron design and the architectures built upon it.

The APTx Neuron is a novel computational unit that unifies linear transformation and non-linear activation into a single, expressive formulation. Inspired by the parametric APTx activation function, this neuron architecture removes the strict separation between computation and activation, allowing both to be learned as a cohesive entity. It is designed to enhance representational flexibility while reducing architectural redundancy.

#### Mathematical Formulation

Traditionally, a neuron computes the output as:

$y = \phi\left( \sum_{i=1}^{n} w_i x_i + b \right)$

where: 
- $x_i$ are the inputs,
- $w_i$ are the weights,
- $b$ is the bias,
- and $\phi$ is an activation function such as ReLU, Swish, or Mish.


The APTx Neuron merges these components into a unified trainable expression as:

$y = \sum_{i=1}^{n} \left[ (\alpha_i + \tanh(\beta_i x_i)) \cdot \gamma_i x_i \right] + \delta$

where:
- $x_i$ is the $i$-th input feature,
- $\alpha_i$, $\beta_i$, and $\gamma_i$ are trainable parameters for each input,
- $\delta$ is a trainable scalar bias.

This equation allows the neuron to modulate each input through a learned, per-dimension non-linearity and scaling operation. The term $(\alpha_i + \tanh(\beta_i x_i))$ introduces adaptive gating, and $\gamma_i x_i$ provides multiplicative control.

---
## 📥 Installation
```bash
pip install aptx_neuron
```
or,

```bash
pip install git+https://github.com/mr-ravin/aptx_neuron.git
```
----

## Usage
<b>1</b>. APTx Neuron-based Layer with all $\alpha_i$, $\beta_i$, $\gamma_i$, and $\delta$ as trainable:

The setting `is_alpha_trainable = True` keeps $\alpha_i$ trainable. Each APTx neuron will have $(3n + 1)$ trainable parameters, where $n$ is input dimension. Note: The default value of `is_alpha_trainable` is `True`.

```
import aptx_neuron
input_dim = 8  # assuming input vector to be of dimension 8.
output_dim = 1 # assuming output dimension equals 1.

aptx_neuron_layer = aptx_neuron.aptx_layer(input_dim=input_dim, output_dim=output_dim, is_alpha_trainable=True)
```

<b>2</b>. APTx Neuron-based Layer with $\alpha_i=1$ (not trainable); While $\beta_i$, $\gamma_i$, and $\delta$ as trainable:

The setting `is_alpha_trainable = False` makes $\alpha_i$ fixed (non-trainable). Each APTx neuron will then have $(2n + 1)$ trainable parameters, thus reducing memory and training time per epoch. Here, $n$ is input dimension.

```
import aptx_neuron
input_dim = 8  # assuming input vector to be of dimension 8.
output_dim = 1 # assuming output dimension equals 1.

aptx_neuron_layer = aptx_neuron.aptx_layer(input_dim=input_dim, output_dim=output_dim, is_alpha_trainable=False)  # α_i is fixed (not trainable)
```

----
#### Conclusion
This work introduced the APTx Neuron, a unified, fully trainable neural unit that integrates linear transformation and non-linear activation into a single expression, extending the APTx activation function. By learning per-input parameters $\alpha_i$, $\beta_i$, and $\gamma_i$ for each input $x_i$, and a shared bias term $\delta$ within a neuron, the APTx Neuron removes the need for separate activation layers and enables fine-grained input transformation. The APTx Neuron generalizes traditional neurons and activations, offering greater representational power. Our experiments show that a fully connected APTx Neuron-based feedforward neural network achieves 96.69% test accuracy on the MNIST dataset within 11 epochs using approximately 332K trainable parameters, demonstrating rapid convergence and high efficiency. This design lays the groundwork for extending APTx Neurons to CNNs and transformers, paving the way for more compact and adaptive deep learning architectures.

----

### 📜 Copyright License
```python
Copyright (c) 2025 Ravin Kumar
Website: https://mr-ravin.github.io

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the 
Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, 
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
