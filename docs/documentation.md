### Commented and Type Hinted Code
---
The code below is the original code from https://github.com/MSU-MLSys-Lab/arch2vec; however,
I have added most of the comments and have attempted to type hint as much of the code involved
in the reproduction as possible. 

### gin/models
> MLP from mlp.py is used in GIN and Model in models/model.py <br>
> GraphConvolution from graphcnn.py is used in VAEncoder and Encoder in models/model.py 
#### MLP
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
        A class that represents a multilayered perceptron consisting of

        Attributes
        ----------
        num_layers : int
            The number of layers of the MLP. This excludes the input layer and
            if num_layers=1, the MLP is linear.
        input_dim : int
            The dimensionality of the input features.
        hidden_dim : int
            The dimensionality of the hidden units for each layer.
        output_dim : int
            The number of classes to produce predictions for.

        Methods
        -------
        forward(x)
            Prints the animals name and what sound it makes

        Raises
        ------
        ValueError
            If the number of layers is not positive.
        """
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        """Represents the forward pass over the multilayered perceptron

        Parameters
        ----------
        x : ?
            The sound the animal makes (default is None)
        """
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
```
