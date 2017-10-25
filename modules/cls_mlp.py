#!/usr/bin/env python
# -*- coding: utf-8 -*-

# imports
from torch import nn

__docformat__ = 'reStructuredText'

__all__ = [
    'MLP'
]


def make_to_list(x, length=1):
    """

    Makes a list of `x` argument.

    :param x: The argument to make list of.
    :type x: list|int|str|float|double
    :param length: How long will be the list.
    :type length: int
    :return: A list from `x` if `x` was not a list in the first place.
    :rtype: list[int|str|float|double]
    """
    to_return = [x] if type(x) not in [list, tuple] else x
    if len(to_return) == 1 and len(to_return) < length:
        to_return = to_return * length
    return to_return


class MLP(nn.Module):

    def __init__(self, initial_input_dim, output_dims, activations, dropouts, use_dropout=True,
                 use_bias=True, bias_value=0, weight_init_function=nn.init.xavier_normal,
                 my_name='linear_layer'):
        """

        A class for making an MLP.

        :param initial_input_dim: The initial input dimensionality to the MLP
        :type initial_input_dim: int
        :param output_dims: The output dimensionalities for the MLP
        :type output_dims: int | list[int]
        :param activations: The activations to be used for each layer of the\
                            MLP. Must be the function or a list of functions. \
                            If it is a list, then the length of the list must be\
                            equal to the length of the output dimensionalities `output_dims`.
        :type activations: callable | list[callable]
        :param dropouts: The dropouts to be used. Can be one dropout (same for all layers) \
                         or a list of dropouts, specifying the dropout for each layer. It if\
                         is a list, then the length must be equal to the output dimensionalities\
                         `output_dims`.
        :type dropouts: float | list[float]
        :param use_dropout: A flag to indicate the usage of dropout. Can be a single value \
                            (applied to all layers) or a list of values, for each layer \
                            specifically. If it is a list, then the length must be equal \
                            to the output dimensionalities `output_dims`.
        :type use_dropout: bool | list[bool]
        :param use_bias: A flag to indicate the usage of bias. Can be a single bool value or a\
                         list. If it is a single value, then this value is used for all layers. \
                         If it is a list, then each value is used for the corresponding layer.
        :type use_bias: bool | list[bool]
        :param bias_value: The value to be used for bias initialization.
        :type bias_value: int | float | list[int] | list[float]
        :param weight_init_function: The function to be used for weight initialization.
        :type weight_init_function: callable | list[callable]
        :param my_name: A string to identify the name of each layer. An index will be appended\
                        after the name for each layer.
        :type my_name: str
        """
        super(MLP, self).__init__()

        self.my_name = my_name

        self.initial_input_dim = initial_input_dim

        if type(output_dims) == int:
            output_dims = [output_dims]

        if type(output_dims) == tuple:
            output_dims = list(output_dims)

        self.dims = [self.initial_input_dim] + output_dims

        self.activations = make_to_list(activations, len(self.dims) - 1)
        self.dropout_values = make_to_list(dropouts, len(self.dims) - 1)
        self.use_dropout = make_to_list(use_dropout, len(self.dims) - 1)
        self.use_bias = make_to_list(use_bias, len(self.dims) - 1)
        self.bias_values = make_to_list(bias_value, len(self.dims) - 1)
        self.weight_init_functions = make_to_list(weight_init_function, len(self.dims) - 1)

        self.layers = []
        self.dropouts = []

        for i_dim in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(
                in_features=self.dims[i_dim],
                out_features=self.dims[i_dim + 1],
                bias=self.use_bias[i_dim]
            ))
            if self.use_dropout[i_dim]:
                self.dropouts.append(nn.Dropout(
                    p=self.dropout_values[i_dim]
                ))
                setattr(
                    self,
                    '{the_name}_dropout_{the_index}'.format(
                        the_name=self.my_name,
                        the_index=i_dim
                    ),
                    self.dropouts[-1]
                )
            else:
                self.dropouts.append(None)

            setattr(
                self,
                '{the_name}_{the_index}'.format(
                    the_name=self.my_name,
                    the_index=i_dim
                ),
                self.layers[-1]
            )

        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        for layer, init_function, bias_value, use_bias in zip(
            self.layers, self.weight_init_functions,
                self.bias_values, self.use_bias
        ):
            init_function(layer.weight.data)
            if use_bias:
                nn.init.constant(layer.bias.data, bias_value)

    def forward(self, x):
        output = self.activations[0](self.layers[0](self.dropouts[0](x)))

        for activation, layer, dropout in zip(
                self.activations[1:], self.layers[1:], self.dropouts[1:]
        ):
            if dropout is not None:
                output = dropout(output)
            output = activation(layer(output))

        return output


def main():
    # Testing of the MLP class
    import torch
    from torch.nn import functional
    from torch import optim
    from torch.autograd import Variable
    import time

    total_examples = 10
    initial_dim = 64
    output_dims = [64, 128, 256, 128, 64]
    epochs = 1000

    init_function = nn.init.xavier_uniform
    bias_values = [0, 1, -1, 0, 0]  # or bias_values = 0
    use_biases = [True, False, True, True, True]  # or use_biases = True
    activations = functional.sigmoid
    the_name = 'Testing MLP'
    dropouts = [.4, .6, .1, .4, .5]  # or dropouts = .5
    use_dropout = [True, True, False, True, True]  # or use_dropout = True

    x = Variable(torch.rand((total_examples, initial_dim)))
    y = Variable(torch.rand((total_examples, output_dims[-1])))

    mlp = MLP(
        initial_input_dim=initial_dim,
        output_dims=output_dims,
        activations=activations,
        dropouts=dropouts,
        use_dropout=use_dropout,
        use_bias=use_biases,
        bias_value=bias_values,
        weight_init_function=init_function,
        my_name=the_name
    )

    optimizer = optim.Adam(mlp.parameters())

    for epoch in range(epochs):
        start_time = time.time()
        y_hat = mlp(x)
        optimizer.zero_grad()
        loss = functional.l1_loss(y_hat, y)
        loss.backward()
        optimizer.step()

        print('Epoch {:5d} | Loss: {:.6f} | Elapsed time: {:.6f} sec(s)'.format(
            epoch,
            loss.data[0],
            time.time() - start_time
        ))

if __name__ == '__main__':
    main()

# EOF
