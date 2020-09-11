import torch
import logging
import math
import Attention


def initCNN(layer):
    if (type(layer) is torch.nn.Conv1d) or (type(layer) is torch.nn.Linear):
        logging.info("Initializing layer");
        if hasattr(layer, 'weight'):
            if layer.weight is not None:
                torch.nn.init.kaiming_uniform_(layer.weight);
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.1);


def SingleLinearLayer(
    in_features,
    out_features,
    dropout=0,
    batch_norm=True,
    activation="ReLU",
    activation_args=dict(),
    use_weight_norm=False,
    norm_type="BatchNorm1d",
):
    if use_weight_norm:
        batch_norm = False

    config = [
        {
            "type": "Linear" if not use_weight_norm else "WeightNormedLinear",
            "kwargs": {
                "in_features": in_features,
                "out_features": out_features,
            }
        },
    ]

    if batch_norm:
        config.append(
            {
                "type": norm_type,
                "kwargs": {
                    "num_features": out_features,
                }
            }
        )

    config.append(
        {
            "type": activation,
            "kwargs": activation_args
        }
    )

    if dropout > 0:
        config.append(
            {
                "type": "Dropout",
                "kwargs": {
                    "p": dropout
                }
            }
        )

    return config


def SingleConvLayer(
    inChannels,
    outChannels,
    kernelSize,
    padding,
    dilation,
    stride,
    groups=1,
    activation="ReLU",
    activation_args=dict(),
    no_batch_norm=False,
    use_weight_norm=False,
    norm_type="BatchNorm1d",
):
    if use_weight_norm:
        no_batch_norm = True

    convBlock = [{
        "type": "Conv1d" if not use_weight_norm else "WeightNormedConv1d",
        "kwargs": {
            "in_channels": inChannels,
            "out_channels": outChannels,
            "kernel_size": kernelSize,
            "padding": padding,
            "dilation": dilation,
            "stride": stride,
            "groups": groups,
        }
    }]

    if not no_batch_norm:
        convBlock += [{
            "type": norm_type,
            "kwargs": {
                "num_features": outChannels,
            }
        }]

    convBlock += [{
        "type": activation,
        "kwargs": activation_args,
    }]

    return convBlock


def ResidualBlockConvShortcut(
    inChannels,
    outChannels,
    kernelSizes,
    paddings,
    dilations,
    strides,
    groups=[1, 1, 1],
    use_weight_norm=False,
    norm_type="BatchNorm1d",
    activation="ReLU",
    activation_args=dict(),
):
    residualBlock = {
        "type": "ResidualBlock",
        "kwargs": {
            "feedforward": [
                {
                    "type": "Conv1d" if not use_weight_norm else "WeightNormedConv1d",
                    "kwargs": {
                        "in_channels": inChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSizes[0],
                        "padding": paddings[0],
                        "dilation": dilations[0],
                        "stride": strides[0],
                        "groups": groups[0],
                    }
                },

                {
                    "type": norm_type,
                    "kwargs": {
                        "num_features": outChannels,
                    }
                } if not use_weight_norm else {
                    "type": "Noop",
                    "kwargs": dict()
                },

                {
                    "type": activation,
                    "kwargs": activation_args,
                },

                {
                    "type": "Conv1d" if not use_weight_norm else "WeightNormedConv1d",
                    "kwargs": {
                        "in_channels": outChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSizes[1],
                        "padding": paddings[1],
                        "dilation": dilations[1],
                        "stride": strides[1],
                        "groups": groups[1],
                    }
                },

                {
                    "type": norm_type,
                    "kwargs": {
                        "num_features": outChannels,
                    }
                } if not use_weight_norm else {
                    "type": "Noop",
                    "kwargs": dict(),
                },

                {
                    "type": activation,
                    "kwargs": activation_args,
                },
            ],

            "shortcut": [
                {
                    "type": "Conv1d" if not use_weight_norm else "WeightNormedConv1d",
                    "kwargs": {
                        "in_channels": inChannels,
                        "out_channels": outChannels,
                        "kernel_size": 1,
                        "stride": strides[2],
                        "groups": groups[2],
                    }
                },
            ]
        }
    }

    return residualBlock


def ResidualBlockFTShortcut(
    inChannels,
    outChannels,
    kernelSizes,
    paddings,
    dilations,
    strides,
    groups=[1, 1],
    use_weight_norm=False,
    norm_type="BatchNorm1d",
    activation="ReLU",
    activation_args=dict(),
):
    residualBlock = {
        "type": "ResidualBlock",
        "kwargs": {
            "feedforward": [
                {
                    "type": "Conv1d" if not use_weight_norm else "WeightNormedConv1d",
                    "kwargs": {
                        "in_channels": inChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSizes[0],
                        "padding": paddings[0],
                        "dilation": dilations[0],
                        "stride": strides[0],
                        "groups": groups[0],
                    }
                },

                {
                    "type": norm_type,
                    "kwargs": {
                        "num_features": outChannels,
                    }
                } if not use_weight_norm else {
                    "type": "Noop",
                    "kwargs": dict(),
                },

                {
                    "type": activation,
                    "kwargs": activation_args,
                },

                {
                    "type": "Conv1d" if not use_weight_norm else "WeightNormedConv1d",
                    "kwargs": {
                        "in_channels": outChannels,
                        "out_channels": outChannels,
                        "kernel_size": kernelSizes[1],
                        "padding": paddings[1],
                        "dilation": dilations[1],
                        "stride": strides[1],
                        "groups": groups[1],
                    }
                },

                {
                    "type": norm_type,
                    "kwargs": {
                        "num_features": outChannels,
                    }
                } if not use_weight_norm else {
                    "type": "Noop",
                    "kwargs": dict(),
                },

                {
                    "type": activation,
                    "kwargs": activation_args,
                },
            ],

            "shortcut": [
                {
                    "type": "Noop",
                    "kwargs": {
                    }
                },
            ]
        }
    }

    return residualBlock


def inceptionBlockA(
    inChannels,
    intChannels,
    outChannelsPerBranch,
    outputStrides,
    poolType="avg"
):
    """
    Inception block with three branches
        - 1x1 conv
        - 1x1 conv + 3x3 conv
        - 1x1 conv + 3x3 conv + 3x3 conv
    """
    def branch1():
        conv = SingleConvLayer(
            inChannels,
            outChannelsPerBranch[0],
            kernelSize=1,
            padding=0,
            dilation=1,
            stride=1,
        )
        if outputStrides > 1:
            conv.append(
                {
                    "type": "AvgPool1d" if poolType == "avg" else "MaxPool1d",
                    "kwargs": {
                        "kernel_size": 3,
                        "stride": outputStrides,
                        "padding": 1,
                    }
                }
            )
        return conv

    inceptionBlock = {
        "type": "Inception",
        "kwargs": {
            "branches": [
                # Branch 1
                branch1(),

                # Branch 2
                SingleConvLayer(
                    inChannels,
                    intChannels[1],
                    kernelSize=1,
                    padding=0,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[1],
                    outChannelsPerBranch[1],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=outputStrides,
                ),

                # Branch 3
                SingleConvLayer(
                    inChannels,
                    intChannels[2],
                    kernelSize=1,
                    padding=0,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[2],
                    intChannels[2],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[2],
                    outChannelsPerBranch[2],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=outputStrides,
                )
            ]
        }
    }

    return inceptionBlock


def inceptionBlockB(
    inChannels,
    intChannels,
    outChannelsPerBranch,
    outputStrides,
    poolType="avg"
):
    """
    Inception block with four branches
        - 1x1 conv
        - 1x1 conv + 3x3 conv
        - 1x1 conv + 3x3 conv + 3x3 conv
        - 1x1 conv + 3x3 conv + 3x3 conv + 3x3 conv
    """
    def branch1():
        # Do pooling as necessary. In case of stride > 1, do pooling.
        conv = SingleConvLayer(
            inChannels,
            outChannelsPerBranch[0],
            kernelSize=1,
            padding=0,
            dilation=1,
            stride=1,
        )
        if outputStrides > 1:
            conv.append(
                {
                    "type": "AvgPool1d" if poolType == "avg" else "MaxPool1d",
                    "kwargs": {
                        "kernel_size": 3,
                        "stride": outputStrides,
                        "padding": 1,
                    }
                }
            )
        return conv

    inceptionBlock = {
        "type": "Inception",
        "kwargs": {
            "branches": [
                # Branch 1
                branch1(),

                # Branch 2
                SingleConvLayer(
                    inChannels,
                    intChannels[1],
                    kernelSize=1,
                    padding=0,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[1],
                    outChannelsPerBranch[1],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=outputStrides,
                ),

                # Branch 3
                SingleConvLayer(
                    inChannels,
                    intChannels[2],
                    kernelSize=1,
                    padding=0,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[2],
                    intChannels[2],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[2],
                    outChannelsPerBranch[2],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=outputStrides,
                ),

                # Branch 4
                SingleConvLayer(
                    inChannels,
                    intChannels[3],
                    kernelSize=1,
                    padding=0,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[3],
                    intChannels[3],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[3],
                    intChannels[3],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=1,
                ) + \
                SingleConvLayer(
                    intChannels[3],
                    outChannelsPerBranch[3],
                    kernelSize=3,
                    padding=1,
                    dilation=1,
                    stride=outputStrides,
                ),
            ]
        }
    }

    return inceptionBlock


def terminus(
    inChannels,
    outChannels,
    dropout=0,
    use_weight_norm=False,
    norm_type="BatchNorm1d"
):
    config = [
        {
            "type": "AdaptiveAvgPool1d",
            "kwargs": {
                "output_size": 1,
            }
        },

        {
            "type": "Flatten",
            "kwargs": dict(),
        },

        # Use a batch norm layer if dropout is zero
        # otherwise use dropout. Using them both is not
        # good for training
        {
            "type": norm_type,
            "kwargs": {
                "num_features": inChannels,
            }
        } 
        if (dropout == 0) and (not use_weight_norm) else
        {
            "type": "Dropout",
            "kwargs": {
                "p": dropout
            }
        } if (dropout > 0) else {
            "type": "Noop",
            "kwargs": dict(),
        },

        {
            "type": "Linear" if not use_weight_norm else "WeightNormedLinear",
            "kwargs": {
                "in_features": inChannels,
                "out_features": outChannels,
            }
        }
    ]

    return config


class ResidualBlock(torch.nn.Module):
    """
    Creates a residual block of convolutional layers
    """
    def __init__(self, **kwargs):
        super().__init__()
        # 'feedforward' is a list of feed-forward layers
        # 'shortcut' indicates the shortcut connection
        ffConfigs = kwargs['feedforward']
        shConfigs = kwargs['shortcut']
        self.ffNetwork = Network(ffConfigs)
        self.shNetwork = Network(shConfigs)

    def forward(self, tensor):
        return self.ffNetwork(tensor) + self.shNetwork(tensor)


class Noop(torch.nn.Module):
    """
    Noop layer does no operation; for shortcut connections
    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, tensor):
        return tensor


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return tensor.view(tensor.shape[0], -1)


class GlobalPool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return torch.sum(tensor, dim=2)


class Inception(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        branches = kwargs['branches']
        self.numBranches = len(branches)

        for i, branch in enumerate(branches):
            branchNetwork = NNTools.Network(branch)
            setattr(self, 'branch%d' % i, branchNetwork)

    def forward(self, tensor):
        branchResults = []

        for i in range(self.numBranches):
            branch = getattr(self, 'branch%d' % i)
            branchResults.append(branch(tensor))

        return torch.cat(branchResults, dim=1)


class Network(torch.nn.Module):
    """
    Base Network class
    """
    def __init__(self, config):
        super().__init__()
        layers = []

        for i, configuration in enumerate(config):
            try:
                layerType = getattr(torch.nn, configuration['type'])
            except Exception:
                logging.error("Failed reading configuration " + str(configuration))
                raise ValueError

            layer = layerType(**configuration['kwargs'])
            # Commented: September 1 2020. pytorch does
            # automatic initialization of weights
            # initCNN(layer)
            layers.append(layer)

        self.network = torch.nn.Sequential(*layers)

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)


class Pad1d(torch.nn.Module):
    def __init__(self, padleft, padright):
        super().__init__()
        self.padleft = padleft
        self.padright = padright

    def forward(self, tensor):
        return torch.nn.functional.pad(
            tensor, pad=(self.padleft, self.padright)
        )


class Compressor(torch.nn.Module):
    """
    A layer which performs compression
    """
    def __init__(self, input_length, num_inputs):
        super().__init__()
        num_layers = math.ceil(math.log2(input_length))
        layers = []

        for i in range(num_layers):
            dilation = 2 ** i
            padding = (dilation - dilation // 2, dilation // 2)
            layers.append(
                Pad1d(*padding)
            )
            layers.append(
                Network(SingleConvLayer(
                    inChannels=num_inputs,
                    outChannels=num_inputs,
                    kernelSize=2,
                    padding=0,
                    dilation=dilation,
                    stride=1,
                ))
            )

        last_layer = torch.nn.Conv1d(
            in_channels=num_inputs,
            out_channels=num_inputs,
            kernel_size=1,
        )

        layers.append(last_layer)

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, tensor):
        res = tensor
        return self.layers(res)


class DotProduct(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        tensora, tensorb = tensors
        res = torch.matmul(
            tensora.view(tensora.shape[0], 1, tensora.shape[1]),
            tensorb.view(tensorb.shape[0], tensorb.shape[1], 1)
        ) / (tensora.shape[1] ** 0.5)
        res = torch.squeeze(res, dim=2)
        return res


class ConcatenateChannels(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        res = torch.cat(tensors, dim=1)
        return res


class AdditiveLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensors):
        tensora, tensorb = tensors
        return tensora + tensorb


class SelectArgument(torch.nn.Module):
    def __init__(self, select):
        super().__init__()
        self.select = select

    def forward(self, args):
        return args[self.select]


class Fork(torch.nn.Module):
    def __init__(self, net_args):
        super().__init__()
        for i, args in enumerate(net_args):
            setattr(self, 'net%d' % i, Network(args))

    def forward(self, args):
        return [
            getattr(self, 'net%d' % i)(args[i]) for i in range(len(args))
        ]


class LinearCombination(torch.nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.coefficients = coefficients

    def forward(self, args):
        result = 0

        for i, a in enumerate(args):
            result += self.coefficients[i] * a

        return result


class WeightNormedLinear(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = torch.nn.utils.weight_norm(
            torch.nn.Linear(*args, **kwargs)
        )

    def forward(self, *args, **kwargs):
        return self.linear(*args, **kwargs)


class WeightNormedConv1d(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1d = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(*args, **kwargs)
        )

    def forward(self, *args, **kwargs):
        return self.conv1d(*args, **kwargs)


class LayerNormModule(torch.nn.Module):
    """
    Layer norm cannot be applied to conv by default as the original paper says, but
    we treat the conv outputs similar to recurrent activations occurring one after the
    other in order as a result of processing a state (which are the layer inputs)
    """
    def __init__(self, num_features):
        super().__init__()
        self.normer = torch.nn.LayerNorm(normalized_shape=num_features)

    def forward(self, tensor):
        """
        :param tensor: torch.Tensor
            Shape: [batch, #channels, length]
        """
        if len(tensor.shape) == 3:
            # For convolutional layers
            transposed = torch.transpose(tensor, 1, 2)  # [b, l, c]
            normed = self.normer(transposed)
            result = torch.transpose(normed, 1, 2)  # [b, c, l]
        elif len(tensor.shape) == 2:
            # For linear layers
            result = self.normer(tensor)
        else:
            raise ValueError("Unknown tensor shape")

        return result


class Transposer(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, tensor):
        return torch.transpose(tensor, *self.dims)


torch.nn.Compressor = Compressor
torch.nn.Flatten = Flatten
torch.nn.GlobalPool = GlobalPool
torch.nn.ResidualBlock = ResidualBlock
torch.nn.Noop = Noop
torch.nn.Inception = Inception
torch.nn.DotProduct = DotProduct
torch.nn.ConcatenateChannels = ConcatenateChannels
torch.nn.SelectArgument = SelectArgument
torch.nn.Fork = Fork
torch.nn.LinearCombination = LinearCombination
torch.nn.WeightNormedConv1d = WeightNormedConv1d
torch.nn.WeightNormedLinear = WeightNormedLinear
torch.nn.LayerNormModule = LayerNormModule
torch.nn.Transposer = Transposer
