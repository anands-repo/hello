import torch
import logging


def initCNN(layer):
    if (type(layer) is torch.nn.Conv1d) or (type(layer) is torch.nn.Linear):
        logging.info("Initializing layer");
        if hasattr(layer, 'weight'):
            if layer.weight is not None:
                torch.nn.init.kaiming_uniform_(layer.weight);
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.1);


def SingleConvLayer(
    inChannels,
    outChannels,
    kernelSize,
    padding,
    dilation,
    stride,
    groups=1,
):
    convBlock = [
        {
            "type": "Conv1d",
            "kwargs": {
                "in_channels": inChannels,
                "out_channels": outChannels,
                "kernel_size": kernelSize,
                "padding": padding,
                "dilation": dilation,
                "stride": stride,
                "groups": groups,
            }
        },

        {
            "type": "BatchNorm1d",
            "kwargs": {
                "num_features": outChannels,
            }
        },

        {
            "type": "ReLU",
            "kwargs": dict(),
        }
    ]
    return convBlock


def ResidualBlockConvShortcut(
    inChannels,
    outChannels,
    kernelSizes,
    paddings,
    dilations,
    strides,
    groups=[1, 1, 1],
):
    residualBlock = {
        "type": "ResidualBlock",
        "kwargs": {
            "feedforward": [
                {
                    "type": "Conv1d",
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
                    "type": "BatchNorm1d",
                    "kwargs": {
                        "num_features": outChannels,
                    }
                },

                {
                    "type": "ReLU",
                    "kwargs": dict(),
                },

                {
                    "type": "Conv1d",
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
                    "type": "BatchNorm1d",
                    "kwargs": {
                        "num_features": outChannels,
                    }
                },

                {
                    "type": "ReLU",
                    "kwargs": dict(),
                },
            ],

            "shortcut": [
                {
                    "type": "Conv1d",
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
):
    residualBlock = {
        "type": "ResidualBlock",
        "kwargs": {
            "feedforward": [
                {
                    "type": "Conv1d",
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
                    "type": "BatchNorm1d",
                    "kwargs": {
                        "num_features": outChannels,
                    }
                },

                {
                    "type": "ReLU",
                    "kwargs": dict(),
                },

                {
                    "type": "Conv1d",
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
                    "type": "BatchNorm1d",
                    "kwargs": {
                        "num_features": outChannels,
                    }
                },

                {
                    "type": "ReLU",
                    "kwargs": dict(),
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


def terminus(inChannels, outChannels, dropout=0):
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
            "type": "BatchNorm1d",
            "kwargs": {
                "num_features": inChannels,
            }
        } 
        if dropout == 0 else
        {
            "type": "Dropout",
            "kwargs": {
                "p": dropout
            }
        },

        {
            "type": "Linear",
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
    def __init__(self):
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


torch.nn.Flatten = Flatten
torch.nn.GlobalPool = GlobalPool
torch.nn.ResidualBlock = ResidualBlock
torch.nn.Noop = Noop
torch.nn.Inception = Inception


class Network(torch.nn.Module):
    """
    Base Network class
    """
    def __init__(self, config):
        super().__init__()
        layers = []

        for i, configuration in enumerate(config):
            layerType = getattr(torch.nn, configuration['type'])
            layer = layerType(**configuration['kwargs'])
            initCNN(layer)
            layers.append(layer)

        self.network = torch.nn.Sequential(*layers)

    def forward(self, tensors, *args, **kwargs):
        return self.network(tensors)
