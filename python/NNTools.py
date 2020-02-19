

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
    ];
    return convBlock;


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
    };

    return residualBlock;


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
    };

    return residualBlock;


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
        );
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
            );
        return conv;

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
    };

    return inceptionBlock;


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
        );
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
            );
        return conv;

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
    };

    return inceptionBlock;


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
    ];

    return config;
