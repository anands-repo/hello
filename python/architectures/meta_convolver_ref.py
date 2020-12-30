# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

"""
Meta convolver using Reference segment for decision making
"""

config = None
weight_norm = False
norm_type = "BatchNorm1d"
activation = "ReLU"


def gen_config():
    global config

    # Select the reference segment item from meta-expert arguments
    config = [
        {
            "type": "SelectArgument",
            "kwargs": {
                "select": 1,
            }
        }
    ]

    # Transpose length <-> channels for convolutional layers
    config += [
        {
            "type": "Transposer",
            "kwargs": {
                "dim0": 1,
                "dim1": 2,
            }
        }
    ]

    # Renormalizing convolutions
    config += NNTools.SingleConvLayer(
        inChannels=5,
        outChannels=16,
        kernelSize=1,
        padding=0,
        dilation=1,
        stride=1,
        groups=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    )  # 16 x 150

    config += [
        NNTools.ResidualBlockConvShortcut(
            inChannels=16,
            outChannels=32,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[2, 1, 2],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 32 x 75

        NNTools.ResidualBlockConvShortcut(
            inChannels=32,
            outChannels=64,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[2, 1, 2],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 64 x 38

        NNTools.ResidualBlockConvShortcut(
            inChannels=64,
            outChannels=128,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[2, 1, 2],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 128 x 19

        NNTools.ResidualBlockConvShortcut(
            inChannels=128,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[2, 1, 2],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 256 x 10
    ]

    config.extend(
        NNTools.terminus(
            256, 3, use_weight_norm=weight_norm, norm_type=norm_type
        )
    )


gen_config()
