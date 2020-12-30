# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

weight_norm = False
config = None
norm_type = "BatchNorm1d"
activation = "ReLU"

def gen_config():
    global config

    # *** ResNet stem *** #
    config = NNTools.SingleConvLayer(
        inChannels=6,
        outChannels=16,
        kernelSize=3,
        padding=0,
        dilation=1,
        stride=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    );  # 16 x 148

    config += NNTools.SingleConvLayer(
        inChannels=16,
        outChannels=16,
        kernelSize=3,
        padding=0,
        dilation=1,
        stride=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    );  # 16 x 146

    config += NNTools.SingleConvLayer(
        inChannels=16,
        outChannels=32,
        kernelSize=3,
        padding=0,
        dilation=1,
        stride=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    );  # 32 x 144

    config.append({
        "type": "MaxPool1d",
        "kwargs": {
            "kernel_size": 3,
            "stride": 2,
            "padding": 0,
        }
    });  # 32 x 72
    # *** ResNet stem end *** #

    # Residual convolutions
    config += [
        NNTools.ResidualBlockFTShortcut(
            inChannels=32,
            outChannels=32,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 32 x 72

        NNTools.ResidualBlockFTShortcut(
            inChannels=32,
            outChannels=32,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 32 x 72

        NNTools.ResidualBlockFTShortcut(
            inChannels=32,
            outChannels=32,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 32 x 72

        NNTools.ResidualBlockConvShortcut(
            inChannels=32,
            outChannels=64,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1, 1],
            strides=[2, 1, 2],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 64 x 36

        NNTools.ResidualBlockFTShortcut(
            inChannels=64,
            outChannels=64,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 64 x 36

        NNTools.ResidualBlockFTShortcut(
            inChannels=64,
            outChannels=64,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 64 x 36

        NNTools.ResidualBlockFTShortcut(
            inChannels=64,
            outChannels=64,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 64 x 36
    ]

gen_config()
