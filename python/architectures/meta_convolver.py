import NNTools

config = None
weight_norm = False
norm_type = "BatchNorm1d"
activation = "ReLU"


def gen_config():
    global config

    # Select an item
    config = [{
        "type": "SelectArgument",
        "kwargs": {
            "select": 0,
        }
    }]

    # Renormalizing convolutions
    config += NNTools.SingleConvLayer(
        inChannels=128,
        outChannels=128,
        kernelSize=1,
        padding=0,
        dilation=1,
        stride=1,
        groups=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    )  # 128 x 16

    config += [
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
        ),  # 256 x 8

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[1, 1, 1],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 256 x 8

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[1, 1, 1],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),  # 256 x 8
    ]

    config.extend(
        NNTools.terminus(
            256, 3, use_weight_norm=weight_norm, norm_type=norm_type
        )
    )


gen_config()
