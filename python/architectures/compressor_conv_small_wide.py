# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

weight_norm = False
config = None
norm_type = "BatchNorm1d"

def gen_config():
    global config

    config = NNTools.SingleConvLayer(
        inChannels=128,
        outChannels=128,
        kernelSize=1,
        padding=0,
        dilation=1,
        stride=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
    )  # [128, 36]

    config += [
        NNTools.ResidualBlockConvShortcut(
            inChannels=128,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[2, 1, 2],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
        ),  # [256, 18]

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
        ),  # [256, 18]

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
        ),  # [256, 18]
    ]

gen_config()
