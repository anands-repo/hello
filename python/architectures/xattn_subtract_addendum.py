# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

weight_norm = False
config = None
norm_type = "BatchNorm1d"
activation = "ReLU"

def gen_config():
    global config

    config = [
        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),
    ]  # [256, 9]

    config += NNTools.terminus(256, 1, use_weight_norm=weight_norm)

gen_config()
