# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

weight_norm = False
config = None

def gen_config():
    global config

    # Re-normalizing convolution
    # After read convolver, we have added evidence from many reads
    # So, we need to normalize the results of the addition
    config = NNTools.SingleConvLayer(
        inChannels=64,
        outChannels=64,
        kernelSize=1,
        padding=0,
        dilation=1,
        stride=1,
        groups=1,
        use_weight_norm=weight_norm,
    );  # 64 x 36

    config += [
        NNTools.ResidualBlockConvShortcut(
            inChannels=64,
            outChannels=128,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[2, 1, 2],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
        ),  # 128 x 18

        NNTools.ResidualBlockFTShortcut(
            inChannels=128,
            outChannels=128,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[1, 1, 1],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
        ),  # 128 x 18

        NNTools.ResidualBlockFTShortcut(
            inChannels=128,
            outChannels=128,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            strides=[1, 1, 1],
            dilations=[1, 1, 1],
            use_weight_norm=weight_norm,
        ),  # 128 x 18
    ];

gen_config()
