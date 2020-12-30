# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

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
);  # 64 x 64

config += [
    NNTools.ResidualBlockConvShortcut(
        inChannels=64,
        outChannels=128,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[2, 1, 2],
        dilations=[1, 1, 1],
    ),  # 128 x 32

    NNTools.ResidualBlockFTShortcut(
        inChannels=128,
        outChannels=128,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 128 x 32

    NNTools.ResidualBlockFTShortcut(
        inChannels=128,
        outChannels=128,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 128 x 32

    NNTools.ResidualBlockFTShortcut(
        inChannels=128,
        outChannels=128,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 128 x 32
];
