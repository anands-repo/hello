# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

# Re-normalizing convolution
config = NNTools.SingleConvLayer(
    inChannels=128,
    outChannels=128,
    kernelSize=1,
    padding=0,
    dilation=1,
    stride=1,
);  # 128 x 18

config += [
    NNTools.ResidualBlockConvShortcut(
        inChannels=128,
        outChannels=256,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[2, 1, 2],
        dilations=[1, 1, 1],
    ),  # 256 x 9

    NNTools.ResidualBlockFTShortcut(
        inChannels=256,
        outChannels=256,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 256 x 9

    NNTools.ResidualBlockFTShortcut(
        inChannels=256,
        outChannels=256,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 256 x 9
];

config.extend(
    NNTools.terminus(256, 3)
);
