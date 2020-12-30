# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

# *** ResNet stem *** #
config = NNTools.SingleConvLayer(
    inChannels=6,
    outChannels=16,
    kernelSize=3,
    padding=0,
    dilation=1,
    stride=1,
);  # 16 x 248

config += NNTools.SingleConvLayer(
    inChannels=16,
    outChannels=16,
    kernelSize=3,
    padding=0,
    dilation=1,
    stride=1,
);  # 16 x 246

config += NNTools.SingleConvLayer(
    inChannels=16,
    outChannels=32,
    kernelSize=3,
    padding=0,
    dilation=1,
    stride=1,
);  # 32 x 244

config.append({
    "type": "MaxPool1d",
    "kwargs": {
        "kernel_size": 3,
        "stride": 2,
        "padding": 0,
    }
});  # 32 x 122
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
    ),  # 32 x 122

    NNTools.ResidualBlockFTShortcut(
        inChannels=32,
        outChannels=32,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[1, 1],
    ),  # 32 x 122

    NNTools.ResidualBlockFTShortcut(
        inChannels=32,
        outChannels=32,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[1, 1],
    ),  # 32 x 122

    NNTools.ResidualBlockConvShortcut(
        inChannels=32,
        outChannels=64,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1, 1],
        strides=[2, 1, 2],
    ),  # 64 x 61

    NNTools.ResidualBlockFTShortcut(
        inChannels=64,
        outChannels=64,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[1, 1],
    ),  # 64 x 61

    NNTools.ResidualBlockFTShortcut(
        inChannels=64,
        outChannels=64,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[1, 1],
    ),  # 64 x 61

    NNTools.ResidualBlockFTShortcut(
        inChannels=64,
        outChannels=64,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[1, 1],
    ),  # 64 x 61

]
