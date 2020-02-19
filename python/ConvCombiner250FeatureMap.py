import NNTools

# Groups = 2 convolutions
config = NNTools.SingleConvLayer(
    inChannels=128 * 2,
    outChannels=128 * 2,
    kernelSize=3,
    padding=1,
    dilation=1,
    stride=1,
    groups=2
);  # 256 x 32

config += [
    NNTools.ResidualBlockConvShortcut(
        inChannels=128 * 2,
        outChannels=128 * 4,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1, 1],
        strides=[2, 1, 2],
        groups=[2, 2, 2],
    ),  # 512 x 16

    NNTools.ResidualBlockFTShortcut(
        inChannels=512,
        outChannels=512,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 512 x 16

    NNTools.ResidualBlockFTShortcut(
        inChannels=512,
        outChannels=512,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
    ),  # 512 x 16
]

# Create output layer
config += NNTools.SingleConvLayer(
    inChannels=128 * 4,
    outChannels=128 * 1,
    kernelSize=1,
    padding=0,
    dilation=1,
    stride=1,
);  # 128 x 16
