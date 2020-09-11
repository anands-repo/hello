import NNTools

config = [NNTools.ResidualBlockConvShortcut(
    inChannels=128,
    outChannels=256,
    kernelSizes=[3, 3],
    paddings=[1, 1],
    dilations=[1, 1, 1],
    strides=[2, 1, 2]
)]  # [256, 9]

config.extend(
    NNTools.terminus(256, 3)
)
