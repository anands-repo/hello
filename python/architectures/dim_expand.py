import NNTools

# Perform simple dimensionality expansion
config = NNTools.SingleConvLayer(
    inChannels=64,
    outChannels=128,
    kernelSize=1,
    padding=0,
    dilation=1,
    stride=1,
)