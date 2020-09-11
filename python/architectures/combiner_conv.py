import NNTools

config = [{
    "type": "ConcatenateChannels",
    "kwargs": dict()
}]

config += NNTools.SingleConvLayer(
    inChannels=128,
    outChannels=256,
    kernelSize=3,
    padding=1,
    dilation=1,
    stride=1,
)

config += NNTools.SingleConvLayer(
    inChannels=256,
    outChannels=64,
    kernelSize=1,
    padding=0,
    dilation=1,
    stride=1,
)
