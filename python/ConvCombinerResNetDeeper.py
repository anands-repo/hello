# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

# Create intermediate sum
config = NNTools.SingleConvLayer(
    inChannels=128 * 2,
    outChannels=128 * 4,
    kernelSize=3,
    padding=1,
    dilation=1,
    stride=1,
);

# Create output layer
config += NNTools.SingleConvLayer(
    inChannels=128 * 4,
    outChannels=128 * 1,
    kernelSize=1,
    padding=0,
    dilation=1,
    stride=1,
);
