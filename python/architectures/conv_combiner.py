# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

config = None
weight_norm = False
norm_type = "BatchNorm1d"
activation = "ReLU"


def gen_config():
    global config

    config = [{
        "type": "ConcatenateChannels",
        "kwargs": dict(),
    }]

    # Create intermediate sum
    config += NNTools.SingleConvLayer(
        inChannels=128 * 2,
        outChannels=128 * 4,
        kernelSize=3,
        padding=1,
        dilation=1,
        stride=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    )

    # Create output layer
    config += NNTools.SingleConvLayer(
        inChannels=128 * 4,
        outChannels=128 * 1,
        kernelSize=1,
        padding=0,
        dilation=1,
        stride=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    )


gen_config()
