# © 2019 University of Illinois Board of Trustees.  All rights reserved
import NNTools

weight_norm = False
config = None
norm_type = "BatchNorm1d"
activation = "ReLU"

def gen_config():
    global config

    # First stage: select compressed_features_allele and
    # expanded_frames_for_site1
    config = [
        {
            "type": "Fork",
            "kwargs": {
                "net_args": [
                    [{
                        "type": "Noop",
                        "kwargs": dict(),
                    }], 
                    [{
                        "type": "SelectArgument",
                        "kwargs": {
                            "select": 1
                        }
                    }]
                ]
            }
        }
    ]

    # Perform 2 * compressed_features_allele - expanded_frames_for_site1
    config.append(
        {
            "type": "LinearCombination",
            "kwargs": {
                "coefficients": [2, -1]
            },
        }
    )  # [256, 18]

    config += NNTools.SingleConvLayer(
        inChannels=256,
        outChannels=256,
        kernelSize=1,
        padding=0,
        dilation=1,
        stride=1,
        groups=1,
        use_weight_norm=weight_norm,
        norm_type=norm_type,
        activation=activation,
    )  # [256, 18]

    config += [
        NNTools.ResidualBlockConvShortcut(
            inChannels=256,
            outChannels=512,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[2, 1, 2],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=512,
            outChannels=512,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=512,
            outChannels=512,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
            norm_type=norm_type,
            activation=activation,
        )
    ]  # [512, 9]

    config += NNTools.terminus(512, 1, use_weight_norm=weight_norm)

gen_config()
