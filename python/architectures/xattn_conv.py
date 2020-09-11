import NNTools

weight_norm = False
config = None
norm_type = "BatchNorm1d"

def gen_config():
    global config

    # Concatenate allele-level signal and site-level signal and reduce
    config = [
        {
            "type": "ConcatenateChannels",
            "kwargs": dict()
        }
    ]  # [256, 36]

    # Perform additional feature-extraction at dim=256
    config += [
        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=256,
            outChannels=256,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
        ),
    ]  # [256, 36]

    # Expand dim to 512, subsample to 9 channel length
    # and perform additional feature extraction
    config += [
        NNTools.ResidualBlockConvShortcut(
            inChannels=256,
            outChannels=512,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1, 1],
            strides=[2, 1, 2],
            use_weight_norm=weight_norm,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=512,
            outChannels=512,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
        ),

        NNTools.ResidualBlockFTShortcut(
            inChannels=512,
            outChannels=512,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1],
            strides=[1, 1],
            use_weight_norm=weight_norm,
        ),
    ]  # [512, 9]

    # Final dim expansion to 1024
    config += [
        NNTools.ResidualBlockConvShortcut(
            inChannels=512,
            outChannels=1024,
            kernelSizes=[3, 3],
            paddings=[1, 1],
            dilations=[1, 1, 1],
            strides=[2, 1, 2],
            use_weight_norm=weight_norm,
        )
    ]

    # Predict
    config.extend(
        NNTools.terminus(1024, 1, use_weight_norm=weight_norm)
    )

gen_config()
