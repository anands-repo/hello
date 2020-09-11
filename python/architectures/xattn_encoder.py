import NNTools

# We are accepting two tensor sets each of size
# 64, 50, so first concatenate
config = [
    {
        "type": "ConcatenateChannels",
        "kwargs": dict(),
    }
]

# Perform self-attention computations
config += [
    {
        "type": "HelloEncoder",
        "kwargs": {
            "n_layers": 8,
            "n_heads": 8,
            "input_dim": 256,
            "embedding_dim": 256,
        }
    }
]  # [128, 36]

# Subsample
config += [
    NNTools.ResidualBlockConvShortcut(
        inChannels=256,
        outChannels=256,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[2, 1, 2],
    ),  # [256, 18]

    NNTools.ResidualBlockConvShortcut(
        inChannels=256,
        outChannels=512,
        kernelSizes=[3, 3],
        paddings=[1, 1],
        dilations=[1, 1],
        strides=[2, 1, 2],
    )  # [512, 9]
]

# Terminate
config += NNTools.terminus(512, 1)
