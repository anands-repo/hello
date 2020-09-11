import NNTools

config = []

# Transpose so that we switch from b, d, l => b, l, d
config += [
    {
        "type": "Transposer",
        "kwargs": {
            "dims": [1, 2]
        }
    }
]

# Attention for 16-dimensional sequences
config += [
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 6,
            "embedding_dim": 16,
            "head_dim": 4,
            "stride": 1,
            "length_first": True,
            "padding": 1,
            "normalized_weighting": False,
        }
    },
]  # [16, 150]

# Attention for 32 dimensional
config += [
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 16,
            "embedding_dim": 32,
            "head_dim": 8,
            "stride": 2,
            "length_first": True,
            "padding": 1,
            "normalized_weighting": False,
        }
    },  # [32, 75]
]

# Attention for 64 dimensional
config += [
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 32,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 2,
            "length_first": True,
            "padding": 1,
            "normalized_weighting": False,
        }
    },  # [64, 37]
]
