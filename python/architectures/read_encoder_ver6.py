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

# Expand dimensions - this is in order to ensure that
# LayerNorm has a larger sample size to obtain sample statistics
# which are more stable
config += [
    {
        "type": "Linear",
        "kwargs": {
            "in_features": 6,
            "out_features": 64,
        }
    },

    {
        "type": "LayerNorm",
        "kwargs": {
            "normalized_shape": 64,
        }
    },

    {
        "type": "ReLU",
        "kwargs": dict()
    },
]

# Bring down sequence size from 150 -> 38 in two steps
config += [
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 5,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "padding": 2,
            "stride": 2,
            "use_mean": True,
        }
    },

    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 5,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "padding": 2,
            "stride": 2,
            "use_mean": True,
        }
    },
]  # [32, 38]

# Attention for 64 dimensional data
config += [
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "padding": 1,
            "stride": 1,
            "use_mean": True,
        }
    },

    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "padding": 1,
            "stride": 1,
            "use_mean": True,
        }
    },
]
