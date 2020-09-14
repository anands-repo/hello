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

# Dim expand using layer norm
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

# Bricked attention
config.append(
    {
        "type": "BrickedAttention",
        "kwargs": {
            "window_size": 6,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 1,
            "use_positional": True,
        }
    }
)  # [64, 150]

# Subsampling
config.append(
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "padding": 1,
            "stride": 2,
            "use_mean": True,
        }
    },
)  # [64, 75]

# Bricked attention
config.append(
    {
        "type": "BrickedAttention",
        "kwargs": {
            "window_size": 6,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 1,
            "use_positional": True,
        }
    }
)  # [64, 75]

# Subsampling
config.append(
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "padding": 1,
            "stride": 2,
            "use_mean": True,
        }
    },
)  # [64, 38]
