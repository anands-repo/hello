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

# Expand dimensions and add positional embeddings
config += [
    {
        "type": "Linear",
        "kwargs": {
            "in_features": 6,
            "out_features": 32,
        }
    },

    {
        "type": "LayerNorm",
        "kwargs": {
            "normalized_shape": 32,
        }
    },

    {
        "type": "ReLU",
        "kwargs": dict()
    },

    {
        "type": "PositionalEmbedding",
        "kwargs": {
            "feature_size": 32
        }
    }
]

# Bring down sequence size from 150 -> 38 in two steps
# while expanding dimensionality to 64
config += [
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 5,
            "n_heads": 4,
            "input_dim": 32,
            "embedding_dim": 32,
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
            "input_dim": 32,
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
