import NNTools

# Renormalizing layer with layer normalization
config = [
    {
        "type": "Linear",
        "kwargs": {
            "in_features": 64,
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
    }
]

config += [
    # Dim expand/subsample to 128 dimensional
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 128,
            "head_dim": 16,
            "padding": 1,
            "stride": 2,
            "use_mean": True,
        }
    },

    # Additional attention modules at 128 x 19
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 128,
            "embedding_dim": 128,
            "head_dim": 16,
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
            "input_dim": 128,
            "embedding_dim": 128,
            "head_dim": 16,
            "padding": 1,
            "stride": 1,
            "use_mean": True,
        }
    },
]  # [128, 19]
