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
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 128,
            "head_dim": 16,
            "stride": 2,
            "length_first": True,
            "padding": 1,
        }
    },  # [128, 19]

    # Additional attention modules at 128 x 19
    {
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 128,
            "embedding_dim": 128,
            "head_dim": 16,
            "stride": 1,
            "length_first": True,
            "padding": 1,
        }
    },  # [128, 19]
    {
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 128,
            "embedding_dim": 128,
            "head_dim": 16,
            "stride": 1,
            "length_first": True,
            "padding": 1,
        }
    },  # [128, 19]
]
