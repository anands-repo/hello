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

# Dim expand using linear layer with layer norm
config += [
    {
        "type": "Linear",
        "kwargs": {
            "in_features": 6,
            "out_features": 16,
        }
    },

    {
        "type": "LayerNorm",
        "kwargs": {
            "normalized_shape": 16,
        }
    },

    {
        "type": "ReLU",
        "kwargs": dict()
    }
]

# Add global positional embeddings
config += [
    {
        "type": "PositionalEmbedding",
        "kwargs": {
            "feature_size": 16
        }
    }
]

# Bring down the sequence size from 150 -> 38. I believe
# Attention modules can do this very well since they have a stronger grasp
# of positional dynamics than Convolutional Neural Networks
config += [{
    "type": "WindowedAttentionLayerCustomSkip",
    "kwargs": {
        "window_size": 7,
        "n_heads": 4,
        "input_dim": 16,
        "embedding_dim": 32,
        "head_dim": 8,
        "stride": 4,
        "length_first": True,
        "padding": 3,
    }
}]

# Attention for 32 dimensional data
config += [
    {
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 32,
            "embedding_dim": 32,
            "head_dim": 8,
            "stride": 2,
            "length_first": True,
            "padding": 1,
        }
    },  # [32, 38]

    {
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 32,
            "embedding_dim": 32,
            "head_dim": 8,
            "stride": 2,
            "length_first": True,
            "padding": 1,
        }
    },  # [32, 38]
]

# Attention for 64 dimensional data
config += [
    {
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 32,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 1,
            "length_first": True,
            "padding": 1,
        }
    },  # [64, 38]

    {
        "type": "WindowedAttentionLayerCustomSkip",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 1,
            "length_first": True,
            "padding": 1,
        }
    },  # [64, 38]
]
