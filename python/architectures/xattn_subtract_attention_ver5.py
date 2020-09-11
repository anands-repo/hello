import NNTools

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
)  # [128, 18]

# Renormalizing linear layer with layer-normalization
config += [
    {
        "type": "Linear",
        "kwargs": {
            "in_features": 128,
            "out_features": 128,
        }
    },

    {
        "type": "LayerNorm",
        "kwargs": {
            "normalized_shape": 128,
        }
    },

    {
        "type": "ReLU",
        "kwargs": dict()
    }
]

# Dim expand/subsample
config.append(
    {
        "type": "WindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 16,
            "input_dim": 128,
            "embedding_dim": 256,
            "head_dim": 16,
            "padding": 1,
            "stride": 2,
            "use_mean": True,
        }
    },
)

# Add positional embeddings here again
config.append(
    {
        "type": "LearntEmbeddings",
        "kwargs": {
            "feature_size": 256,
            "feature_length": 10,
        }
    }
)

# At a dimensionality of 9, there is no difference in computations
# between doing a windowed attention with window-size of 3 and a full
# global attention computation. So we will simply perform a full global
# attention computation
config.append(
    {
        "type": "FullAttentionLayer",
        "kwargs": {
            "input_dim": 256,
            "head_dim": 16,
            "n_heads": 16,
            "embedding_dim": 256
        }
    }
)
config.append(
    {
        "type": "FullAttentionLayer",
        "kwargs": {
            "input_dim": 256,
            "head_dim": 16,
            "n_heads": 16,
            "embedding_dim": 256
        }
    }
)

config.append(
    {
        "type": "Transposer",
        "kwargs": {
            "dims": [1, 2]
        }
    }
)

config += NNTools.terminus(256, 1, use_weight_norm=False, norm_type="Noop")
