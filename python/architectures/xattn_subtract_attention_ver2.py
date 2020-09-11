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

# Renormalizing attention layer
config += [
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 1,
            "n_heads": 8,
            "input_dim": 128,
            "embedding_dim": 128,
            "head_dim": 16,
            "stride": 1,
            "length_first": True,
            "padding": 0,
            "normalized_weighting": False,
        }
    },  # [128, 18]
]

# Dim expand/subsample
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 16,
            "input_dim": 128,
            "embedding_dim": 256,
            "head_dim": 16,
            "stride": 2,
            "length_first": True,
            "padding": 1,
            "normalized_weighting": False,
        }
    }  # [256, 9]
)

# Add two more attention modules for 256 x 9
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 16,
            "input_dim": 256,
            "embedding_dim": 256,
            "head_dim": 16,
            "stride": 1,
            "length_first": True,
            "padding": 1,
            "normalized_weighting": False,
        }
    }  # [256, 9]
)
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 16,
            "input_dim": 256,
            "embedding_dim": 256,
            "head_dim": 16,
            "stride": 1,
            "length_first": True,
            "padding": 1,
            "normalized_weighting": False,
        }
    }  # [256, 9]
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
