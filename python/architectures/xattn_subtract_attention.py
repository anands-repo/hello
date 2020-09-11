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

# Subsample
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
        }
    }  # [256, 9]
)

# Perform additional [256 x 8] attention computations
config += list([
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
        }
    }  # [256, 9]
])

config.append(
    {
        "type": "Transposer",
        "kwargs": {
            "dims": [1, 2]
        }
    }
)

config += NNTools.terminus(256, 1, use_weight_norm=False, dropout=0.1)
