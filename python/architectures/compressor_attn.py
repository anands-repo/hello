import NNTools

config = []

# Renormalizing attention
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 1,
            "n_heads": 8,
            "input_dim": 64,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 1,
            "length_first": True,
            "padding": 0,
        }
    }  # [64, 36]
)

# Subsample
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
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
    }  # [128, 18]
)

# Perform additional [128 x 18] attention computations
config += list([
    {
        "type": "SingleWindowedAttentionLayer",
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
    }  # [128, 18]
])
