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
]  # [128, 150]

# Initial dimensionality expansion
config += [
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 1,
            "n_heads": 4,
            "input_dim": 6,
            "embedding_dim": 16,
            "head_dim": 4,
            "stride": 1,
            "length_first": True,
            "padding": 0,
        }
    },  # [16, 150]
]

# Convolutions for 16-dimensional sequences
config += [
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 7,
            "n_heads": 4,
            "input_dim": 16,
            "embedding_dim": 16,
            "head_dim": 4,
            "stride": 1,
            "length_first": True,
            "padding": 3,
        }
    },]  # [16, 150]

config += list([
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 16,
            "embedding_dim": 16,
            "head_dim": 4,
            "stride": 1,
            "length_first": True,
            "padding": 1,
        }
    }  # [16, 150]
])

# Subsample
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 16,
            "embedding_dim": 32,
            "head_dim": 8,
            "stride": 2,
            "length_first": True,
            "padding": 1,
        }
    }  # [32, 75]
)

# Perform 32x32 computations
config += list([
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 4,
            "input_dim": 32,
            "embedding_dim": 32,
            "head_dim": 8,
            "stride": 1,
            "length_first": True,
            "padding": 1,
        }
    }  # [32, 75]
])

# Subsample
config.append(
    {
        "type": "SingleWindowedAttentionLayer",
        "kwargs": {
            "window_size": 3,
            "n_heads": 8,
            "input_dim": 32,
            "embedding_dim": 64,
            "head_dim": 8,
            "stride": 2,
            "length_first": True,
            "padding": 1,
        }
    }  # [64, 36]
)

# Perform 64x35 computations
config += list([
    {
        "type": "SingleWindowedAttentionLayer",
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
    }  # [64, 36]
])
