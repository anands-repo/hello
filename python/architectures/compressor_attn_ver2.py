import NNTools

config = [
    # Renormalizing attention
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
            "normalized_weighting": False,
        }
    },  # [64, 37]

    # Add two more attention layers for [64 x 37]
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
            "normalized_weighting": False,
        }
    },  # [64, 37]
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
            "normalized_weighting": False,
        }
    },  # [64, 37]

    # Dim expand/subsample to 128 dimensional
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
            "normalized_weighting": False,
        }
    },  # [128, 18]

    # Additional attention modules at 128 x 18
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
            "normalized_weighting": False,
        }
    },  # [128, 18]
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
            "normalized_weighting": False,
        }
    },  # [128, 18]
]
