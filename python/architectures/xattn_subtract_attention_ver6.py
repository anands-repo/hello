import NNTools
import torch


class Mean(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return torch.mean(tensor, dim=self.dim)


torch.nn.Mean = Mean

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

# Add positional embeddings again for full attention computation
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

# Final attention layer with max stride to average everything
config.append(
    {
        "type": "Mean",
        "kwargs": {
            "dim": 1
        }
    }
)   # Take the mean across all items

config.extend([
    {
        "type": "LayerNorm",
        "kwargs": {
            "normalized_shape": 256
        }
    },

    {
        "type": "Linear",
        "kwargs": {
            "in_features": 256,
            "out_features": 1,
        }
    }
])
