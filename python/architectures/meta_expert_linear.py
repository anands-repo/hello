import NNTools

config = NNTools.SingleLinearLayer(
    in_features=128,
    out_features=64
)

config += NNTools.SingleLinearLayer(
    in_features=64,
    out_features=3
)