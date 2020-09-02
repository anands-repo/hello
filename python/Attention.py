import torch


class PositionalEncoding(torch.nn.Module):
    """
    This is based on the "Attention is all you need" paper, and the
    Transformer-XL paper
    Code reference: https://github.com/kimiyoung/transformer-xl
    """
    def __init__(self, feature_size):
        """
        :param feature_size: int
            Size of the feature into which to add the
            positional encoding
        """
        super().__init__()
        self.feature_size = feature_size
        stepper = 10000 ** (torch.arange(0, self.feature_size, 2).float() / self.feature_size)
        stepper = 1 / stepper
        stepper = torch.unsqueeze(stepper, dim=0)   # [1, feature_size//2]
        self.register_buffer("stepper", stepper)

    def forward(self, indices):
        """
        :param indices: torch.LongTensor
            Indices for which we should create a positional embedding. Indices
            are of dimensionality [length]

        :return: torch.Tensor
            Tensor of size [length, self.feature_size]
        """
        indices_prep = torch.unsqueeze(indices, dim=1).repeat(1, self.feature_size // 2)
        sinusoid_args = indices_prep.float() * self.stepper  # [length, self.feature_size // 2]
        sinusoid_args = sinusoid_args.float()
        return torch.cat([sinusoid_args.sin(), sinusoid_args.cos()], dim=1)


class MultiheadAttention(torch.nn.Module):
    """
    Performs bidirectional multi-head attention. This is based on the
    Attention is all you need paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self,
        head_dim,
        n_heads,
        embedding_dim,
        add_positional=False,
    ):
        """
        :param input_dim: int
            Input dimensions

        :param n_heads: int
            Number of attention heads

        :param embedding_dim: int
            Size of embedding dimensions

        :param add_positional: bool
            Add positional embeddings to input
        """
        super().__init__()

        assert(
            n_heads * head_dim == embedding_dim
        ), "Dimensionality of each head doesn't sum up to total embedding size"

        self.add_positional = add_positional

        if self.add_positional:
            self.positional = PositionalEncoding(embedding_dim)

        self.W_q = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_k = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_o = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.layer_norm_attn = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
        )
        self.layer_norm_linear = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self._init_params()

    def _init_params(self):
        torch.nn.init.kaiming_uniform_(self.W_q)
        torch.nn.init.kaiming_uniform_(self.W_k)
        torch.nn.init.kaiming_uniform_(self.W_v)
        torch.nn.init.kaiming_uniform_(self.W_o)

    def forward(self, sequences):
        """
        Sequences in the format [batch, length, #channels]
        """
        # Add positional embeddings if needed
        if self.add_positional:
            if sequences.is_cuda:
                indices = torch.arange(0, sequences.shape[1], device=sequences.get_device())
            else:
                indices = torch.arange(0, sequences.shape[1])

            positional_embeddings = torch.unsqueeze(
                self.positional(indices), dim = 0)

            sequences += positional_embeddings

        W_qkv = torch.cat((self.W_q, self.W_k, self.W_v), dim=1)  # [e, 3e]
        qkv = torch.matmul(sequences, W_qkv)  # [l, b, 3e]
        q, k, v = torch.split(qkv, split_size_or_sections=self.W_q.shape[0], dim=2)

        # Split into n heads - reshape value so that head id
        # appears earlier
        def split_into_heads(signal):
            signal = signal.view(
                signal.shape[0],
                signal.shape[1],
                self.n_heads,
                signal.shape[2] // self.n_heads
            )
            return torch.transpose(
                signal,
                1, 2
            )  # [b, n_heads, l, hdim]

        q = split_into_heads(q)
        k = split_into_heads(k)
        v = split_into_heads(v)  # [b, n_heads, l, h_dim]
        qxk = torch.matmul(
            q,  # [b, n_heads, l, h_dim]
            torch.transpose(k, 2, 3)  # [b, n_heads, h_dim, l]
        ) / (self.head_dim ** 0.5)  # [b, n_heads, l, l]
        qxkxv = torch.matmul(qxk, v) # Shape [b, n_heads, l, head_dim]

        # Concatenate all heads: just reshape so that heads line-up
        # one after the other in the last two dims
        def concatenate_heads(signal):
            signal = torch.transpose(signal, 1, 2)  # [b, l, n_heads, head_dim]
            signal = signal.reshape(signal.shape[0], signal.shape[1], -1)
            return signal

        attn_output = torch.matmul(concatenate_heads(qxkxv), self.W_o)
        multihead_attention = self.layer_norm_attn(sequences + attn_output)
        linear_results = self.linear(multihead_attention)
        output_embedding = self.layer_norm_linear(linear_results + multihead_attention)

        return output_embedding


class HelloEncoder(torch.nn.Module):
    """
    Encoder network for HELLO
    """
    def __init__(
        self,
        n_layers,
        n_heads,
        input_dim,
        embedding_dim,
    ):
        super().__init__()
        self.attention_layers = torch.nn.Sequential(
            *[
                MultiheadAttention(
                    head_dim=embedding_dim // n_heads,
                    n_heads=n_heads,
                    embedding_dim=embedding_dim,
                    add_positional=True if i == 0 else False,
                )
                for i in range(n_layers)
            ]
        )

        self.embedding_layer = torch.nn.Linear(
            input_dim, embedding_dim
        )

    def forward(self, tensor):
        # Reshape tensor from [b, c, l] => [b, l, c]
        tensor = torch.transpose(tensor, 1, 2)

        # Perform embedding
        seq = self.embedding_layer(tensor)  # [b, l, e]

        # Pass through attention layers
        result = self.attention_layers(seq)  # [b, l, e]

        # Reshape to obtain appropriate shaping
        return torch.transpose(result, 1, 2)  # [b, e, l]


# Set Hello's encoder here
torch.nn.HelloEncoder = HelloEncoder
