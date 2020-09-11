import torch
import logging


class FullAttentionLayer(torch.nn.Module):
    """
    Performs bidirectional multi-head attention. This is based on the
    Attention is all you need paper: https://arxiv.org/abs/1706.03762
    """
    def __init__(
        self,
        input_dim,
        head_dim,
        n_heads,
        embedding_dim,
        activation="ReLU",
        activation_kwargs=dict(),
    ):
        """
        :param input_dim: int
            Input dimensions

        :param head_dim: int
            Dimension of an attention head

        :param n_heads: int
            Number of attention heads

        :param embedding_dim: int
            Size of embedding dimensions

        :param activation: str
            Activation function to use

        :param activation_kwargs: str
            Arguments to be used for the activation function
        """
        super().__init__()

        assert(
            n_heads * head_dim == embedding_dim
        ), "Dimensionalities of heads do not sum up to total embedding size"

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        # Embed input data
        if input_dim != embedding_dim:
            self.input_layer = torch.nn.Linear(input_dim, embedding_dim)
        else:
            self.input_layer = None

        # Attention parameters
        self.W_q = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_k = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_o = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))

        # Output layer
        self.output_layer = torch.nn.Linear(embedding_dim, embedding_dim)

        # LayerNorms
        self.layer_norm_attn = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm_linear = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        # Activation layer
        self.activation = getattr(torch.nn, activation)(**activation_kwargs)

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
        # Copy over input
        orig_input = sequences

        # Perform embedding operation
        sequences = self.input_layer(sequences) if (self.input_layer is not None) else sequences

        # Extract attention parameters
        W_qkv = torch.cat((self.W_q, self.W_k, self.W_v), dim=1)  # [e, 3e]
        qkv = torch.matmul(sequences, W_qkv)  # [l, b, 3e]
        q, k, v = torch.split(qkv, split_size_or_sections=self.W_q.shape[0], dim=-1)

        # Split into n heads - reshape value so that head id appears earlier
        def split_into_heads(signal):
            signal = signal.view(
                signal.shape[0],  # b
                signal.shape[1],  # l
                self.n_heads,  # n_heads
                signal.shape[2] // self.n_heads  # head_dim
            )
            return torch.transpose(
                signal,
                1, 2
            )  # [b, n_heads, l, hdim]

        # Perform attention computations
        q, k, v = (split_into_heads(i) for i in [q, k, v])
        qxk = torch.matmul(
            q,  # [b, n_heads, l, h_dim]
            torch.transpose(k, 2, 3)  # [b, n_heads, h_dim, l]
        ) / (self.head_dim ** 0.5)  # [b, n_heads, l, l]
        qxk = torch.softmax(qxk, dim=-1)
        qxkxv = torch.matmul(qxk, v) # Shape [b, n_heads, l, head_dim]

        # Concatenate all heads: just reshape so that the heads line-up
        # one after the other in the last two dimensions
        def concatenate_heads(signal):
            signal = torch.transpose(signal, 1, 2)  # [b, l, n_heads, head_dim]
            signal = signal.reshape(signal.shape[0], signal.shape[1], -1)
            return signal

        attn_output = concatenate_heads(qxkxv)

        # Multihead attention outputs
        multihead_attention = self.layer_norm_attn(
            sequences + torch.matmul(attn_output, self.W_o)
        )

        # Layer outputs
        linear_results = self.output_layer(multihead_attention)
        output_embedding = self.activation(
            self.layer_norm_linear(linear_results + multihead_attention)
        )

        return output_embedding


torch.nn.FullAttentionLayer = FullAttentionLayer


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


class WindowedAttentionLayer(torch.nn.Module):
    """
    Performs attention-like computations within windowed-copies of the data. This allows
    us to use this block like a convolutional neural network
    """

    def __init__(
        self,
        window_size,
        n_heads,
        input_dim,
        embedding_dim,
        head_dim,
        padding,
        stride,
        use_mean=False,
        activation="ReLU",
        activation_kwargs=dict(),
    ):
        """
        :param window_size: int
            Size of a window

        :param n_heads: int
            Number of attention heads

        :param input_dim: int
            Dimension of the input

        :param embedding_dim: int
            Dimension of the output

        :param head_dim: int
            Dimension of each head

        :param padding: int
            Number of padding zeros to apply to both sides

        :param stride: int
            Step size

        :param use_mean: bool
            Use mean for collating results for a single window

        :param activation: str
            Activation function name

        :param activation_kwargs: dict
            Arguments to activation function
        """
        super().__init__()

        self.window_size = window_size
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.padding = padding
        self.stride = stride
        self.use_mean = use_mean

        assert(n_heads * head_dim == embedding_dim), "Dimensionality of heads do not sum up to embedding"

        # Embed input data if needed
        if self.input_dim != self.embedding_dim:
            self.input_layer = torch.nn.Linear(input_dim, embedding_dim)
        else:
            self.input_layer = None

        # Attention parameters
        self.W_q = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_k = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_o = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))

        # Output linear layer
        self.output_layer = torch.nn.Linear(embedding_dim, embedding_dim)

        # Have learnt positional embeddings add to the data
        self.positional_embeddings = torch.nn.Parameter(torch.Tensor(
            1, 1, self.window_size, self.embedding_dim)
        )

        # Initialize layer norms
        self.layer_norm_attn = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm_output = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        # Activation function
        self.activation = getattr(torch.nn, activation)(**activation_kwargs)

        # Initialize all parameters
        self._init_params()

    def _init_params(self):
        torch.nn.init.kaiming_uniform_(self.W_q)
        torch.nn.init.kaiming_uniform_(self.W_k)
        torch.nn.init.kaiming_uniform_(self.W_v)
        torch.nn.init.kaiming_uniform_(self.W_o)
        torch.nn.init.kaiming_uniform_(self.positional_embeddings)

    def forward(self, tensors):
        """
        :param tensors: torch.Tensor
            Tensors are of shape [batch, length, #dim]
        """
        embedded = self.input_layer(tensors) if (self.input_layer is not None) else tensors

        # Zero pad
        if self.padding > 0:
            padded = torch.nn.functional.pad(embedded, (0, 0, self.padding, self.padding))
        else:
            padded = embedded

        # Unfold the data
        unfolded = padded.unfold(
            dimension=1, size=self.window_size, step=self.stride
        )  # [b, l, d, L]

        # Transpose so that we have L x d on the last two dimensions
        transposed = torch.transpose(unfolded, -1, -2)  # [b, l, L, d]

        # Add positional embeddings
        transposed = transposed + self.positional_embeddings

        # Initialize attention parameters
        W_qkv = torch.cat((self.W_q, self.W_k, self.W_v), dim=1)
        qkv = torch.matmul(transposed, W_qkv)  # [b, l, L, 3d]
        q, k, v = torch.split(qkv, split_size_or_sections=self.W_q.shape[0], dim=-1)

        # Split into n heads - reshape value so that head id appears earlier
        def split_into_heads(signal):
            signal = signal.view(
                signal.shape[0],  # b
                signal.shape[1],  # l
                signal.shape[2],  # L
                self.n_heads,  # n_heads
                signal.shape[3] // self.n_heads,  # head_dim
            )
            return torch.transpose(
                signal,
                2, 3
            )  # [b, l, n_heads, L, head_dim]

        # Perform attention computations
        q, k, v = (split_into_heads(x) for x in [q, k, v])
        qxk = torch.matmul(
            q,  # [b, l, n_heads, L, head_dim]
            torch.transpose(k, 3, 4)  # [b, l, n_heads, head_dim, L]
        ) / (self.head_dim ** 0.5)
        qxk = torch.softmax(qxk, dim=-1)
        qxkxv = torch.matmul(qxk, v)  # [b, l, n_heads, L, head_dim]

        # Add/Average over all of L for each window
        # result shape is [b, l, n_heads, head_dim]
        if hasattr(self, 'use_mean') and self.use_mean:
            attn_result_pre = torch.mean(qxkxv, dim=3)
        else:
            attn_result_pre = torch.sum(qxkxv, dim=3)

        # Combine all heads together
        attn_result_pre = attn_result_pre.reshape(
            attn_result_pre.shape[0], attn_result_pre.shape[1], -1
        )  # [b, l, d]

        # Determine attention skip connection
        if self.stride == 1:
            attn_skip_connection = embedded
        else:
            attn_skip_connection = embedded[:, ::self.stride]

        # Determine multi-head attention result
        attn_result = self.layer_norm_attn(
            attn_skip_connection + torch.matmul(attn_result_pre, self.W_o)
        )

        # Perform output layer operations
        return self.activation(
            self.layer_norm_output(
                self.output_layer(attn_result) + attn_result
            )
        )


torch.nn.WindowedAttentionLayer = WindowedAttentionLayer


class SingleWindowedAttentionLayer(torch.nn.Module):
    """
    Wrapper class for windowed attention layers
    """
    def __init__(
        self,
        window_size,
        n_heads,
        input_dim,
        embedding_dim,
        head_dim,
        padding,
        stride,
        length_first,
        activation="ReLU",
        activation_kwargs=dict(),
        normalized_weighting=True,
    ):
        super().__init__()
        self.attention_layer = WindowedAttentionLayer(
            window_size=window_size,
            n_heads=n_heads,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            head_dim=head_dim,
            padding=padding,
            stride=stride,
            normalized_weighting=normalized_weighting,
        )
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.activation = getattr(torch.nn, activation)(**activation_kwargs)
        self.length_first = length_first
    
    def forward(self, tensor):
        """
        :param tensor: torch.Tensor
            [batch, length, #dim] shaped tensor

        :return: torch.Tensor
            Output tensor from layer
        """
        if not self.length_first:
            tensor = torch.transpose(tensor, 1, 2)  # re-shape for the sake of attention

        attn = self.attention_layer(tensor)

        # Keeping this in forward allows for compatibility
        skip_connection = (
            (self.attention_layer.input_dim == self.attention_layer.embedding_dim) and
            (self.attention_layer.stride == 1)
        )

        if skip_connection:
            pre = self.activation(self.layer_norm(attn + tensor))
        else:
            pre = self.activation(self.layer_norm(attn))

        if not self.length_first:
            result = torch.transpose(pre, 1, 2)  # re-shape for the sake of transferring to the next layer
        else:
            result = pre

        return result


torch.nn.SingleWindowedAttentionLayer = SingleWindowedAttentionLayer


class CustomSkip(torch.nn.Module):
    """
    Actually, we could simply use an AttentionLayer here,
    but such an attention layer would simply perform two linear
    operations on the input first by the linear layer, and next
    by the value matrix multiplication. This may be equivalently
    captured into a single conv1d layer.
    """
    def __init__(self, input_dim, output_dim, stride):
        super().__init__()
        self.skipper = torch.nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            stride=stride,
            kernel_size=1
        )

    def forward(self, tensor):
        """
        Note: custom skip always assumes that the input
        is in length-first format
        """
        tensor = torch.transpose(tensor, 1, 2)  # [b, dim, l]
        pre = self.skipper(tensor)
        result = torch.transpose(pre, 1, 2)  # [b, l, dim]
        return result


class WindowedAttentionLayerCustomSkip(torch.nn.Module):
    def __init__(
        self,
        window_size,
        n_heads,
        input_dim,
        embedding_dim,
        head_dim,
        padding,
        stride,
        length_first,
        activation="ReLU",
        activation_kwargs=dict(),
        normalized_weighting=True,
    ):
        super().__init__()
        self.attention_layer = WindowedAttentionLayer(
            window_size=window_size,
            n_heads=n_heads,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            head_dim=head_dim,
            padding=padding,
            stride=stride,
            normalized_weighting=normalized_weighting,
        )
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.activation = getattr(torch.nn, activation)(**activation_kwargs)
        self.length_first = length_first

        # If input and output cannot be combined into a skip connection
        # directly, then we want to make a custom skip connection for this
        # purpose
        if stride != 1 or embedding_dim != input_dim:
            self.skip_connector = CustomSkip(
                input_dim,
                embedding_dim,
                stride,
            )
        else:
            self.skip_connector = None

    def forward(self, tensor):
        """
        :param tensor: torch.Tensor
            [batch, length, #dim tensor]

        :return torch.Tensor
            Output tensor from layer
        """
        tensor = torch.transpose(tensor, 1, 2) if not self.length_first else tensor
        attn = self.attention_layer(tensor)
        skip_value = self.skip_connector(tensor) if self.skip_connector else tensor
        pre = self.activation(self.layer_norm(attn + skip_value))
        result = torch.transpose(pre, 1, 2) if not self.length_first else pre
        return result


class PositionalEmbedding(torch.nn.Module):
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

    def forward(self, signal):
        """
        :param signal: torch.Tensor
            Input data of shape [batch, length, #features]

        :return: torch.Tensor
            Input data with added positional embeddings
        """
        # Produce [L, 1] input
        if signal.is_cuda:
            indices = torch.arange(signal.shape[1], device=signal.get_device()).view(signal.shape[1], 1)
        else:
            indices = torch.arange(signal.shape[1]).view(signal.shape[1], 1)

        # Produce L x feature_size // 2 indices
        indices_prep = indices.repeat(1, self.feature_size // 2)

        # Convert that to sinusoidal arguments
        sinusoid_args = indices_prep.float() * self.stepper  # [length, self.feature_size // 2]
        sinusoid_args = sinusoid_args.float()

        # Create positional embeddings
        positional_embeddings = torch.cat([sinusoid_args.sin(), sinusoid_args.cos()], dim=1)

        # Add to input data
        return signal + torch.unsqueeze(positional_embeddings, dim=0)


class LearntEmbeddings(torch.nn.Module):
    """
    Add learnt positional embeddings to an incoming tensor of fixed size
    """
    def __init__(self, feature_size, feature_length):
        super().__init__()
        self.feature_size = feature_size
        self.feature_length = feature_length
        self.embeddings = torch.nn.Parameter(
            torch.Tensor(1, feature_length, feature_size)
        )
        torch.nn.init.kaiming_uniform_(self.embeddings)

    def forward(self, tensor):
        try:
            return tensor + self.embeddings
        except Exception:
            logging.error("Tensor sizes mismatch: tensor = %s, embeddings = %s" % (str(tensor.shape), str(self.embeddings.shape)))
            raise ValueError()


torch.nn.WindowedAttentionLayerCustomSkip = WindowedAttentionLayerCustomSkip
torch.nn.PositionalEmbedding = PositionalEmbedding
torch.nn.LearntEmbeddings = LearntEmbeddings


class BrickedAttention(torch.nn.Module):
    """
    Performs split-windowed attention over non-overlapping windows
    """
    def __init__(
        self,
        window_size,
        n_heads,
        input_dim,
        embedding_dim,
        head_dim,
        stride,
        activation="ReLU",
        activation_kwargs=dict(),
        use_positional=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.stride = stride
        self.use_positional = use_positional

        assert(n_heads * head_dim == embedding_dim), "Dimensionality of heads do not sum up to embedding"

        # Embed input data if needed
        if self.input_dim != self.embedding_dim:
            self.input_layer = torch.nn.Linear(input_dim, embedding_dim)
        else:
            self.input_layer = None

        # Attention parameters
        self.W_q = torch.nn.Parameter(torch.Tensor(2, embedding_dim, embedding_dim))
        self.W_k = torch.nn.Parameter(torch.Tensor(2, embedding_dim, embedding_dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(2, embedding_dim, embedding_dim))
        self.W_o = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))

        # Output linear layer
        self.output_layer = torch.nn.Linear(embedding_dim, embedding_dim)

        # Have learnt positional embeddings add to the data
        self.positional_embeddings = torch.nn.Parameter(torch.Tensor(
            1, 1, self.window_size, self.embedding_dim)
        )

        # Initialize layer norms
        self.layer_norm_attn = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm_output = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        # Activation function
        self.activation = getattr(torch.nn, activation)(**activation_kwargs)

        # Initialize all parameters
        self._init_params()

    def _init_params(self):
        torch.nn.init.kaiming_uniform_(self.W_q)
        torch.nn.init.kaiming_uniform_(self.W_k)
        torch.nn.init.kaiming_uniform_(self.W_v)
        torch.nn.init.kaiming_uniform_(self.W_o)
        torch.nn.init.kaiming_uniform_(self.positional_embeddings)

    def _pad_to_divisible_size(self, tensor):
        to_pad = self.window_size - tensor.shape[1] % self.window_size

        if to_pad == 0:
            return tensor, 0
        else:
            return torch.nn.functional.pad(tensor, (0, 0, 0, to_pad)), to_pad

    def _unpad(self, tensor, padding):
        if padding == 0:
            return tensor
        else:
            return tensor[:, :-padding]

    def _attn_computations(self, tensor, brick_id):
        # Split tensor into bricks
        bricks = tensor.reshape(
            tensor.shape[0],
            tensor.shape[1] // self.window_size,
            self.window_size,
            tensor.shape[2]
        )  # [b, #bricks, brick-length, embedding dimension]

        if hasattr(self, 'use_positional') and self.use_positional:
            bricks = bricks + self.positional_embeddings

        # Concatenate for single matmul computation
        W_qkv = torch.cat(
            (self.W_q[brick_id], self.W_k[brick_id], self.W_v[brick_id]),
            dim=1
        )

        # Perform matmul and split to obtain attention parameters
        qkv = torch.matmul(bricks, W_qkv)  # [b, nbricks, w, d]
        q, k, v = torch.split(qkv, split_size_or_sections=self.W_q[brick_id].shape[0], dim=-1)  # [b, nbricks, w, d]

        # Split into heads
        def split_into_heads(signal):
            signal = signal.reshape(
                signal.shape[0],  # b
                signal.shape[1],  # nbricks
                signal.shape[2],  # window
                self.n_heads,  # n_heads
                -1,  # head_dim
            )
            return torch.transpose(signal, 2, 3)  # [b, nbricks, n_heads, window, head_dim]

        # Perform attention computations
        q, k, v = (split_into_heads(i) for i in [q, k, v])
        qxk = torch.softmax(
            torch.matmul(q, torch.transpose(k, 3, 4)) / (self.head_dim ** 0.5),
            dim=-1
        )
        qxkxv = torch.matmul(qxk, v)  # [b, nbricks, n_heads, window, head_dim]

        heads_concatenated_pre = torch.transpose(qxkxv, 2, 3)  # [b, nbricks, window, n_heads, head_dim]
        heads_concatenated = heads_concatenated_pre.reshape(
            heads_concatenated_pre.shape[0],  # b
            heads_concatenated_pre.shape[1],  # nbricks
            heads_concatenated_pre.shape[2],  # window-size
            -1,  # embedding size
        )

        bricks_concatenated = heads_concatenated.reshape(
            heads_concatenated.shape[0],  # b
            heads_concatenated.shape[1] * heads_concatenated.shape[2],  # l
            heads_concatenated.shape[-1]  # e
        )

        return bricks_concatenated

    def forward(self, tensor):
        # Pad sequence as needed for perfect divisibility
        orig_seq_length = tensor.shape[1]
        tensor_padded, padding = self._pad_to_divisible_size(tensor)
        new_seq_length = tensor_padded.shape[1]

        # Embed the tensors
        embedded = self.input_layer(tensor_padded) if (self.input_layer is not None) else tensor_padded

        # Obtain first layer of bricks
        first_brick_layer = self._attn_computations(embedded, 0)

        # To obtain second layer of bricks, pad the signal so that the two layers of bricks are staggered
        left_zero_pad = self.window_size // 2
        right_zero_pad = self.window_size - left_zero_pad
        padded_embed = torch.nn.functional.pad(embedded, (0, 0, left_zero_pad, right_zero_pad))
        second_brick_layer = self._attn_computations(padded_embed, 1)[:, left_zero_pad: -right_zero_pad]

        # Obtain multihead attention
        multihead_attention = self.layer_norm_attn(
            embedded + torch.matmul((first_brick_layer + second_brick_layer) / 2, self.W_o)
        )

        # Obtain final linear result
        pre_result = self.activation(
            self.layer_norm_output(
                self.output_layer(multihead_attention) + multihead_attention
            )
        )

        # Unpad the result
        pre_result = self._unpad(pre_result, padding)

        # Stride the result if needed
        return pre_result[:, ::self.stride]


torch.nn.BrickedAttention = BrickedAttention


class SingleBrickedAttention(torch.nn.Module):
    """
    Non-overlapping windowed attention without overlapping bricks
    """
    def __init__(
        self,
        window_size,
        n_heads,
        input_dim,
        embedding_dim,
        head_dim,
        phase,
        stride=1,
        activation="ReLU",
        activation_kwargs=dict(),
        use_positional=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.stride = stride
        self.phase = phase
        self.use_positional = use_positional

        assert(n_heads * head_dim == embedding_dim), "Dimensionality of heads do not sum up to embedding"

        # Embed input data if needed
        if self.input_dim != self.embedding_dim:
            self.input_layer = torch.nn.Linear(input_dim, embedding_dim)
        else:
            self.input_layer = None

        # Attention parameters
        self.W_q = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_k = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_v = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))
        self.W_o = torch.nn.Parameter(torch.Tensor(embedding_dim, embedding_dim))

        # Output linear layer
        self.output_layer = torch.nn.Linear(embedding_dim, embedding_dim)

        # Have learnt positional embeddings add to the data
        self.positional_embeddings = torch.nn.Parameter(torch.Tensor(
            1, 1, self.window_size, self.embedding_dim)
        )

        # Initialize layer norms
        self.layer_norm_attn = torch.nn.LayerNorm(normalized_shape=embedding_dim)
        self.layer_norm_output = torch.nn.LayerNorm(normalized_shape=embedding_dim)

        # Activation function
        self.activation = getattr(torch.nn, activation)(**activation_kwargs)

        # Initialize all parameters
        self._init_params()

    def _init_params(self):
        torch.nn.init.kaiming_uniform_(self.W_q)
        torch.nn.init.kaiming_uniform_(self.W_k)
        torch.nn.init.kaiming_uniform_(self.W_v)
        torch.nn.init.kaiming_uniform_(self.W_o)
        torch.nn.init.kaiming_uniform_(self.positional_embeddings)

    def _pad_to_divisible_size(self, tensor):
        to_pad = self.window_size - tensor.shape[1] % self.window_size

        if to_pad == 0:
            return tensor, 0
        else:
            return torch.nn.functional.pad(tensor, (0, 0, 0, to_pad)), to_pad

    def _unpad(self, tensor, padding):
        if padding == 0:
            return tensor
        else:
            return tensor[:, :-padding]

    def _attn_computations(self, tensor):
        # Split tensor into bricks
        bricks = tensor.reshape(
            tensor.shape[0],
            tensor.shape[1] // self.window_size,
            self.window_size,
            tensor.shape[2]
        )  # [b, #bricks, brick-length, embedding dimension]

        if self.use_positional:
            bricks = bricks + self.positional_embeddings

        # Concatenate for single matmul computation
        W_qkv = torch.cat(
            (self.W_q, self.W_k, self.W_v),
            dim=1
        )

        # Perform matmul and split to obtain attention parameters
        qkv = torch.matmul(bricks, W_qkv)  # [b, nbricks, w, d]
        q, k, v = torch.split(qkv, split_size_or_sections=self.W_q.shape[0], dim=-1)  # [b, nbricks, w, d]

        # Split into heads
        def split_into_heads(signal):
            signal = signal.reshape(
                signal.shape[0],  # b
                signal.shape[1],  # nbricks
                signal.shape[2],  # window
                self.n_heads,  # n_heads
                -1,  # head_dim
            )
            return torch.transpose(signal, 2, 3)  # [b, nbricks, n_heads, window, head_dim]

        # Perform attention computations
        q, k, v = (split_into_heads(i) for i in [q, k, v])
        qxk = torch.softmax(
            torch.matmul(q, torch.transpose(k, 3, 4)) / (self.head_dim ** 0.5),
            dim=-1
        )
        qxkxv = torch.matmul(qxk, v)  # [b, nbricks, n_heads, window, head_dim]

        heads_concatenated_pre = torch.transpose(qxkxv, 2, 3)  # [b, nbricks, window, n_heads, head_dim]
        heads_concatenated = heads_concatenated_pre.reshape(
            heads_concatenated_pre.shape[0],  # b
            heads_concatenated_pre.shape[1],  # nbricks
            heads_concatenated_pre.shape[2],  # window-size
            -1,  # embedding size
        )

        bricks_concatenated = heads_concatenated.reshape(
            heads_concatenated.shape[0],  # b
            heads_concatenated.shape[1] * heads_concatenated.shape[2],  # l
            heads_concatenated.shape[-1]  # e
        )

        return bricks_concatenated

    def forward(self, tensor):
        # Pad sequence as needed for perfect divisibility
        tensor_padded, padding = self._pad_to_divisible_size(tensor)

        # Embed the tensors
        embedded = self.input_layer(tensor_padded) if (self.input_layer is not None) else tensor_padded

        # Pad so that we can phase the attention computation accordingly
        if self.phase != 0:
            right_zero_pad = self.phase
            left_zero_pad = self.window_size - right_zero_pad
            pembedded = torch.nn.functional.pad(
                embedded, (0, 0, left_zero_pad, right_zero_pad)
            )
        else:
            pembedded = embedded

        brick_layer_pre = self._attn_computations(pembedded)

        if self.phase != 0:
            brick_layer = brick_layer_pre[:, left_zero_pad: -right_zero_pad]
        else:
            brick_layer = brick_layer_pre

        # Obtain multihead attention
        multihead_attention = self.layer_norm_attn(
            embedded + torch.matmul(brick_layer, self.W_o)
        )

        # Obtain final linear result
        pre_result = self.activation(
            self.layer_norm_output(
                self.output_layer(multihead_attention) + multihead_attention
            )
        )

        # Unpad the result
        pre_result = self._unpad(pre_result, padding)

        # Stride the result if needed
        return pre_result[:, ::self.stride]


torch.nn.SingleBrickedAttention = SingleBrickedAttention


class PoolAttention(torch.nn.Module):
    """
    A pooling module that works with attention modules above
    """
    def __init__(self, stride, window_size, padding):
        super().__init__()
        self.stride = stride
        self.window_size = window_size
        self.padding = padding
        self.pooler = torch.nn.AvgPool1d(
            kernel_size=self.window_size,
            stride=self.stride,
            padding=self.padding,
        )

    def forward(self, tensor):
        return torch.transpose(
            self.pooler(
                torch.transpose(tensor, 1, 2)
            ), 1, 2
        )


torch.nn.PoolAttention = PoolAttention
