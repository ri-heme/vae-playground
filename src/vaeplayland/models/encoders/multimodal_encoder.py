__all__ = ["MultimodalEncoder"]

from typing import Sequence, Union

from vaeplayland.models.encoders.simple_encoder import SimpleEncoder


class MultimodalEncoder(SimpleEncoder):
    """Parameterize q(z|x). Note that x is multimodal, having a discrete and a
    continuous part.

    Args:
        disc_dims:
            Dimensions (excluding batch dimension) of each categorical dataset,
            each shape expected to be a two-element tuple (num. features times
            cardinality).
        cont_dims:
            Dimensions (excluding batch dimension) of each continuous dataset.
        compress_dims:
            Size of each layer.
        embedding_dim:
            Size of latent space.
        activation_fun_name:
            Name of activation function torch module. Default is "ReLU".
        batch_norm:
            Apply batch normalization.
        dropout_rate:
            Fraction of elements to zero between activations. Default is 0.5.
    """

    def __init__(
        self,
        disc_dims: Sequence[tuple[int, int]],
        cont_dims: Sequence[int],
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        batch_norm: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        disc_dims_1d = [int.__mul__(*shape) for shape in disc_dims]
        input_dim = sum(disc_dims_1d) + sum(cont_dims)
        super().__init__(
            input_dim,
            compress_dims,
            embedding_dim,
            activation_fun_name,
            batch_norm,
            dropout_rate,
        )
