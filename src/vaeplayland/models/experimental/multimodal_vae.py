__all__ = []

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from vaeplayland.models.decoders.simple_decoder import create_decoder_network
from vaeplayland.models.encoders.simple_encoder import SimpleEncoder
from vaeplayland.models.loss import (
    BimodalELBODict,
    compute_cross_entropy,
    compute_gaussian_log_prob,
    compute_kl_div,
)
from vaeplayland.models.vae import BimodalVAE


class ExperimentalMultimodalEncoder(SimpleEncoder):
    def __init__(
        self,
        categorical_shapes: Sequence[tuple[int, int]],
        continuous_shapes: Sequence[int],
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        dropout_rate: float = 0.5,
    ) -> None:
        cat_shapes_1d = [int.__mul__(*shape) for shape in categorical_shapes]
        input_dim = sum(cat_shapes_1d) + sum(continuous_shapes)
        super().__init__(
            input_dim, compress_dims, embedding_dim, activation_fun_name, False, dropout_rate
        )


class ExperimentalMultimodalDecoder(nn.Module):
    def __init__(
        self,
        categorical_shapes: Sequence[tuple[int, int]],
        continuous_shapes: Sequence[int],
        compress_dims: Union[int, Sequence[int]],
        embedding_dim: int,
        activation_fun_name: str = "ReLU",
        dropout_rate: float = 0.5,
    ) -> None:
        """Parameterize p(x|z). Note that x is bimodal, having two distinct
        distributions. The output of this network is split into the parameters
        of each distribution (a Categorical for each categorical dataset and
        Normal distribution for each continuous dataset).

        Args:
            compress_dims:
                Size of each layer.
            embedding_dim:
                Size of latent space.
            split:
                Index of input at which the two data modalities can be split.
            activation_fun_name:
                Name of activation function torch module. Default is "ReLU".
            dropout_rate:
                Fraction of elements to zero between activations. Default is
                0.5.
        """
        super().__init__()
        self.cat_shapes = categorical_shapes
        self.cat_shapes_1d = [int.__mul__(*shape) for shape in categorical_shapes]
        self.con_shapes = continuous_shapes
        self.cat_sz = sum(self.cat_shapes_1d)
        self.con_sz = sum(continuous_shapes)

        output_dim = self.cat_sz + 4 * self.con_sz

        layers = create_decoder_network(
            output_dim, compress_dims, embedding_dim, activation_fun_name, False, dropout_rate
        )
        self.network = nn.Sequential(*layers)

    def reshape_data(
        self, x: torch.Tensor, *, n_chunks: int
    ) -> tuple[list[torch.Tensor], ...]:
        cat_subset, con_subset = torch.tensor_split(x, [self.cat_sz], dim=-1)
        # Split/reshape categorical distributions
        split_indices = np.cumsum(self.cat_shapes_1d)[:-1].tolist()
        cat_subsets = torch.tensor_split(cat_subset, split_indices, dim=-1)
        x_cat = []
        for subset, shape in zip(cat_subsets, self.cat_shapes):
            x_cat.append(subset.reshape(-1, *shape))
        # Split continuous distributions
        con_shapes = self.con_shapes
        if n_chunks > 1:
            con_shapes = np.multiply(con_shapes, n_chunks)
        split_indices = np.cumsum(con_shapes)[:-1].tolist()
        con_subsets = torch.tensor_split(con_subset, split_indices, dim=-1)
        if n_chunks > 1:
            x_con = []
            for subset in con_subsets:
                chunks = [*torch.chunk(subset, n_chunks, dim=-1)]
                for i in range(1, n_chunks, 2):
                    chunks[i] = chunks[i].exp()
                x_con.append(chunks)
        else:
            x_con = con_subsets
        return x_cat, x_con

    def forward(self, batch: torch.Tensor) -> Sequence[Sequence[torch.Tensor]]:
        x_params = self.network(batch)
        return self.reshape_data(x_params, n_chunks=4)


class ExperimentalMultimodalVAE(BimodalVAE):
    def __init__(
        self, encoder: nn.Module, decoder: ExperimentalMultimodalDecoder
    ) -> None:
        self.decoder: ExperimentalMultimodalDecoder
        super().__init__(encoder, decoder)

    @property
    def split(self):
        return self.decoder.cat_sz

    def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        x, *_ = batch
        z_loc, z_scale = self.encode(x)
        z = self.reparameterize(z_loc, z_scale)
        x_params = self.decode(z)
        return *x_params, z, z_loc, z_scale

    def compute_loss(
        self, batch: tuple[torch.Tensor, ...], kl_weight: float
    ) -> BimodalELBODict:
        x, *_ = batch
        x_cat, x_con = self.decoder.reshape_data(x, n_chunks=1)

        x_cat_params, x_con_params, z, qz_loc, qz_scale = self(batch)

        cat_rec_loss = torch.tensor(0.0)
        for i, logits in enumerate(x_cat_params):
            y = torch.argmax(x_cat[i], dim=-1)
            cat_rec_loss += compute_cross_entropy(y, logits).mean()

        con_rec_loss = torch.tensor(0.0)
        for i, (loc1, scale1, loc2, scale2) in enumerate(x_con_params):
            loc = Normal(loc1, scale1).rsample()
            scale = Normal(loc2, scale2).rsample().exp()
            con_rec_loss += compute_gaussian_log_prob(x_con[i], loc, scale).mean()

        rec_loss = cat_rec_loss + con_rec_loss
        reg_loss = compute_kl_div(qz_loc, qz_scale).mean()

        elbo = cat_rec_loss + con_rec_loss - kl_weight * reg_loss
        return {
            "elbo": elbo,
            "reg_loss": reg_loss,
            "rec_loss": rec_loss,
            "cat_rec_loss": cat_rec_loss,
            "con_rec_loss": con_rec_loss,
            "kl_weight": kl_weight,
        }
