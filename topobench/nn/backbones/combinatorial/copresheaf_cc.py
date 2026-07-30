"""Higher-order copresheaf message passing on combinatorial complexes."""

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from topobench.nn.layers.copresheaf import (
    CopresheafRoute,
    HigherOrderCopresheafLayer,
)


class CopresheafCC(nn.Module):
    r"""A rank-aware Copresheaf Topological Neural Network.

    The configured TopoBench neighborhoods induce directed graphs on cells.
    Every route gets its own learned maps :math:`\rho_{y\to x}` and message
    function, while messages arriving at the same rank are combined before a
    rank-specific update. This realizes the message-passing form in
    Definitions 9--10 of Copresheaf Topological Neural Networks
    (arXiv:2505.21251).

    Parameters
    ----------
    in_channels : int
        Feature dimension shared by all input ranks.
    hidden_channels : int
        Hidden stalk feature dimension.
    out_channels : int
        Output feature dimension for every represented rank.
    neighborhoods : sequence of str
        TopoBench neighborhood names, for example ``up_incidence-0``.
    map_type : str or mapping
        One map family for all routes or a neighborhood-to-family mapping.
    message_schedule : sequence of str, optional
        If provided, apply neighborhood updates sequentially in this order
        inside each layer instead of aggregating all incoming routes
        simultaneously.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        neighborhoods: Sequence[str],
        num_layers: int = 3,
        heads: int = 1,
        stalk_dimension: int | None = None,
        map_type: str | Mapping[str, str] = "shared_local",
        aggr: str = "mean",
        neighborhood_aggr: str = "sum",
        message_type: str = "identity",
        update_type: str = "residual",
        dropout: float = 0.0,
        map_kwargs: Mapping | None = None,
        num_heads: int | None = None,
        message_gate_init: float | None = None,
        route_self_bias: float = 1.0,
        message_schedule: Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if num_heads is not None:
            if heads != 1:
                raise ValueError("pass heads or num_heads, not both")
            heads = num_heads

        self.neighborhoods = list(neighborhoods)
        self.message_schedule = (
            tuple(message_schedule) if message_schedule is not None else None
        )
        self.routes = [
            CopresheafRoute.from_neighborhood(name)
            for name in self.neighborhoods
        ]
        self.ranks = sorted(
            {route.source_rank for route in self.routes}
            | {route.target_rank for route in self.routes}
        )
        self.input_projections = nn.ModuleDict(
            {
                str(rank): nn.Linear(in_channels, hidden_channels)
                for rank in self.ranks
            }
        )
        self.layers = nn.ModuleList(
            [
                HigherOrderCopresheafLayer(
                    channels=hidden_channels,
                    neighborhoods=self.neighborhoods,
                    heads=heads,
                    stalk_dimension=stalk_dimension,
                    map_type=map_type,
                    aggr=aggr,
                    neighborhood_aggr=neighborhood_aggr,
                    message_type=message_type,
                    update_type=update_type,
                    dropout=dropout,
                    map_kwargs=map_kwargs,
                    message_gate_init=message_gate_init,
                    route_self_bias=route_self_bias,
                    message_schedule=message_schedule,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_projections = nn.ModuleDict(
            {
                str(rank): nn.Linear(hidden_channels, out_channels)
                for rank in self.ranks
            }
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: Mapping[int, torch.Tensor],
        connectivities: Mapping[str, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        """Encode features on all configured cell ranks."""
        missing_ranks = set(self.ranks) - set(features)
        if missing_ranks:
            raise KeyError(
                f"missing features for ranks {sorted(missing_ranks)}"
            )

        hidden = {
            rank: self.dropout(
                self.activation(
                    self.input_projections[str(rank)](features[rank])
                )
            )
            for rank in self.ranks
        }
        for layer in self.layers:
            hidden = layer(hidden, connectivities)
            hidden = {
                rank: self.dropout(self.activation(value))
                for rank, value in hidden.items()
            }
        return {
            rank: self.output_projections[str(rank)](value)
            for rank, value in hidden.items()
        }
