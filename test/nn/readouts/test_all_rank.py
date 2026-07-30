"""Tests for the all-rank copresheaf readout."""

import torch
from torch_geometric.data import Data

from topobench.nn.readouts.all_rank import AllRankReadout


def _sparse_incidence(rows: int, cols: int) -> torch.Tensor:
    indices = torch.stack([torch.arange(min(rows, cols))] * 2)
    values = torch.ones(indices.shape[1])
    return torch.sparse_coo_tensor(indices, values, (rows, cols)).coalesce()


def test_all_rank_graph_readout_pools_every_rank():
    """Graph tasks should predict from concatenated pooled rank features."""
    readout = AllRankReadout(
        hidden_dim=4,
        out_channels=1,
        task_level="graph",
        pooling_type="sum",
        num_cell_dimensions=3,
        readout_name="AllRankReadout",
    )
    model_out = {
        "x_0": torch.randn(5, 4),
        "x_1": torch.randn(4, 4),
        "x_2": torch.randn(2, 4),
    }
    batch = Data(
        batch_0=torch.tensor([0, 0, 0, 1, 1]),
        batch_1=torch.tensor([0, 0, 1, 1]),
        batch_2=torch.tensor([0, 1]),
        incidence_1=_sparse_incidence(5, 4),
        incidence_2=_sparse_incidence(4, 2),
    )

    output = readout(model_out, batch)

    assert output["graph_features"].shape == (2, 12)
    assert output["logits"].shape == (2, 1)
    assert set(output["readout_rank_pool_norms"]) == {"0", "1", "2"}


def test_all_rank_graph_readout_handles_missing_high_rank_cells():
    """Graphs with no high-rank cells should get zero pooled features."""
    readout = AllRankReadout(
        hidden_dim=4,
        out_channels=1,
        task_level="graph",
        pooling_type="sum",
        num_cell_dimensions=3,
        readout_name="AllRankReadout",
    )
    model_out = {
        "x_0": torch.randn(5, 4),
        "x_1": torch.randn(4, 4),
        "x_2": torch.randn(1, 4),
    }
    batch = Data(
        batch_0=torch.tensor([0, 0, 0, 1, 1]),
        batch_1=torch.tensor([0, 0, 1, 1]),
        batch_2=torch.tensor([0]),
        incidence_1=_sparse_incidence(5, 4),
        incidence_2=_sparse_incidence(4, 1),
    )

    output = readout(model_out, batch)

    assert output["graph_features"].shape == (2, 12)
    assert output["logits"].shape == (2, 1)
    torch.testing.assert_close(
        output["graph_features"][1, 8:],
        torch.zeros(4),
    )


def test_all_rank_node_readout_uses_rank_zero_logits():
    """Node tasks should retain the standard propagate-down behavior."""
    readout = AllRankReadout(
        hidden_dim=4,
        out_channels=3,
        task_level="node",
        pooling_type="sum",
        num_cell_dimensions=2,
        readout_name="AllRankReadout",
    )
    model_out = {
        "x_0": torch.randn(5, 4),
        "x_1": torch.randn(4, 4),
    }
    batch = Data(
        batch_0=torch.zeros(5, dtype=torch.long),
        incidence_1=_sparse_incidence(5, 4),
    )

    output = readout(model_out, batch)

    assert output["logits"].shape == (5, 3)
    assert "graph_features" not in output
