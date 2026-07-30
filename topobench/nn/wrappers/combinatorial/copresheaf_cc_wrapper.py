"""TopoBench wrapper for the combinatorial copresheaf backbone."""

from topobench.nn.wrappers.base import AbstractWrapper


class CopresheafCCWrapper(AbstractWrapper):
    """Extract rank features and neighborhoods from a TopoBench batch."""

    def forward(self, batch):
        """Run higher-order copresheaf message passing on ``batch``."""
        features = {rank: batch[f"x_{rank}"] for rank in self.backbone.ranks}
        connectivities = {
            name: batch[name] for name in self.backbone.neighborhoods
        }
        output = self.backbone(features, connectivities)

        model_out = {f"x_{rank}": value for rank, value in output.items()}
        model_out.update(
            {f"batch_{rank}": batch[f"batch_{rank}"] for rank in output}
        )
        model_out["labels"] = batch.y
        return model_out
