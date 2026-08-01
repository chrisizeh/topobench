"""Triangle-clique lifting of graphs to combinatorial complexes."""

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CombinatorialComplex

from topobench.data.utils import (
    get_combinatorial_complex_connectivity,
    select_neighborhoods_of_interest,
)
from topobench.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class GraphTriangleCliqueCCLifting(Graph2CombinatorialLifting):
    r"""Lift graph triangles as explicit rank-2 combinatorial cells.

    Rank 0 is the input nodes, rank 1 is the input graph edges, and rank 2
    contains one cell for every graph triangle. The lift gives the submitted
    copresheaf model a deterministic topological domain while keeping the
    preprocessing small and easy to audit.

    Parameters
    ----------
    latent_rank3_cells : int, optional
        Number of latent rank-3 cells to add on top of the triangle
        complex. A value of ``0`` (default) disables rank-3 cells.
    latent_rank3_memberships : int, optional
        Number of nearest latent regions each node is softly
        assigned to when building rank-3 memberships (default:
        ``1``).
    latent_rank3_iterations : int, optional
        Number of k-means-style refinement steps used to place the
        latent region centers (default: ``8``).
    latent_rank3_assignment_temperature : float, optional
        Temperature used to turn negative center distances into
        softmax membership weights (default: ``1.0``).
    latent_rank3_features : str, optional
        Descriptors used to build latent regions: one of
        ``"features"``, ``"structure"``, or ``"features_structure"``
        (default: ``"features_structure"``).
    latent_rank3_direct_node_routes : bool, optional
        If ``True``, also add direct rank-3 to rank-0 incidence
        tensors for ablations (default: ``False``).
    **kwargs : optional
        Additional arguments for the base lifting class.
    """

    def __init__(
        self,
        latent_rank3_cells: int = 0,
        latent_rank3_memberships: int = 1,
        latent_rank3_iterations: int = 8,
        latent_rank3_assignment_temperature: float = 1.0,
        latent_rank3_features: str = "features_structure",
        latent_rank3_direct_node_routes: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if latent_rank3_cells < 0:
            raise ValueError("latent_rank3_cells must be non-negative")
        if latent_rank3_memberships < 1:
            raise ValueError("latent_rank3_memberships must be at least one")
        if latent_rank3_iterations < 1:
            raise ValueError("latent_rank3_iterations must be at least one")
        if latent_rank3_assignment_temperature <= 0:
            raise ValueError(
                "latent_rank3_assignment_temperature must be positive"
            )
        if latent_rank3_features not in {
            "features",
            "structure",
            "features_structure",
        }:
            raise ValueError(
                "latent_rank3_features must be one of: features, "
                "structure, features_structure"
            )
        self.latent_rank3_cells = int(latent_rank3_cells)
        self.latent_rank3_memberships = int(latent_rank3_memberships)
        self.latent_rank3_iterations = int(latent_rank3_iterations)
        self.latent_rank3_assignment_temperature = float(
            latent_rank3_assignment_temperature
        )
        self.latent_rank3_features = latent_rank3_features
        self.latent_rank3_direct_node_routes = bool(
            latent_rank3_direct_node_routes
        )

    def lift_topology(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data | dict:
        """Lift one graph into a triangle combinatorial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data, expected to carry undirected edges
            and node features ``data.x``.

        Returns
        -------
        dict
            The lifted topology: rank-0/1/2 (and optional rank-3)
            connectivity, node features, and structural features.
        """
        graph = self._generate_graph_from_data(data)
        if graph.is_directed():
            raise ValueError(
                "GraphTriangleCliqueCCLifting expects undirected graphs"
            )

        combinatorial_complex = CombinatorialComplex(graph)
        for triangle in self._triangles(graph):
            combinatorial_complex.add_cell(triangle, rank=2)

        lifted_topology = self._connectivity(combinatorial_complex)
        if self.latent_rank3_cells:
            lifted_topology.update(
                self._latent_rank3_connectivity(
                    data,
                    graph,
                    lifted_topology,
                    dtype=data.x.dtype,
                    device=data.x.device,
                )
            )
        if self.neighborhoods is not None:
            lifted_topology.update(
                select_neighborhoods_of_interest(
                    lifted_topology, self.neighborhoods
                )
            )
        lifted_topology["x_0"] = data.x
        lifted_topology.update(
            self._structural_features(
                lifted_topology,
                dtype=data.x.dtype,
                device=data.x.device,
            )
        )
        return lifted_topology

    @staticmethod
    def _triangles(graph: nx.Graph) -> list[tuple[int, int, int]]:
        """Return sorted node triples that form graph triangles.

        Parameters
        ----------
        graph : nx.Graph
            The input undirected graph.

        Returns
        -------
        list of tuple of int
            Sorted node triples, one per triangle (3-clique) found
            in ``graph``.
        """
        triangles = set()
        for clique in nx.enumerate_all_cliques(graph):
            if len(clique) < 3:
                continue
            if len(clique) > 3:
                break
            triangles.add(tuple(sorted(clique)))
        return sorted(triangles)

    def _connectivity(self, combinatorial_complex) -> dict[str, torch.Tensor]:
        """Return the raw rank-0/1/2 connectivity of a complex.

        Parameters
        ----------
        combinatorial_complex : CombinatorialComplex
            The complex whose connectivity tensors are computed.

        Returns
        -------
        dict
            Connectivity tensors (incidence, adjacency, and
            Laplacian matrices) for ranks ``0`` through
            ``self.complex_dim``.
        """
        connectivity = get_combinatorial_complex_connectivity(
            combinatorial_complex,
            self.complex_dim,
            neighborhoods=None,
        )
        return connectivity

    def _latent_rank3_connectivity(
        self,
        data: torch_geometric.data.Data,
        graph: nx.Graph,
        connectivity: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Create K latent rank-3 cells from a KNN-style node partition.

        The latent cells are not inserted as graph nodes or triangle cliques.
        They attach to rank-2 triangles; long ``0 <-> 3`` neighborhoods are
        then induced by the incidence chain unless direct node-region routes
        are explicitly requested for ablations.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data, used to build node descriptors.
        graph : nx.Graph
            The input undirected graph, used to build node
            descriptors.
        connectivity : dict
            Rank-0/1/2 connectivity tensors, including
            ``"incidence_1"``, ``"incidence_2"``, and ``"shape"``.
        dtype : torch.dtype
            Dtype used for the returned rank-3 tensors.
        device : torch.device
            Device used for the returned rank-3 tensors.

        Returns
        -------
        dict
            Rank-3 tensors: updated ``"shape"``, ``"incidence_3"``,
            empty ``"adjacency_3"``, ``"coadjacency_3"``,
            ``"down_laplacian_3"``, ``"up_laplacian_3"``, and
            ``"hodge_laplacian_3"``, plus direct node-region
            incidence tensors when
            ``latent_rank3_direct_node_routes`` is enabled.
        """
        if self.complex_dim < 3:
            raise ValueError(
                "latent_rank3_cells requires complex_dim >= 3 so rank-3 "
                "features, batches, and structural channels are represented"
            )
        num_rank2 = int(connectivity["incidence_2"].size(1))
        num_rank3 = self.latent_rank3_cells

        membership = self._latent_node_membership(data, graph).to(
            device=device, dtype=dtype
        )
        incidence_3 = self._dense_to_sparse(
            self._triangle_membership(connectivity, membership, num_rank2)
        )

        shape = list(connectivity["shape"])
        if len(shape) < 4:
            shape.extend([0] * (4 - len(shape)))
        shape[3] = num_rank3

        output = {
            "shape": torch.tensor(shape, dtype=torch.long),
            "incidence_3": incidence_3,
            "adjacency_3": self._empty_sparse(
                num_rank3, num_rank3, dtype=dtype, device=device
            ),
            "coadjacency_3": self._empty_sparse(
                num_rank3, num_rank3, dtype=dtype, device=device
            ),
            "down_laplacian_3": self._empty_sparse(
                num_rank3, num_rank3, dtype=dtype, device=device
            ),
            "up_laplacian_3": self._empty_sparse(
                num_rank3, num_rank3, dtype=dtype, device=device
            ),
            "hodge_laplacian_3": self._empty_sparse(
                num_rank3, num_rank3, dtype=dtype, device=device
            ),
        }
        if self.latent_rank3_direct_node_routes:
            output["3-up_incidence-0"] = self._dense_to_sparse(membership.t())
            output["3-down_incidence-3"] = self._dense_to_sparse(membership)
        return output

    def _latent_node_membership(
        self, data: torch_geometric.data.Data, graph: nx.Graph
    ) -> torch.Tensor:
        """Return node-to-latent-region membership weights.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data, used to build node descriptors.
        graph : nx.Graph
            The input undirected graph, used to build node
            descriptors.

        Returns
        -------
        torch.Tensor
            A dense ``(num_nodes, latent_rank3_cells)`` tensor whose
            rows hold softmax weights over each node's nearest
            latent regions.
        """
        num_nodes = int(data.x.size(0))
        num_rank3 = self.latent_rank3_cells
        if num_nodes == 0:
            return data.x.new_zeros((0, num_rank3))

        descriptors = self._node_descriptors(data, graph)
        centers = self._initial_centers(descriptors, num_rank3)
        for _ in range(self.latent_rank3_iterations):
            distances = torch.cdist(descriptors, centers)
            assignments = distances.argmin(dim=1)
            updated = centers.clone()
            for idx in range(num_rank3):
                mask = assignments == idx
                if bool(mask.any()):
                    updated[idx] = descriptors[mask].mean(dim=0)
            centers = updated

        distances = torch.cdist(descriptors, centers)
        memberships = min(
            self.latent_rank3_memberships,
            num_rank3,
            num_nodes,
        )
        nearest = distances.topk(memberships, largest=False, dim=1)
        weights = torch.softmax(
            -nearest.values / self.latent_rank3_assignment_temperature,
            dim=1,
        )
        output = descriptors.new_zeros((num_nodes, num_rank3))
        output.scatter_(1, nearest.indices, weights)
        return output

    def _node_descriptors(
        self, data: torch_geometric.data.Data, graph: nx.Graph
    ) -> torch.Tensor:
        """Build unsupervised descriptors for latent KNN regions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data; ``data.x`` is used when
            ``latent_rank3_features`` includes node features.
        graph : nx.Graph
            The input undirected graph; node degree and clustering
            coefficient are used when ``latent_rank3_features``
            includes structural descriptors.

        Returns
        -------
        torch.Tensor
            A ``(num_nodes, num_descriptors)`` tensor of per-node
            descriptors, standardized to zero mean and unit
            variance.
        """
        parts = []
        if self.latent_rank3_features in {"features", "features_structure"}:
            parts.append(data.x.float())
        if self.latent_rank3_features in {"structure", "features_structure"}:
            degree = torch.tensor(
                [float(graph.degree(node)) for node in graph.nodes()],
                dtype=torch.float,
                device=data.x.device,
            )
            clustering = torch.tensor(
                [float(nx.clustering(graph, node)) for node in graph.nodes()],
                dtype=torch.float,
                device=data.x.device,
            )
            parts.append(
                torch.stack((torch.log1p(degree), clustering), dim=-1)
            )
        descriptors = torch.cat(parts, dim=-1)
        mean = descriptors.mean(dim=0, keepdim=True)
        std = descriptors.std(dim=0, keepdim=True, unbiased=False)
        return (descriptors - mean) / std.clamp_min(1e-6)

    @staticmethod
    def _initial_centers(
        descriptors: torch.Tensor, num_centers: int
    ) -> torch.Tensor:
        """Deterministically initialize centers by farthest-point sampling.

        Parameters
        ----------
        descriptors : torch.Tensor
            A ``(num_nodes, num_descriptors)`` tensor of per-node
            descriptors.
        num_centers : int
            Number of centers to select.

        Returns
        -------
        torch.Tensor
            A ``(num_centers, num_descriptors)`` tensor of initial
            center coordinates.
        """
        num_nodes = descriptors.size(0)
        centers = descriptors.new_zeros((num_centers, descriptors.size(1)))
        first = (
            torch.cdist(
                descriptors,
                descriptors.mean(dim=0, keepdim=True),
            )
            .squeeze(-1)
            .argmax()
        )
        chosen = [int(first)]
        centers[0] = descriptors[chosen[0]]
        min_dist = torch.cdist(descriptors, centers[:1]).squeeze(-1)
        for idx in range(1, num_centers):
            if idx < num_nodes:
                next_idx = int(min_dist.argmax())
                chosen.append(next_idx)
                centers[idx] = descriptors[next_idx]
                min_dist = torch.minimum(
                    min_dist,
                    torch.cdist(descriptors, centers[idx : idx + 1]).squeeze(
                        -1
                    ),
                )
            else:
                centers[idx] = descriptors[chosen[idx % len(chosen)]]
        return centers

    def _triangle_membership(
        self,
        connectivity: dict[str, torch.Tensor],
        node_membership: torch.Tensor,
        num_rank2: int,
    ) -> torch.Tensor:
        """Average node memberships over every rank-2 triangle.

        Parameters
        ----------
        connectivity : dict
            Rank-0/1/2 connectivity tensors, including
            ``"incidence_1"`` and ``"incidence_2"``.
        node_membership : torch.Tensor
            A ``(num_nodes, latent_rank3_cells)`` tensor of node-to-
            latent-region membership weights.
        num_rank2 : int
            Number of rank-2 triangle cells.

        Returns
        -------
        torch.Tensor
            A ``(num_rank2, latent_rank3_cells)`` tensor with each
            triangle's node memberships averaged over its incident
            nodes.
        """
        if num_rank2 == 0:
            return node_membership.new_zeros((0, self.latent_rank3_cells))
        incidence_1 = connectivity["incidence_1"].to(
            device=node_membership.device
        )
        incidence_2 = connectivity["incidence_2"].to(
            device=node_membership.device
        )
        edge_to_triangle = (
            incidence_2.to_dense().abs().to(node_membership.dtype)
        )
        node_to_edge = incidence_1.to_dense().abs().to(node_membership.dtype)
        node_to_triangle = (node_to_edge @ edge_to_triangle).clamp_max(1.0)
        counts = node_to_triangle.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
        return node_to_triangle.t() @ node_membership / counts

    @staticmethod
    def _dense_to_sparse(matrix: torch.Tensor) -> torch.Tensor:
        """Convert a dense matrix to a coalesced sparse tensor.

        Parameters
        ----------
        matrix : torch.Tensor
            The dense matrix to convert.

        Returns
        -------
        torch.Tensor
            A coalesced sparse COO tensor with the same shape,
            device, and non-zero values as ``matrix``.
        """
        indices = matrix.nonzero(as_tuple=False).t()
        if indices.numel() == 0:
            return torch.sparse_coo_tensor(
                matrix.new_zeros((2, 0), dtype=torch.long),
                matrix.new_zeros((0,)),
                matrix.shape,
                device=matrix.device,
            ).coalesce()
        values = matrix[indices[0], indices[1]]
        return torch.sparse_coo_tensor(
            indices,
            values,
            matrix.shape,
            device=matrix.device,
        ).coalesce()

    @staticmethod
    def _empty_sparse(
        rows: int,
        columns: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return an empty sparse matrix with stable dtype/device.

        Parameters
        ----------
        rows : int
            Number of rows of the returned matrix.
        columns : int
            Number of columns of the returned matrix.
        dtype : torch.dtype
            Dtype of the returned matrix's values.
        device : torch.device
            Device of the returned matrix.

        Returns
        -------
        torch.Tensor
            A coalesced sparse COO tensor of shape
            ``(rows, columns)`` with no non-zero entries.
        """
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=dtype, device=device),
            (rows, columns),
            device=device,
        ).coalesce()

    def _structural_features(
        self,
        connectivity: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Compute scale-stable size/degree features per complex rank.

        Parameters
        ----------
        connectivity : dict
            Rank connectivity tensors, including ``"shape"`` and,
            for each rank, ``"incidence_{rank}"`` and
            ``"adjacency_{rank}"``.
        dtype : torch.dtype
            Dtype used for the returned feature tensors.
        device : torch.device
            Device used for the returned feature tensors.

        Returns
        -------
        dict
            A ``"structure_{rank}"`` tensor for every rank from
            ``0`` to ``self.complex_dim``, each stacking log1p-
            scaled cell size, lower degree, same-rank degree, and
            upper degree.
        """
        shape = list(connectivity["shape"])
        output = {}
        node_membership = None
        for rank in range(self.complex_dim + 1):
            num_cells = int(shape[rank]) if rank < len(shape) else 0
            if rank == 0:
                cell_size = torch.ones(num_cells, dtype=dtype, device=device)
                lower_degree = torch.zeros(
                    num_cells, dtype=dtype, device=device
                )
                node_membership = torch.eye(
                    num_cells, dtype=dtype, device=device
                )
            else:
                incidence = (
                    connectivity[f"incidence_{rank}"].coalesce().to(device)
                )
                lower_degree = self._fit_length(
                    torch.sparse.sum(incidence, dim=0).to_dense().to(dtype),
                    num_cells,
                )
                if node_membership is None:
                    node_membership = torch.zeros(
                        shape[0], num_cells, dtype=dtype, device=device
                    )
                else:
                    node_membership = (
                        node_membership @ incidence.to_dense().abs().to(dtype)
                    ).clamp_max(1)
                cell_size = node_membership.sum(0)

            adjacency = connectivity[f"adjacency_{rank}"].coalesce().to(device)
            same_degree = self._fit_length(
                torch.sparse.sum(adjacency, dim=1).to_dense().to(dtype),
                num_cells,
            )
            if rank < self.complex_dim:
                upper_incidence = (
                    connectivity[f"incidence_{rank + 1}"].coalesce().to(device)
                )
                upper_degree = self._fit_length(
                    torch.sparse.sum(upper_incidence, dim=1)
                    .to_dense()
                    .to(dtype),
                    num_cells,
                )
            else:
                upper_degree = torch.zeros(
                    num_cells, dtype=dtype, device=device
                )

            output[f"structure_{rank}"] = torch.stack(
                (
                    torch.log1p(cell_size),
                    torch.log1p(lower_degree),
                    torch.log1p(same_degree),
                    torch.log1p(upper_degree),
                ),
                dim=-1,
            )
        return output

    @staticmethod
    def _fit_length(values: torch.Tensor, length: int) -> torch.Tensor:
        """Trim TopoNetX placeholder degrees or pad missing degrees.

        Parameters
        ----------
        values : torch.Tensor
            A 1-D tensor of degree values to resize.
        length : int
            Target number of entries.

        Returns
        -------
        torch.Tensor
            ``values`` trimmed to ``length`` entries, or right-
            padded with zeros if it has fewer than ``length``
            entries.
        """
        if values.numel() == length:
            return values
        if values.numel() > length:
            return values[:length]
        return torch.cat((values, values.new_zeros(length - values.numel())))
