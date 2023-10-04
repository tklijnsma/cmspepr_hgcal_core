import numpy as np
import torch
from torch_scatter import scatter_max
from torch_geometric.data import Data

from cmspepr_hgcal_core.utils import (
    huber,
    scatter_count,
    LossResult,
)


def oc_loss(model_output: torch.FloatTensor, data: Data) -> LossResult:
    """Calculator for the object condensation loss function *per batch*.

    Creates an ObjectCondensation per event in the batch and adds up individual losses.
    """
    n_events = data.batch.max() + 1
    loss_result = LossResult()
    for i_event in range(n_events):
        select = data.batch == i_event
        loss_result += ObjectCondensation(model_output[select], data.y[select]).loss()
    return loss_result / float(n_events)


def oc_loss_with_noise_filter(
    model_output: torch.FloatTensor, data: Data, pass_noise_filter: torch.IntTensor
) -> LossResult:
    n_events = data.batch.max() + 1
    loss_result = LossResult()
    y = data.y[pass_noise_filter]
    batch = data.batch[pass_noise_filter]
    for i_event in range(n_events):
        select = batch == i_event
        loss_result += ObjectCondensation(
            model_output[select], reincrementalize(y[select])
        ).loss()
    return loss_result / float(n_events)


class ObjectCondensation:
    """Calculator for the object condensation loss function.

    Every instance is specific to a particular event.

    Args:
        x (torch.FloatTensor): Output of the model. The first column is assumed to be
            the beta's. All other columns are assumed coordintes in the latent
            clustering space. There is currently no support for property determination.
        y (torch.LongTensor): The truth cluster index per hit.
    """

    # Optional huberization for V_att distances
    huberize_norm_for_V_att = True

    # Specific OC parameter, representing the minimum charge per point.
    qmin = 1.0

    # Specific OC parameter, balancing the weight the noise should have when calculating
    # the repulsive loss term.
    sB = 0.1

    # How to calculate the charge q from beta.
    # Options are 'paper' or 'betaclip'
    beta_stabilizing = 'betaclip'

    # The model must be encouraged to pick high beta values for signal points. There are
    # multiple options to achieve this.
    # Options are 'paper' or 'short-range-potential'
    beta_term_option = 'paper'

    def __init__(self, x: torch.FloatTensor, y: torch.LongTensor):
        assert not torch.isnan(x).any()
        # Quantities that will surely be used can be calculated now:
        self.beta = torch.sigmoid(x[:, 0])
        self.x = x[:, 1:]  # Cluster space coordinates
        self.y = y

        # Terminology: "Signal" points are points belonging to an actual shape (i.e. NOT
        # noise).
        self.is_noise = y == 0
        self.is_sig = ~self.is_noise
        self.n = x.size(0)
        self.n_sig = self.is_sig.sum()  # Number of signal hits (i.e. non-noise hits)
        self.n_cond = torch.max(
            y
        )  # Number of condensation points == number of clusters

        self.x_sig = self.x[self.is_sig]

        # Make it 0-indexed. In self.y, 0 is the noise cluster; in self.y_sig, 0 is the
        # first shape.
        self.y_sig = y[self.is_sig] - 1

        # Number of points per cluster / cond point
        self.n_per_cond = scatter_count(self.y_sig)

    def loss(self) -> LossResult:
        # Calculate q
        if self.beta_stabilizing == 'betaclip':
            self.calc_q_betaclip()
        elif self.beta_stabilizing == 'paper':
            self.calc_q_paper()
        else:
            raise Exception(
                f'Unknown beta_stabilizing option {self.beta_stabilizing};'
                ' Pick from "betaclip" or "paper"'
            )

        if self.beta_term_option == 'paper':
            L_beta_sig = self.L_beta_sig_paper
        elif self.beta_term_option == 'short_range_potential':
            L_beta_sig = self.L_beta_sig_short_range_potential
        else:
            raise Exception(
                f'Unknown beta_term_option option {self.beta_term_option};'
                ' Pick from "short_range_potential" or "paper"'
            )

        return LossResult(
            V_att=self.V_att,
            V_rep=self.V_rep,
            L_beta_sig=L_beta_sig,
            L_beta_sig_logterm=self.L_beta_sig_logterm,
            L_beta_noise=self.L_beta_noise,
        )

    def calc_q_paper(self):
        """
        Calculates the charge q from beta.
        """
        self.q = self.beta.arctanh() ** 2 + self.qmin
        self.calc_q_dependent_quantities()

    def calc_q_betaclip(self):
        """
        Performs a clip on beta before calling arctanh**2.
        Very often necessary to avoid NaN's in the arctanh calc.
        Known as "soft q scaling".
        """
        self.q = (self.beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + self.qmin
        self.calc_q_dependent_quantities()

    def calc_q_dependent_quantities(self):
        """
        Does a set of calculations that can only be done after self.q is set.
        """
        self.q_sig = self.q[self.is_sig]
        # Select the condensation points: The points with the largest charge, per shape.
        self.q_cond, self.i_cond = scatter_max(self.q_sig, self.y_sig)
        self.x_cond = self.x_sig[self.i_cond]
        self.beta_cond = self.beta[self.i_cond]

    @property
    def d(self) -> torch.FloatTensor:
        """
        Distance matrix of every point to every condensation point.

        (n_hits, 1, cluster_space_dim) - (1, n_cond, cluster_space_dim) gives
        (n_hits, n_cond, cluster_space_dim).
        The norm reduces the last dimension, so the result is (n_hits, n_cond).

        Returns:
            torch.FloatTensor: (n_hits, n_cond)-shaped tensor containing the distances
                of every point to every condensation point.
        """
        if not hasattr(self, "_d"):
            self._d = (self.x.unsqueeze(1) - self.x_cond.unsqueeze(0)).norm(dim=-1)
        return self._d

    @property
    def M(self) -> torch.LongTensor:
        """
        Connectivity matrix for sig hits: Only 1 of hit belongs to cond point,
        otherwise 0
        """
        return torch.nn.functional.one_hot(self.y_sig).long()

    @property
    def V_att(self) -> torch.float:
        """Calculates the attractive potential loss.

        Returns:
            torch.float: The attractive potential loss.
        """
        if self.huberize_norm_for_V_att:
            # Parabolic (like normal L2-norm) where distances < threshold,
            # but linear outside.
            # This prevents unreasonably high losses when misassigning
            # singular points, and allows the network to space clusters
            # more.
            d = huber(self.d[self.is_sig] + 1e-5, 4.0)
        else:
            d = self.d[self.is_sig] ** 2
        V = self.M * self.q_sig.unsqueeze(-1) * self.q_cond.unsqueeze(0) * d
        assert V.size() == (self.n_sig, self.n_cond)
        V = V.sum() / self.n
        return V

    @property
    def V_rep(self) -> torch.float:
        """Calculates the repulsive potential loss.

        Returns:
            torch.float: The repulsive potential loss.
        """
        # Anti-connectivity matrix
        M_inv = 1 - torch.nn.functional.one_hot(self.y).long()
        # Throw away the noise column; there is no cond point for noise
        M_inv = M_inv[:, 1:]

        # Power-scale the norms: Gaussian scaling term instead of a cone
        d = torch.exp(-4.0 * self.d**2)

        # (n, 1) * (1, n_cond) * (n, n_cond)
        V = self.q.unsqueeze(1) * self.q_cond.unsqueeze(0) * M_inv * d
        assert V.size() == (self.n, self.n_cond)

        V = torch.maximum(V, torch.tensor(0.0)).sum() / self.n
        return V

    @property
    def L_beta_noise(self) -> torch.float:
        """Loss term to suppress large charge for noise points.

        Returns:
            torch.float: Loss term
        """
        return self.sB * self.beta[self.is_noise].mean()

    @property
    def L_beta_sig_paper(self) -> torch.float:
        """Loss term to encourage large charge for shape points.

        Returns:
            torch.float: Loss term
        """
        return (1 - self.beta[self.i_cond]).mean()

    @property
    def L_beta_sig_short_range_potential(self) -> torch.float:
        # (N, 1): Inverse scaled distance to the cond point every hit belongs to
        # low d -> high closeness, and vice versa
        # Keep only distances w.r.t. belonging cond point
        # Then sum over _hits_, so the result is (n_cond,)
        closeness = (1.0 / (20.0 * self.d[self.is_sig] ** 2 + 1.0) * self.M).sum(dim=0)
        assert torch.all(closeness >= 1.0) and torch.all(closeness <= self.n_per_cond)

        # closeness of the cond point w.r.t. itself will be 1., by definition
        # Remove that one, then divide by number of hits per cluster
        # to obtain average closeness per cluster
        closeness = (closeness - 1.0) / self.n_per_cond
        assert torch.all(closeness >= 0.0) and torch.all(closeness <= 1.0)

        # Multiply by the beta of the cond point and take the average, invert
        L = -(closeness * self.beta_cond).mean()
        assert -1 <= L <= 0.0

        # Summary: For a good prediction,
        # small d -> large closeness -> more neg L -> low L
        # high beta_cond -> more neg L -> low L
        return L

    @property
    def L_beta_sig_logterm(self) -> torch.float:
        # For a good prediction: large beta_cond -> more negative -> low L
        return (-0.2 * torch.log(self.beta_cond + 1e-9)).mean()


def get_clustering_np(
    betas: np.array, X: np.array, tbeta: float = 0.1, td: float = 1.0
) -> np.array:
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes numpy arrays as input.
    """
    n_points = betas.shape[0]
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = np.nonzero(select_condpoints)[0]
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[np.argsort(-betas[select_condpoints])]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for index_condpoint in indices_condpoints:
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.1, td=1.0):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def reincrementalize(y: torch.Tensor) -> torch.Tensor:
    """Re-indexes y so that missing clusters are no longer counted.

    Example:
        >>> reincrementalize(torch.LongTensor([0, 6, 1, 0, 1, 6, 3, 0, 3]))
        tensor([0, 3, 1, 0, 1, 3, 2, 0, 2])
    """
    vals, indices = torch.unique(y, return_inverse=True)
    return torch.arange(len(vals))[indices]


# def reincrementalize(y: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
#     """Re-indexes y so that missing clusters are no longer counted.

#     Example:
#         >>> y = torch.LongTensor([
#             0, 0, 0, 1, 1, 3, 3,
#             0, 0, 0, 0, 0, 2, 2, 3, 3,
#             0, 0, 1, 1
#             ])
#         >>> batch = torch.LongTensor([
#             0, 0, 0, 0, 0, 0, 0,
#             1, 1, 1, 1, 1, 1, 1, 1, 1,
#             2, 2, 2, 2,
#             ])
#         >>> print(reincrementalize(y, batch))
#         tensor([0, 0, 0, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1])
#     """
#     y_offset, n_per_event = batch_cluster_indices(y, batch)
#     offset = y_offset - y
#     n_clusters = n_per_event.sum()
#     holes = (~torch.isin(torch.arange(n_clusters, device=y.device), y_offset)).nonzero().squeeze(-1)
#     n_per_event_without_holes = n_per_event.clone()
#     n_per_event_cumsum = n_per_event.cumsum(0)
#     for hole in holes.sort(descending=True).values:
#         y_offset[y_offset > hole] -= 1
#         i_event = (hole > n_per_event_cumsum).long().argmin()
#         n_per_event_without_holes[i_event] -= 1
#     offset_per_event = torch.zeros_like(n_per_event_without_holes)
#     offset_per_event[1:] = n_per_event_without_holes.cumsum(0)[:-1]
#     offset_without_holes = torch.gather(offset_per_event,0, batch).long()
#     reincrementalized = y_offset - offset_without_holes
#     return reincrementalized
