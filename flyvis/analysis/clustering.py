import logging
import pickle
from dataclasses import dataclass
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

import flyvis
from flyvis.utils.activity_utils import CentralActivity

from .stimulus_responses import naturalistic_stimuli_responses
from .visualization import plt_utils
from .visualization.plt_utils import check_markers

__all__ = ["Embedding", "Clustering", "GaussianMixtureClustering", "EmbeddingPlot"]

INVALID_INT = -99999

logging = logging.getLogger(__name__)
if TYPE_CHECKING:
    from umap import UMAP
else:
    UMAP = TypeVar("UMAP")


@dataclass
class Embedding:
    """
    Embedding of the ensemble responses.

    Attributes:
        embedding (npt.NDArray): The embedded data.
        mask (npt.NDArray): Mask for valid data points.
        reducer (object): The reduction object used for embedding.
    """

    embedding: npt.NDArray = None
    mask: npt.NDArray = None
    reducer: object = None

    @property
    def cluster(self) -> "Clustering":
        """Returns a Clustering object for this embedding."""
        return Clustering(self)

    @property
    def embedding(self) -> npt.NDArray:  # noqa: F811
        """Returns the embedded data."""
        return getattr(self, "_embedding", None)

    @embedding.setter
    def embedding(self, value: npt.NDArray) -> None:
        """
        Sets the embedding and scales it to range (0, 1).

        Args:
            value: The embedding array to set.
        """
        self._embedding, self.minmaxscaler = scale_tensor(value)

    def plot(
        self,
        fig: Figure = None,
        ax: Axes = None,
        figsize: tuple = None,
        plot_mode: str = "paper",
        fontsize: int = 5,
        colors: npt.NDArray = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot the embedding.

        Args:
            fig: Existing figure to plot on.
            ax: Existing axes to plot on.
            figsize: Size of the figure.
            plot_mode: Mode for plotting ('paper', 'small', or 'large').
            fontsize: Font size for annotations.
            colors: Colors for data points.
            **kwargs: Additional arguments passed to plot_embedding.

        Returns:
            A tuple containing the figure and axes objects.

        Raises:
            AssertionError: If the embedding is not 2-dimensional.
        """
        if self.embedding.shape[1] != 2:
            raise AssertionError("Embedding must be 2-dimensional for plotting")
        if figsize is None:
            figsize = [0.94, 2.38]
        return plot_embedding(
            self.embedding,
            colors=colors,
            task_error=None,
            labels=None,
            mask=self.mask,
            fit_gaussians=False,
            annotate=False,
            title="",
            fig=fig,
            ax=ax,
            mode=plot_mode,
            figsize=figsize,
            fontsize=fontsize,
            **kwargs,
        )


def scale_tensor(tensor: npt.NDArray) -> tuple[npt.NDArray, MinMaxScaler]:
    """
    Scale tensor to range (0, 1).

    Args:
        tensor: Input tensor to be scaled.

    Returns:
        A tuple containing the scaled tensor and the MinMaxScaler object.
    """
    s = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
    return s.fit_transform(tensor), s


@dataclass
class GaussianMixtureClustering:
    """
    Gaussian Mixture Clustering of the embeddings.

    Attributes:
        embedding (Embedding): The embedding to cluster.
        range_n_clusters (Iterable[int]): Range of number of clusters to try.
        n_init (int): Number of initializations for GMM.
        max_iter (int): Maximum number of iterations for GMM.
        random_state (int): Random state for reproducibility.
        labels (npt.NDArray): Cluster labels.
        gm (object): Fitted GaussianMixture object.
        scores (list): Scores for each number of clusters.
        n_clusters (list): Number of clusters tried.
    """

    embedding: Embedding = None
    range_n_clusters: Iterable[int] = None
    n_init: int = 1
    max_iter: int = 1000
    random_state: int = 0
    labels: npt.NDArray = None
    gm: object = None
    scores: list = None
    n_clusters: list = None

    def __call__(
        self,
        range_n_clusters: Iterable[int] = None,
        n_init: int = 1,
        max_iter: int = 1000,
        random_state: int = 0,
        **kwargs,
    ) -> "GaussianMixtureClustering":
        """
        Perform Gaussian Mixture clustering.

        Args:
            range_n_clusters: Range of number of clusters to try.
            n_init: Number of initializations for GMM.
            max_iter: Maximum number of iterations for GMM.
            random_state: Random state for reproducibility.
            **kwargs: Additional arguments for gaussian_mixture function.

        Returns:
            Self with updated clustering results.
        """
        self.labels, self.gm, self.scores, self.n_clusters = gaussian_mixture(
            self.embedding.embedding,
            self.embedding.mask,
            range_n_clusters=range_n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs,
        )
        self.range_n_clusters = range_n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.kwargs = kwargs
        return self

    def task_error_sort_labels(self, task_error: npt.NDArray, mode: str = "mean") -> None:
        """
        Sort cluster labels based on task error.

        Args:
            task_error: Array of task errors.
            mode: Method to compute task error ('mean', 'min', or 'median').
        """
        self.labels = task_error_sort_labels(task_error, self.labels, mode=mode)

    def plot(
        self,
        task_error: npt.NDArray = None,
        colors: npt.NDArray = None,
        annotate: bool = True,
        annotate_scores: bool = False,
        fig: Figure = None,
        ax: Axes = None,
        figsize: tuple = None,
        plot_mode: str = "paper",
        fontsize: int = 5,
        **kwargs,
    ) -> "EmbeddingPlot":
        """
        Plot the clustering results.

        Args:
            task_error: Array of task errors.
            colors: Colors for data points.
            annotate: Whether to annotate clusters.
            annotate_scores: Whether to annotate BIC scores.
            fig: Existing figure to plot on.
            ax: Existing axes to plot on.
            figsize: Size of the figure.
            plot_mode: Mode for plotting ('paper', 'small', or 'large').
            fontsize: Font size for annotations.
            **kwargs: Additional arguments for plot_embedding function.

        Returns:
            An EmbeddingPlot object.

        Raises:
            AssertionError: If the embedding is not 2-dimensional.
        """
        if self.embedding.embedding.shape[1] != 2:
            raise AssertionError("Embedding must be 2-dimensional for plotting")
        if figsize is None:
            figsize = [0.94, 2.38]
        fig, ax = plot_embedding(
            self.embedding.embedding,
            colors=colors,
            task_error=task_error,
            labels=self.labels,
            gm=self.gm,
            mask=self.embedding.mask,
            fit_gaussians=True,
            annotate=annotate,
            title="",
            fig=fig,
            ax=ax,
            mode=plot_mode,
            figsize=figsize,
            fontsize=fontsize,
            range_n_clusters=self.range_n_clusters,
            n_init_gaussian_mixture=self.n_init,
            gm_kwargs=self.kwargs,
            **kwargs,
        )
        if annotate_scores:
            ax.annotate(
                "BIC: {:.2f}".format(np.min(self.scores)),
                xy=(ax.get_xlim()[0], ax.get_ylim()[1]),
                ha="left",
                va="top",
                fontsize=fontsize,
            )
        return EmbeddingPlot(fig, ax, None, None, self.gm.n_components, self)


@dataclass
class EmbeddingPlot:
    fig: Figure = None
    ax: Axes = None
    cmap: Colormap = None
    norm: Normalize = None
    n_clusters: int = None
    cluster: GaussianMixtureClustering = None


@dataclass
class Clustering:
    """Clustering of the embedding.

    Attributes:
        embedding (Embedding): The embedding to be clustered.
    """

    embedding: Embedding = None

    @property
    def gaussian_mixture(self) -> GaussianMixtureClustering:
        """Create a GaussianMixtureClustering object for the embedding.

        Returns:
            GaussianMixtureClustering: A clustering object for Gaussian mixture models.
        """
        return GaussianMixtureClustering(self.embedding)


def gaussian_mixture(
    X: np.ndarray,
    mask: np.ndarray,
    range_n_clusters: Optional[Union[List[int], np.ndarray]] = None,
    n_init: int = 1,
    max_iter: int = 1000,
    random_state: int = 0,
    criterion: str = "bic",
    **kwargs,
) -> Tuple[np.ndarray, GaussianMixture, np.ndarray, np.ndarray]:
    """Fit Gaussian Mixtures to the data.

    Args:
        X: Input data with shape (n_samples, n_features).
        mask: Boolean mask for valid samples.
        range_n_clusters: Range of number of components to fit.
        n_init: Number of initializations for each number of components.
        max_iter: Maximum number of iterations for the fitting process.
        random_state: Random state for reproducibility.
        criterion: Criterion to use for selecting the number of components.
            Options are "bic", "aic", or "score".
        **kwargs: Additional keyword arguments for GaussianMixture.

    Returns:
        A tuple containing:

        - labels: Cluster labels for each sample.
        - gm: Fitted GaussianMixture object.
        - metric: Metric values (BIC, AIC, or score) for each number of components.
        - range_n_clusters: Range of number of components tried.

    Raises:
        ValueError: If an unknown criterion is provided.
    """
    if range_n_clusters is None:
        range_n_clusters = np.array([1, 2, 3, 4, 5])

    if isinstance(range_n_clusters, list):
        range_n_clusters = np.array(range_n_clusters)

    n_samples = X.shape[0]
    labels = np.ones([n_samples], dtype=int) * INVALID_INT
    X = X[mask]

    n_valid_samples = X.shape[0]
    range_n_clusters = range_n_clusters[range_n_clusters <= n_valid_samples]

    scores = []
    aic = []
    bic = []
    for n_clusters in range_n_clusters:
        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            max_iter=max_iter,
            n_init=n_init,
            **kwargs,
        )
        gm.fit(X)
        scores.append(-gm.score(X))
        aic.append(gm.aic(X))
        bic.append(gm.bic(X))

    if criterion == "bic":
        metric = bic
    elif criterion == "aic":
        metric = aic
    elif criterion == "score":
        metric = scores
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    n_components = range_n_clusters[np.argmin(metric)]
    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
        **kwargs,
    )

    labels[mask] = gm.fit_predict(X).astype(int)

    return labels, gm, metric, range_n_clusters


class EnsembleEmbedding:
    """Embedding of the ensemble responses.

    Args: responses (CentralActivity): CentralActivity object
    """

    def __init__(self, responses: CentralActivity):
        self.responses = responses

    def from_cell_type(
        self,
        cell_type,
        embedding_kwargs=None,
    ) -> Embedding:
        """Umap Embedding of the responses of a specific cell type."""

        embedding_kwargs = embedding_kwargs or {}
        return Embedding(*umap_embedding(self.responses[cell_type], **embedding_kwargs))

    def __call__(
        self,
        arg: Union[str, Iterable],
        embedding_kwargs=None,
    ):
        if isinstance(arg, str):
            return self.from_cell_type(arg, embedding_kwargs)
        else:
            raise ValueError("arg")


def umap_embedding(
    X: np.ndarray,
    n_neighbors: int = 5,
    min_dist: float = 0.12,
    spread: float = 9.0,
    random_state: int = 42,
    n_components: int = 2,
    metric: str = "correlation",
    n_epochs: int = 1500,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, UMAP]:
    """
    Perform UMAP embedding on input data.

    Args:
        X: Input data with shape (n_samples, n_features).
        n_neighbors: Number of neighbors to consider for each point.
        min_dist: Minimum distance between points in the embedding space.
        spread: Determines how spread out all embedded points are overall.
        random_state: Random seed for reproducibility.
        n_components: Number of dimensions in the embedding space.
        metric: Distance metric to use.
        n_epochs: Number of training epochs for embedding optimization.
        **kwargs: Additional keyword arguments for UMAP.

    Returns:
        A tuple containing:
        - embedding: The UMAP embedding.
        - mask: Boolean mask for valid samples.
        - reducer: The fitted UMAP object.

    Raises:
        ValueError: If n_components is too large relative to sample size.

    Note:
        This function handles reshaping of input data and removes constant rows.
    """
    # umap import would slow down whole library import
    from umap import UMAP
    from umap.utils import disconnected_vertices

    if n_components > X.shape[0] - 2:
        raise ValueError(
            "number of components must be 2 smaller than sample size. "
            "See: https://github.com/lmcinnes/umap/issues/201"
        )

    if len(X.shape) > 2:
        shape = X.shape
        X = X.reshape(X.shape[0], -1)
        logging.info("reshaped X from %s to %s", shape, X.shape)

    embedding = np.ones([X.shape[0], n_components]) * np.nan
    # umap doesn't like contant rows
    mask = ~np.isclose(X.std(axis=1), 0)
    X = X[mask]
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=n_components,
        metric=metric,
        spread=spread,
        n_epochs=n_epochs,
        **kwargs,
    )
    _embedding = reducer.fit_transform(X)

    # gaussian mixture doesn't like nans through disconnected vertices in umap
    connected_vertices_mask = ~disconnected_vertices(reducer)
    mask[mask] = mask[mask] & connected_vertices_mask
    embedding[mask] = _embedding[connected_vertices_mask]
    return embedding, mask, reducer


def plot_embedding(
    X: np.ndarray,
    colors: Optional[np.ndarray] = None,
    task_error: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    gm: Optional[GaussianMixture] = None,
    mask: Optional[np.ndarray] = None,
    fit_gaussians: bool = True,
    annotate: bool = True,
    contour_gaussians: bool = True,
    range_n_clusters: List[int] = [1, 2, 3, 4, 5],
    n_init_gaussian_mixture: int = 10,
    title: str = "",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    mode: Optional[str] = None,
    figsize: List[float] = [3, 3],
    s: float = 20,
    fontsize: int = 5,
    ax_lim_pad: float = 0.2,
    task_error_sort_mode: str = "mean",
    err_x_offset: float = 0.025,
    err_y_offset: float = -0.025,
    gm_kwargs: Optional[Dict] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the embedding of data points with optional Gaussian mixture clustering.

    Args:
        X: Input data with shape (n_samples, 2).
        colors: Color values for each data point.
        task_error: Task error values for each data point.
        labels: Cluster labels for each data point.
        gm: Fitted GaussianMixture object.
        mask: Boolean mask for valid samples.
        fit_gaussians: Whether to fit Gaussian mixtures.
        annotate: Whether to annotate clusters with performance metrics.
        contour_gaussians: Whether to plot contours for Gaussian mixtures.
        range_n_clusters: Range of number of clusters to try.
        n_init_gaussian_mixture: Number of initializations for Gaussian mixture.
        title: Title of the plot.
        fig: Existing figure to plot on.
        ax: Existing axes to plot on.
        mode: Plotting mode ('paper', 'small', or 'large').
        figsize: Size of the figure.
        s: Size of scatter points.
        fontsize: Font size for annotations.
        ax_lim_pad: Padding for axis limits.
        task_error_sort_mode: Mode for sorting task errors ('mean' or 'min').
        err_x_offset: X-offset for error annotations.
        err_y_offset: Y-offset for error annotations.
        gm_kwargs: Additional keyword arguments for Gaussian mixture.

    Returns:
        A tuple containing the figure and axes objects.

    Note:
        This function handles various plotting scenarios including Gaussian mixture
        clustering and task error annotation.
    """
    if mask is None:
        mask = slice(None)

    if colors is None:
        colors = np.array(X.shape[0] * ["#779eaa"])
    colors = colors[mask]

    if task_error is not None:
        task_error = task_error[mask]

    if mode == "paper":
        figsize = [2.68, 2.38]
        s = 20
        fontsize = 5
    if mode == "small":
        figsize = [3, 3]
        s = 20
        fontsize = 5
    elif mode == "large":
        figsize = [10, 10]
        s = 60
        fontsize = 12

    fig, ax = plt_utils.init_plot(figsize=figsize, fontsize=fontsize, fig=fig, ax=ax)

    xc = X[:, 0]
    yc = X[:, 1]
    x_min, x_max = plt_utils.get_lims(xc, ax_lim_pad)
    y_min, y_max = plt_utils.get_lims(yc, ax_lim_pad)

    if fit_gaussians:
        # breakpoint()
        if labels is not None and gm:
            _labels = labels
        elif labels is not None:
            _labels, gm, scores = gaussian_mixture(
                X,
                mask,
                range_n_clusters,
                n_init=n_init_gaussian_mixture,
                **(gm_kwargs or {}),
            )
        if labels is not None:
            assert (_labels == labels).all()
        X = X[mask]
        labels = _labels[mask]

        if task_error is not None:
            labels = task_error_sort_labels(task_error, labels, mode=task_error_sort_mode)
        else:
            # because the sorting is later based on this to plot better
            # performing models on top of worse performing models
            task_error = np.arange(X.shape[0])
            annotate = False

        if contour_gaussians:
            x = np.linspace(x_min, x_max, 1000)
            y = np.linspace(y_min, y_max, 1000)
            _X, _Y = np.meshgrid(x, y)
            XX = np.array([_X.ravel(), _Y.ravel()]).T
            Z = np.exp(gm.score_samples(XX))
            Z /= Z.sum()
            Z = Z.reshape(_X.shape)
            # print(Z.min(), Z.max())
            ax.contourf(
                _X,
                _Y,
                Z,
                norm=LogNorm(vmin=1e-25, vmax=1e0),
                levels=np.logspace(-25, 0, 15),
                cmap=plt.cm.binary,
                alpha=0.1,
                zorder=-1,
            )

        MARKERS = check_markers(len(np.unique(labels)))
        for label in np.unique(labels).astype(int):
            # to plot best performing models on top
            _argsort = np.argsort(task_error)[::-1]
            _X = X[_argsort]
            _colors = colors[_argsort]
            _labels = labels[_argsort]

            ax.scatter(
                _X[_labels == label, 0],
                _X[_labels == label, 1],
                marker=MARKERS[label],
                c=_colors[_labels == label],
                s=s,
                alpha=0.6,
                edgecolors="none",
            )

            if annotate:
                # annotate performance average
                x, y = X[labels == label, 0].mean(), X[labels == label, 1].mean()
                if task_error_sort_mode == "mean":
                    _perf = task_error[labels == label].mean().item()
                elif task_error_sort_mode == "min":
                    _perf = task_error[labels == label].min().item()
                else:
                    raise ValueError()

                ax.annotate(
                    f"{_perf:.3f}",
                    (
                        x + err_x_offset * (x_max - x_min),
                        y + err_y_offset * (y_max - y_min),
                    ),
                    fontsize=fontsize,
                    ha="left",
                    va="top",
                )

    elif labels is not None:
        X = X[mask]
        labels = labels[mask]
        MARKERS = check_markers(len(np.unique(labels)))
        for label in np.unique(labels).astype(int):
            ax.scatter(
                X[labels == label, 0],
                X[labels == label, 1],
                marker=MARKERS[label],
                c=colors[labels == label],
                s=s,
                alpha=0.6,
                edgecolors="none",
            )

            if annotate and task_error is not None:
                # annotate performance average
                x, y = X[labels == label, 0].mean(), X[labels == label, 1].mean()
                _perf = task_error[labels == label].mean().item()
                ax.annotate(
                    f"{_perf:.3G}",
                    (
                        x + err_x_offset * (x_max - x_min),
                        y - err_y_offset * (y_max - y_min),
                    ),
                    fontsize=fontsize,
                    ha="left",
                    va="top",
                )
    else:
        X = X[mask]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=s, alpha=0.6, edgecolors="none")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title(title, fontsize=fontsize)
    plt_utils.rm_spines(ax)

    return fig, ax


def task_error_sort_labels(
    task_error: np.ndarray, labels: np.ndarray, mode: str = "mean"
) -> np.ndarray:
    """
    Permute cluster label IDs based on task error.

    Args:
        task_error: Array of task errors for each sample.
        labels: Array of cluster labels.
        mode: Method to compute task error. Options: "mean", "min", "median".

    Returns:
        Array of sorted labels.

    Raises:
        ValueError: If an invalid mode is provided.

    Note:
        Sorts labels so the best performing cluster has label 0, second best 1, etc.
    """
    sorted_labels = np.ones_like(labels, dtype=int) * INVALID_INT

    task_error_cluster = []
    for label_id in np.unique(labels[labels != INVALID_INT]):
        if mode == "mean":
            _error = task_error[labels == label_id].mean()
        elif mode == "min":
            _error = task_error[labels == label_id].min()
        elif mode == "median":
            _error = np.median(task_error[labels == label_id])
        else:
            raise ValueError("Invalid mode for task error calculation")
        task_error_cluster.append(_error)

    for i, x in enumerate(np.argsort(task_error_cluster)):
        sorted_labels[np.isclose(labels, x)] = i

    return sorted_labels


def get_cluster_to_indices(
    mask: np.ndarray, labels: np.ndarray, task_error: Optional[np.ndarray] = None
) -> Dict[int, np.ndarray]:
    """
    Map cluster labels to corresponding indices.

    Args:
        mask: Boolean mask for valid samples.
        labels: Array of cluster labels.
        task_error: Optional array of task errors for sorting labels.

    Returns:
        Dictionary mapping cluster labels to arrays of indices.
    """
    indices = np.arange(len(labels)).astype(int)
    if task_error is not None:
        labels = task_error_sort_labels(task_error.values, labels)

    indices = np.arange(len(labels))[mask]
    labels = labels[mask]
    cluster_indices = {
        label_id: indices[labels == label_id] for label_id in np.unique(labels)
    }
    return cluster_indices


def compute_umap_and_clustering(
    ensemble: "flyvis.network.EnsembleView",
    cell_type: str,
    embedding_kwargs: Optional[Dict] = None,
    gm_kwargs: Optional[Dict] = None,
    subdir: str = "umap_and_clustering",
) -> GaussianMixtureClustering:
    """
    Compute UMAP embedding and Gaussian Mixture clustering of responses.

    Args:
        ensemble: EnsembleView object.
        cell_type: Type of cell to analyze.
        embedding_kwargs: UMAP embedding parameters.
        gm_kwargs: Gaussian Mixture clustering parameters.
        subdir: Subdirectory for storing results.

    Returns:
        GaussianMixtureClustering object.

    Note:
        Results are cached to disk for faster subsequent access.
    """
    if embedding_kwargs is None:
        embedding_kwargs = {
            "min_dist": 0.105,
            "spread": 9.0,
            "n_neighbors": 5,
            "random_state": 42,
            "n_epochs": 1500,
        }
    if gm_kwargs is None:
        gm_kwargs = {
            "range_n_clusters": [2, 3, 3, 4, 5],
            "n_init": 100,
            "max_iter": 1000,
            "random_state": 42,
            "tol": 0.001,
        }

    destination = ensemble.path / subdir

    def load_from_disk():
        with open((destination / cell_type).with_suffix(".pickle"), "rb") as f:
            embedding_and_clustering = pickle.load(f)

        logging.info("Loaded %s embedding and clustering from %s", cell_type, destination)
        return embedding_and_clustering

    if (destination / cell_type).with_suffix(".pickle").exists():
        return load_from_disk()

    def create_embedding_object(responses):
        central_responses = CentralActivity(
            responses['responses'].values, ensemble[0].connectome, keepref=True
        )
        embeddings = EnsembleEmbedding(central_responses)
        return embeddings

    responses = naturalistic_stimuli_responses(ensemble)
    embeddings = create_embedding_object(responses)

    embedding = embeddings.from_cell_type(cell_type, embedding_kwargs=embedding_kwargs)
    embedding_and_clustering = embedding.cluster.gaussian_mixture(**gm_kwargs)
    return embedding_and_clustering


@wraps(compute_umap_and_clustering)
def umap_and_clustering_generator(
    ensemble: "flyvis.network.EnsembleView", **kwargs
) -> Generator[Tuple[str, GaussianMixtureClustering], None, None]:
    """
    Generate UMAP and clustering for all cell types.

    Args:
        ensemble: EnsembleView object.
        **kwargs: Additional arguments for compute_umap_and_clustering.

    Yields:
        Tuple of cell type and corresponding GaussianMixtureClustering object.
    """
    for cell_type in ensemble[0].connectome_view.cell_types_sorted:
        yield cell_type, compute_umap_and_clustering(ensemble, cell_type, **kwargs)
