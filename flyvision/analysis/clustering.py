import logging
import pickle
from dataclasses import dataclass
from functools import wraps
from typing import Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.figure import Figure
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

# umap import slows down whole library import
from umap.umap_ import UMAP
from umap.utils import disconnected_vertices

import flyvision
from flyvision.analysis.stimulus_responses import naturalistic_stimuli_responses
from flyvision.plots import plt_utils
from flyvision.plots.plt_utils import check_markers
from flyvision.utils.activity_utils import CentralActivity

INVALID_INT = -99999

logging = logging.getLogger(__name__)


@dataclass
class Embedding:
    """Embedding of the ensemble responses."""

    embedding: npt.NDArray = None
    mask: npt.NDArray = None
    reducer: object = None

    @property
    def cluster(self) -> "Clustering":
        return Clustering(self)

    @property
    def embedding(self) -> npt.NDArray:  # noqa: F811
        return getattr(self, "_embedding", None)

    @embedding.setter
    def embedding(self, value):
        self._embedding, self.minmaxscaler = scale_tensor(value)

    def plot(
        self,
        fig=None,
        ax=None,
        figsize=None,
        plot_mode="paper",
        fontsize=5,
        colors=None,
        **kwargs,
    ):
        """Plot the embedding."""

        if self.embedding.shape[1] != 2:
            raise AssertionError
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


def scale_tensor(tensor):
    """Scale tensor to range (0, 1)."""

    s = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
    return s.fit_transform(tensor), s


@dataclass
class GaussianMixtureClustering:
    """Gaussian Mixture Clustering of the embeddings."""

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
        range_n_clusters=None,
        n_init=1,
        max_iter=1000,
        random_state=0,
        **kwargs,
    ):
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

    def task_error_sort_labels(self, task_error, mode="mean"):
        self.labels = task_error_sort_labels(task_error, self.labels, mode=mode)

    def plot(
        self,
        task_error=None,
        colors=None,
        annotate=True,
        annotate_scores=False,
        fig=None,
        ax=None,
        figsize=None,
        plot_mode="paper",
        fontsize=5,
        **kwargs,
    ):
        if self.embedding.embedding.shape[1] != 2:
            raise AssertionError
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
    embedding: Embedding = None
    """Clustering of the embedding."""

    @property
    def gaussian_mixture(
        self,
    ) -> GaussianMixtureClustering:
        """See gaussian_mixture for kwargs."""
        return GaussianMixtureClustering(self.embedding)


def gaussian_mixture(
    X,
    mask,
    range_n_clusters=None,
    n_init=1,
    max_iter=1000,
    random_state=0,
    criterion="bic",
    **kwargs,
):
    """
    Fitting Gaussian Mixtures to the data.

        Args:
            X (Array): (#samples, here ensemble size, 2)
            range_n_clusters (Array): (#components) range of components to fit
            n_init (int): number of initializations
            max_iter (int): maximum number of iterations
            criterion (str): criterion to use for selecting the number of components

        Returns:
            labels (Array): (#samples) cluster labels
            gm (object): GaussianMixture object
            metric (Array): (#components) metric values
            range_n_clusters (Array): (#components) range of components
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
        raise ValueError(f"unknown criterion {criterion}")

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
    X,
    n_neighbors=5,
    min_dist=0.12,
    spread=9.0,
    random_state=42,
    n_components=2,
    metric="correlation",
    n_epochs=1500,
    **kwargs,
):
    """Embedding of X using UMAP.

    Args:
        X (Array): (#samples, 2)
        n_neighbors (int): number of neighbors
        min_dist (float): minimum distance
        spread (float): spread
        random_state (int): random state
        n_components (int): number of components
        metric (str): metric
        n_epochs (int): number of epochs
    """

    if n_components > X.shape[0] - 2:
        raise ValueError(
            "number of components must be 2 smaller than sample size."
            " See: https://github.com/lmcinnes/umap/issues/201"
        )

    if len(X.shape) > 2:
        shape = X.shape
        X = X.reshape(X.shape[0], -1)
        logging.info(f"reshaped X from {shape} to {X.shape}")

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
    X,
    colors=None,
    task_error=None,
    labels=None,
    gm=None,
    mask=None,
    fit_gaussians=True,
    annotate=True,
    contour_gaussians=True,
    range_n_clusters=[1, 2, 3, 4, 5],
    n_init_gaussian_mixture=10,
    title="",
    fig=None,
    ax=None,
    mode=None,
    figsize=[3, 3],
    s=20,
    fontsize=5,
    ax_lim_pad=0.2,  # relative
    task_error_sort_mode="mean",
    err_x_offset=0.025,
    err_y_offset=-0.025,
    gm_kwargs=None,
):
    """
    Args:
        X (Array): (#samples, here ensemble size, 2)
        colors (Array): (#samples) color values. Used to color the scatter points.
            Optional.
        task_error (Array): (#samples) task error values.
            Used to sort the label id's and to annotate the clusters. Optional.
        mask (Array): (#samples) Used to remove broken sampled.
    """

    # to remove broken samples (e.g. through disconnected vertices in umap or
    # model responses in nullspace)
    if mask is None:
        mask = slice(None)

    # to keep colors optional init default if not given
    if colors is None:
        colors = np.array(X.shape[0] * ["#779eaa"])
    colors = colors[mask]

    # to keep the task error optional
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


def task_error_sort_labels(task_error, labels, mode="mean"):
    """
    Permute the cluster label id's according to which cluster has the smallest
    average task error.

    We sort the labels such that the best
    performing cluster has label 0, the second best label 1 and so on.
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
            raise ValueError
        task_error_cluster.append(_error)

    for i, x in enumerate(np.argsort(task_error_cluster)):
        sorted_labels[np.isclose(labels, x)] = i

    return sorted_labels


def get_cluster_to_indices(mask, labels, task_error=None):
    indices = np.arange(len(labels)).astype(int)
    if task_error is not None:
        labels = task_error_sort_labels(task_error.values, labels)

    # to remove models from the index that are invalid
    indices = np.arange(len(labels))[mask]
    labels = labels[mask]
    cluster_indices = {
        label_id: indices[labels == label_id] for label_id in np.unique(labels)
    }
    return cluster_indices


def compute_umap_and_clustering(
    ensemble: "flyvision.EnsembleView",
    cell_type: str,
    embedding_kwargs=None,
    gm_kwargs=None,
    subdir="umap_and_clustering",
):
    """Compute UMAP embedding and Gaussian Mixture clustering of the responses."""

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

        logging.info(
            "Loaded %s embedding and clustering from %s.", cell_type, destination
        )
        return embedding_and_clustering

    # Load the embedding and clustering from disk if it exists
    if (destination / cell_type).with_suffix(".pickle").exists():
        return load_from_disk()

    def create_embedding_object(responses):
        """Return embedding object from cache or create and write cache."""
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
def umap_and_clustering_generator(ensemble: "flyvision.EnsembleView", **kwargs):
    """UMAP and clustering of all cell types."""
    for cell_type in ensemble[0].connectome_view.cell_types_sorted:
        yield cell_type, compute_umap_and_clustering(ensemble, cell_type, **kwargs)


if __name__ == "__main__":
    import flyvision

    ensemble = flyvision.EnsembleView("flow/0000")
    clustering = ensemble.clustering("T4c")
    print(clustering)
