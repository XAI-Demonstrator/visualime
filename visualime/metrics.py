import numpy as np


def cosine_distance(samples: np.ndarray):
    """Cosine distance as in the original LIME implementation."""
    return 1 - np.sum(samples, axis=1) / (np.linalg.norm(samples, axis=1) + 1e-6)


def exponential_kernel(distances: np.ndarray, kernel_width: float = 0.25):
    """Exponential kernel as in the original LIME implementation.

    Parameters
    ----------
    distances : np.ndarray
        One-dimensional array as returned by :meth:`visualime.lime.compute_distances`.

    kernel_width : float, default 0.25
        Width of the exponential kernel.
    """
    return np.sqrt(np.exp(-(distances**2) / kernel_width**2))


DISTANCES_KERNEL_DOC = """distances : np.ndarray, optional
        The distances between the images and the original images used as sample weights when
        fitting the linear model.

        If not given, the cosine distance between a sample and the original image is used.
        Note that this is only a rough approximation and not a good measure if the image
        contains a lot of variation or the segments are of very different size.

    kernel : callable, default exponential_kernel
        Kernel function to weigh the samples based on the `distances`.

        Operates on the `distances` and returns an array of the same shape:
        `kernel(distances: np.ndarray) -> np.ndarray`

        Defaults to an exponential kernel with width `.25` as in the original LIME
        implementation."""
