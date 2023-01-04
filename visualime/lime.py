from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, quickshift, slic, watershed

__all__ = [
    "create_segments",
    "generate_images",
    "generate_samples",
    "compute_distances",
    "predict_images",
    "weigh_segments",
]

from visualime._models import (
    LINEAR_MODEL_TYPES,
    MODEL_TYPE_PARAMS_DOC,
    instantiate_model,
)
from visualime.metrics import DISTANCES_KERNEL_DOC, cosine_distance, exponential_kernel

SAMPLES_PREDICTIONS_LABEL_IDX_DOC = """samples : np.ndarray
        The samples generated by :meth:`visualime.lime.generate_samples`:
        An array of shape `(num_of_samples, num_of_segments)`.

    predictions : np.ndarray
        The predictions produced by :meth:`visualime.lime.predict_images`:
        An array of shape `(num_of_samples, num_of_classes)`.

    label_idx : int
        The index of the label to explain in the output of `predict_fn()`.
        Can be the class predicted by the model, or a different class."""


def _watershed(image: np.ndarray, **kwargs):
    gradient = sobel(rgb2gray(image))
    return watershed(image=gradient, **kwargs)


SEGMENTATION_METHOD_TYPES = Literal["felzenszwalb", "slic", "quickshift", "watershed"]

SEGMENTATION_METHODS: Dict[
    SEGMENTATION_METHOD_TYPES, Tuple[Callable, Dict[str, Any]]
] = {
    "felzenszwalb": (felzenszwalb, {"scale": 250, "sigma": 0.6, "min_size": 45}),
    "slic": (
        slic,
        {
            "n_segments": 250,
            "compactness": 2,
            "convert2lab": True,
            "sigma": 1,
            "start_label": 0,
        },
    ),
    "quickshift": (quickshift, {"kernel_size": 5, "max_dist": 6, "ratio": 0.7}),
    "watershed": (_watershed, {"markers": 250, "compactness": 0.001}),
}


def create_segments(
    image: np.ndarray,
    segmentation_method: SEGMENTATION_METHOD_TYPES,
    segmentation_settings: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Divide the image into segments (superpixels).

    Proper segmentation of the images is key to producing meaningful explanations with LIME.
    Which method and settings are appropriate is highly use-case specific.

    For an introduction into image segmentation and a comparison of the different methods, see
    `this tutorial
    <https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html>`_
    in the `scikit-image` documentation.

    Parameters
    ----------
    image : np.ndarray
        The image to segment as a three-dimensional array of shape `(image_width, image_height, 3)`
        where the last dimension are the RGB channels.

    segmentation_method : str
        The method used to segment the image into superpixels.
        Available options are `"felzenszwalb"`, `"slic"`, `"quickshift"`, and `"watershed"`.

        See the `scikit-image documentation
        <https://scikit-image.org/docs/stable/api/skimage.segmentation.html>`_
        for details.

    segmentation_settings : dict, optional
        Keyword arguments to pass to the segmentation method.

        See the `scikit-image documentation
        <https://scikit-image.org/docs/stable/api/skimage.segmentation.html>`_
        for details.

    Returns
    -------
    np.ndarray
        An array of shape `(image_width, image_height)` where each entry is an integer
        that corresponds to the segment number.

        Segment numbers start at 0 and are continuous. The number of segments can be computed
        by determining the maximum value in the array and adding 1.
    """
    segmentation_settings = segmentation_settings or {}

    try:
        segmentation_fn, default_settings = SEGMENTATION_METHODS[segmentation_method]
    except KeyError:
        raise ValueError(
            f"Unknown segmentation_method '{segmentation_method}'."
            f" Available options: {list(SEGMENTATION_METHODS.keys())}"
        )

    settings = {**default_settings, **segmentation_settings}

    _segment_mask = segmentation_fn(image=image, **settings)

    return _segment_mask - np.min(_segment_mask)


def generate_samples(
    segment_mask: np.ndarray, num_of_samples: int = 64, p: float = 0.5
) -> np.ndarray:
    """Generate samples by randomly selecting a subset of the segments.

    Parameters
    ----------
    segment_mask : np.ndarray
        The mask generated by :meth:`visualime.lime.create_segments`:
        An array of shape `(image_width, image_height)`.

    num_of_samples : int
        The number of samples to generate.

    p : float
        The probability for each segment to be removed from a sample.

    Returns
    -------
    np.ndarray
        A two-dimensional array of shape `(num_of_samples, num_of_segments)`.
    """
    num_of_segments = int(np.max(segment_mask) + 1)

    return np.random.binomial(n=1, p=p, size=(num_of_samples, num_of_segments))


def generate_images(
    image: np.ndarray,
    segment_mask: np.ndarray,
    samples: np.ndarray,
    background: Optional[Union[np.ndarray, int, float]] = None,
) -> np.ndarray:
    """Generate images from a list of samples.

    Parameters
    ----------
    image : np.ndarray
        The image to explain: An array of shape `(image_width, image_height, 3)`.

    segment_mask : np.ndarray
        The mask generated by :meth:`visualime.lime.create_segments`:
        An array of shape `(image_width, image_height)`.

    samples : np.ndarray
        The samples generated by :meth:`visualime.lime.generate_samples`:
        An array of shape `(num_of_samples, num_of_segments)`.

    background : {np.ndarray, int}, optional
        The background to replace the excluded segments with.
        Can be a single number or an array of the same shape as the image.
        If not given, excluded segments are replaced with `0`.

    Returns
    -------
    np.ndarray
        An array of shape `(num_of_samples, image_width, image_height, 3)`.
    """
    binary_segment_mask = np.zeros(
        shape=(samples.shape[0], image.shape[0], image.shape[1]), dtype=np.uint8
    )

    for sample_idx in range(binary_segment_mask.shape[0]):
        binary_segment_mask[sample_idx, :, :] = np.isin(
            segment_mask, np.nonzero(samples[sample_idx])
        ).astype(np.uint8)

    images = binary_segment_mask[:, :, :, None] * image

    if background is not None:
        images += (1 - binary_segment_mask)[:, :, :, None] * background

    return images


def predict_images(
    images: np.ndarray, predict_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Obtain model predictions for all images.

    Parameters
    ----------
    images : np.ndarray
        Images as an array of shape `(num_of_samples, image_width, image_height, 3)`.

    predict_fn : callable
        A function that takes an input of shape `(num_of_samples, image_width, image_height, 3)`
        and returns an array of shape `(num_of_samples, num_of_classes)`, where `num_of_classes`
        is the number of output classes (labels) assigned by the model.

        Commonly, `predict_fn()` feeds the images to the image classification model to be
        explained and takes care of any preprocessing and batching.
        When building explanation pipelines, it is generally preferable to replace `predict_images()`
        entirely.

    Returns
    -------
    np.ndarray
        An array of shape `(num_of_samples, num_of_classes)`.
    """
    return predict_fn(images)


def compute_distances(
    image: np.ndarray,
    images: np.ndarray,
    norm: Optional[Union[str, int]] = None,
    select: str = "sum",
) -> np.ndarray:
    """Calculate the distances between the original `image` and the generated `images`.

    Parameters
    ----------
    image : np.ndarray
        The original image.

    images : np.ndarray
        The sample images.

    norm : {non-zero int, np.inf, -np.inf, str}, optional
        The norm used to compute the distance between two images.
        It is calculated for the difference between each color channel.

        Defaults to the Frobenius norm if not given.

        For all available options, see
        `the documentation for numpy.linalg.norm
        <https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html>`_.

    select : {"sum", "max"}, default "sum"
        Method to combine the channel-wise distances to the final distance.

        There are two options:

        - `"sum"` (the default): Sum the channel-wise distances

        - `"max"`: Take the maximum of the channel-wise distances

    Returns
    -------
    np.ndarray
        Array of length `images.shape[0]` containing the distances of each image to the original image.
    """
    distances_per_channel = np.linalg.norm(images - image, axis=(1, 2), ord=norm)
    # TODO: Re-scaling for other norms
    if select == "sum":
        return np.sum(distances_per_channel, axis=1) / (
            255 * image.shape[2] * np.sqrt(np.prod(image.shape[0:2]))
        )
    elif select == "max":
        return np.max(distances_per_channel, axis=1) / (
            255 * np.sqrt(np.prod(image.shape[0:2]))
        )
    else:
        raise ValueError(f"Invalid value '{select}' for parameter 'select'.")


def weigh_segments(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    model_type: LINEAR_MODEL_TYPES = "bayesian_ridge",
    model_params: Optional[Dict[str, Any]] = None,
    distances: Optional[np.ndarray] = None,
    kernel: Callable[[np.ndarray], np.ndarray] = exponential_kernel,
    segment_subset: Optional[List[int]] = None,
) -> np.ndarray:
    linear_model = instantiate_model(model_type=model_type, model_params=model_params)

    if distances is None:
        distances = cosine_distance(samples)
    sample_weight = kernel(distances)

    if segment_subset is not None:
        if min(segment_subset) < 0:
            raise ValueError("Indices in segment subset cannot be below 0.")
        if max(segment_subset) > samples.shape[1]:
            raise ValueError(
                "Indices in segment subset exceed number of available segments."
            )
        reduced_samples = samples[:, segment_subset]
    else:
        reduced_samples = samples
        segment_subset = list(range(samples.shape[1]))

    linear_model.fit(
        reduced_samples, predictions[:, label_idx], sample_weight=sample_weight
    )

    reduced_weights = list(linear_model.coef_)

    segment_weights = np.array(
        [
            reduced_weights[segment_subset.index(feature_idx)]
            if feature_idx in segment_subset
            else 0.0
            for feature_idx in range(samples.shape[1])
        ]
    )

    return segment_weights


weigh_segments.__doc__ = f"""Generating list of coefficients to weigh segments.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}

    {MODEL_TYPE_PARAMS_DOC}

    {DISTANCES_KERNEL_DOC}

    segment_subset : list of ints, optional
        List of the indices of the segments to consider when fitting the linear model.
        Note that the resulting array will nevertheless have length `num_of_segments`.
        The weights of segments not in `segment_subset` will be `0.0`.

        If not given, all segments will be used.

    Returns
    -------
    np.ndarray
        Array of length `num_of_segments` where each entry corresponds to the segment's coefficient
        in the fitted linear model.
"""
