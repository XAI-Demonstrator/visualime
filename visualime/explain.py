from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as PIL_Image

from .feature_selection import forward_selection, select_by_weight
from .lime import (
    compute_distances,
    create_segments,
    generate_images,
    generate_samples,
    predict_images,
    weigh_segments,
)
from .visualize import generate_overlay, scale_opacity, select_segments


def explain_classification(
    image: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    label_idx: Optional[int] = None,
    segmentation_method: str = "slic",
    segmentation_settings: Optional[Dict[str, Any]] = None,
    num_of_samples: int = 64,
    p: float = 0.33,
    segment_selection_method: str = "by_weight",
    num_segments_to_select: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Explain why the classifier called through `predict_fn` classifies the `image` into
    a particular class using the LIME algorithm.

    For more detailed control, we recommend you create your own function, using this
    function as a template.

    Parameters
    ----------
    image : np.ndarray
        The image to explain as a three-dimensional array of shape
        `(image_width, image_height, 3)` representing an RGB image.

    predict_fn : Callable
        A function that takes an input of shape `(num_of_samples, image_width, image_height, 3)`
        and returns an array of shape `(num_of_samples, num_of_classes)`.

    label_idx : int, optional
        The index of the label to explain in the output of `predict_fn()`.
        If not given, this corresponds to the class that `predict_fn()` assigns
        to the image.

    segmentation_method : str, default "slic"
        The method used to segment the image into superpixels.
        See :meth:`visualime.lime.create_segments` for available methods.

    segmentation_settings : dict, optional
        Keyword arguments to pass to the segmentation method.
        See :meth:`visualime.lime.create_segments` for details.

    num_of_samples : int, default 64
        The number of sample images to generate for calculating the explanation.

    p : float, default 0.33
        The probability of a segment to be replaced in a sample.

    segment_selection_method : str, default "by_weight"
        The segment selection method.
        Possible choices are "by_weight" and "forward_selection".

    num_segments_to_select : int, optional
        The number of segments to be considered when fitting the linear model to determine the `segment_weights`.
        If not given, half of the generated segments are selected.

    Returns
    -------
    segment_mask : np.ndarray
        A two-dimensional array of shape `(image_width, image_height)`.

    segment_weights : np.ndarray
        A one-dimensional array whose length is equal to the number of segments.

    Examples
    --------
    TODO: Add end-to-end example
    """
    model_type = "bayesian_ridge"

    if label_idx is None:
        label_idx = int(np.argmax(predict_fn(image[None, :, :, :])))

    segment_mask = create_segments(
        image=image,
        segmentation_method=segmentation_method,
        segmentation_settings=segmentation_settings,
    )

    samples = generate_samples(
        segment_mask=segment_mask, num_of_samples=num_of_samples, p=p
    )

    images = generate_images(image=image, segment_mask=segment_mask, samples=samples)

    predictions = predict_images(images=images, predict_fn=predict_fn)

    distances = compute_distances(image=image, images=images)

    num_segments_to_select = num_segments_to_select or int(samples.shape[1] / 2)

    if segment_selection_method == "by_weight":
        segment_subset = select_by_weight(
            samples=samples,
            predictions=predictions,
            label_idx=label_idx,
            model_type=model_type,
            num_segments_to_select=num_segments_to_select,
        )
    elif segment_selection_method == "forward_selection":
        segment_subset = forward_selection(
            samples=samples,
            predictions=predictions,
            label_idx=label_idx,
            model_type=model_type,
            num_segments_to_select=num_segments_to_select,
        )
    else:
        raise ValueError(
            "Segment selection method has to be either 'by_weight' or 'forward_selection'."
        )

    segment_weights = weigh_segments(
        samples=samples,
        predictions=predictions,
        label_idx=label_idx,
        model_type=model_type,
        distances=distances,
        segment_subset=segment_subset,
    )

    return segment_mask, segment_weights


def render_explanation(
    image: np.ndarray,
    segment_mask: np.ndarray,
    segment_weights: np.ndarray,
    positive: Optional[Union[Tuple[int, int, int], str]] = "green",
    negative: Optional[Union[Tuple[int, int, int], str]] = None,
    opacity: float = 0.7,
    coverage: float = 0.2,
) -> PIL_Image:
    """Render a visual explanation from the `segment_mask` and `segment_weights`
     produced by :meth:`visualime.explain.explain_classification`.

    Parameters
    ----------
    image : np.ndarray
        The image to explain the classification for as a three-dimensional array
        of shape `(image_width, image_height, 3)` representing an RGB image.

    segment_mask : np.ndarray
        The mask generated by :meth:`visualime.lime.create_segments`:
        An array of shape `(image_width, image_height)`.

    segment_weights : np.ndarray
        The weights produced by :meth:`visualime.lime.weigh_segments`:
        A one-dimensional array of length `num_of_segments`.

    positive : str or int 3-tuple (RGB), optional, default "green"
        The color for the segments that contribute positively towards the classification.
        If `None`, these segments are not colored.

    negative : str or int 3-tuple (RGB), optional
        The color for the segments that contribute negatively towards the classification.
        If `None` (the default), these segments are not colored.

    opacity : float, default 0.7
        The opacity of the explanation overlay.

    coverage : float, default 0.2
        The coverage of each overlay relative to the area of the image.
        E.g., if set to 0.2 (the default), about 20% of the image are colored.

    Returns
    -------
    PIL.Image
        The rendered explanation as a PIL Image object.

    Examples
    --------
    TODO: Add end-to-end example
    """
    final_img = Image.fromarray(image.astype(np.uint8), "RGB").convert("RGBA")

    if positive is not None:
        positive_segments = select_segments(
            segment_weights, segment_mask, coverage=coverage
        )
        positive_overlay = generate_overlay(
            segment_mask, positive_segments, color=positive, opacity=opacity
        )

        positive_overlay = scale_opacity(
            overlay=positive_overlay,
            segment_weights=segment_weights,
            segment_mask=segment_mask,
            segments_to_color=positive_segments,
        )

        overlay_image = Image.fromarray(positive_overlay.astype(np.uint8), "RGBA")
        final_img.alpha_composite(overlay_image)

    if negative is not None:
        negative_segments = select_segments(
            -segment_weights, segment_mask, coverage=coverage
        )
        negative_overlay = generate_overlay(
            segment_mask, negative_segments, color=negative, opacity=opacity
        )

        negative_overlay = scale_opacity(
            overlay=negative_overlay,
            segment_weights=segment_weights,
            segment_mask=segment_mask,
            segments_to_color=negative_segments,
        )

        overlay_image = Image.fromarray(negative_overlay.astype(np.uint8), "RGBA")
        final_img.alpha_composite(overlay_image)

    return final_img
