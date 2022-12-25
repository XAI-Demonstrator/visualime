import itertools

import numpy as np
import pytest

from visualime.lime import (
    SEGMENTATION_METHODS,
    compute_distances,
    create_segments,
    generate_images,
    generate_samples,
    weigh_segments,
)

MAX_SEGMENT_INDEX = 25

img_size_x, img_size_y = (224, 224)

image = np.random.randint(0, 255, size=(img_size_x, img_size_y, 3))
segment_mask = np.random.randint(0, MAX_SEGMENT_INDEX, size=(img_size_x, img_size_y))
samples = generate_samples(segment_mask, 100, p=0.5)


def test_that_create_segments_only_accepts_known_methods():
    with pytest.raises(ValueError):
        _ = create_segments(
            image=image, segmentation_method="this-method-does-not-exist-and-never-will"
        )


@pytest.mark.parametrize("segmentation_method", list(SEGMENTATION_METHODS.keys()))
def test_image_segmentation(segmentation_method):
    segment_mask = create_segments(image=image, segmentation_method=segmentation_method)

    assert np.min(segment_mask) == 0


def test_that_replacing_background_with_image_generates_original_image():
    images = generate_images(image, segment_mask, samples, background=image)

    for img_idx in range(images.shape[0]):
        assert (images[img_idx, :, :, :] == image).all()


@pytest.mark.parametrize(
    "norm,select",
    list(
        itertools.product(
            [None, "fro", "nuc", np.inf, -np.inf, 1, -1, 2, -2], ["sum", "max"]
        )
    ),
)
def test_that_image_distances_are_computed(norm, select):
    images = generate_images(image, segment_mask, samples, background=1)

    distances = compute_distances(image, images, norm=norm, select=select)

    assert distances.shape == (100,)


def test_that_invalid_select_method_raises_exception():
    images = generate_images(image, segment_mask, samples, background=1)

    with pytest.raises(ValueError):
        _ = compute_distances(image, images, select="this-method-does-not-exist")


def test_that_weigh_segments_only_accepts_known_models():
    with pytest.raises(ValueError):
        _ = weigh_segments(
            samples=samples,
            predictions=np.zeros((samples.shape[0], 10)),
            label_idx=0,
            model_type="this-model-does-not-exist-and-never-will",
        )


def test_segment_subset_validation_in_weigh_segments():
    with pytest.raises(ValueError):
        _ = weigh_segments(
            samples=samples,
            predictions=np.zeros((samples.shape[0], 5)),
            label_idx=0,
            segment_subset=[-5, 0, 1],
        )

    with pytest.raises(ValueError):
        _ = weigh_segments(
            samples=samples,
            predictions=np.zeros((samples.shape[0], 5)),
            label_idx=0,
            segment_subset=[0, 1, 2, MAX_SEGMENT_INDEX + 1],
        )
