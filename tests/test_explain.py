from urllib.request import urlopen

import numpy as np
import pytest
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from visualime.explain import explain_classification, render_explanation

model = MobileNetV2()


def predict_fn(image):
    return model.predict(preprocess_input(image))


FULL_IMAGE = Image.open(
    urlopen(
        "https://storage.googleapis.com/xai-demo-assets/visual-inspection/images/table.jpg"
    )
)

camera_img = FULL_IMAGE.crop((766, 90, 990, 314))

image = np.array(camera_img)
assert image.shape == (224, 224, 3)

prediction = predict_fn(image[None, :, :, :])
assert np.argmax(prediction, axis=1)[0] == 759


@pytest.mark.parametrize("segment_selection_method", ["forward_selection", "by_weight"])
def test_end_to_end(segment_selection_method):
    segment_mask, segment_weights = explain_classification(
        image=image,
        predict_fn=predict_fn,
        num_of_samples=128,
        segment_selection_method=segment_selection_method,
    )

    assert segment_mask.shape == (224, 224)

    _ = render_explanation(
        image,
        segment_mask,
        segment_weights,
        positive="green",
        negative="red",
        coverage=0.2,
        min_num_of_segments=1,
        max_num_of_segments=10,
    )

    _ = render_explanation(
        image,
        segment_mask,
        segment_weights,
        positive="green",
        negative="red",
        coverage=None,
        num_of_segments=2,
    )


def test_that_unknown_selection_method_raises_exception():
    with pytest.raises(ValueError):
        _, _ = explain_classification(
            image=image,
            predict_fn=predict_fn,
            num_of_samples=1,
            segment_selection_method="this-method-does-not-exist",
        )


def test_that_either_coverage_or_num_of_segments_has_to_be_specified():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([0])
    with pytest.raises(ValueError):
        _ = render_explanation(
            image=image,
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            coverage=None,
            num_of_segments=None,
        )

    with pytest.raises(ValueError):
        _ = render_explanation(
            image=image,
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            coverage=0.5,
            num_of_segments=1,
        )
