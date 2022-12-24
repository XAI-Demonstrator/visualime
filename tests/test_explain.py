from urllib.request import urlopen

import numpy as np
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


def test_end_to_end():
    camera_img = FULL_IMAGE.crop((766, 90, 990, 314))

    image = np.array(camera_img)
    assert image.shape == (224, 224, 3)

    prediction = predict_fn(image[None, :, :, :])
    assert np.argmax(prediction, axis=1)[0] == 759

    segment_mask, segment_weights = explain_classification(
        image=image,
        predict_fn=predict_fn,
        num_of_samples=128,
    )

    assert segment_mask.shape == (224, 224)

    _ = render_explanation(
        image,
        segment_mask,
        segment_weights,
        positive="green",
        negative="red",
        coverage=0.2,
    )
