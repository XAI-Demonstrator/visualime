"""XAI Demonstrator LIME explainer"""
from typing import Dict
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.filters import sobel
from xaidemo.tracing import traced

from .generate import generate_visual_explanation
from .lime import create_segments, generate_samples, generate_images, predict_images, weigh_segments
from ..config import settings

@traced
def explain_image(img: np.ndarray, seg_method: str, seg_settings: Dict, num_of_samples: int, samples_p: float,
                  model_: tf.keras.models.Model, threshold: float, volume: int, colour: str,
                  transparency: float) -> np.ndarray:
    segment_mask = create_segments(img=img, seg_method=seg_method, settings=seg_settings)
    samples_theo = generate_samples(segment_mask=segment_mask, num_of_samples=num_of_samples, p=samples_p)
    samples_imgs = generate_images(image=img, segment_mask=segment_mask, samples=samples_theo)
    samples_imgs_predictions = predict_images(images=samples_imgs, model_=model_)
    weighted_segments = weigh_segments(samples=samples_theo, predictions=samples_imgs_predictions)
    visual_explanation = generate_visual_explanation(weighted_segments=weighted_segments, segment_mask=segment_mask,
                                                     image=img, threshold=threshold, volume=volume, colour=colour,
                                                     transparency=transparency)

    return visual_explanation

