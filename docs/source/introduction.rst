Introduction to VisuaLIME
#########################

Getting Started
***************

To install *VisuaLIME*, run:

.. code-block:: bash

   pip install visualime

*VisuaLIME* provides two convenience functions (`explain_classification` and `render_explanation`)
that package its building blocks into a reference explanation pipeline.
While you'll likely want to create your own version later on, they're a great starting point.

.. note::
   If you're new to *LIME*, you might want to check out our
   `Grokking LIME <https://github.com/ionicsolutions/grokking-lime>`_
   talk/tutorial for a general introduction.

   It covers the fundamentals of loading and processing images,
   classifying images using a pre-trained deep learning model,
   and walks through the essential steps that *LIME* uses to
   generate an explanation.
   The simplified version of *LIME* you'll create as part of
   tutorial is directly derived from *VisuaLIME*.

Under the hood, *VisuaLIME* uses `numpy`.
Hence, you need to provide the image whose classification you want to explain as a `numpy` array
of shape `(width, height, 3)` representing an RGB image.

Since *LIME* is a model-agnostic explanation method, it does not make any assumptions about the
classifier you're using.
It's on you to provide a function that takes a `numpy` array of shape
`(num_of_samples, width, height, 3)` representing a collection of `num_of_samples` RGB images
and returns a `numpy` array of shape `(num_of_samples, num_of_classes)` where each entry
corresponds to the classifiers output for the respective image:

.. code-block:: python

   import numpy as np

   def predict_fn(images: np.ndarray) -> np.ndarray:
      predictions = ...  # call your classifier
      return predictions

To check that you've prepared everything correctly, try to generate a prediction for your image
as follows:

.. code-block:: python

   image = ...  # load your image as a three-dimensional numpy array

   predict_fn(image.reshape(1, -1))

*VisuaLIME* conceptually structures the explanation process into two steps:

1. Computing an explanation, which consists of a `segment_mask` and corresponding `segments_weights`
2. Rendering a visualization of the explanation

This is what generating an explanation with *VisuaLIME* looks like in code:

.. code-block:: python

    from visualime.explain import explain_classification, render_explanation

    segment_mask, segment_weights = explain_classification(image, predict_fn)

    explanation = render_explanation(
            image,
            segment_mask,
            segment_weights,
            positive="green",
            negative="red",
            coverage=0.2,
        )

For a full, interactive example with more detailed instructions, see
`the example notebook on GitHub <https://github.com/xai-demonstrator/visualime/blob/main/example.ipynb>`_.

Package overview
****************

*VisuaLIME* is currently structured into four modules:

- :doc:`explain`: Pre-configured explanation pipelines.
  Ideal as a starting point for new users and reference for custom implementations.
- :doc:`lime`: Contains each step of *LIME* implemented as a separate, independent function.
  Custom explanations pipelines can be created by chaining these functions.
- :doc:`feature_selection`: Functions to select which features (image segments) to consider for the explanation.
- :doc:`visualize`: Functions to render visual explanations that are comprehensible for humans.

.. automodule:: visualime
   :members:
   :undoc-members:
   :show-inheritance:
