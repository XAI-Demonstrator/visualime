# VisuaLIME

_VisuaLIME_ is an implementation of _LIME_ [1] focused on producing visual local explanations
for image classifiers created as part of the
[XAI Demonstrator project](https://github.com/XAI-Demonstrator/xai-demonstrator).

In contrast to the [reference implementation](https://github.com/marcotcr/lime), _VisuaLIME_
exclusively supports image classification and gives its users full control over the
properties of the generated explanations.
It was written to produce stable, reliable, and expressive explanations at scale.

## Getting Started

💡 _If you're new to_ LIME, _you might want to check out the_
[Grokking LIME](https://github.com/ionicsolutions/grokking-lime)
_talk/tutorial for a general introduction prior to diving into_ VisuaLIME.

To install _VisuaLIME_, run:

```shell
pip install visualime
```

_VisuaLIME_ provides two functions that package its building blocks into a reference explanation
pipeline:

```python
import numpy as np
from visualime.explain import explain_classification, render_explanation

image = ...  # a numpy array of shape (width, height, 3) representing an RGB image

def predict_fn(images: np.ndarray) -> np.ndarray:
    # a function that takes a numpy array of shape (num_of_samples, width, height, 3)
    # representing num_of_samples RGB images and returns a numpy array of
    # shape (num_of_samples, num_of_classes) where each entry corresponds to the
    # classifiers output for the respective image
    predictions = ...
    return predictions

segment_mask, segment_weights = explain_classification(image, predict_fn)

explanation = render_explanation(
        image,
        segment_mask,
        segment_weights,
        positive="green",
        negative="red",
        coverage=0.2,
    )
```

For a full example, see
[the example notebook on GitHub](https://github.com/xai-demonstrator/visualime/blob/main/example.ipynb).

## Roadmap

- Verify that the algorithm matches the original LIME and document differences
- Build out a comprehensive unit test suite
- Add performance benchmarks and optimize implementation of the algorithm
- Include utilities to assess and tune explanations for stability and truthfulness
- Provide a best practice example of implementing a fully configurable LIME explainer
- Add support for BayLIME

## References

[1] Ribeiro et al.: _"Why Should I Trust You?": Explaining the Predictions of Any Classifier_
    ([arXiv:1602.04938](https://arxiv.org/abs/1602.04938), 2016)
