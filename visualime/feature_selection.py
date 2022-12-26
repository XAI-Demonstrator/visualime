from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import lars_path

from ._models import MODEL_TYPE_PARAMS_DOC, instantiate_model
from .lime import SAMPLES_PREDICTIONS_LABEL_IDX_DOC
from .metrics import DISTANCES_KERNEL_DOC, cosine_distance, exponential_kernel


def _get_num_segments(samples: np.ndarray, num_segments_to_select: Optional[int]):
    num_segments = samples.shape[1]
    num_segments_to_select = num_segments_to_select or num_segments
    if num_segments_to_select > num_segments:
        raise ValueError(
            f"Number of features to select ({num_segments_to_select}) cannot exceed "
            f"number of features in data ({num_segments})"
        )
    return num_segments, num_segments_to_select


def select_by_weight(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    model_type: str = "bayesian_ridge",
    model_params: Optional[Dict[str, Any]] = None,
    distances: Optional[np.ndarray] = None,
    kernel: Callable[[np.ndarray], np.ndarray] = exponential_kernel,
    num_segments_to_select: Optional[int] = None,
) -> List[int]:
    num_segments, num_segments_to_select = _get_num_segments(
        samples, num_segments_to_select
    )

    if distances is None:
        distances = cosine_distance(samples)
    sample_weight = kernel(distances)

    linear_model = instantiate_model(model_type=model_type, model_params=model_params)

    selector = SelectFromModel(
        estimator=linear_model, threshold=-np.inf, max_features=num_segments_to_select
    )
    selector.fit(X=samples, y=predictions[:, label_idx], sample_weight=sample_weight)

    return list(selector.get_support(indices=True))


select_by_weight.__doc__ = f"""Select the `num_segments_to_select` segments with the highest weight.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}

    {MODEL_TYPE_PARAMS_DOC}

        It is generally advisable to use the same model as for the final
        :meth:`visualime.lime.weigh_segments` function.

    {DISTANCES_KERNEL_DOC}

    num_segments_to_select : int, optional
        The number of segments to select. If not given, select all segments.

    Returns
    -------
    list of ints
        List of the indices of the selected segments.
        Segments are ordered by descending weight.
"""


def forward_selection(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    model_type: str = "ridge",
    model_params: Optional[Dict[str, Any]] = None,
    distances: Optional[np.ndarray] = None,
    kernel: Callable[[np.ndarray], np.ndarray] = exponential_kernel,
    num_segments_to_select: Optional[int] = None,
) -> List[int]:
    num_segments, num_segments_to_select = _get_num_segments(
        samples, num_segments_to_select
    )

    if distances is None:
        distances = cosine_distance(samples)
    sample_weight = kernel(distances)

    # TODO: Understand and account for the implications of regularization
    linear_model = instantiate_model(model_type=model_type, model_params=model_params)

    # TODO: Wait for https://github.com/scikit-learn/scikit-learn/issues/25236
    def score(current_features: List[int], next_feature_idx: int):
        linear_model.fit(
            samples[:, current_features + [next_feature_idx]],
            predictions[:, label_idx],
            sample_weight=sample_weight,
        )

        return linear_model.score(
            samples[:, current_features + [next_feature_idx]],
            predictions[:, label_idx],
            sample_weight=sample_weight,
        )

    selected_segments = []
    for _ in range(num_segments_to_select):
        selectable_segments = set(range(num_segments)) - set(selected_segments)
        scores = (
            (score(selected_segments, segment_idx), segment_idx)
            for segment_idx in selectable_segments
        )
        segment_with_highest_score = max(scores, key=lambda x: x[0])[1]
        selected_segments.append(segment_with_highest_score)

    return selected_segments


forward_selection.__doc__ = f"""Select `num_segments_to_select` through forward selection.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}

    {MODEL_TYPE_PARAMS_DOC}

        It is generally advisable to use the same model as for the final
        :meth:`visualime.lime.weigh_segments` function.

    {DISTANCES_KERNEL_DOC}

    num_segments_to_select : int, optional
        The number of segments to select. If not given, select all segments.

    Returns
    -------
    list of ints
        List of the indices of the selected segments.
        The segments are ordered as they were selected.
"""


def lars_selection(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    num_segments_to_select: Optional[int] = None,
) -> List[int]:
    num_segments, num_segments_to_select = _get_num_segments(
        samples, num_segments_to_select
    )

    _, _, coefs, num_of_iterations = lars_path(
        samples, predictions[:, label_idx], return_path=True, return_n_iter=True
    )

    for iteration in range(num_of_iterations, 0, -1):
        segments_with_nonzero_coefficients = coefs.T[iteration].nonzero()[0]
        if len(segments_with_nonzero_coefficients) <= num_segments_to_select:
            break
    else:
        raise RuntimeError(
            f"Could not find subset of {num_segments_to_select} features"
        )

    return list(segments_with_nonzero_coefficients)


lars_selection.__doc__ = f"""Select up to `num_segments_to_select` segments using the LARS path method.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}

    num_segments_to_select : int, optional
        The maximum number of segments to select.
        If not given, this value is set to the total number of segments.

    Returns
    -------
    list of ints
        List of the indices of the selected segments.
        The segment indices are in ascending order.
"""
