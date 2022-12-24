from typing import Optional, List

import numpy as np
from sklearn.linear_model import lars_path

from .lime import weigh_segments, default_distance, LINEAR_MODELS, SAMPLES_PREDICTIONS_LABEL_IDX_DOC, MODEL_TYPE_DOC, DISTANCES_DOC


def highest_weight(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    model_type: str = "bayesian_ridge",
    distances: Optional[np.ndarray] = None,
    num_segments_to_select: Optional[int] = None,
) -> List[int]:
    f"""Select the `num_segments_to_select` with the highest weight.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}
    
    {MODEL_TYPE_DOC}
    
        It is generally advisable to use the same model as for the final `weigh_segments()` function.
    
    {DISTANCES_DOC}
    
    num_segments_to_select : int, optional
        The number of segments to select.

    Returns
    -------
    List of the indices of the selected segments.
    
    """
    num_segments = samples.shape[1]
    num_segments_to_select = num_segments_to_select or num_segments
    if num_segments_to_select > num_segments:
        raise ValueError(
            f"Number of features to select ({num_segments_to_select}) cannot exceed "
            f"number of features in data ({num_segments})"
        )

    segment_weights = weigh_segments(
        samples=samples,
        predictions=predictions,
        label_idx=label_idx,
        model_type=model_type,
        distances=distances,
        segment_subset=None,
    )

    return list(np.argsort(-np.abs(segment_weights))[:num_segments_to_select])


def forward_selection(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    model_type: str = "ridge",
    distances: Optional[np.ndarray] = None,
    num_segments_to_select: Optional[int] = None,
) -> List[int]:
    f"""Select `num_segments_to_select` through forward selection.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}
    
    {MODEL_TYPE_DOC}
    
        It is generally advisable to use the same model as for the final `weigh_segments()` function.
    
    {DISTANCES_DOC}
      
    num_segments_to_select : int, optional
        The number of segments to select.

    Returns
    -------
    List of the indices of the selected segments.

    """
    num_segments = samples.shape[1]
    num_segments_to_select = num_segments_to_select or num_segments
    if num_segments_to_select > num_segments:
        raise ValueError(
            f"Number of features to select ({num_segments_to_select}) cannot exceed "
            f"number of features in data ({num_segments})"
        )

    distances = distances or default_distance(samples)

    try:
        linear_model = LINEAR_MODELS[model_type](alpha=0, fit_intercept=True)
    except KeyError:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Available options: {list(LINEAR_MODELS.keys())}"
        )

    def score(current_features: List[int], next_feature_idx: int):
        linear_model.fit(
            samples[:, current_features + [next_feature_idx]],
            predictions[:, label_idx],
            sample_weight=distances,
        )

        return linear_model.score(
            samples[:, current_features + [next_feature_idx]],
            predictions[:, label_idx],
            sample_weight=distances,
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


def lars_selection(
    samples: np.ndarray,
    predictions: np.ndarray,
    label_idx: int,
    num_segments_to_select: Optional[int] = None,
) -> List[int]:
    f"""Select up to `num_segments_to_select` segments using the LARS path method.

    Parameters
    ----------
    {SAMPLES_PREDICTIONS_LABEL_IDX_DOC}
    
    num_segments_to_select : int, optional
        The maximum number of segments to select.

    Returns
    -------
    List of the indices of the selected segments.

    """
    num_segments = samples.shape[1]
    num_segments_to_select = num_segments_to_select or num_segments
    if num_segments_to_select > num_segments:
        raise ValueError(
            f"Number of features to select ({num_segments_to_select}) cannot exceed "
            f"number of features in data ({num_segments})"
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
