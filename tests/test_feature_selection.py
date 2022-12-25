import numpy as np

from visualime.feature_selection import (
    forward_selection,
    lars_selection,
    select_by_weight,
)

NUM_SAMPLES = 100
NUM_SEGMENTS = 10
LABEL_IDX = 0
TRUE_COEFS = np.array([0.0, 0.0, 0.05, 0.0, 0.5, 0.15, 0.0, 0.0, 0.3, 0.0])
assert TRUE_COEFS.shape[0] == NUM_SEGMENTS

samples = (np.random.rand(NUM_SAMPLES, NUM_SEGMENTS) > 0.5).astype(int)

predictions = np.array(np.sum(TRUE_COEFS * samples, axis=1)).reshape((NUM_SAMPLES, 1))
assert predictions.shape == (NUM_SAMPLES, 1)


def test_that_select_by_weight_recovers_top_feature():
    segment_subset = select_by_weight(
        samples=samples,
        predictions=predictions,
        label_idx=LABEL_IDX,
        num_segments_to_select=1,
    )

    assert len(segment_subset) == 1
    assert segment_subset == [int(np.argmax(TRUE_COEFS))]


def test_that_select_by_weight_recovers_top_3_features():
    segment_subset = select_by_weight(
        samples=samples,
        predictions=predictions,
        label_idx=LABEL_IDX,
        num_segments_to_select=3,
    )

    assert len(segment_subset) == 3
    assert segment_subset == list(map(int, np.argsort(-TRUE_COEFS)[:3]))


def test_that_forward_selection_recovers_top_feature():
    segment_subset = forward_selection(
        samples=samples,
        predictions=predictions,
        label_idx=LABEL_IDX,
        num_segments_to_select=1,
    )

    assert len(segment_subset) == 1
    assert segment_subset == [int(np.argmax(TRUE_COEFS))]


def test_that_forward_selection_recovers_top_3_features():
    segment_subset = forward_selection(
        samples=samples,
        predictions=predictions,
        label_idx=LABEL_IDX,
        num_segments_to_select=3,
    )

    assert len(segment_subset) == 3
    assert segment_subset == list(map(int, np.argsort(-TRUE_COEFS)[:3]))


def test_that_lars_selection_recovers_top_feature():
    segment_subset = lars_selection(
        samples=samples,
        predictions=predictions,
        label_idx=LABEL_IDX,
        num_segments_to_select=1,
    )

    assert len(segment_subset) == 1
    assert segment_subset == [int(np.argmax(TRUE_COEFS))]
