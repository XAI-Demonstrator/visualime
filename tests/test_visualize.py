import numpy as np
import pytest

from visualime.visualize import (
    _get_color,
    generate_overlay,
    mark_boundaries,
    scale_opacity,
    scale_overlay,
    select_segments,
)


def test_that_either_coverage_or_num_of_segments_has_to_be_specified():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([0])
    with pytest.raises(ValueError):
        _ = select_segments(segment_mask=segment_mask, segment_weights=segment_weights)

    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            coverage=0.5,
            num_of_segments=1,
        )


def test_that_min_coverage_has_to_be_smaller_than_max_coverage():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([0])
    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            num_of_segments=1,
            min_coverage=0.8,
            max_coverage=0.2,
        )


def test_that_min_coverage_cannot_equal_max_coverage():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([0])
    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            num_of_segments=1,
            min_coverage=0.8,
            max_coverage=0.8,
        )


def test_that_min_num_of_segments_has_to_be_smaller_than_max_num_of_segments():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([0])
    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            coverage=0.5,
            min_num_of_segments=15,
            max_num_of_segments=5,
        )


def test_that_min_num_of_segments_cannot_equal_max_num_of_segments():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([0])
    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            coverage=0.5,
            min_num_of_segments=5,
            max_num_of_segments=5,
        )


def test_that_each_segment_needs_a_weight():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_weights = np.array([1.0])
    too_many_segment_weights = np.array([0.5, 0.2, 0.5])

    _ = select_segments(
        segment_mask=segment_mask, segment_weights=segment_weights, coverage=0.5
    )

    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask,
            segment_weights=too_many_segment_weights,
            coverage=0.5,
        )

    segment_mask[1, 1] = 1

    with pytest.raises(ValueError):
        _ = select_segments(
            segment_mask=segment_mask, segment_weights=segment_weights, coverage=0.5
        )


def test_that_user_is_warned_if_all_segments_are_selected_to_reach_coverage_threshold():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[5:10, 5:10] = 1
    segment_weights = np.array([0.7, 0.3])

    with pytest.warns(RuntimeWarning):
        _ = select_segments(
            segment_mask=segment_mask, segment_weights=segment_weights, coverage=0.99
        )


def test_that_correct_number_of_segments_is_selected():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[5:10, 5:10] = 1
    segment_weights = np.array([0.2, 0.8])

    selected_segments = select_segments(
        segment_mask=segment_mask, segment_weights=segment_weights, num_of_segments=1
    )

    assert selected_segments.shape[0] == 1


def test_that_more_segments_are_selected_if_coverage_is_below_threshold():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[5:10, 5:10] = 1
    segment_weights = np.array([0.2, 0.8])

    selected_segments = select_segments(
        segment_mask=segment_mask,
        segment_weights=segment_weights,
        num_of_segments=1,
        min_coverage=0.99,
    )

    assert selected_segments.shape[0] == 2


def test_that_fewer_segments_are_selected_if_coverage_is_above_threshold():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[5:10, 0:10] = 1
    segment_mask[1:5, 1:5] = 2
    segment_weights = np.array([0.1, 0.5, 0.4])

    with pytest.warns(UserWarning):
        selected_segments = select_segments(
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            num_of_segments=2,
            max_coverage=0.3,
        )

    assert selected_segments.shape[0] == 1


def test_that_overlay_is_transparent_if_no_segments_are_colored():
    segment_mask = np.zeros((128, 128), dtype=int)
    overlay = generate_overlay(
        segment_mask=segment_mask, segments_to_color=[], color="red", opacity=1.0
    )

    assert overlay.shape == (128, 128, 4)
    assert np.all(overlay[:, :, 3] == 0)


def test_that_selected_segments_are_colored():
    segment_mask = np.zeros((128, 128), dtype=int)
    segment_mask[5, 5:8] = 1
    segment_mask[12:24, 12:24] = 2
    segment_mask[99, 101] = 3

    overlay = generate_overlay(
        segment_mask=segment_mask, segments_to_color=[1, 3], color="blue", opacity=1.0
    )

    # basic properties
    assert overlay.shape == (128, 128, 4)
    assert np.min(overlay) == 0
    assert np.max(overlay) == 255
    assert np.any(overlay[:, :, 3] == 255)
    # segment 1 is colored in blue
    assert np.all(overlay[5, 5:8, 2:4] == 255)
    assert np.all(overlay[5, 5:8, 0:2] == 0)
    # segment 2 is not colored
    assert np.all(overlay[12:24, 12:24] == 0)
    # segment 3 is colored in blue
    assert np.all(overlay[99, 101, 2:4] == 255)
    assert np.all(overlay[99, 101, 0:2] == 0)


def test_that_opacity_is_rescaled_for_selected_segments():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[2:3, 2:3] = 1
    segments_to_color = [1]
    segment_weights = np.array([0.0, 3.0])

    overlay = generate_overlay(
        segment_mask=segment_mask,
        segments_to_color=segments_to_color,
        color="red",
        opacity=1.0,
    )

    assert np.all(overlay[2:3, 2:3, 3] == 255)

    scaled_overlay = scale_opacity(
        overlay=overlay,
        segment_mask=segment_mask,
        segment_weights=segment_weights,
        segments_to_color=segments_to_color,
        max_opacity=0.5,
    )

    assert np.all(scaled_overlay[2:3, 2:3, 3] == 127)


def test_that_opacity_is_scaled_to_custom_reference():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[2:5, 2:5] = 1
    segments_to_color = [1]
    segment_weights = np.array([0.0, 3.0])

    overlay = generate_overlay(
        segment_mask=segment_mask,
        segments_to_color=segments_to_color,
        color="red",
        opacity=1.0,
    )

    assert np.all(overlay[2:5, 2:5, 3] == 255)

    scaled_overlay = scale_opacity(
        overlay=overlay,
        segment_mask=segment_mask,
        segment_weights=segment_weights,
        segments_to_color=segments_to_color,
        max_opacity=0.5,
        relative_to=0.5,
    )

    assert np.all(scaled_overlay[2:5, 2:5, 3] == 127)


def test_that_opacity_is_scaled_to_maximum():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[2:5, 2:5] = 1
    segments_to_color = [1]
    segment_weights = np.array([0.0, 3.0])

    overlay = generate_overlay(
        segment_mask=segment_mask,
        segments_to_color=segments_to_color,
        color="red",
        opacity=1.0,
    )

    assert np.all(overlay[2:5, 2:5, 3] == 255)

    scaled_overlay = scale_opacity(
        overlay=overlay,
        segment_mask=segment_mask,
        segment_weights=segment_weights,
        segments_to_color=segments_to_color,
        max_opacity=0.5,
    )

    assert np.all(scaled_overlay[2:5, 2:5, 3] == 127)


@pytest.mark.parametrize("number", [1.1, 1, np.float32(0.5), np.uint8(1)])
def test_that_relative_to_can_be_passed_as_float_compatible(number):
    segment_mask = np.zeros((10, 10), dtype=int)
    segments_to_color = [0]
    segment_weights = np.array([1.0])

    overlay = generate_overlay(
        segment_mask=segment_mask,
        segments_to_color=segments_to_color,
        color="red",
        opacity=1.0,
    )

    _ = scale_opacity(
        overlay=overlay,
        segment_mask=segment_mask,
        segment_weights=segment_weights,
        segments_to_color=segments_to_color,
        relative_to=number,
    )


def test_that_invalid_relative_to_values_are_handled():
    segment_mask = np.zeros((10, 10), dtype=int)
    segments_to_color = [0]
    segment_weights = np.array([1.0])

    overlay = generate_overlay(
        segment_mask=segment_mask,
        segments_to_color=segments_to_color,
        color="red",
        opacity=1.0,
    )

    with pytest.raises(ValueError):
        _ = scale_opacity(
            overlay=overlay,
            segment_mask=segment_mask,
            segment_weights=segment_weights,
            segments_to_color=segments_to_color,
            relative_to="this-option-does-not-exist-and-never-will",
        )


def test_scale_overlay():
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[2, 2:2] = 1
    segment_mask[3, 2:3] = 1
    segment_mask[4, 2:4] = 1
    segment_mask[5, 2:5] = 1
    segment_mask[6, 2:6] = 1

    segments_to_color = [1]
    color = "red"
    opacity = 1.0

    overlay = generate_overlay(
        segment_mask=segment_mask,
        segments_to_color=segments_to_color,
        color=color,
        opacity=opacity,
    )
    # show the overlay

    scaled_overlay = scale_overlay(overlay=overlay, shape=(100, 100))

    assert scaled_overlay.shape == (100, 100, 4)


@pytest.mark.parametrize(
    "color_name,rgb",
    [("green", [0, 128, 0]), ("magenta", [255, 0, 255]), ("purple", [128, 0, 128])],
)
def test_that_color_can_be_obtained_by_name(color_name, rgb):
    assert np.array_equal(_get_color(color_name, 1.0), np.array(rgb + [255]))


def test_that_invalid_color_names_raise_exception():
    with pytest.raises(ValueError):
        _get_color("this-color-does-not-exist-and-never-will", 0.5)


def test_that_invalid_channel_values_raise_exception():
    with pytest.raises(ValueError):
        _get_color((-5, 2, 10), 0.5)

    with pytest.raises(ValueError):
        _get_color((50, 2000, 0), 0.5)


@pytest.mark.parametrize("color", ["salmon", [12, 24, 48], [0, 0, 255, 40]])
def test_that_color_is_always_returned_as_rgba_array(color):
    assert _get_color(color, 1.0).shape == (4,)


def test_that_mark_boundaries_rejects_mismatched_mask():
    with pytest.raises(ValueError):
        mark_boundaries(image=np.zeros((10, 11)), segment_mask=np.zeros((10, 10)))

    with pytest.raises(ValueError):
        mark_boundaries(image=np.zeros((10, 10)), segment_mask=np.zeros((10, 11)))


def test_that_mark_boundaries_works_for_rgb_images():
    mark_boundaries(image=np.zeros((10, 10, 3)), segment_mask=np.zeros((10, 10)))


def test_that_mark_boundaries_highlights_boundaries():
    image = np.zeros((10, 10, 3))
    segment_mask = np.zeros((10, 10), dtype=int)
    segment_mask[2:5, 2:5] = 1

    marked = mark_boundaries(image=image, segment_mask=segment_mask)

    assert np.all(marked[3:5, 3:5, :] == 0)
    assert np.all(marked[2:5, 2, 0] == 255)
    assert np.all(marked[2:5, 5, 0] == 255)
    assert np.all(marked[2, 2:5, 0] == 255)
    assert np.all(marked[5, 2:5, 0] == 255)
