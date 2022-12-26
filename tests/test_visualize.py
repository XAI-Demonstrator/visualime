import numpy as np
import pytest

from visualime.visualize import _get_color, generate_overlay


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

    assert overlay.shape == (128, 128, 4)
    assert np.any(overlay[:, :, 3] == 255)
    # segment 1 is colored in blue
    assert np.all(overlay[5, 5:8, 2:4] == 255)
    assert np.all(overlay[5, 5:8, 0:2] == 0)
    # segment 2 is not colored
    assert np.all(overlay[12:24, 12:24] == 0)
    # segment 3 is colored in blue
    assert np.all(overlay[99, 101, 2:4] == 255)
    assert np.all(overlay[99, 101, 0:2] == 0)


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
