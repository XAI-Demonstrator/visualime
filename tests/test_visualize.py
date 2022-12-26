import numpy as np
import pytest

from visualime.visualize import _get_color


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
