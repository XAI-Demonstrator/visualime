import numpy as np


def generate_visual_explanation(weighted_segments: np.ndarray, segment_mask: np.ndarray, image: np.ndarray,
                                threshold: float, volume: int, colour: str, transparency: float = 0) -> np.ndarray:
    """Generating image with visual explanation
    Parameters
    ----------
    weighted_segments
    segment_mask
    image
    threshold
    volume
    colour
    transparency
    Returns
    -------
    """
    # set explanation colour
    colours = {"green": [0, 255, 0], "blue": [38, 55, 173], "red": [173, 38, 38], "white": [255, 255, 255],
               "black": [0, 0, 0], "violet": [215, 102, 255]}
    colour = colour.lower()
    if colour not in colours.keys():
        colour = "green"

    # handle outliers
    """
    weighted_segments = np.where(weighted_segments > 1.0, 1, weighted_segments)
    weighted_segments = np.where(weighted_segments < -1.0, -1, weighted_segments)
    # normalize coefficients: coefficient_i ∈ [0.0, 1.0]
    n_weighted_segments = (weighted_segments - weighted_segments.min()) / (
            weighted_segments.max() - weighted_segments.min())
    """
    n_weighted_segments = 1 / (1 + np.exp(-weighted_segments))

    # check if volume is bigger than the amount of segments
    max_volume = len(np.unique(segment_mask))
    if volume > max_volume:
        volume = max_volume

    # differentiate n_weighted_segments with respect to threshold and volume
    # values less than max(limit, threshold) are set to 0
    limit = np.sort(np.unique(n_weighted_segments))[-volume]
    n_d_weighted_segments = np.where(n_weighted_segments >= max(limit, threshold), n_weighted_segments,
                                     0)
    # manipulate the original image (quick and dirty)
    c = np.array(colours[colour])
    image_c = image.copy()
    indices = np.argwhere(n_d_weighted_segments != 0)
    for i, row in enumerate(segment_mask):
        for j, el in enumerate(row):
            if el in indices:
                image_c[i, j] = ((round(n_d_weighted_segments[el], 1) * c / 127.5) - 1)

    return image_c * transparency + image * (1 - transparency)
