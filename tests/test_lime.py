import numpy as np

from visualime.lime import generate_images, generate_samples


def test_that_replacing_background_with_image_generates_original_image():
    img_size_x, img_size_y = (128, 255)

    image = np.random.randint(0, 255, size=(img_size_x, img_size_y, 3))
    segment_mask = np.random.randint(0, 25, size=(img_size_x, img_size_y))
    samples = generate_samples(segment_mask, 100, p=0.5)

    images = generate_images(image, segment_mask, samples, background=image)

    for img_idx in range(images.shape[0]):
        assert (images[img_idx, :, :, :] == image).all()
