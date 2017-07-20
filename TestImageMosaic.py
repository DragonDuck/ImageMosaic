import unittest
import numpy as np
import ImageMosaic


class TestImageMosaic(unittest.TestCase):

    def test_create_mosaic_same_image_sizes(self):
        inputImages = [
            50 * np.ones((5, 5)),
            100 * np.ones((5, 5)),
            150 * np.ones((5, 5)),
        ]

        mosaic = ImageMosaic.create_mosaic(inputImages)


def test_images():
    input_images = [
        50 * np.ones((45, 45)),
        100 * np.ones((30, 60)),
        150 * np.ones((50, 20)),
    ]
    return input_images

images = test_images()
border_val=0
border_size=5
nrows=None
ncols=None