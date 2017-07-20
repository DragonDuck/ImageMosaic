import numpy as np
import ImageMosaic


def test_create_2d_mosaic():
    images = [
        50 * np.ones((45, 45)),
        100 * np.ones((30, 60)),
        150 * np.ones((50, 20)),
        200 * np.ones((40, 30)),
        250 * np.ones((10, 10)),
    ]

    return ImageMosaic.create_mosaic(images=images)


def test_create_3d_mosaic():
    images = [
        50 * np.ones((45, 45, 3), dtype=np.uint8),
        100 * np.ones((30, 60, 3), dtype=np.uint8),
        150 * np.ones((50, 20, 3), dtype=np.uint8),
        200 * np.ones((40, 30, 3), dtype=np.uint8),
        250 * np.ones((10, 10, 3), dtype=np.uint8),
    ]

    return ImageMosaic.create_mosaic(images=images, rows_first=False)

