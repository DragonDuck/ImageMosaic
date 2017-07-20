# NumpyMosaic
Create a mosaic of a list of numpy arrays, essentially a 2D concatenation, with an optional border inbetween images. It can take images of different sizes and centers them in their individual fields.

The package consists of a single function, 'create_mosaic(...)', which takes the following arguments:
* images: An iterable of 2D or 3D numpy arrays. If they're 3D then they must have the shape (spatial, spatial, channels).
* nrows (default: None): The number of rows of the mosaic.
* ncols (default: None): The number of columns of the mosaic.
* border_val (default: 0): The value of the background and border
* border_size (default: 5): The size of the border between images
* rows_first (default: True): A boolean indicating if the mosaic should be populated by rows (True) or columns (False) first

## Installation
Install this package via PyPI:

    pip install ImageMosaic

Or by downloading this repo and running

    python setup.py install

## Usage Example
An example is provided below (this code is also available in the file TestImageMosaic.py):

    import numpy as np
    import ImageMosaic
    import matplotlib.pyplot as plt
    
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

    mosaic2d = test_create_2d_mosaic()
    mosaic3d = test_create_3d_mosaic()
    
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(mosaic2d)
    plt.subplot(1, 2, 2)
    plt.imshow(mosaic3d)
