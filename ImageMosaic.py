import numpy as np


def create_mosaic(images, nrows=None, ncols=None):
    """
    Create a mosaic of 'images' where 'images' has the shape 
    (fields, spatial, spatial, [channels]). The function creates a square 
    mosaic, unless nrows or ncols is set. If both are set, then 
    nrows*ncols must be >= len(images) 
    :param images: An iterable of 2D or 3D numpy arrays
    :param nrows: Custom number of rows
    :param ncols: Custom number of columns
    :return: 
    """

    if len(images.shape) != 4 and len(images.shape) != 3:
        raise Exception("'images' must be a 3D or 4D numpy array")

    n_fields = images.shape[0]
    if n_fields == 1:
        return images[0, ...]

    if ncols is None and nrows is None:
        n_cols = int(np.ceil(np.sqrt(n_fields)))
        n_rows = int(np.ceil(n_fields / n_cols))
    elif ncols is None and nrows is not None:
        n_rows = nrows
        n_cols = int(np.ceil(n_fields / n_rows))
    elif ncols is not None and nrows is None:
        n_cols = ncols
        n_rows = int(np.ceil(n_fields / n_cols))
    else:
        n_cols = ncols
        n_rows = nrows
    img_w = images.shape[2]
    img_h = images.shape[1]
    border = 5
    if len(images.shape) == 3:
        out_block = np.ones((img_h * n_rows + border * (n_rows - 1),
                             img_w * n_cols + border * (n_cols - 1)),
                            dtype=images.dtype)
    else:
        out_block = np.ones((img_h * n_rows + border * (n_rows - 1),
                             img_w * n_cols + border * (n_cols - 1),
                             images.shape[3]), dtype=images.dtype)
    for fld in range(n_fields):
        nr = int(np.floor(fld / n_rows))
        nc = fld - n_rows * nr
        x_start = nc * (img_w + border)
        x_end = x_start + img_w
        y_start = nr * (img_h + border)
        y_end = y_start + img_h
        out_block[x_start:x_end, y_start:y_end, ...] = images[fld, ...]

    return out_block
