
from specula import float_dtype_list

def calc_spatialfrequency(dimension, xp, precision):
    """
    This function returns a square matrix of size [dimension X dimension]
    with the spatial frequencies calculated as row^2 + column^2.
    The null frequency is in [dim-1,dim-1].

    Parameters:
    - dimension: The dimension of the square matrix.
    - precision: If True, use double precision; otherwise, use single precision.

    Returns:
    - matrix: A square matrix of size [dimension X dimension].
    """

    dtype = float_dtype_list[precision]

    half_dim = dimension // 2

    temp_matrix = xp.zeros((half_dim + 1, half_dim + 1), dtype=dtype)
    matrix = xp.zeros((dimension, dimension), dtype=dtype)

    x = xp.arange(half_dim + 1, dtype=dtype)  # Make a row
    for i in range(half_dim + 1):
        temp_matrix[:, i] = x**2 + i**2  # Generate 1/4 of matrix

    # Place temp_matrix in the top-right corner and then
    # fill the rest of the matrix by rotating temp_matrix appropriately
    matrix[half_dim - 1:, half_dim - 1:] = temp_matrix
    matrix[:half_dim, :half_dim] = xp.rot90(temp_matrix, 2)[1:, 1:]
    matrix[:half_dim, half_dim - 1:] = xp.rot90(temp_matrix, 1)[1:, :]
    matrix[half_dim -1 :, :half_dim] = xp.rot90(temp_matrix, 3)[:, 1:]

    return matrix
