import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter


#@njit
def diamond_square(
        scale: int,
        amplitude: float,
        smoothness: float,
        random_seed: int,
        raised_corners: np.ndarray = np.array([[False, False], [False, False]]),
) -> np.ndarray:
    """
    Creates a square grid where the values are determined by the diamond/square algorithm.

    :param scale: grid scale. the grid size (n) is determined by 2**scale + 1.
        So, if scale=5, then n=2**5 + 1 = 33.
    :param amplitude: amplitude of the randomness. If 0, the grid values are a bilinear
        interpolation of the initial corner values.
    :param smoothness: determines how quickly the noisiness of the values decreases
        for each consecutive step. The higher this value, the faster the noise decreases.
        If smoothness=0, the noise amplitude does not decrease.
    :param random_seed: random seed.
    :param raised_corners: indicates if corner elements of the grid are raised, to create
        a sloped terrain. If raised, the corner values are set to 2**p.
    :return: Numpy array.
    """
    np.random.seed(random_seed)
    n = 2**scale + 1  # grid si
    raised_corners = raised_corners * 2**scale

    z = np.zeros((n, n))
    z[0, 0] = raised_corners[0, 0]
    z[0, -1] = raised_corners[0, 1]
    z[-1, 0] = raised_corners[1, 0]
    z[-1, -1] = raised_corners[1, 1]

    # z[0:-1:n//2, 0:-1:n//2] = raised_corners * 2**scale  # initialize corners

    for q in range(scale, 0, -1):
        d = 2**q  # step size
        h = 2 ** (q - 1)  # half step

        size = 1 + 2 ** (scale - q)

        # random corners
        r = np.random.normal(0, amplitude * d**smoothness, (size, size))

        # add to grid
        z[0::d, 0::d] += r

        # interpolate edges
        z[h:n:d, 0:n:d] = 0.5 * z[0 : n - d : d, 0:n:d] + 0.5 * z[d:n:d, 0:n:d]
        z[0:n:d, h:n:d] = 0.5 * z[0:n:d, 0 : n - d : d] + 0.5 * z[0:n:d, d:n:d]

        # interpolate middle
        z[h:n:d, h:n:d] = 0.25 * (
            z[0 : n - d : d, 0 : n - d : d]
            + z[0 : n - d : d, d:n:d]
            + z[d:n:d, 0 : n - d : d]
            + z[d:n:d, d:n:d]
        )

    return z


def generate(
        scale: int = 6,
        amplitude: float = 1,
        raised_corners: np.ndarray = np.array([[False, False], [False, False]]),
        random_seed: int = 42,
        smoothness: float = 0.6,
        final_blur: int = None,
):
    """
    Generates a height map using the diamond/square algorithm.

    If the specified map dimensions each are less than 2**p+1 (for integer p >= 0),
    the output of the diamond/square algorithm will be cropped.

    Optionally, a blurr is applied to the output.

    :param scale: grid scale. the grid size (n) is determined by 2**scale + 1.
        So, if scale=5, then n=2**5 + 1 = 33.
    :param amplitude: amplitude of the randomness. If 0, the grid values are a bilinear
        interpolation of the initial corner values.
    :param raised_corners: indicates if corner elements of the grid should be raised
        to create a sloped terrain.
    :param random_seed: random seed
    :param smoothness: determines noisiness of the grid.
        - high value => more noise but less "blocky"
        - low value => less noise but also more artificial looking
    :param final_blur: blur parameter to further smooth out noise
    :return: height map
    """
    z = diamond_square(scale, amplitude, smoothness, random_seed, raised_corners)
    if final_blur:
        z = gaussian_filter(z, sigma=final_blur)

    return z
