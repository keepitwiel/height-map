import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter


@njit
def diamond_square(
        scale: int,
        smoothness: float,
        random_seed: int,
) -> np.ndarray:
    """
    Creates a square grid where the values are determined by the diamond/square algorithm.

    :param scale: grid scale. the grid size (n) is determined by 2**scale + 1.
        So, if scale=5, then n=2**5 + 1 = 33.
    :param smoothness: determines how quickly the noisiness of the values decreases
        for each consecutive step. The higher this value, the faster the noise decreases.
        If smoothness=0, the noise amplitude does not decrease.
    :param random_seed: random seed.
    :return: Numpy array.
    """
    np.random.seed(random_seed)
    n = 2**scale + 1  # grid size
    z = np.zeros((n, n))
    for q in range(scale, 0, -1):
        d = 2**q  # step size
        h = 2**(q-1)  # half step
        size = 2**(scale-q) + 1

        # add random corners to grid
        r = np.random.normal(0, 2**(q - smoothness), (size, size))
        z[0::d, 0::d] += r

        # interpolate edges
        z[h:n:d, 0:n:d] = 0.5 * z[0:n-d:d, 0:n:d] + 0.5 * z[d:n:d, 0:n:d]
        z[0:n:d, h:n:d] = 0.5 * z[0:n:d, 0:n-d:d] + 0.5 * z[0:n:d, d:n:d]

        # interpolate middle
        z[h:n:d, h:n:d] = 0.25 * (
            z[0:n-d:d, 0:n-d:d]
            + z[0:n-d:d, d:n:d]
            + z[d:n:d, 0:n-d:d]
            + z[d:n:d, d:n:d]
        )
    return z


@njit
def bilinear(slope, scale):
    n = 2 ** scale
    result = np.zeros((n + 1, n + 1))
    for j in range(n + 1):
        for i in range(n + 1):
            x = i / n
            y = j / n
            result[j, i] = (
                (1 - x) * (1 - y) * slope[0, 0]
                + x * (1 - y) * slope[0, 1]
                + (1 - x) * y * slope[1, 0]
                + x * y * slope[1, 1]
            )
    return result * n


def generate(
        scale: int = 6,
        random: float = 1.0,
        slope: np.ndarray = np.array([[False, False], [False, False]]),
        random_seed: int = 42,
        smoothness: float = 0.6,
        final_blur: int = None,
):
    """
    Generates a height map using a linear combination between
        the diamond/square algorithm and a slope.

    Optionally, a blur is applied to the output.

    :param slope: indicates if corner elements of the grid should be raised
        to create a sloped terrain.
    :param random: a number between 0 and 1. If 0, output will be wholly determined
        by the slope parameter; if 1, there is no slope and output will be wholly
        determined by the diamond/square algorithm.
    :param scale: grid scale. the grid size (n) is determined by 2**scale + 1.
        So, if scale=5, then n=2**5 + 1 = 33.
    :param random_seed: random seed
    :param smoothness: determines noisiness of the grid.
        - high value => more noise but less "blocky"
        - low value => less noise but also more artificial looking
    :param final_blur: blur parameter to further smooth out noise
    :return: height map
    """
    try:
        assert 0 <= random <= 1
    except AssertionError:
        print(f"'Random' parameter needs to be between 0 and 1. Value found: {random}")
    if random == 0:
        z = bilinear(slope, scale)
    elif 0 < random < 1:
        z_ = bilinear(slope, scale)
        ds = diamond_square(scale, smoothness, random_seed)
        z = random * ds + (1 - random) * z_
    else:
        z = diamond_square(scale, smoothness, random_seed)

    if final_blur:
        z = gaussian_filter(z, sigma=final_blur)

    return z
