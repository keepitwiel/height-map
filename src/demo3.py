import numpy as np
import matplotlib.pyplot as plt

from height_map import generate

"""
Demo 3
"""


fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i, random in enumerate([0.0, 0.5, 1.0]):
    for j, smoothness in enumerate([-1.0, 1.0, 3.0]):
        z = generate(
            scale=8,
            random=random,
            smoothness=smoothness,
            slope=np.array([[False, True], [True, False]]),
        )
        mh, mw = z.shape
        x, y = np.meshgrid(np.arange(0, mw), np.arange(0, mh))
        axes[i, j].set_title(f"rnd: {random:2.1f}, smooth: {smoothness:2.1f}")
        cs = axes[i, j].contour(x, y, z)
        axes[i, j].clabel(cs, inline=True, fontsize=12)
        axes[i, j].imshow(z, cmap="terrain", vmin=-32, vmax=2**8 + 32)
        axes[i, j].xaxis.set_visible(False)
        axes[i, j].yaxis.set_visible(False)

plt.show()
