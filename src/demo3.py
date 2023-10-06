import numpy as np
import matplotlib.pyplot as plt

from height_map import generate

"""
Demo 3
"""


z = generate(
    scale=8,
    amplitude=1.0,
    smoothness=0.5,
    raised_corners=np.array([[False, True], [True, False]]),
    final_blur=2.0,
)
mh, mw = z.shape
x, y = np.meshgrid(np.arange(0, mw), np.arange(0, mh))
plt.figure(figsize=(4, 4))
cs = plt.contour(x, y, z, levels=4)
plt.title(f"Height map ({mw}x{mh}), min: {np.min(z):2.2f}, max: {np.max(z):2.2f}")
plt.clabel(cs, inline=True, fontsize=12)
plt.imshow(z, cmap="terrain")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
