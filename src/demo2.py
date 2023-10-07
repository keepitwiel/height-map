import numpy as np
import matplotlib.pyplot as plt

from height_map import generate

"""
Demo 2: additional features

Same as Demo 1, but with additional features enabled:
- amplitude: here we set this parameter to 0, making the result a smooth grid.
- slope: we indicate that the top left corner should be elevated.
"""


z = generate(
    random=0,
    slope=np.array([[True, False], [False, False]])
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
