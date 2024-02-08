import numpy as np
from matplotlib import patches, path, pyplot as plt
import jellybamm


class RoundedPolygon(patches.PathPatch):
    def __init__(self, xy, pad, **kwargs):
        p = path.Path(*self.__round(xy=xy, pad=pad))
        super().__init__(path=p, **kwargs)

    def __round(self, xy, pad):
        n = len(xy)

        for i in range(0, n):
            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

            d01, d12 = x1 - x0, x2 - x1
            d01, d12 = d01 / np.linalg.norm(d01), d12 / np.linalg.norm(d12)

            x00 = x0 + pad * d01
            x01 = x1 - pad * d01
            x10 = x1 + pad * d12

            if i == 0:
                verts = [x00, x01, x1, x10]
            else:
                verts += [x01, x1, x10]
        codes = [path.Path.MOVETO] + n * [
            path.Path.LINETO,
            path.Path.CURVE3,
            path.Path.CURVE3,
        ]

        return np.atleast_1d(verts, codes)


alpha = 0.5
c1 = np.array([75 / 255, 139 / 255, 190 / 255, alpha])  # Cyan-Blue Azure
c1e = np.array([48 / 255, 105 / 255, 152 / 255, alpha])  # Lapis Lazuli
c2 = np.array([1, 232 / 255, 115 / 255, alpha])  # Shandy
c2e = np.array([1, 212 / 255, 59 / 255, alpha])  # Sunglow
c3 = np.array([100 / 255, 100 / 255, 100 / 255, alpha])  # Granite Gray
# Test
xy = np.array(
    [
        (0, 0),
        (0.25, 0),
        (0.5, -0.25),
        (0.75, 0),
        (1, 0),
        (1, 0.25),
        (1.25, 0.5),
        (1, 0.75),
        (1, 1),
        (0.75, 1),
        (0.5, 1.25),
        (0.25, 1),
        (0, 1),
        (0, 0.75),
        (-0.25, 0.5),
        (0, 0.25),
    ]
)
r = 1
dr = 0.5
ntheta = 36 * 4
n = 3


(x1, y1, r1, pos1) = jellybamm.spiral(r, dr, ntheta, n)
(x2, y2, r2, pos2) = jellybamm.spiral(r + dr / 2, dr, ntheta, n)
(x3, y3, r3, pos3) = jellybamm.spiral(r + dr, dr, ntheta, n)

xy12 = np.vstack((np.hstack((x1, x2[::-1])), np.hstack((y1, y2[::-1])))).T
rp1 = RoundedPolygon(xy=xy12, pad=dr / 10, facecolor=c1, edgecolor=c1e, lw=1)
xy23 = np.vstack((np.hstack((x2, x3[::-1])), np.hstack((y2, y3[::-1])))).T
rp2 = RoundedPolygon(xy=xy23, pad=0.1, facecolor=c2, edgecolor=c2e, lw=1)

fig, ax = plt.subplots(figsize=(10, 10))
ax.add_patch(rp1)
ax.add_patch(rp2)

ax.set_aspect(1)
ax.axis("off")
limit = x3.max() * 1.1
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
# OpenPNM project
project, arc_edges = jellybamm.make_spiral_net(
    Nlayers=n,
    dtheta=10,
    spacing=dr / 2,
    inner_r=1,
    pos_tabs=[0],
    neg_tabs=[-1],
    length_3d=1,
    tesla_tabs=False,
)
net = project.network
jellybamm.plot_topology(net, ax=ax)


plt.tight_layout()
plt.savefig("jellybamm_logo.png", transparent=True)
