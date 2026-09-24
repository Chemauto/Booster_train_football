"""Generate a 16x11 m grass texture (256 px/m, 4096x2816) with white markings
baked in for a 14x9 m indoor soccer field. Playing area x:[-7,7], y:[-4.5,4.5];
the texture covers a 16x11 m plane so there is 1 m of grass margin around.
"""
import numpy as np
from PIL import Image

PXM = 256                      # pixels per meter
W, H = 16 * PXM, 11 * PXM      # 4096 x 2816

# world coords (meters) for every pixel
xs = (np.arange(W) + 0.5) / PXM - 8.0            # [-8, 8]
ys = 5.5 - (np.arange(H) + 0.5) / PXM            # [5.5, -5.5] (top = +y)
X, Y = np.meshgrid(xs, ys)

# ---- grass base: alternating mow stripes (1.4 m) + noise + vignette ----
stripe = ((X + 8.0) // 1.4).astype(np.int64) % 2 == 0
base = np.where(stripe[..., None],
                np.array([68, 148, 62], np.float32),
                np.array([54, 125, 50], np.float32))
rng = np.random.default_rng(7)
base += rng.normal(0, 5, (H, W, 1)) + rng.normal(0, 2, (H, W, 3))
r2 = ((X / 8.0) ** 2 + (Y / 5.5) ** 2) / 2.0
base *= (1.0 - 0.18 * r2)[..., None]
base = np.clip(base, 0, 255)

# ---- white markings ----
def seg_dist(px, py, ax, ay, bx, by):
    abx, aby = bx - ax, by - ay
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / (abx * abx + aby * aby), 0, 1)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))

LW = 0.03  # line half width (m) -> 6 cm lines
alpha = np.zeros((H, W), np.float32)
def line(ax, ay, bx, by):
    d = seg_dist(X, Y, ax, ay, bx, by)
    np.maximum(alpha, np.clip((LW + 0.004 - d) * PXM, 0, 1), out=alpha)

# boundary 14x9
line(-7, 4.5, 7, 4.5); line(-7, -4.5, 7, -4.5)
line(-7, -4.5, -7, 4.5); line(7, -4.5, 7, 4.5)
# halfway line
line(0, -4.5, 0, 4.5)
# goal areas 3.0 x 1.2
for s in (-1, 1):
    line(s * 7, -1.5, s * 5.8, -1.5); line(s * 7, 1.5, s * 5.8, 1.5)
    line(s * 5.8, -1.5, s * 5.8, 1.5)
# penalty areas 5.0 x 2.4
for s in (-1, 1):
    line(s * 7, -2.5, s * 4.6, -2.5); line(s * 7, 2.5, s * 4.6, 2.5)
    line(s * 4.6, -2.5, s * 4.6, 2.5)

# circles/arcs as ring masks
def ring(cx, cy, r, keep=None, tol=LW):
    d = np.hypot(X - cx, Y - cy)
    m = np.clip((tol + 0.004 - np.abs(d - r)) * PXM, 0, 1)
    if keep is not None:
        m *= keep
    np.maximum(alpha, m, out=alpha)

ring(0, 0, 1.5)                                    # centre circle
ring(-3.5, 0, 1.5, keep=(X > -4.6))                # left penalty arc
ring(3.5, 0, 1.5, keep=(X < 4.6))                  # right penalty arc
for cx in (-7, 7):                                 # corner arcs
    for cy in (-4.5, 4.5):
        keep = (np.abs(X - np.clip(X, -7, 7)) < 1) & (np.abs(Y - np.clip(Y, -4.5, 4.5)) < 1)
        keep = (np.sign(X - cx) == np.sign(np.clip(X, -7, 7) - cx)) if abs(cx - np.clip(X, -7, 7)).max() > 0 else keep
        ring(cx, cy, 0.3, keep=(np.abs(X) <= 7) & (np.abs(Y) <= 4.5) &
                                (np.abs(X - cx) < 0.31) & (np.abs(Y - cy) < 0.31))

# spots: centre + penalty marks
for (sx, sy, r) in ((0, 0, 0.09), (-3.5, 0, 0.08), (3.5, 0, 0.08)):
    d = np.hypot(X - sx, Y - sy)
    np.maximum(alpha, np.clip((r + 0.004 - d) * PXM, 0, 1), out=alpha)

img = base * (1 - alpha[..., None]) + np.array([246, 246, 246], np.float32) * alpha[..., None]
out = Image.fromarray(img.astype(np.uint8))
out = out.quantize(colors=192, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.FLOYDSTEINBERG)
out.save('grass_14x9.png', optimize=True)
print('saved grass_14x9.png', W, 'x', H)
