# Checks the 2x upscaled TIFF: size, bit depth, colour profile, content and tile seams.
import sys
import numpy as np
from PIL import Image

path, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
im = Image.open(path)
print("size", im.size, "mode", im.mode, "icc bytes", len(im.info.get("icc_profile") or b""))
ok = True
if im.size != (2 * w, 2 * h):
    print("FAIL: expected", (2 * w, 2 * h)); ok = False
import tifffile  # 16 bit RGB is not handled by Pillow
a = tifffile.imread(path).astype(np.float32)
print("array", a.shape, "min", a.min(), "max", a.max(), "mean", a.mean(axis=(0, 1)).round(1))
if a.shape != (2 * h, 2 * w, 3):
    print("FAIL: unexpected array shape"); ok = False
if a.max() - a.min() < 500:
    print("FAIL: the image is blank"); ok = False
if not im.info.get("icc_profile"):
    print("FAIL: no colour profile embedded"); ok = False
# tiles are 448 input pixels apart; a seam would show as an outlier in the row/column steps
gy = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
gx = np.abs(np.diff(a, axis=1)).mean(axis=(0, 2))
for name, g in (("rows", gy), ("columns", gx)):
    seam = g[870:920]
    ref = np.median(g[700:1100])
    print("%s: largest step near the seam %.1f, typical %.1f" % (name, seam.max(), ref))
    if seam.max() > 4 * ref + 50:
        print("FAIL: visible seam between tiles (%s)" % name); ok = False
sys.exit(0 if ok else 1)
