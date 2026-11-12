# Checks that a local adjustment with an AI mask changes the selection and nothing else.
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
off = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32)
on = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.float32)
mask = Image.open(sys.argv[3])
print("images", off.shape, on.shape, "mask", mask.size)
if off.shape != on.shape:
    sys.exit("FAIL: image sizes differ")
m = np.asarray(mask.resize((off.shape[1], off.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
area = float((m > 0.5).mean())
print("selected fraction of the image: %.3f" % area)
ok = True
if not 0.02 < area < 0.95:
    print("FAIL: the selection is implausibly small or large"); ok = False
diff = np.abs(on - off).mean(axis=2)
inside = float(diff[m > 0.9].mean()) if (m > 0.9).any() else 0.0
outside = float(diff[m < 0.02].mean()) if (m < 0.02).any() else 0.0
print("mean change inside the selection %.2f, outside %.3f (of 255)" % (inside, outside))
if inside < 1.5:
    print("FAIL: the local adjustment did not change the selection"); ok = False
if outside > 0.3 or outside > 0.1 * inside:
    print("FAIL: the local adjustment leaked outside the selection"); ok = False
sys.exit(0 if ok else 1)
