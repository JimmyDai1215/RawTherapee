# Checks that a developed AI-denoised DNG matches the developed original raw in geometry,
# brightness and colour, and that it is actually less noisy.
import sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.float32)
print("original :", a.shape, "mean", a.mean(axis=(0, 1)).round(2))
print("denoised :", b.shape, "mean", b.mean(axis=(0, 1)).round(2))
if a.shape != b.shape:
    sys.exit("FAIL: the two images have different sizes")

def box(img, n):
    h, w = img.shape[0] // n * n, img.shape[1] // n * n
    return img[:h, :w].reshape(h // n, n, w // n, n, 3).mean(axis=(1, 3))

la, lb = box(a, 16), box(b, 16)
diff = np.abs(la - lb)
print("low-frequency difference: mean %.2f, 99th percentile %.2f (of 255)" % (diff.mean(), np.percentile(diff, 99)))

def noise(img):
    # energy that does not survive a 2x2 average: mostly noise
    h, w = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2
    c = img[:h, :w]
    m = c.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    return float(np.sqrt(((c - np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)) ** 2).mean()))

na, nb = noise(a), noise(b)
print("fine-grain energy: original %.3f, denoised %.3f" % (na, nb))
ok = True
if diff.mean() > 2.0 or np.percentile(diff, 99) > 12.0:
    print("FAIL: brightness or colour differs too much"); ok = False
if not nb < na * 0.8:
    print("FAIL: the denoised image is not clearly less noisy"); ok = False
sys.exit(0 if ok else 1)
