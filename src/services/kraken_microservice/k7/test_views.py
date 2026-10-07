"""Tests for the test-time view geometry (run: python -m pytest src/services/kraken_microservice/k7/test_views.py)."""
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from views import bbox_to_original, point_to_original, rotate, seg_view  # noqa: E402


def _marked(w: int = 7, h: int = 4, at: tuple = (5, 1)) -> Image.Image:
    """Black image with one white pixel.

    :param w: Width.
    :type w: int
    :param h: Height.
    :type h: int
    :param at: Marked pixel.
    :type at: tuple
    :return: Image.
    :rtype: Image.Image
    """
    im = Image.new("L", (w, h), 0)
    im.putpixel(at, 255)
    return im


def test_rotation_round_trip_every_pixel() -> None:
    """Every pixel of a rotated view maps back to the original pixel it came from."""
    w, h = 7, 4
    for rot in (0, 90, 180, 270):
        for xo in range(w):
            for yo in range(h):
                view = rotate(_marked(w, h, (xo, yo)), rot)
                (xv, yv), = [(x, y) for x in range(view.width) for y in range(view.height) if view.getpixel((x, y))]
                assert point_to_original(xv, yv, rot, w, h) == (xo, yo), (rot, xo, yo)


def test_bbox_round_trip() -> None:
    """A box drawn in a rotated view maps back onto the same original pixels."""
    w, h = 9, 5
    orig = Image.new("L", (w, h), 0)
    for x in range(2, 5):
        for y in range(1, 3):
            orig.putpixel((x, y), 255)
    for rot in (0, 90, 180, 270):
        v = rotate(orig, rot)
        pts = [(x, y) for x in range(v.width) for y in range(v.height) if v.getpixel((x, y))]
        box = [min(p[0] for p in pts), min(p[1] for p in pts), max(p[0] for p in pts), max(p[1] for p in pts)]
        assert bbox_to_original(box, rot, w, h) == [2, 1, 4, 2], rot


def test_seg_view_scale_and_padding() -> None:
    """A half-scale view of a tall page is model height / 0.5 tall, content on top, geometry scale kept."""
    im = Image.new("RGB", (1200, 3600), (200, 180, 150))
    view, f, content_h = seg_view(im, 0.5)
    assert abs(f - 0.5) < 1e-9 and content_h == 1800 and view.size == (600, 3600)
    view1, f1, h1 = seg_view(im, 1.0)
    assert view1.size == (600, 1800) and h1 == 1800 and abs(f1 - 0.5) < 1e-9
    small = Image.new("RGB", (800, 900), (10, 10, 10))
    v2, f2, h2 = seg_view(small, 0.5)
    assert f2 == 1.0 and h2 == 900 and v2.size == (800, 1800) and v2.getpixel((5, 1700)) == (10, 10, 10)


def test_scaled_view_geometry_maps_inside_the_page() -> None:
    """A polygon point in a shrunk view's padding maps (after clamping) inside the full-resolution page."""
    im = Image.new("L", (1400, 3000), 255)
    view, f, content_h = seg_view(im, 0.5)
    w, h = im.size
    to_full = lambda p: (min(max(int(p[0] / f), 0), w - 1), min(max(int(p[1] / f), 0), h - 1))  # noqa: E731
    # a seam point 12 px into the padding (as blla produces for a bottom line) and the view's far corner
    for p in [(100, content_h + 12), (view.width - 1, view.height - 1)]:
        x, y = to_full(p)
        assert 0 <= x < w and 0 <= y < h
    assert int((content_h + 12) / f) >= h        # unclamped it would fall outside the page
