"""Page views for test-time scale and orientation search (pure PIL geometry, no kraken).

blla resizes every page to its fixed input height (1800 px), so the size of
the writing at the model input depends only on how tall the page is relative
to its text.  Large-script fragments therefore reach the model with text far
bigger than anything it was trained on.  A *view* shrinks the text by a factor
``scale`` without losing resolution where it matters: the page is resized to
the model height and padded at the bottom to ``height / scale``; segmentation
geometry is mapped back to full resolution, and recognition always runs on the
full-resolution page.  Rotated views (90° / 270°) handle scans whose text runs
vertically.

Rotations follow PIL's ``Image.transpose`` naming: ``90`` = ``ROTATE_90``
(counter-clockwise), ``270`` = ``ROTATE_270`` (clockwise), ``180``.
"""
from typing import List, Sequence, Tuple

from PIL import Image

MODEL_HEIGHT = 1800
_TRANSPOSE = {90: Image.Transpose.ROTATE_90, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_270}


def rotate(im: Image.Image, rot: int) -> Image.Image:
    """Rotate a page for a view (lossless transpose).

    :param im: Page image.
    :type im: Image.Image
    :param rot: 0, 90 (counter-clockwise), 180 or 270 (clockwise).
    :type rot: int
    :return: Rotated image (the same object for 0).
    :rtype: Image.Image
    """
    return im if rot == 0 else im.transpose(_TRANSPOSE[rot])


def point_to_original(x: float, y: float, rot: int, width: int, height: int) -> Tuple[float, float]:
    """Map a point from a rotated view back to the original page.

    :param x: View x, px.
    :type x: float
    :param y: View y, px.
    :type y: float
    :param rot: View rotation (see :func:`rotate`).
    :type rot: int
    :param width: Original page width, px.
    :type width: int
    :param height: Original page height, px.
    :type height: int
    :return: Original ``(x, y)``.
    :rtype: Tuple[float, float]
    """
    if rot == 0:
        return x, y
    if rot == 90:                      # original (xo, yo) -> view (yo, W-1-xo)
        return width - 1 - y, x
    if rot == 270:                     # original (xo, yo) -> view (H-1-yo, xo)
        return y, height - 1 - x
    if rot == 180:
        return width - 1 - x, height - 1 - y
    raise ValueError(f"unsupported rotation {rot}")


def bbox_to_original(bbox: Sequence[float], rot: int, width: int, height: int) -> List[int]:
    """Map a view bounding box ``[x0, y0, x1, y1]`` back to the original page (clipped).

    :param bbox: View box.
    :type bbox: Sequence[float]
    :param rot: View rotation.
    :type rot: int
    :param width: Original page width, px.
    :type width: int
    :param height: Original page height, px.
    :type height: int
    :return: Original integer box.
    :rtype: List[int]
    """
    x0, y0, x1, y1 = bbox
    pts = [point_to_original(x, y, rot, width, height) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    clip = lambda v, hi: int(min(max(v, 0), hi - 1))  # noqa: E731
    return [clip(min(xs), width), clip(min(ys), height), clip(max(xs), width), clip(max(ys), height)]


def border_fill(im: Image.Image) -> Tuple:
    """Median colour of the bottom 2 % of a page (padding colour for a view).

    :param im: Page image (any mode).
    :type im: Image.Image
    :return: A fill value valid for ``im.mode``.
    :rtype: Tuple
    """
    h = im.height
    strip = im.crop((0, max(0, h - max(2, h // 50)), im.width, h)).resize((64, 4))
    px = sorted(strip.getdata(), key=lambda v: sum(v) if isinstance(v, tuple) else v)
    return px[len(px) // 2]


def seg_view(im: Image.Image, scale: float) -> Tuple[Image.Image, float, int]:
    """Build the segmentation input for a text-shrinking view.

    :param im: (Rotated) full-resolution page.
    :type im: Image.Image
    :param scale: Text shrink factor in ``(0, 1]``.
    :type scale: float
    :return: ``(view image, resize factor f, content height)``: view coordinates / f = full-resolution
        coordinates; rows at or below ``content height`` are padding.
    :rtype: Tuple[Image.Image, float, int]
    """
    f = min(1.0, MODEL_HEIGHT / im.height)
    small = im if f == 1.0 else im.resize((max(1, round(im.width * f)), max(1, round(im.height * f))),
                                          Image.Resampling.LANCZOS)
    if scale >= 1.0:
        return small, f, small.height
    view = Image.new(small.mode, (small.width, int(round(small.height / scale))), border_fill(small))
    view.paste(small, (0, 0))
    return view, f, small.height
