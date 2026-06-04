"""Measure how well the bed is covered, from a top-down camera.

PhysX keeps deformed particle-cloth positions in its own buffer rather than the
USD mesh, so instead of reading vertex positions we estimate coverage straight
from a bird's-eye render: classify each pixel over the bed as *sheet* or *bare
bed* by colour, and report the covered fraction plus a centreline check.

This gives the demo a **tolerant, physical goal**: the bed counts as "made"
when it is *good enough*, not Figure-perfect (matching issue #2's note to keep
the placement intentionally imperfect).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Coverage:
    coverage: float            # fraction of the bed top covered by the sheet
    centerline_aligned: bool   # sheet straddles the head->foot centreline
    midpoint_covered: bool     # the centre of the bed is covered
    good_enough: bool
    bed_px: int

    def as_dict(self) -> dict:
        return {
            "coverage_pct": round(self.coverage * 100, 1),
            "centerline_aligned": self.centerline_aligned,
            "midpoint_covered": self.midpoint_covered,
            "good_enough": self.good_enough,
        }


# Colour heuristics (tuned for the demo's light-lilac sheet on a brown bed).
def _masks(rgb):

    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    # Sheet: light and slightly bluish (high in all channels, blue >= red).
    sheet = (r > 140) & (g > 140) & (b > 150) & (b >= r - 10)
    # Bare bed: brown (red dominant, clearly warmer than blue).
    bed = (r > 60) & (r < 210) & (g < r) & (b < g) & ((r - b) > 25)
    return sheet, bed


def estimate_coverage(rgb, coverage_threshold: float = 0.5) -> Coverage:
    """Estimate bed coverage from a top-down RGB image framed on the bed.

    ``rgb`` is an (H, W, 3) uint8 array. The bed long axis (head->foot) is
    assumed horizontal in the image (width). Returns a :class:`Coverage`.
    """

    sheet, bed = _masks(rgb)
    bedtop = sheet | bed
    bed_px = int(bedtop.sum())
    coverage = float(sheet.sum()) / max(1, bed_px)

    H, W = sheet.shape
    # Midpoint: central patch of the bed is covered.
    cy0, cy1 = int(H * 0.42), int(H * 0.58)
    cx0, cx1 = int(W * 0.42), int(W * 0.58)
    centre = sheet[cy0:cy1, cx0:cx1]
    midpoint_covered = bool(centre.mean() > 0.5)

    # Centreline (head->foot = horizontal): a central horizontal band should be
    # mostly sheet across most of its length.
    by0, by1 = int(H * 0.45), int(H * 0.55)
    band = sheet[by0:by1, :]
    col_has_sheet = band.any(axis=0)
    centerline_aligned = bool(col_has_sheet.mean() > 0.6)

    good_enough = (coverage >= coverage_threshold) or (centerline_aligned and midpoint_covered)
    return Coverage(coverage, centerline_aligned, midpoint_covered, good_enough, bed_px)
