from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class CylindricalRangeFilter:
    """
    円筒形に点群をフィルタリング
    """
    radius_m: float = 10.0
    z_min_m: float = 0.0
    z_max_m: float = 30.0
    cx: float = 0.0
    cy: float = 0.0
    keep_nans: bool = False

    def apply(self, points: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if points is None:
            return points

        # キー確認
        if not all(k in points for k in ("x", "y", "z")):
            raise ValueError("points must have keys 'x','y','z'")

        x = np.asarray(points["x"])
        y = np.asarray(points["y"])
        z = np.asarray(points["z"])

        if x.ndim != 1 or y.ndim != 1 or z.ndim != 1:
            raise ValueError(f"x,y,z must be 1-D arrays; got {x.shape},{y.shape},{z.shape}")
        if not (len(x) == len(y) == len(z)):
            raise ValueError(f"x,y,z length mismatch: {len(x)}, {len(y)}, {len(z)}")

        dx = x - self.cx
        dy = y - self.cy
        r2 = dx * dx + dy * dy

        m = (r2 <= (self.radius_m * self.radius_m)) & (z >= self.z_min_m) & (z <= self.z_max_m)

        if self.keep_nans:
            nan_row = np.isnan(x) | np.isnan(y) | np.isnan(z)
            m = m | nan_row

        out: Dict[str, np.ndarray] = {}
        N = len(x)
        for k, v in points.items():
            a = np.asarray(v)
            if a.ndim == 1 and len(a) == N:
                out[k] = a[m]
            else:
                out[k] = a
        return out