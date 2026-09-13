"""Physical curve geometry with lumen-checked cubic fits and arc integration."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy import ndimage as ndi

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(12)


class Centerline:
    """A bounded curve in physical image axes (ZYX mm).

    Arc length integrates ||r'(t)||. Seed placement inverts that integral.
    Translation/rotation into LPS is unnecessary for lengths and curvature.
    Unsupported fits fall back to the original physical polyline.
    """
    def __init__(self, path, spacing, lumen=None, parent=None, max_mm=np.inf):
        self.spacing = np.asarray(spacing, float)
        physical = np.asarray(path, float) * self.spacing
        if physical.ndim != 2 or physical.shape[1] != 3 or not np.isfinite(physical).all():
            raise ValueError("centerline requires finite Nx3 points")
        keep = np.r_[True, np.linalg.norm(np.diff(physical, axis=0), axis=1) > 1e-8]
        physical = physical[keep]
        if len(physical) < 2:
            raise ValueError("centerline requires two distinct points")
        self.raw = physical
        self.raw_arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(physical, axis=0), axis=1))]
        self.raw_length_mm = float(self.raw_arc[-1])
        self.spline = None
        self.fallback_reason = "too_few_points"
        if len(physical) >= 4:
            # Controls approximately one native voxel apart suppress subvoxel
            # recentering chatter without asserting additional image resolution.
            interval = max(1.0, float(self.spacing.min()))
            knots = np.linspace(0, self.raw_length_mm,
                                max(4, int(np.ceil(self.raw_length_mm / interval)) + 1))
            controls = self._linear(knots)
            spline = CubicSpline(knots, controls, axis=0, extrapolate=False)
            check_t = np.linspace(0, self.raw_length_mm,
                                  max(2, int(np.ceil(self.raw_length_mm / 0.1)) + 1))
            fit = spline(check_t)
            deviation = np.linalg.norm(fit - self._linear(check_t), axis=1)
            self.fallback_reason = None
            if deviation.max() > 0.5 * self.spacing.min():
                self.fallback_reason = "excessive_displacement"
            elif lumen is not None:
                in_lumen = ndi.map_coordinates(lumen, (fit / self.spacing).T,
                                                order=0, mode="constant", cval=0).astype(bool)
                if parent is not None:
                    in_parent = ndi.map_coordinates(parent, (fit / self.spacing).T,
                                                     order=0, mode="constant", cval=0).astype(bool)
                    in_lumen |= in_parent & (check_t <= self.spacing.max() / 2 + 1e-8)
                if not in_lumen.all():
                    self.fallback_reason = "fit_leaves_lumen"
            if self.fallback_reason is None:
                self.spline = spline
                self.knots = knots
                lengths = self._integral(knots[:-1], knots[1:])
                if lengths.sum() > self.raw_length_mm * 1.05:
                    self.spline = None
                    self.fallback_reason = "length_inflation"
                else:
                    self.arc = np.r_[0., np.cumsum(lengths)]
        if self.spline is None:
            self.knots = self.raw_arc
            self.arc = self.raw_arc
        self.length_mm = min(float(self.arc[-1]), float(max_mm))
        self.method = "cubic_arc_integral" if self.spline is not None else "physical_polyline"

    def _linear(self, t):
        return np.stack([np.interp(t, self.raw_arc, self.raw[:, axis]) for axis in range(3)], axis=-1)

    def _integral(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        t = (a[..., None] + b[..., None]) / 2 + (b - a)[..., None] / 2 * _NODES
        speed = np.linalg.norm(self.spline(t, 1), axis=-1)
        return (b - a) / 2 * np.sum(speed * _WEIGHTS, axis=-1)

    def _parameters(self, distances):
        values = np.asarray(distances, float)
        if not np.isfinite(values).all() or np.any(values < -1e-8) or np.any(values > self.length_mm + 1e-8):
            raise ValueError("requested arc distance lies outside the observed centerline")
        values = np.clip(values, 0, self.length_mm)
        if self.spline is None:
            return values
        answer = []
        for distance in values.ravel():
            i = min(max(0, int(np.searchsorted(self.arc, distance, side="right")) - 1),
                    len(self.knots) - 2)
            a, b = self.knots[i:i + 2]
            residual = distance - self.arc[i]
            if residual <= 1e-10:
                answer.append(a)
            elif self.arc[i + 1] - distance <= 1e-10:
                answer.append(b)
            else:
                answer.append(brentq(lambda t: float(self._integral(a, t)) - residual,
                                     a, b, xtol=1e-10))
        return np.array(answer).reshape(values.shape)

    def points_at(self, distances):
        t = self._parameters(distances)
        return (self.spline(t) if self.spline is not None else self._linear(t)) / self.spacing

    def tangents_at(self, distances):
        t = self._parameters(distances)
        if self.spline is not None:
            first = self.spline(t, 1)
        else:
            i = np.clip(np.searchsorted(self.raw_arc, t, side="right") - 1, 0, len(self.raw) - 2)
            first = self.raw[i + 1] - self.raw[i]
        return first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)

    def curvature_at(self, distances):
        if self.spline is None:
            return None  # a polyline has no finite smooth curvature at its corners
        t = self._parameters(distances)
        first, second = self.spline(t, 1), self.spline(t, 2)
        return np.linalg.norm(np.cross(first, second), axis=-1) / np.maximum(
            np.linalg.norm(first, axis=-1) ** 3, 1e-12)

    def sampled_path(self, step_mm=0.25):
        return self.points_at(np.linspace(0, self.length_mm,
                              max(2, int(np.ceil(self.length_mm / step_mm)) + 1)))

    def diagnostics(self):
        return dict(method=self.method, raw_length_mm=self.raw_length_mm,
                    length_mm=self.length_mm, fallback_reason=self.fallback_reason)
