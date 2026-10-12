"""Elementary bounds that turn field measurements into falsifiable quantitative laws."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta

BOLTZMANN = 1.380649e-23  # J/K, exact in the 2019 SI


def contraction_horizons(q: float, *, diameter: float = 2.0, decoder: np.ndarray | None = None,
                         margin: float | None = None, damage: float | None = None,
                         tolerance: float | None = None) -> dict:
    """Bound forgetting and repair of one sup-norm contraction under a shared fixed input."""
    if not (math.isfinite(q) and 0 < q < 1 and math.isfinite(diameter) and diameter > 0):
        raise ValueError("Strict contraction 0 < q < 1 and positive diameter required")
    rate = math.log(1 / q)
    result = dict(q=q, steps_per_e_fold=1 / rate, diameter=diameter, norm="sup norm over the whole field",
                  scope="Corollary of the contraction certificate; inputs after time zero must not carry the item")
    if decoder is not None:
        weights = np.asarray(decoder, dtype=np.float64)
        if (weights.ndim != 2 or len(weights) < 2 or not np.isfinite(weights).all()
                or margin is None or not math.isfinite(margin) or margin <= 0):
            raise ValueError("Finite decoder with two or more classes and a positive margin required")
        pair = max(float(np.abs(weights[a] - weights[b]).sum())
                   for a in range(len(weights)) for b in range(a + 1, len(weights)))
        # Different decisions, each with margin >= m, need pair * ||state difference|| >= 2m.
        # The relative slack keeps an exact equality step admitted despite floating-point rounding.
        last = (math.floor(math.log(pair * diameter / (2 * margin)) / rate * (1 + 1e-12) + 1e-12)
                if pair * diameter >= 2 * margin else None)
        result.update(decoder_pair_lipschitz=pair, margin=margin,
                      max_separable_step=None if last is None or last < 0 else last,
                      memory_statement="After max_separable_step no two histories can be read as different "
                                       "classes with the registered margin")
    if damage is not None or tolerance is not None:
        if not (damage is not None and tolerance is not None and math.isfinite(damage) and damage >= 0
                and math.isfinite(tolerance) and tolerance > 0):
            raise ValueError("Finite nonnegative damage and positive tolerance required together")
        steps = 0 if damage <= tolerance else math.ceil(math.log(damage / tolerance) / rate)
        result.update(damage=damage, tolerance=tolerance, repair_steps_bound=steps,
                      repair_statement="The damaged trajectory is within tolerance of the clean one "
                                       "from repair_steps_bound onwards")
    return result


def capacity_ceiling_bits(cells: int, channels: int, erased_cells: int, noise_sd: float,
                          amplitude: float = 1.0) -> dict:
    """Upper bound on item information surviving erasure plus Gaussian noise in a bounded field."""
    if not (min(cells, channels) >= 1 and 0 <= erased_cells <= cells and math.isfinite(noise_sd)
            and noise_sd > 0 and math.isfinite(amplitude) and amplitude > 0):
        raise ValueError("Invalid field size, erasure count, noise or amplitude")
    surviving = (cells - erased_cells) * channels
    per_value = 0.5 * math.log2(1 + amplitude**2 / noise_sd**2)
    return dict(cells=cells, channels=channels, erased_cells=erased_cells, surviving_values=surviving,
                noise_sd=noise_sd, amplitude=amplitude, bits_per_value_bound=per_value,
                ceiling_bits=surviving * per_value,
                assumptions=["item-independent erasure pattern and noise",
                             "independent Gaussian noise on every surviving value",
                             "|state| <= amplitude, so power <= amplitude^2",
                             "no input after the damage carries the item"],
                scope="Data processing plus Gaussian capacity; elementary, not a new theorem")


def _binary_entropy(p: float) -> float:
    """Return h2(p) in bits."""
    return 0.0 if p <= 0 or p >= 1 else -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _fano_bound(error: float, classes: int) -> float:
    """Fano lower bound on I(item; estimate) for a uniform item over the classes."""
    if error >= (classes - 1) / classes:
        return 0.0
    return max(0.0, math.log2(classes) - _binary_entropy(error) - error * math.log2(classes - 1))


def fano_information_lower_bound(correct: int, total: int, classes: int, *, confidence: float = 0.95) -> dict:
    """Conservative retained-information estimate from balanced, independent recall trials."""
    if not (isinstance(correct, int) and isinstance(total, int) and isinstance(classes, int)
            and 0 <= correct <= total and total >= 1 and classes >= 2 and 0 < confidence < 1):
        raise ValueError("Integer counts 0 <= correct <= total, at least two classes and a confidence required")
    errors = total - correct
    upper = 1.0 if errors == total else float(beta.ppf(confidence, errors + 1, total - errors))
    return dict(correct=correct, total=total, classes=classes, observed_error=errors / total,
                error_upper_bound=upper, confidence=confidence,
                point_bits=_fano_bound(errors / total, classes),
                lower_bound_bits=_fano_bound(upper, classes),
                assumptions=["uniform item over the classes", "independent trials", "one-sided Clopper-Pearson"],
                scope="Lower bound on information retained by the field and read by this decoder")


def max_log2_items(ceiling_bits: float, error_rate: float) -> float:
    """Largest log2(K) compatible with Fano at a given error rate and an information ceiling."""
    if not (math.isfinite(ceiling_bits) and ceiling_bits >= 0 and 0 <= error_rate < 1):
        raise ValueError("Nonnegative ceiling and 0 <= error_rate < 1 required")
    h = _binary_entropy(error_rate)

    def needed(x: float) -> float:
        # Fano is vacuous at or beyond chance error; on its domain the bound increases with x.
        if error_rate >= 1 - 2.0**-x:
            return 0.0
        return (1 - error_rate) * x - h - error_rate * math.log2(1 - 2.0**-x)

    if needed(1.0) > ceiling_bits:
        return 0.0
    low, high = 1.0, (ceiling_bits + h + 1) / (1 - error_rate) + 1
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if needed(middle) <= ceiling_bits else (low, middle)
    return low


def landauer_reference(energy_joules: float, bits: float, temperature_kelvin: float = 300.0) -> dict:
    """Express measured energy against kT ln 2 per declared irreversible bit."""
    if not all(math.isfinite(v) and v > 0 for v in (energy_joules, bits, temperature_kelvin)):
        raise ValueError("Positive finite energy, bit count and temperature required")
    minimum = bits * BOLTZMANN * temperature_kelvin * math.log(2)
    ratio = energy_joules / minimum
    return dict(energy_joules=energy_joules, bits=bits, temperature_kelvin=temperature_kelvin,
                landauer_minimum_joules=minimum, ratio=ratio, log10_ratio=math.log10(ratio),
                scope="Reference scale only; declare the bit convention and the measured boundary")


def _law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Saturating power law in scale units relative to the smallest registered scale."""
    return c + a * np.power(x, -b)


def fit_saturating_power_law(scales: list[float], errors: list[float], *, min_r_squared: float = 0.9) -> dict:
    """Fit error = c + a (scale/scale_min)^(-b) and flag fits that do not describe the data."""
    x = np.asarray(scales, dtype=np.float64)
    y = np.asarray(errors, dtype=np.float64)
    if (x.ndim != 1 or x.shape != y.shape or len(np.unique(x)) < 4 or (x <= 0).any()
            or not np.isfinite(x).all() or not np.isfinite(y).all() or (y < 0).any()):
        raise ValueError("At least four distinct positive scales with finite nonnegative errors required")
    reference = float(x.min())
    upper_c = float(y.max()) if y.max() > 0 else 1e-12
    start = (max(float(y.max() - 0.5 * y.min()), 1e-6), 0.5, 0.5 * float(y.min()))
    params, _ = curve_fit(_law, x / reference, y, p0=start,
                          bounds=([0.0, 0.0, 0.0], [np.inf, 8.0, upper_c]), maxfev=20000)
    residual = y - _law(x / reference, *params)
    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - float(residual @ residual) / total if total > 0 else 0.0
    ordered = y[np.argsort(x, kind="stable")]
    flags = dict(r_squared=r_squared, monotone_nonincreasing=bool(np.all(np.diff(ordered) <= 0)),
                 exponent_at_bound=bool(params[1] < 1e-3 or params[1] > 8 - 1e-3),
                 floor_at_upper_bound=bool(params[2] >= upper_c * (1 - 1e-6)))
    adequate = (r_squared >= min_r_squared and flags["monotone_nonincreasing"]
                and not flags["exponent_at_bound"] and not flags["floor_at_upper_bound"])
    return dict(a=float(params[0]), b=float(params[1]), c=float(params[2]), reference_scale=reference,
                residual_sum_squares=float(residual @ residual), degrees_of_freedom=len(x) - 3,
                min_r_squared=min_r_squared, adequate=adequate, **flags,
                form="error = c + a * (scale / reference_scale) ** (-b)")


def predict_saturating_power_law(fit: dict, scale: float) -> float:
    """Evaluate a registered fit at a new scale."""
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("Positive finite scale required")
    return float(_law(np.array([scale / fit["reference_scale"]]), fit["a"], fit["b"], fit["c"])[0])


def extrapolation_test(units_by_scale: dict[float, list[float]], heldout_scale: float,
                       heldout_units: list[float], tolerance: float, *, replicates: int = 2000,
                       confidence: float = 0.95, seed: int = 20261012) -> dict:
    """Predict a held-out larger scale before reading it, then score the registered tolerance."""
    scales = sorted(float(scale) for scale in units_by_scale)
    if not (math.isfinite(heldout_scale) and heldout_scale > max(scales) and math.isfinite(tolerance)
            and tolerance > 0 and 0 < confidence < 1 and replicates >= 200):
        raise ValueError("A larger held-out scale, positive tolerance and at least 200 replicates required")
    groups = [np.asarray(units_by_scale[scale], dtype=np.float64) for scale in sorted(units_by_scale)]
    if any(len(group) < 2 or not np.isfinite(group).all() for group in groups):
        raise ValueError("Two or more finite units per development scale required")
    observed = np.asarray(heldout_units, dtype=np.float64)
    if len(observed) < 2 or not np.isfinite(observed).all():
        raise ValueError("Two or more finite held-out units required")
    fit = fit_saturating_power_law(scales, [float(group.mean()) for group in groups])
    prediction = predict_saturating_power_law(fit, heldout_scale)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws, failures = [], 0
    for _ in range(replicates):
        means = [float(group[rng.integers(0, len(group), size=len(group))].mean()) for group in groups]
        try:
            draws.append(predict_saturating_power_law(fit_saturating_power_law(scales, means), heldout_scale))
        except (RuntimeError, ValueError):
            failures += 1
    tail = (1 - confidence) / 2
    interval = [float(np.quantile(draws, tail)), float(np.quantile(draws, 1 - tail))] if draws else None
    observed_mean = float(observed.mean())
    reliable = failures <= 0.05 * replicates and fit["adequate"]
    return dict(fit=fit, heldout_scale=heldout_scale, prediction=prediction, tolerance=tolerance,
                observed_mean=observed_mean, absolute_error=abs(observed_mean - prediction),
                within_tolerance=abs(observed_mean - prediction) <= tolerance,
                passed=reliable and abs(observed_mean - prediction) <= tolerance,
                fit_interval=interval,
                observed_inside_fit_interval=None if interval is None else interval[0] <= observed_mean <= interval[1],
                bootstrap_failures=failures, reliable=reliable,
                scope="Passing needs an adequate registered fit and the registered tolerance; "
                      "the interval is fit uncertainty only")
