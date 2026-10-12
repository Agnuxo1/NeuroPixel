"""Paired training-unit statistics; examples never inflate the replication count."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import t


def paired_interval(differences: list[float], confidence: float = 0.95) -> dict:
    """Calculate a two-sided Student interval for independent paired-unit effects."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent-unit effects required")
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * sd / math.sqrt(len(values))
    return dict(n=len(values), df=len(values) - 1, mean=mean, sample_sd=sd,
                interval=[mean - half, mean + half], half_width=half,
                confidence=confidence, scope="Training-unit variation; declared population only")


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original registered contrast order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Finite p-values in [0,1] required")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * float(values[index])))
        adjusted[index] = running
    return adjusted.tolist()


def plan_precision(sd_pp: float, half_width_pp: float, *, minimum: int = 5,
                   maximum: int = 200, confidence: float = 0.95) -> dict:
    """Plan a NEW fixed cohort from an explicit between-unit SD assumption."""
    if not (math.isfinite(sd_pp) and sd_pp >= 0 and half_width_pp > 0
            and 2 <= minimum <= maximum and 0 < confidence < 1):
        raise ValueError("Invalid planning assumptions")
    for n in range(minimum, maximum + 1):
        width = float(t.ppf((1 + confidence) / 2, n - 1)) * sd_pp / math.sqrt(n)
        if width <= half_width_pp:
            return dict(n=n, assumed_sd_pp=sd_pp, expected_half_width_pp=width,
                        target_pp=half_width_pp, scope="Prospective planning, not optional stopping")
    return dict(n=None, assumed_sd_pp=sd_pp, target_pp=half_width_pp,
                reason="Requested precision exceeds the declared maximum cohort")


def factorial_effect(cells: dict[tuple[int, ...], float], factor: int) -> float:
    """Compute one balanced full-factorial main effect from all binary cells."""
    from itertools import product
    if not cells:
        raise ValueError("No cells")
    dimensions = len(next(iter(cells)))
    expected = set(product((0, 1), repeat=dimensions))
    if set(cells) != expected or not 0 <= factor < dimensions:
        raise ValueError("Complete binary factorial required")
    values = list(cells.values())
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite factorial endpoint")
    positive = [v for key, v in cells.items() if key[factor] == 1]
    negative = [v for key, v in cells.items() if key[factor] == 0]
    return float(np.mean(positive) - np.mean(negative))
