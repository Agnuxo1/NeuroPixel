"""Programme-level inference: equivalence, clustered resampling, planned looks and online error control."""
from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm, t


def _paired_summary(differences: list[float]) -> tuple[int, float, float]:
    """Validate independent paired-unit effects and return n, mean and standard error."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite independent-unit effects required")
    # Identical effects have exactly zero spread; rounding must not turn them into a tiny variance.
    sd = 0.0 if float(np.ptp(values)) == 0 else float(values.std(ddof=1))
    return len(values), float(values.mean()), sd / math.sqrt(len(values))


def _upper_tail(statistic: float, df: int) -> float:
    """Return P(T >= statistic), handling the degenerate zero-variance limit."""
    if math.isinf(statistic):
        return 0.0 if statistic > 0 else 1.0
    return float(t.sf(statistic, df))


def tost_paired(differences: list[float], low: float, high: float, *, alpha: float = 0.05) -> dict:
    """Two one-sided tests: equivalence is declared only inside registered bounds."""
    if not (math.isfinite(low) and math.isfinite(high) and low < high and 0 < alpha < 0.5):
        raise ValueError("Registered finite bounds low < high and 0 < alpha < 0.5 required")
    n, mean, se = _paired_summary(differences)
    df = n - 1
    if se == 0:
        lower_stat = math.inf if mean > low else -math.inf
        upper_stat = -math.inf if mean < high else math.inf
    else:
        lower_stat, upper_stat = (mean - low) / se, (mean - high) / se
    p_lower = _upper_tail(lower_stat, df)
    p_upper = _upper_tail(-upper_stat, df)
    critical = float(t.ppf(1 - alpha, df))
    interval = [mean - critical * se, mean + critical * se]
    p_value = max(p_lower, p_upper)
    return _flag_degenerate(dict(n=n, df=df, mean=mean, standard_error=se, bounds=[low, high],
                                 p_lower=p_lower, p_upper=p_upper, p_tost=p_value,
                                 interval_1_minus_2alpha=interval, equivalent=p_value < alpha, alpha=alpha,
                                 scope="Equivalence within registered bounds for the declared unit population"))


def _flag_degenerate(result: dict) -> dict:
    """Mark decisions that rest on identical unit effects rather than on estimated variance."""
    if result["standard_error"] == 0:
        result["warning"] = ("Zero between-unit variance: the decision depends on measurement resolution; "
                             "register a resolution floor before relying on it")
    return result


def noninferiority_paired(differences: list[float], margin: float, *, alpha: float = 0.05) -> dict:
    """One-sided test of H0: mean difference <= -margin against the registered margin."""
    if not (math.isfinite(margin) and margin > 0 and 0 < alpha < 0.5):
        raise ValueError("Positive registered margin and 0 < alpha < 0.5 required")
    n, mean, se = _paired_summary(differences)
    statistic = (mean + margin) / se if se > 0 else (math.inf if mean > -margin else -math.inf)
    p_value = _upper_tail(statistic, n - 1)
    lower = mean - float(t.ppf(1 - alpha, n - 1)) * se
    return _flag_degenerate(dict(n=n, mean=mean, standard_error=se, margin=margin, p_value=p_value,
                                 one_sided_lower_bound=lower, noninferior=p_value < alpha, alpha=alpha,
                                 scope="Noninferiority only; superiority needs its own registered test"))


def hierarchical_bootstrap(groups: dict[str, list[float]], *, replicates: int = 10000,
                           confidence: float = 0.95, seed: int = 20261012) -> dict:
    """Resample clusters, then units inside them; clusters receive equal weight."""
    if len(groups) < 2 or not 0 < confidence < 1 or replicates < 1000:
        raise ValueError("Two or more clusters, 0 < confidence < 1 and at least 1000 replicates required")
    arrays = []
    for name in sorted(groups):
        values = np.asarray(groups[name], dtype=np.float64)
        if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
            raise ValueError("Finite non-empty unit values required in cluster " + name)
        arrays.append(values)
    rng = np.random.Generator(np.random.PCG64(seed))
    k = len(arrays)
    estimates = np.empty(replicates)
    for index in range(replicates):
        chosen = rng.integers(0, k, size=k)
        estimates[index] = np.mean([arrays[c][rng.integers(0, len(arrays[c]), size=len(arrays[c]))].mean()
                                    for c in chosen])
    tail = (1 - confidence) / 2
    point = float(np.mean([values.mean() for values in arrays]))
    result = dict(clusters=k, units=int(sum(len(values) for values in arrays)), estimate=point,
                  interval=[float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1 - tail))],
                  confidence=confidence, replicates=replicates, seed=seed,
                  scope="Percentile interval of the mean of cluster means")
    if k < 10:
        result["warning"] = "Fewer than ten clusters: bootstrap coverage is unreliable; report it as descriptive"
    return result


def spending_function(fraction: float, alpha: float, kind: str = "obrien_fleming") -> float:
    """Lan-DeMets cumulative error spent at information fraction t in (0, 1]."""
    if not (0 < fraction <= 1 and 0 < alpha < 1):
        raise ValueError("Information fraction in (0,1] and alpha in (0,1) required")
    if kind == "obrien_fleming":
        return float(2 - 2 * norm.cdf(norm.isf(alpha / 2) / math.sqrt(fraction)))
    if kind == "pocock":
        return float(alpha * math.log(1 + (math.e - 1) * fraction))
    raise ValueError("Unknown spending function")


def conservative_sequential_plan(fractions: list[float], alpha: float = 0.05,
                                 kind: str = "obrien_fleming") -> dict:
    """Per-look one-sided p thresholds valid by the union bound under any dependence."""
    values = [float(value) for value in fractions]
    if not values or any(b <= a for a, b in zip(values, values[1:])) or values[-1] != 1.0 or values[0] <= 0:
        raise ValueError("Strictly increasing registered fractions ending at 1 required")
    cumulative = [spending_function(value, alpha, kind) for value in values]
    increments = [cumulative[0]] + [b - a for a, b in zip(cumulative, cumulative[1:])]
    return dict(kind=kind, alpha=alpha, fractions=values, cumulative_alpha=cumulative,
                p_thresholds=increments, total_alpha=float(sum(increments)),
                scope="Conservative Bonferroni split of spent alpha; exact boundaries need dedicated software")


def online_bonferroni(p_values: list[float], alpha: float = 0.05, gamma: list[float] | None = None) -> dict:
    """Familywise control for a stream of registered tests under arbitrary dependence."""
    values = _stream(p_values, alpha)
    weights = _gamma(len(values), gamma)
    thresholds = [alpha * weight for weight in weights]
    return dict(thresholds=thresholds, rejected=[0 < a and p <= a for p, a in zip(values, thresholds)],
                alpha=alpha, error_rate="FWER", scope="Valid under arbitrary dependence")


def lond(p_values: list[float], alpha: float = 0.05, gamma: list[float] | None = None) -> dict:
    """LOND online FDR (Javanmard-Montanari); thresholds grow with prior discoveries."""
    values = _stream(p_values, alpha)
    weights = _gamma(len(values), gamma)
    discoveries, thresholds, rejected = 0, [], []
    for p_value, weight in zip(values, weights):
        threshold = alpha * weight * (discoveries + 1)
        thresholds.append(threshold)
        rejected.append(0 < threshold and p_value <= threshold)
        discoveries += int(rejected[-1])
    return dict(thresholds=thresholds, rejected=rejected, discoveries=discoveries, alpha=alpha,
                error_rate="FDR", scope="Stated for independent p-values; use online_bonferroni otherwise")


def _stream(p_values: list[float], alpha: float) -> list[float]:
    """Validate an ordered stream of registered p-values and its error budget."""
    values = [float(value) for value in p_values]
    if not values or not all(math.isfinite(v) and 0 <= v <= 1 for v in values) or not 0 < alpha < 1:
        raise ValueError("Finite p-values in [0,1] and 0 < alpha < 1 required")
    return values


def _gamma(length: int, gamma: list[float] | None) -> list[float]:
    """Default to 6/(pi^2 j^2), a nonnegative sequence whose infinite sum is one."""
    if gamma is None:
        return [6 / (math.pi**2 * j**2) for j in range(1, length + 1)]
    weights = [float(value) for value in gamma]
    if len(weights) < length or any(w < 0 or not math.isfinite(w) for w in weights) or sum(weights) > 1 + 1e-12:
        raise ValueError("Registered nonnegative gamma covering every test with total at most one required")
    return weights[:length]


def specification_curve(results: list[dict]) -> dict:
    """Describe every registered analysis specification without picking a favourite."""
    rows = []
    for row in results:
        effect, low, high = float(row["effect"]), float(row["low"]), float(row["high"])
        if not (math.isfinite(effect) and math.isfinite(low) and math.isfinite(high) and low <= effect <= high):
            raise ValueError("Each specification needs finite low <= effect <= high")
        rows.append(dict(specification=str(row["specification"]), effect=effect, low=low, high=high))
    if len(rows) < 2 or len({row["specification"] for row in rows}) != len(rows):
        raise ValueError("At least two uniquely named specifications required")
    rows.sort(key=lambda row: row["effect"])
    effects = np.array([row["effect"] for row in rows])
    return dict(specifications=len(rows), curve=rows, median_effect=float(np.median(effects)),
                share_positive=float(np.mean(effects > 0)),
                share_interval_above_zero=float(np.mean([row["low"] > 0 for row in rows])),
                share_interval_below_zero=float(np.mean([row["high"] < 0 for row in rows])),
                scope="Descriptive robustness across registered choices; not a joint significance test")
