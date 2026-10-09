"""Prespecified full-factorial contrast weights, without neural-model dependencies."""
import itertools
import math
import statistics
from scipy.stats import t


def factorial_contrasts(cells):
    levels = tuple(itertools.product((8192, 16384), ('repeated', 'paired'), (0., .3)))
    if set(cells) != set(levels) or any(len(cells[key]) != 3 for key in levels):
        raise ValueError('Complete eight cells and three initializations required')
    if any(not math.isfinite(value) for values in cells.values() for value in values):
        raise ValueError('Nonfinite contrast input')
    factors = {'budget_primary': (0,), 'query': (1,), 'school': (2,),
               'query_school': (1, 2), 'budget_query': (0, 1), 'budget_school': (0, 2),
               'triple_interaction': (0, 1, 2)}
    result = {}
    for name, indices in factors.items():
        values = []
        for seed in range(3):
            total = 0.
            for level in levels:
                signs = (1 if level[0] == 16384 else -1,
                         1 if level[1] == 'paired' else -1,
                         1 if level[2] == .3 else -1)
                total += math.prod(signs[index] for index in indices) * cells[level][seed]
            values.append(total * 2**len(indices) / 8)
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        half = float(t.ppf(.975, 2)) * sd / math.sqrt(3)
        result[name] = {'values': values, 'mean': mean, 'sd': sd,
                        't95': [mean - half, mean + half], 'training_initializations': 3}
    return result
