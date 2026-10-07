"""NumPy-only Soil contract fixtures; --legacy-witness runs preserved failures."""
from __future__ import annotations
import ast
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIZES = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2., 6.3, 20., 63., 200.]


def legacy_suite():
    source = (ROOT / 'results/research/08_validation/soil_source_original.py').read_text()
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    mean = next(n for n in ast.walk(main) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'mean_cdf' for t in n.targets))
    log = next(n for n in ast.walk(main) if isinstance(n, ast.Dict) and [k.value for k in n.keys if isinstance(k, ast.Constant)] == ['fold', 'EMD_np', 'EMD_media'])
    pdf = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cdf_to_pdf')
    ns = {'np': np}
    exec(compile(ast.Module(body=[pdf], type_ignores=[]), '<preserved-cdf_to_pdf>', 'exec'), ns)

    class LegacyWitness(unittest.TestCase):
        def test_reference_must_not_depend_on_heldout_label(self):
            labs = {'train': np.array([0.] * 10 + [100.]), 'heldout': np.full(11, 100.)}
            env = {'np': np, 'samples': ['train', 'heldout'], 'labs': labs}
            expression = compile(ast.Expression(mean.value), '<preserved-mean_cdf>', 'eval')
            before = eval(expression, env)
            labs['heldout'] = np.array([0.] * 10 + [100.])
            after = eval(expression, env)
            np.testing.assert_array_equal(before, after)

        def test_fold_log_must_report_current_fold(self):
            env = {'np': np, 'k': 1, 'errs': [0., 0., 90.], 'base': [1., 1., 1.]}
            result = eval(compile(ast.Expression(log), '<preserved-fold-log>', 'eval'), env)
            self.assertEqual(result['EMD_np'], 90.)

        def test_invalid_cdf_must_not_be_silently_repaired(self):
            with self.assertRaises(ValueError):
                ns['cdf_to_pdf'](np.array([0., 20., 10.] + [100.] * 8))
    return unittest.defaultTestLoader.loadTestsFromTestCase(LegacyWitness)


if __name__ == '__main__':
    if '--legacy-witness' in sys.argv:
        result = unittest.TextTestRunner(verbosity=2).run(legacy_suite())
        sys.exit(not result.wasSuccessful())
    unittest.main()
