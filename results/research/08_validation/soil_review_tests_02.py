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


sys.path.insert(0, str(ROOT))
from kaggle.soil import evaluation as ev


class SoilEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fine = np.full(11, 100.)
        self.coarse = np.array([0.] * 10 + [100.])

    def test_metric_identity_symmetry_and_known_extremes(self):
        self.assertEqual(ev.emd(self.fine, self.fine), 0.)
        expected = 100. * (np.log10(200. / .002) - .5 * np.log10(200. / 63.))
        self.assertAlmostEqual(ev.emd(self.fine, self.coarse), expected, places=12)
        self.assertEqual(ev.emd(self.fine, self.coarse), ev.emd(self.coarse, self.fine))

    def test_trapezoid_is_not_discrete_wasserstein(self):
        # Point mass at first vs last diameter: exact log-W1×100 is 500.
        discrete = 100. * np.log10(200. / .002)
        self.assertEqual(discrete, 500.)
        self.assertNotAlmostEqual(ev.emd(self.fine, self.coarse), discrete)

    def test_matches_independent_numpy_trapezoid(self):
        a = np.array([0., 2., 20., 20., 50., 51., 60., 88., 90., 99., 100.])
        b = np.array([1., 10., 10., 30., 40., 70., 80., 80., 90., 95., 100.])
        expected = np.trapezoid(np.abs(a-b), x=np.log10(SIZES))
        self.assertEqual(ev.emd(a, b), expected)

    def test_invalid_cdf_cases_rejected_without_projection(self):
        cases = [np.zeros(11), self.coarse[:-1], self.coarse.reshape(1,11),
                 np.array([0., 20., 10.] + [100.] * 8),
                 np.array([-1.] + [100.] * 10), np.array([0.] * 10 + [101.]),
                 np.array([np.nan] + [100.] * 10), np.array([np.inf] + [100.] * 10)]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError): ev.validate_cdf(case)
                with self.assertRaises(ValueError): ev.cdf_to_pdf(case)

    def test_terminal_tolerance_explicit_and_unmodified(self):
        curve = self.coarse.copy();curve[-1] -= ev.CDF_ENDPOINT_ATOL / 2
        np.testing.assert_array_equal(ev.validate_cdf(curve), curve)
        curve[-1] = 100. - 2 * ev.CDF_ENDPOINT_ATOL
        with self.assertRaises(ValueError): ev.validate_cdf(curve)

    def test_monotonicity_and_bounds_have_no_hidden_tolerance(self):
        for curve in [np.array([-1e-12] + [100.] * 10),
                      np.array([0.] * 10 + [100. + 1e-12]),
                      np.array([0., 1e-12, 0.] + [100.] * 8)]:
            with self.assertRaises(ValueError): ev.validate_cdf(curve)

    def test_pdf_roundtrip_and_input_not_mutated(self):
        original = np.linspace(0,100,11); before = original.copy()
        pdf = ev.cdf_to_pdf(original)
        np.testing.assert_allclose(np.cumsum(pdf)*100, original, atol=1e-14)
        np.testing.assert_array_equal(original, before)
        self.assertAlmostEqual(pdf.sum(), 1.)

    def test_normalization_collisions_and_duplicates_rejected(self):
        self.assertEqual(ev.normalize_sample_id('Müller_01'), 'mueller01')
        for names in [('A-B','A B'), ('Müller','Mueller'), ('x','x'), ('---',)]:
            with self.subTest(names=names):
                with self.assertRaises(ValueError): ev.unique_normalized_ids(names)

    def test_valid_ids_preserve_raw_names(self):
        self.assertEqual(ev.unique_normalized_ids(['Müller 1', 'H374']), {'mueller1':'Müller 1','h374':'H374'})

    def test_folds_cover_once_with_nonempty_training(self):
        ids = ['a','b','c','d','e'];folds = ev.make_folds(ids,3,seed=0)
        self.assertEqual(sorted(map(len,folds)), [1,2,2])
        self.assertEqual(sorted(s for f in folds for s in f), ids)
        self.assertEqual(folds,ev.make_folds(ids,3,seed=0))
        for fold in folds:self.assertTrue(set(ids)-set(fold))

    def test_invalid_fold_counts_rejected(self):
        for n in [0,1,4,True,2.0]:
            with self.subTest(n=n):
                with self.assertRaises(ValueError):ev.make_folds(['a','b','c'],n)

    def test_duplicate_missing_unknown_overlapping_empty_folds_rejected(self):
        for folds in [[['a'],[]], [['a'],['a','b','c']], [['a'],['b']],
                      [['a'],['b','d']], [['a','a'],['b','c']]]:
            with self.subTest(folds=folds):
                with self.assertRaises(ValueError):ev.validate_folds(['a','b','c'],folds)
        with self.assertRaises(ValueError):ev.make_folds(['a','a','c'],2)

    def test_reference_never_reads_heldout_label(self):
        class TrainOnly(dict):
            def __getitem__(inner,key):
                if key=='heldout':raise AssertionError('heldout label accessed')
                return super().__getitem__(key)
        labels = TrainOnly(a=self.fine,b=self.coarse)
        ref = ev.train_mean_cdf(labels,['a','b'],['heldout'])
        np.testing.assert_array_equal(ref, (self.fine+self.coarse)/2)

    def test_reference_invariant_under_heldout_label_change(self):
        labels={'a':self.coarse,'heldout':self.fine}
        before=ev.train_mean_cdf(labels,['a'],['heldout'])
        labels['heldout']=self.coarse
        np.testing.assert_array_equal(before,ev.train_mean_cdf(labels,['a'],['heldout']))

    def test_reference_rejects_overlap_and_empty_train(self):
        with self.assertRaises(ValueError):ev.train_mean_cdf({'a':self.fine},['a'],['a'])
        with self.assertRaises(ValueError):ev.train_mean_cdf({},[],['a'])

    def test_per_sample_records_preserve_unrounded_values_and_group(self):
        pred = np.linspace(0.,100.,11);pred[1] = 10.123456789
        r=ev.score_sample('a',0,self.coarse,pred,self.fine)
        self.assertEqual(r['prediction_cdf'][1],10.123456789)
        self.assertEqual(r['group_id'],'a')
        self.assertEqual(r['scores']['model'],ev.emd(self.coarse,pred))
        json.dumps(r,allow_nan=False)

    def test_pooled_mean_weights_samples_not_folds(self):
        records=[ev.score_sample('a',0,self.coarse,self.coarse,self.fine),
                 ev.score_sample('b',0,self.coarse,self.coarse,self.fine),
                 ev.score_sample('c',1,self.coarse,self.fine,self.fine)]
        first=ev.summarize_scores(records[:2]);last=ev.summarize_scores(records[2:]);all_=ev.summarize_scores(records)
        self.assertEqual(first['means']['model'],0.)
        self.assertEqual(last['means']['model'],ev.emd(self.coarse,self.fine))
        self.assertEqual(all_['means']['model'],last['means']['model']/3)
        self.assertNotEqual(all_['means']['model'],(first['means']['model']+last['means']['model'])/2)

    def test_summary_rejects_empty_duplicate_nonfinite_scores(self):
        r=ev.score_sample('a',0,self.coarse,self.fine,self.fine)
        for rows in [[],[r,r]]:
            with self.assertRaises(ValueError):ev.summarize_scores(rows)
        for invalid in [float('nan'),float('inf'),-1.]:
            r['scores']['model']=invalid
            with self.assertRaises(ValueError):ev.summarize_scores([r])

    def _run_main_fixture(self, fail_after=None):
        # Execute only the AST-extracted CLI body, with inert train/predict stubs.
        # This checks real CV wiring and persistence without importing Torch.
        import argparse, contextlib, hashlib, io, tempfile, time
        from types import SimpleNamespace
        from unittest.mock import patch
        source=(ROOT/'kaggle/soil/soil.py').read_text();tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        calls=[]; prediction_calls=[];labs={'a':self.coarse,'b':self.fine,'c':self.coarse}
        def labels(raw):raw.update({s:s.upper() for s in labs});return labs
        def photos(folder,table,metadata):
            metadata.extend({'file':s+'.png','phone':'phone','sample_id':s} for s in labs)
            return [(s,np.zeros((1,1,3))) for s in labs]
        def train(images,fold_labs,iters,dev,seed):
            initial=json.loads(next((here/'runs/np_votes').glob('cv_evaluation_*.json')).read_text())
            calls.append((set(s for s,_ in images),set(fold_labs),seed,initial));return seed
        def predict(*args,**kwargs):
            if fail_after is not None and len(prediction_calls)==fail_after:
                raise RuntimeError('injected prediction failure')
            prediction_calls.append(True);return self.coarse
        with tempfile.TemporaryDirectory() as td:
            here=Path(td)/'kaggle'/'soil';here.mkdir(parents=True)
            (here/'soil.py').write_text(source);(here/'evaluation.py').write_bytes((ROOT/'kaggle/soil/evaluation.py').read_bytes())
            (Path(td)/'neuropixel').mkdir()
            (Path(td)/'neuropixel/model.py').write_bytes((ROOT/'neuropixel/model.py').read_bytes())
            ns={name:getattr(ev,name) for name in dir(ev) if not name.startswith('_')}
            ns.update(argparse=argparse,json=json,hashlib=hashlib,time=time,Path=Path,HERE=here,D=here/'data',
                      __file__=str(here/'soil.py'),choose_device=lambda *a,**k:'no_device',ppm_table=lambda:{},
                      labels=labels,load_photos=photos,train=train,predict_sample=predict,
                      torch=SimpleNamespace(__version__='not_imported'))
            exec(compile(ast.Module(body=[main],type_ignores=[]),'<soil-main-only>','exec'),ns)
            stdout=io.StringIO()
            caught=None
            try:
                with patch.object(sys,'argv',['soil.py','--cv','--folds','2','--iters','1']),contextlib.redirect_stdout(stdout):ns['main']()
            except RuntimeError as exc:
                caught=exc
            evidence=json.loads(next((here/'runs/np_votes').glob('cv_evaluation_*.json')).read_text())
            logs=[json.loads(line) for line in stdout.getvalue().splitlines() if line.startswith('{')]
            return evidence,calls,[r for r in logs if 'fold_metrics' in r],caught,labs

    def test_main_cv_integration_with_stubbed_training_and_predictions(self):
        evidence,calls,foldlogs,caught,labs=self._run_main_fixture()
        self.assertIsNone(caught)
        self.assertEqual(evidence['status'],'complete');self.assertEqual(len(evidence['per_sample']),3)
        self.assertEqual(calls[0][3]['status'],'running')
        self.assertEqual(calls[0][3]['per_sample'],[])
        self.assertIn('neuropixel/model.py',evidence['source_sha256'])
        for fold,call in zip(evidence['folds'],calls):
            self.assertEqual(call[0],set(fold['train_ids']));self.assertEqual(call[1],set(fold['train_ids']))
            self.assertFalse(call[1]&set(fold['validation_ids']))
            expected=np.mean([labs[s] for s in fold['train_ids']],axis=0)
            np.testing.assert_array_equal(fold['train_mean_cdf'],expected)
        self.assertEqual([r['fold_metrics']['n_samples'] for r in foldlogs],[2,1])
        self.assertEqual([r['pooled_to_date_metrics']['n_samples'] for r in foldlogs],[2,3])
        self.assertEqual({r['group_id'] for r in evidence['per_sample']},set(labs))
        self.assertEqual(evidence['pooled_metrics'],ev.summarize_scores(evidence['per_sample']))
        self.assertEqual(evidence['torch_version'],'not_imported')

    def test_failed_prediction_preserves_initial_plan_and_completed_sample(self):
        evidence,calls,_,caught,_=self._run_main_fixture(fail_after=1)
        self.assertIsInstance(caught,RuntimeError)
        self.assertEqual(evidence['status'],'failed')
        self.assertEqual(len(evidence['per_sample']),1)
        self.assertEqual(evidence['failure']['fold'],0)
        self.assertEqual(evidence['failure']['sample_id'],evidence['validation_folds'][0][1])
        self.assertEqual(evidence['failure']['type'],'RuntimeError')
        self.assertIn('injected prediction failure',evidence['failure']['message'])
        self.assertEqual(calls[0][3]['status'],'running')
        self.assertEqual(calls[0][3]['per_sample'],[])
        self.assertEqual(evidence['pooled_metrics']['n_samples'],1)

    def test_submission_uniform_fallback_constructor_is_valid(self):
        tree=ast.parse((ROOT/'kaggle/soil/soil.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        conditional=next(n for n in ast.walk(main) if isinstance(n,ast.IfExp) and isinstance(n.test,ast.Name) and n.test.id=='arrs')
        ns={name:getattr(ev,name) for name in dir(ev) if not name.startswith('_')};ns['np']=np
        fallback=eval(compile(ast.Expression(conditional.orelse),'<submission-fallback>','eval'),ns)
        self.assertEqual(float(fallback[-1]),100.)
        ev.validate_cdf(fallback)



if __name__ == '__main__':
    if '--legacy-witness' in sys.argv:
        result = unittest.TextTestRunner(verbosity=2).run(legacy_suite())
        sys.exit(not result.wasSuccessful())
    unittest.main()
