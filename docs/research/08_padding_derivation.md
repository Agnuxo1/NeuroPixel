# Item 8: analytical recount of the saved legacy PAD witness

This post-run saved-output audit uses only Python's standard library. It does
not import the project or Torch, execute a model, draw samples, run a test suite
or apply an optimizer. It is not another experiment or an external independent
replication. Its scope is the two explicit fixtures in frozen source commit
`d01e20625068a2ec19bf25554e106c1fc2cc4420`, not general model performance.

The source-bound helper is
[`research_verify_item8_padding.py`](../../scripts/research_verify_item8_padding.py).
Its input is `legacy_padding.json` from cloud run `37584119610-1`, SHA-256
`3f542c4c96898475510e5f67520645a316ab729e9e7c569acd5d7042f40fea62`.
The [recount receipt](../../results/research/08_validation/padding_analytic_recount.json)
records the input, fixture source, preserved model and plan hashes, every
comparison, formulas, numerical tolerance and resource observations.

## Reduction from the explicit fixture

Write the first PAD embedding coordinate as \(x\), initially zero. The dictionary
rows are \((x,0,0,0)\), \((1,0,0,0)\), \((-1,0,0,0)\) and
\((1/2,0,0,0)\). All parameters were zeroed except the stated dictionary entries,
one first-layer bias, one identity-input coefficient, one update coefficient
and one read coefficient. There is no nonzero perception weight or stochastic
firing. The optimizer fixture uses plain SGD with learning rate 0.1.

**Answer reinjection.** At the empty output cell the zero seed gives initial
state zero. Each of the two updates contributes \(\operatorname{ReLU}(1+x)\)
to the first state channel. Near \(x=0\), the read value is therefore
\(u=2(1+x)\). The effective answer logits are
\((-10000,u,-u,u/2)\), with target index 2. The PAD answer logit is replaced by
a constant, so its direct derivative is zero. With \(p_i\) denoting the softmax
probabilities at \(x=0\),

\[
L_{\rm answer}=4+\log(1+e^{-4}+e^{-1}),\qquad
\frac{\partial L_{\rm answer}}{\partial x}
=2\left(1+p_1-p_2+\frac{p_3}{2}\right).
\]

The displayed loss omits only the masked-class contribution \(e^{-10002}\),
smaller than \(10^{-4000}\); it underflows in both float32 and binary64.
The gradient reaches PAD through the repeated identity input. The other empty
cells do not add to this objective: there is no spatial coupling and the loss
reads only the output cell. The separately calculated activity statistic is not
part of the fixture objective.

**Lens readout.** Every one of the four supplied state positions has read vector
\((1,0,0,0)\), giving logits \((x,1,-1,1/2)\), with target index 1. Mean
cross-entropy cancels the four identical contributions. Unlike the answer
readout, this lens does not mask the PAD logit. Hence

\[
L_{\rm lens}=\log(1+e+e^{-1}+e^{1/2})-1,\qquad
\frac{\partial L_{\rm lens}}{\partial x}
=p_0=\frac{1}{1+e+e^{-1}+e^{1/2}}.
\]

In both routes the full PAD gradient is \((g,0,0,0)\). One SGD step changes that
row from zero to \((-0.1g,0,0,0)\). The preserved dictionary applies no zero-PAD
projection, so those stored and effective PAD-row values agree in the witness.

## Numerical comparison

| Route / quantity | Binary64 analytical value | Saved float32 result |
|---|---:|---:|
| Answer loss | 4.32656264126747 | 4.326562881469727 |
| Answer PAD first-coordinate gradient | 3.6817605234126005 | 3.681760549545288 |
| Answer PAD first coordinate after SGD | -0.36817605234126005 | -0.36817607283592224 |
| Lens loss | 0.746567269173791 | 0.7465673089027405 |
| Lens PAD first-coordinate gradient | 0.17437148764032928 | 0.17437149584293365 |
| Lens PAD first coordinate after SGD | -0.01743714876403293 | -0.017437150701880455 |

The audit declares `rtol = 16 × 2^-23 = 1.9073486328125e-6`, with `atol = 0`.
The relative envelope allows accumulated rounding in the short, well-conditioned
three/four-term softmax, log, two positive ReLU updates, weighted derivative sum
and scalar SGD multiplication. It is a stated comparison allowance, not a
universal error theorem for arbitrary kernels. All twenty structurally zero
entries require exact zero; they receive no absolute-error allowance.

All **26 comparisons passed**, with zero issues. Maximum absolute error was
**2.402022563074979e-7**, maximum relative error was
**1.1113327935842836e-7**, and the largest discrepancy was approximately
**1.040374 float32 ULP** at the expected magnitude. Available RAM was
9.12637710571289 GiB at both observations; the recount used one thread. These
are sampled RAM observations, not a continuous minimum or peak RSS.

The results analytically explain the two saved legacy defects: indirect input
reinjection and direct dictionary lens readout can both change PAD despite the
embedding module's `padding_idx=0` metadata. Evidence for the corrected model
comes from the separately executed regression suite, not a corrected-model
execution within this recount.
