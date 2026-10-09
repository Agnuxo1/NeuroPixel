# Item 4 — primary reference and implementation checks

Accessed 2026-10-06. These checks document the frozen implementation; they do not
change its configuration after selection or add hyperparameter trials.

## Convolutional recurrent reference

Ballas, Yao, Pal and Courville, *Delving Deeper into Convolutional Networks for
Learning Video Representations*, first posted 2015; version 4 dated 2016-03-01.
[Primary full text](https://arxiv.org/html/1511.06432v4), section 3.1, equations 5–8.

The paper replaces the dense input/state transformations of a GRU with spatial
convolutions. Its reset-gated state enters a convolution when constructing the
candidate, while an update gate mixes old state and candidate. This supports the
form of the local recurrent comparator. Our token task, widths, initial state,
readout and training budget are specified independently; the video experiments
and their published scores are not being reproduced.

## Relative-position reference

Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted
Windows*, 2021. [Primary paper](https://arxiv.org/pdf/2103.14030), section 3.2,
relative-position bias.

The paper adds a learned bias indexed by two-dimensional relative displacement
to attention scores. Our comparator uses this type of spatial information over
the full small grid. It does not implement the Swin hierarchy, patch merging or
shifted-window schedule, and is not called a Swin replication. Its exact table
index, mask and attention equations are in `04_model_specification.md`.

## Target-runtime reproducibility

PyTorch 2.6 primary documentation:

- [Reproducibility](https://docs.pytorch.org/docs/2.6/notes/randomness.html).
- [torch.manual_seed](https://docs.pytorch.org/docs/2.6/generated/torch.manual_seed.html).

The documentation states that manual seeding affects all devices and describes
deterministic algorithms, cuDNN configuration and the cuBLAS workspace setting.
It also warns that releases, platforms and CPU/GPU execution need not be bitwise
equivalent. The pilot therefore compares runs under the same recorded target
environment; the CPU unit suite is a compatibility check, not a claim of numerical
identity with CUDA. Unsupported deterministic operations must fail visibly.

## Inference-mode qualification

Direct code inspection established that both NCA families use Bernoulli firing
only during training. Evaluation applies all 16 updates synchronously. For a fixed
state, the conditional expected training increment is half the unmasked increment;
nonlinear recurrent trajectories cannot be replaced by multiplying a final result
by that factor. The experiment measures deterministic full-update inference of
models trained with random firing. It does not measure accuracy averaged over
random inference trajectories or robustness to different inference masks.

The training probe uses this same evaluation mode. Its predeclared below-95%
flag identifies a budget-limited operational result, but alone cannot separate
optimization difficulty from sensitivity to the training/evaluation update change.
No additional tuning or repeated final evaluation is authorized by that flag.
