# Item 4 — frozen pilot execution

This implementation executes protocol `NP-SCI-20261006-v1`. The protocol files remain
unchanged. The implementation commit is recorded separately before any scientific training.
The two-update unit smoke uses temporary, reduced validation data and is software testing,
not a scientific result or an additional hyperparameter trial.

## Prospective execution decisions

- Four families and two learning rates produce exactly eight pilot training runs.
  Initialization seed 7 is reserved for configuration selection. Each run uses 1,024
  updates, batches of 64, answer cross-entropy only, and the final checkpoint.
- AdamW uses constant learning rate, betas (0.9, 0.999), epsilon 1e-8 and the frozen
  weight decay/gradient clipping. There is no learning-rate schedule or early stopping.
- Model initialization, stochastic updates, and CPU data sampling have separate seeds.
  Evaluation uses the model's documented evaluation mode. Its fire mask is disabled in
  evaluation, as in the original implementation; this behavior is common to both NCA
  references and is explicitly tested. Validation and final inputs are identical across
  families, with four equally sized role blocks.
- Each family's rate is selected on validation binding accuracy, then validation cross
  entropy, then the smaller rate. The strongest non-NeuroPixel family is selected by
  the frozen validation rule. Selection is persisted and hashed before final examples
  are first constructed by this runner. The test is opened once for descriptive pilot
  results; H1 is decided only by the later five-pair initialization panel.
- Incomplete runs, resource stops, non-finite values and interrupted outputs are retained.
  No automatic continuation overwrites a partial run or reopens a frozen selection.
  Recovery requires an explicit recorded implementation decision.

## Resource reservation and environment

The local RTX 3090 is used through `D:/PROJECTS/.cognition/gpu_queue/gpuq.py`, with a
single queued worker, a nominal 6 GiB VRAM request and at least 9 GiB RAM available at
admission. During the run, a guard checks that at least 8 GiB RAM and 3.5 GiB GPU memory
remain available, and that GPU temperature is below 83 C. CPU threads are limited to
four. Other processes are not stopped. Code, output, temporary files and CUDA cache
reside on D:. No paid compute is used.

The source is deployed to an isolated directory; the user's working repository is not
overwritten. The worker disables automatic loading of unrelated installed pytest
plugins and first runs the existing and new tests with the installed Python
3.13 / PyTorch 2.6 CUDA environment. Training starts only if those tests pass. Package
versions, deterministic flags, hardware, source hashes, datasets, model weights and
individual predictions are saved with every run.

The current source passed 87 tests with two skips in 2.67 seconds on Python 3.12 / PyTorch
2.14.1 CPU. The skipped tests require a local CIFAR dataset and CUDA, respectively.
CUDA compatibility remains unverified until the queued preflight completes.

## Interpretation limits

See `04_model_specification.md` and `04_data_specification.md` for exact architecture and
split details. Nominal model sizes are close, but capacity, receptive fields, FLOPs and
elapsed time are not equalized. The NCA reference is an adaptation, not a complete
replication of a published NCA experiment. Eight short runs are a bounded pilot, not an
exhaustive search over each architecture. High training-probe error is explicitly marked
as optimization-budget limited. Inference latency and physical energy remain unmeasured
until the corresponding later items. The elementary exact controls already solve this
generator perfectly; a neural advantage here would have a correspondingly narrow scope.
