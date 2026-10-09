# Item 4 — Neural reference model specification

**Scope:** executable model definitions for the resource-bounded pilot frozen in
`02_protocol.md` and `protocol.json`. These definitions were selected before
outcome inspection. This document records architecture and unit verification;
it reports no fitted-model accuracy or completed pilot comparison.

The implementation is [models.py](../../neuropixel/research/models.py), with
behavioral checks in [test_research_models.py](../../tests/test_research_models.py).
The legacy mechanism being preserved is the NeuroPixel implementation at
`7da18d1cbd36c40c48228118abb20a7b5ab032f4`, specified in item 1. The four primary
families receive the same visible token grid and answer target. Targets and
generator metadata are not accepted by any model interface. Answer
cross-entropy is computed by the experiment runner, not by these models.

## 1. Factory, configuration, and parameter counts

```python
from neuropixel.research.models import build_model, model_config

model = build_model("neuropixel", vocab=35, h=8, w=8, steps=16)
configuration = model_config("neuropixel", vocab=35, h=8, w=8, steps=16)
output = model(canvas)  # canvas: int64 [batch, height, width]
```

`build_model(family, vocab=35, h=8, w=8, steps=16, **variant_kwargs)` constructs a
model with normal PyTorch initialization on the caller's default device (CPU
under the ordinary PyTorch default). The output position defaults to
`(h - 1, w - 1)` and may be overridden explicitly. Configuration overrides are
passed as named architecture arguments; an unknown argument raises an error
from the relevant constructor.

`model_config` accepts the same arguments. It returns the resolved `family`,
`vocab`, `h`, `w`, `steps`, `recurrent_steps`, `out_pos`, architecture `kwargs`,
actual `parameters`, `dropout`, and `primary_supervision`. It constructs the
model explicitly on CPU inside an RNG-preservation context to count parameters
without advancing the caller's initialization stream. This CPU override also
applies when the caller has selected another default device. Callers must retain the resolved
configuration with weights: boolean NCA intervention flags are configuration,
not entries in `state_dict`.

| Family name | Class and exact default widths | Parameters at V=35, H=W=8 | Primary settings |
|---|---|---:|---|
| `neuropixel` | `ResearchNCA`: identity 16, state 48, hidden 128 | 29,824 | Tied decoder; reinjection; legacy PAD policy; 16 updates; training fire rate 0.5. |
| `standard_nca` | `ResearchNCA`: identity 16, state 48, hidden 128 | 30,384 | Independent factorized decoder; no reinjection; legacy PAD policy; 16 updates; training fire rate 0.5. |
| `convgru` | `SpatialConvGRU`: identity 16, state 24 | 27,835 | Learned seed; local reset/update/candidate gates; 16 updates; deterministic recurrence. |
| `relative_transformer` | `RelativeTransformer`: width 32, 2 layers, 4 heads, feedforward width 128 | 29,547 | Learned 2D relative bias; empty key positions masked; deterministic encoder. |

These counts were obtained from the executable factory with
`sum(p.numel() for p in model.parameters() if p.requires_grad)` and are asserted
in the tests. They are nominal trainable-parameter counts. The models are
approximately sized alike; exact parameter equality is not claimed.

Every family returns:

- `logits`: `[batch, vocab]`, with PAD logit overwritten by `-1e4`.
- `state`: `[batch, channels, height, width]`.
- `activity`: a scalar. For recurrent models it is mean absolute state update;
  for the Transformer it is an explicitly zero placeholder, not an energy
  measurement and not a claim of inactive computation.

`forward(canvas, trace=False, lens_every=0)` is common to all families. Positive
`lens_every` is supported only by `ResearchNCA`; other models reject it. No
primary model silently receives a school loss. NCA and ConvGRU accept an
explicit update count and post-update hook in their forward methods. The
Transformer has fixed encoder depth: `steps` is accepted by the common factory
for configuration uniformity, but `recurrent_steps` is `None` and its forward
method does not accept a recurrent `steps` argument. It always uses the number
of encoder layers in its resolved configuration.

## 2. ResearchNCA: compatibility and intervention semantics

### Preserved legacy case

`ResearchNCA` subclasses `NeuroPixel`, retaining the legacy parameter names and
initialization. With `tied=True`, `reinject=True`, `freeze_pad=False`, its state
dictionary loads legacy weights strictly. Its own dictionary, lens, and forward
methods explicitly preserve the audited recipe. They do not delegate lookup or
decoding to a subsequently modified parent implementation.

The identity lookup remains `F.embedding(canvas, dictionary())`. Initialization
is the original pointwise seed, masked to nonempty or camera-present cells.
Each update applies the same learned depthwise 3×3 perception, pointwise
ReLU network, stochastic cell mask during training, and residual addition.
The supplied identity is constant during a rollout. No dropout or normalization
was added to this core. Optional grounding and camera/retina arguments are
retained for legacy equivalence, but are not part of the primary symbolic pilot.

The default PAD policy intentionally preserves the audited lookup behavior.
`freeze_pad=True` is an explicit opt-in projection of the effective input and
decoder PAD rows to zero. It does not delete parameter rows or promise that the
stored raw parameter value is zero after arbitrary checkpoint loading. Its
effective zero row has zero gradient through the lookup/decoder projection.
The frozen primary configuration uses `False`; this file does not alter the
legacy core's PAD behavior globally.

### Tied versus independent decoder

Let the identity table be \(D\in\mathbb R^{V\times16}\), read projection
\(R\in\mathbb R^{16\times48}\), and read bias \(r\). The tied logits are

\[
\ell(s)=D(Rs+r).
\]

With `tied=False`, the model creates an independent parameter
\(U\in\mathbb R^{V\times16}\), initialized as an exact clone of the effective
input dictionary, and computes

\[
\ell(s)=U(Rs+r).
\]

Cloning does not add an extra random initialization draw. Input and output
tables thereafter have independent storage and gradients. Both decoders keep
the same factorization rank limit; the independent decoder adds exactly
\(35\times16=560\) parameters at the primary vocabulary size. It is not a
replacement by a full \(V\times48\) head.

### Reinjection and observation timing

With `reinject=False`, the initial state still uses the genuine input identity
field. Only the identity channels of the update input are replaced by zeros of
the same shape. The recurrent state and perception channels remain intact.
This preserves tensor widths and dense arithmetic while removing repeated
access through that input path.

This choice retains \(16\times128=2,048\) weights in the first pointwise layer
whose inputs are always zero. They are included in nominal parameter counts
but cannot affect a no-reinjection forward pass through those channels. This is
an explicit limitation of interpreting nominal counts as equal effective
capacity; keeping the tensor width is the controlled intervention specified in
the protocol.

The lens preserves legacy **pre-update** timing. With 16 updates and
`lens_every=4`, it reads states 3, 7, 11, and 15. Trace frames contain the seed
and each post-hook state, giving `[batch, T+1, C, H, W]`. State damage hooks do
not erase the persistent identity field. Mean update activity excludes any
state change subsequently applied by a hook.

### Meaning of the `standard_nca` name

This is an **adapted local cellular reference**, sharing the NeuroPixel
perception/update widths while using an independent factorized output table
and no repeated identity input. The name is a protocol identifier. It does not
claim a full replication of the model, data, training procedure, or results of
Growing Neural Cellular Automata or Self-classifying MNIST Digits. The two
changes occur together in this primary family comparison; their separate
effects require the previously specified factorial contrasts.

## 3. Spatial ConvGRU reference

Let \(z_p\) be a learned 16-channel embedding of the visible token at cell
\(p\). The module uses standard `nn.Embedding(..., padding_idx=0)` lookup. Its
24-channel initial state is

\[
s^0_p=\mathbf1\{X_p\ne0\}\tanh(Az_p+a).
\]

The same input embedding field is supplied at every recurrent update. Using
learned 3×3 convolutions with zero padding, the cell computes

\[
(r^t,u^t)=\sigma\bigl(G*[z;s^{t-1}]+b_G\bigr),
\]

\[
\widetilde s^t=\tanh\bigl(C*[z;r^t\odot s^{t-1}]+b_C\bigr),
\]

\[
s^t=(1-u^t)\odot s^{t-1}+u^t\odot\widetilde s^t.
\]

`gates` is one convolution with 48 outputs, split into reset and update gates;
`candidate` has 24 outputs. A direct independent linear head maps the final
24-channel output cell to vocabulary scores. It has no shared dictionary
decoder, lens loss, stochastic firing, or dropout. A post-update hook can be
applied explicitly. Its count decomposes into 560 embedding parameters, 408
seed parameters, 17,328 gate parameters, 8,664 candidate parameters, and 875
head parameters.

The candidate convolution reads a reset-multiplied state whose reset gate
already used a 3×3 neighbourhood. Consequently a state dependency can travel
up to Chebyshev radius **2** in one ConvGRU update. This differs from the
radius-1 NCA update. Equal update counts are not equal receptive-field depth or
equal computation; the architecture and actual resource use must accompany
any accuracy comparison.

## 4. Relative-position Transformer reference

The encoder uses a 32-channel token embedding, two pre-normalized attention/
feedforward blocks, a final LayerNorm, and an independent linear vocabulary
head at the configured output cell. Each block has four attention heads with
head width 8, and a 32→128→32 feedforward network with GELU. All dropout rates
are zero. There is no learned absolute-position embedding.

### Relative coordinates

Coordinates are row-major `[row, column]`. For query cell \(i\) and key cell
\(j\), define the displacement as **key minus query**:

\[
\Delta r=r_j-r_i,\qquad\Delta c=c_j-c_i.
\]

The bias-table index is

\[
\operatorname{index}(i,j)=(\Delta r+H-1)(2W-1)+(\Delta c+W-1).
\]

Each head in each block has an independent learned table of
\((2H-1)(2W-1)\) entries, initialized to zero. At 8×8, the two blocks therefore
have \(2\times4\times225=1,800\) relative-bias parameters. Attention scores are

\[
a_{b,h,i,j}=q_{b,h,i}^{\top}k_{b,h,j}/\sqrt{8}
+b_{h,\operatorname{index}(i,j)}.
\]

The tests use a non-square 3×4 grid to distinguish rows from columns and
positive from negative displacements. They also use nonzero biases so a
permutation check cannot pass merely because position information is absent.

### Empty keys and the output query

The key mask is derived solely from the original visible grid:
`valid_keys = tokens != 0`. It is reused in every block. Empty positions,
including the output cell, remain valid **query** positions and can read
nonempty keys. They do not become keys in later blocks merely because their
intermediate state is nonzero. No target-derived or generator-derived mask is
used.

An all-empty input receives zero attention mass. The implementation avoids
softmax over an all-negative-infinity row, so this boundary case has finite
forward values and gradients. Projection and feedforward biases may still
produce a constant representation; zero attention mass is not a promise of a
zero final state.

`encode_tokens(tokens, positions=None)` exposes the encoder for checking
serialization permutations. If tokens are reordered, their coordinates must
be reordered with them, and an output must be read from the corresponding
token index. This property is equivariance to a change in serialization; it is
not invariance to moving words to different physical grid positions. Ordinary
`forward(canvas)` always uses the configured row-major grid and readout cell.

The final `state` has shape `[batch, 32, H, W]`, after the final LayerNorm.
Optional trace frames are the embedding and each block state before that final
normalization, with shape `[batch, layers+1, 32, H, W]`. This trace axis counts
encoder layers, not recurrent NCA updates. Relative-bias table size depends on
the configured grid; resizing a trained model is not claimed here.

## 5. Training and randomness contract

The primary experiment uses only answer cross-entropy for every family, as
frozen in item 2. Models return states and optional diagnostics without adding
losses themselves. The caller controls training/evaluation mode, initialization
seed, optimizer, examples, and update-mask stream.

Only the NCA families sample random masks during forward training. They retain
the legacy `torch.rand_like` route. The runner must seed the device's update
stream separately **after** model initialization; dataset sampling uses its own
generator. ConvGRU and Transformer forwards do not consume randomness. No
dropout silently makes one family's sample stream depend on another's model
construction. NCA models disable firing randomness in evaluation mode.

The factory's similar parameter counts do not equalize FLOPs, memory, wall
time, propagation radius, decoder rank, tuning difficulty, or effective active
parameter count. The frozen primary budget equalizes optimization updates and
example exposure. Claims about measured cost require the runner's actual
records and the separate efficiency protocol.

## 6. Verification and limits

The focused CPU command was:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  ../neuropixel-env/bin/python -m pytest tests/test_research_models.py -p no:cacheprovider
```

It completed with **22 passed in 1.50 seconds**, using Python 3.12.14 and
PyTorch 2.14.1+cpu, with at most two CPU threads. The tests verify:

- Exact legacy state, logits, activity, traces, and lens outputs after loading
  identical parameters, including nonzero dynamics, a nonzero PAD row,
  grounding, direct camera input, retina input, and a fixed damage hook.
- Pre-update lens timing and preservation of the seed without reinjection.
- Exact initial cloning and independent gradient routes for the untied table.
- Explicit PAD behavior independent of the parent class's dictionary method.
- Finite forward values and meaningful finite gradients for all four families.
- Actual default parameter counts, preservation of the initialization RNG
  during metadata queries, and CPU construction inside an alternate default
  device context without requiring a GPU.
- Relative-coordinate orientation, coordinated token permutations, empty key
  masking, and finite all-empty-input behavior.
- Explicit rejection of unavailable auxiliary lens supervision.

These are functional and numerical unit checks; no model was fitted to a
benchmark for this specification. The implementation uses ordinary PyTorch
operations intended to remain compatible with the project's PyTorch 2.6 CUDA
environment. That environment and GPU execution were **not exercised by these
CPU tests**. Model accuracy, optimization sufficiency, selected learning rates,
and any comparison outcome remain the responsibility of the declared pilot.
