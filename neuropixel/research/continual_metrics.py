"""Descriptive continual-learning summaries; no model or producer imports."""
from __future__ import annotations
import numpy as np

ROLE_NAMES = ("AGENTE", "ACCION", "PACIENTE", "LUGAR")
MEASURES = ("accuracy", "binding", *ROLE_NAMES)


def matrix_summary(values):
    """Summarize a complete post-task performance matrix.

    Rows are checkpoints after tasks; columns are fixed task evaluation sets.
    BWT compares final with first post-acquisition scores. Max-past forgetting
    uses all earlier observed checkpoints. The separately named
    post-acquisition variant excludes checkpoints before that task was trained.
    Both forgetting quantities are signed; improvement can make them negative.
    No FWT is computed without an independently recorded random-init baseline.
    """
    r = np.asarray(values, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or len(r) < 1:
        raise ValueError("a nonempty square post-task matrix is required")
    if not np.all(np.isfinite(r)) or np.any((r < 0) | (r > 1)):
        raise ValueError("all performance values must be finite proportions")
    t = len(r)
    stages = []
    for end in range(t):
        k = end + 1
        row = {"after_task": end, "tasks_seen": k,
               "average_seen_accuracy": float(r[end, :k].mean()),
               "BWT": None, "max_past_forgetting": None,
               "post_acquisition_forgetting": None, "per_old_task": []}
        if end:
            for j in range(end):
                row["per_old_task"].append({
                    "task": j, "at_acquisition": float(r[j, j]),
                    "current": float(r[end, j]),
                    "backward_change": float(r[end, j] - r[j, j]),
                    "max_past_forgetting": float(r[:end, j].max() - r[end, j]),
                    "post_acquisition_forgetting": float(r[j:end, j].max() - r[end, j])})
            row["BWT"] = float(np.mean([v["backward_change"] for v in row["per_old_task"]]))
            for field in ("max_past_forgetting", "post_acquisition_forgetting"):
                row[field] = float(np.mean([v[field] for v in row["per_old_task"]]))
        stages.append(row)
    return {"matrix": r.tolist(), "acquisition_diagonal": np.diag(r).tolist(),
            "stages": stages, "final": stages[-1], "FWT": None,
            "FWT_reason": "No random-initialization baseline was recorded for these saved predictions.",
            "uncertainty": "Descriptive per seed; no additional interval, p-value or bootstrap."}


def score_matrices(log_probabilities, target, role, topic):
    """Count every checkpoint/topic/role; caller validates archive identity."""
    logp = np.asarray(log_probabilities)
    y, role, topic = np.asarray(target), np.asarray(role), np.asarray(topic)
    if logp.ndim != 3 or logp.shape[0] < 1 or logp.shape[2] < 2 or not logp.shape[1]:
        raise ValueError("log probabilities must have shape checkpoint,example,vocabulary")
    k, n, vocab = logp.shape
    if any(a.shape != (n,) or a.dtype.kind not in "iu" for a in (y, role, topic)):
        raise ValueError("integer target/role/topic arrays must align")
    if (not np.all(np.isfinite(logp)) or np.any((y < 0) | (y >= vocab))
            or np.any((role < 0) | (role > 3)) or np.any((topic < 0) | (topic >= k))):
        raise ValueError("nonfinite probabilities or invalid labels")
    z = logp.astype(np.float64)
    maximum = z.max(axis=-1, keepdims=True)
    lse = maximum[..., 0] + np.log(np.exp(z - maximum).sum(axis=-1))
    if not np.allclose(lse, 0.0, atol=1e-5, rtol=0):
        raise ValueError("saved expert log probabilities are not normalized")
    prediction = z.argmax(axis=-1)
    correct = prediction == y[None, :]
    matrices = {name: np.zeros((k, k), dtype=np.float64) for name in MEASURES}
    rows = []
    for i in range(k):
        for j in range(k):
            mask = topic == j
            counts = {}
            for r, name in enumerate(ROLE_NAMES):
                take = mask & (role == r)
                total = int(take.sum())
                if not total:
                    raise ValueError("each task must contain every role")
                hit = int(correct[i, take].sum())
                counts[name] = {"n": total, "correct": hit, "accuracy": hit / total}
                matrices[name][i, j] = hit / total
            total, hit = int(mask.sum()), int(correct[i, mask].sum())
            global_accuracy = hit / total
            binding = (counts["AGENTE"]["accuracy"] + counts["PACIENTE"]["accuracy"]) / 2
            matrices["accuracy"][i, j] = global_accuracy
            matrices["binding"][i, j] = binding
            rows.append({"checkpoint_after_task": i, "evaluation_task": j, "n": total,
                         "correct": hit, "accuracy": global_accuracy, "binding": binding,
                         "cross_entropy": float(-z[i, np.flatnonzero(mask), y[mask]].mean()),
                         "per_role": counts})
    oracle = np.any(correct, axis=0)
    oracle_rows = []
    for j in range(k):
        counts = {}
        for r, name in enumerate(ROLE_NAMES):
            take = (topic == j) & (role == r)
            total, hit = int(take.sum()), int(oracle[take].sum())
            counts[name] = {"n": total, "correct": hit, "accuracy": hit / total}
        take = topic == j
        total, hit = int(take.sum()), int(oracle[take].sum())
        oracle_rows.append({"task": j, "n": total, "correct": hit, "accuracy": hit / total,
                            "binding": (counts["AGENTE"]["accuracy"] + counts["PACIENTE"]["accuracy"]) / 2,
                            "per_role": counts})
    return {"rows": rows, "measures": {key: matrix_summary(v) for key, v in matrices.items()},
            "hard_selection_oracle_by_topic": oracle_rows,
            "oracle_scope": "Label-aware any-individual-argmax-correct bound for hard selectors only.",
            "prediction": prediction}
