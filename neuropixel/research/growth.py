"""Item-6 growth/routing utilities with explicit topic filters and deterministic routers."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from neuropixel.research.data import ResearchRoleTask
from neuropixel.research.experiment import classification_metrics
from neuropixel.research.models import ResearchNCA
from neuropixel.task import ROLES

TOPICS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11))
TOPIC_NAMES = ("A", "B", "C")
STAGE_UPDATES = 512
BATCH_SIZE = 64
SCHOOL_WEIGHT = 0.3
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 1.0
TRAIN_SAMPLE_SEEDS = (9002, 9102, 9202)
FIRING_SEEDS = (9001, 9101, 9201)
DECISION_PROBE_SEEDS = (62010, 62110, 62210)
GATE_DATA_SEEDS = (62011, 62111, 62211)
FINAL_DATA_SEEDS = (62012, 62112, 62212)
RANDOM_ROUTER_SEED = 62013
NOVELTY_THRESHOLD = 0.1
NOVELTY_TOKEN_THRESHOLD = 0.5
RESONANCE_MULTIPLIER = 0.75
GATE_STEPS = 128
GATE_LR = 0.1
GATE_L2 = 1e-4


class TopicRoleTask(ResearchRoleTask):
    """Frozen split-0 task restricted to triples whose agent and patient share one topic."""

    def __init__(self, topic: int, h: int = 8, w: int = 8, seed: int = 0):
        if topic not in range(len(TOPICS)):
            raise ValueError("topic must be 0, 1, or 2")
        super().__init__(h=h, w=w, seed=seed)
        allowed = set(TOPICS[topic])
        self.topic = topic
        self.topic_name = TOPIC_NAMES[topic]
        filtered = {}
        for split, triples in (("train", self.train_triples), ("validation", self.validation_triples), ("test", self.test_triples)):
            values = tuple(t for t in triples if int(t[0]) in allowed and int(t[2]) in allowed)
            if not values:
                raise ValueError(f"topic {topic} has no {split} triples")
            filtered[split] = values
        self.train_triples = filtered["train"]
        self.validation_triples = filtered["validation"]
        self.test_triples = filtered["test"]
        self._triples_cpu = {split: torch.tensor(values, dtype=torch.long, device="cpu") for split, values in filtered.items()}

    def split_counts(self) -> dict[str, int]:
        return {name: len(values) for name, values in (("train", self.train_triples), ("validation", self.validation_triples), ("test", self.test_triples))}


def balanced_topic_dataset(task: TopicRoleTask, split: str, n: int, seed: int):
    if n <= 0 or n % len(ROLES):
        raise ValueError("n must be a positive multiple of four")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    per_role = n // len(ROLES)
    canvases, targets, roles = [], [], []
    for role in range(len(ROLES)):
        canvas, target = task.sample(per_role, split, generator, device="cpu", query_role=role)
        canvases.append(canvas)
        targets.append(target)
        roles.append(torch.full((per_role,), role, dtype=torch.long))
    return torch.cat(canvases), torch.cat(targets), torch.cat(roles)


def tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        arr = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(str(arr.dtype).encode())
        digest.update(json.dumps(list(arr.shape)).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def make_core(init_seed: int) -> ResearchNCA:
    torch.manual_seed(init_seed)
    return ResearchNCA(35, (7, 7), c_id=16, c=48, hidden=128, steps=16, fire_rate=0.5, tied=True, reinject=True, freeze_pad=False)


def school_loss(output: dict, canvas: torch.Tensor) -> torch.Tensor:
    if "lens" not in output:
        raise ValueError("school loss requires lens outputs")
    count = output["lens"].shape[1]
    occupied = (canvas != 0).unsqueeze(1).expand(-1, count, -1, -1)
    labels = canvas.unsqueeze(1).expand(-1, count, -1, -1)
    return F.cross_entropy(output["lens"][occupied], labels[occupied])


def train_stage(model: ResearchNCA, task: TopicRoleTask, stage: int, device: torch.device, resource_check=None) -> dict:
    """Train exactly one 512-update topic stage with a fresh optimizer."""
    if stage not in range(3):
        raise ValueError("stage must be 0, 1, or 2")
    model.to(device)
    model.train()
    torch.manual_seed(FIRING_SEEDS[stage])
    generator = torch.Generator(device="cpu").manual_seed(TRAIN_SAMPLE_SEEDS[stage])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    curve = []
    answer_window, total_window = [], []
    for update in range(1, STAGE_UPDATES + 1):
        canvas, target = task.sample(BATCH_SIZE, "train", generator, device="cpu")
        canvas, target = canvas.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(canvas, lens_every=4)
        answer = F.cross_entropy(output["logits"], target)
        school = school_loss(output, canvas)
        total = answer + SCHOOL_WEIGHT * school
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite growth loss at stage {stage}, update {update}")
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
        optimizer.step()
        answer_window.append(float(answer.detach().cpu()))
        total_window.append(float(total.detach().cpu()))
        if update % 128 == 0:
            row = {"update": update, "mean_answer_loss": float(np.mean(answer_window)), "mean_total_loss": float(np.mean(total_window)), "last_unclipped_gradient_norm": float(grad.detach().cpu())}
            curve.append(row)
            answer_window.clear()
            total_window.clear()
            if resource_check is not None:
                resource_check()
    model.eval()
    return {"stage": stage, "topic": task.topic, "updates": STAGE_UPDATES, "curve": curve}


@torch.inference_mode()
def own_token_statistics(model: ResearchNCA, canvas: torch.Tensor, device: torch.device, batch_size: int = 256) -> tuple[np.ndarray, float]:
    """Return per-example resonance and global occupied-token novelty fraction."""
    model.to(device)
    model.eval()
    resonance = []
    novel_count = 0
    occupied_count = 0
    for start in range(0, len(canvas), batch_size):
        batch = canvas[start:start + batch_size].to(device)
        state = model(batch)["state"]
        probabilities = model.lens_logits(state).softmax(-1)
        own = probabilities.gather(-1, batch.unsqueeze(-1)).squeeze(-1)
        occupied = batch != 0
        per_example = (own * occupied).sum((1, 2)) / occupied.sum((1, 2)).clamp_min(1)
        resonance.append(per_example.cpu())
        novel_count += int(((own < NOVELTY_TOKEN_THRESHOLD) & occupied).sum().cpu())
        occupied_count += int(occupied.sum().cpu())
    return torch.cat(resonance).numpy(), novel_count / occupied_count


@torch.inference_mode()
def expert_probabilities(experts: list[ResearchNCA], canvas: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Return [experts, examples, vocab] answer probabilities."""
    outputs = []
    for model in experts:
        model.to(device)
        model.eval()
        values = []
        for start in range(0, len(canvas), batch_size):
            logits = model(canvas[start:start + batch_size].to(device))["logits"]
            values.append(logits.softmax(-1).cpu())
        outputs.append(torch.cat(values))
    return torch.stack(outputs).numpy()


@torch.inference_mode()
def reconstruction_scores(experts: list[ResearchNCA], canvas: torch.Tensor, device: torch.device, batch_size: int = 256) -> np.ndarray:
    """Return [examples, experts] unrounded mean own-token probabilities."""
    columns = []
    for model in experts:
        values, _ = own_token_statistics(model, canvas, device, batch_size)
        columns.append(values)
    return np.stack(columns, axis=1)


def visible_features(canvas: torch.Tensor, role_ids: torch.Tensor, query_pos=(7, 6), vocab: int = 35) -> torch.Tensor:
    """35 token-presence bits + four query-role one-hots + bias = 40 features."""
    if canvas.ndim != 3:
        raise ValueError("canvas must have [examples,height,width]")
    n = len(canvas)
    presence = torch.zeros(n, vocab, dtype=torch.float32)
    rows = torch.arange(n)[:, None].expand(-1, canvas.shape[1] * canvas.shape[2])
    flat = canvas.reshape(n, -1)
    presence[rows, flat] = 1.0
    query_token = canvas[:, query_pos[0], query_pos[1]]
    query = torch.zeros(n, 4, dtype=torch.float32)
    for index, token in enumerate(role_ids.tolist()):
        query[:, index] = (query_token == token).float()
    if not torch.allclose(query.sum(1), torch.ones(n)):
        raise ValueError("every example must carry exactly one query role")
    bias = torch.ones(n, 1, dtype=torch.float32)
    return torch.cat([presence, query, bias], dim=1)


def fit_gate(features: torch.Tensor, expert_probs: np.ndarray, targets: torch.Tensor, *, steps: int = GATE_STEPS, lr: float = GATE_LR, l2: float = GATE_L2) -> dict:
    """Fit the fixed 40xK mixture gate with full-batch gradient descent."""
    probs = torch.as_tensor(expert_probs, dtype=torch.float64).permute(1, 0, 2)
    targets = targets.to(dtype=torch.long, device="cpu")
    features = features.to(dtype=torch.float64, device="cpu")
    n, k, _ = probs.shape
    if features.shape != (n, 40) or targets.shape != (n,):
        raise ValueError("gate inputs have incompatible shapes")
    target_probs = probs.gather(2, targets[:, None, None].expand(-1, k, 1)).squeeze(-1)
    target_logp = target_probs.clamp_min(1e-300).log()
    weight = torch.zeros(40, k, dtype=torch.float64, requires_grad=True)
    losses = []
    for _ in range(steps):
        log_router = torch.log_softmax(features @ weight, dim=1)
        nll = -torch.logsumexp(log_router + target_logp, dim=1).mean()
        loss = nll + 0.5 * l2 * weight.square().sum()
        loss.backward()
        with torch.no_grad():
            weight -= lr * weight.grad
            weight.grad.zero_()
        losses.append(float(loss.detach()))
    return {"weight": weight.detach().to(dtype=torch.float32).numpy(), "steps": steps, "learning_rate": lr, "l2": l2, "initial_loss": losses[0], "final_loss": losses[-1]}


def _metric(prediction: np.ndarray, target: np.ndarray, roles: np.ndarray) -> dict:
    return classification_metrics(prediction, target, roles, intervals=False)


def score_routers(experts: list[ResearchNCA], canvas: torch.Tensor, target: torch.Tensor, roles: torch.Tensor, role_ids: torch.Tensor, gate_weight: np.ndarray, device: torch.device, *, random_seed: int = RANDOM_ROUTER_SEED) -> dict:
    """Evaluate legacy, random, uniform and learned routers on one concatenated dataset."""
    probs = expert_probabilities(experts, canvas, device)
    argmax = probs.argmax(-1)
    target_np = target.numpy()
    roles_np = roles.numpy()
    scores = reconstruction_scores(experts, canvas, device)
    legacy_index = scores.argmax(1)
    rng = np.random.default_rng(random_seed)
    random_index = rng.integers(len(experts), size=len(canvas))
    examples = np.arange(len(canvas))
    legacy_pred = argmax[legacy_index, examples]
    random_pred = argmax[random_index, examples]
    uniform_pred = probs.mean(0).argmax(-1)
    features = visible_features(canvas, role_ids).numpy()
    logits = features @ gate_weight
    router = np.exp(logits - logits.max(1, keepdims=True))
    router /= router.sum(1, keepdims=True)
    learned_prob = np.einsum("nk,knv->nv", router, probs)
    learned_pred = learned_prob.argmax(-1)
    oracle = (argmax == target_np[None, :]).any(0)
    return {"predictions": {"legacy": legacy_pred, "random": random_pred, "uniform": uniform_pred, "learned": learned_pred}, "indices": {"legacy": legacy_index, "random": random_index}, "gate_weights_mean": router.mean(0).tolist(), "metrics": {"legacy": _metric(legacy_pred, target_np, roles_np), "random": _metric(random_pred, target_np, roles_np), "uniform": _metric(uniform_pred, target_np, roles_np), "learned": _metric(learned_pred, target_np, roles_np)}, "routing_frequency": {"legacy": np.bincount(legacy_index, minlength=len(experts)).tolist(), "random": np.bincount(random_index, minlength=len(experts)).tolist()}, "hard_selection_oracle": {"correct": int(oracle.sum()), "n": len(oracle), "accuracy": float(oracle.mean()), "binding_accuracy": float(oracle[np.isin(roles_np, [0, 2])].mean())}, "_expert_probs": probs}


def state_sha256(model: ResearchNCA) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(array.tobytes())
    return digest.hexdigest()


def clone_cpu(model: ResearchNCA) -> ResearchNCA:
    return deepcopy(model).cpu().eval()
