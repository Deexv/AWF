"""
Gradient Event-Driven Training for AWF.

The fundamental insight: in a transformer, most layers' gradients on most tokens
are SMALL. We can skip computing/applying them.

AWF's UNIQUE advantage: the shared CoordGenerator is the most expensive part
of training. If we detect that NO layer needs an update, we skip the generator
backward entirely.

This implementation actually SAVES backward FLOPs by setting requires_grad=False
on layers we want to skip BEFORE the forward pass. PyTorch will then skip
backward through those layers.

3 levels (per layer, per step):
  Level 0 (REUSE): freeze layer (requires_grad=False). No backward, no update.
  Level 1 (APPROX): keep layer trainable but scale its gradients by 0.3.
  Level 2 (FULL): normal full backward.

Decision signal: activation novelty. If layer k's activations on this batch are
similar to recent activations, the gradient through layer k is similar too.
We use the LAST step's activations to decide the CURRENT step's levels
(one-step lookahead). This means we run forward ONCE to capture activations,
then decide, then run forward+backward selectively.

To amortize the cost of the lookahead, we run it every K steps and reuse the
decision for K-1 steps after that.

Novelty = 1 - max_cosine_similarity(activation, recent_activations)
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation hooks for novelty detection
# ---------------------------------------------------------------------------
class ActivationCollector:
    """Collects per-layer activations during forward pass for novelty detection."""
    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles = []

    def attach(self, model: nn.Module) -> List[nn.Module]:
        """Attach forward hooks to capture activations. Returns list of AWF layers."""
        awf_layers = []
        if hasattr(model, "awf_layers"):
            awf_layers = model.awf_layers
        elif hasattr(model, "layers_list"):
            awf_layers = model.layers_list
        elif hasattr(model, "blocks"):
            # Transformer — collect q,k,v,o,up,down from each block
            for blk in model.blocks:
                if hasattr(blk, "attn"):
                    for name in ["q", "k", "v", "o"]:
                        awf_layers.append(blk.attn.projs[name])
                if hasattr(blk, "up"):
                    awf_layers.append(blk.up)
                if hasattr(blk, "down"):
                    awf_layers.append(blk.down)
            if hasattr(model, "head"):
                awf_layers.append(model.head)

        self._layer_to_id = {layer: i for i, layer in enumerate(awf_layers)}
        self.n_layers = len(awf_layers)

        for layer in awf_layers:
            lid = self._layer_to_id[layer]
            handle = layer.register_forward_hook(self._make_hook(lid))
            self.handles.append(handle)

        return awf_layers

    def _make_hook(self, lid):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                act = out.detach().mean(dim=0).flatten()
                if act.numel() > 256:
                    act = act[:256]
                self.activations[lid] = act
        return hook

    def get_activation(self, lid):
        return self.activations.get(lid)

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# Event-Driven Trainer (actually saves backward FLOPs)
# ---------------------------------------------------------------------------
class GradientEventTrainer:
    """3-level gradient event-driven trainer that ACTUALLY skips backward.

    Strategy: every K steps, run a "scout" forward pass with hooks to capture
    activations and decide per-layer levels. Then for the next K-1 steps,
    freeze Level-0 layers (requires_grad=False) so PyTorch skips backward
    through them — saving real compute.

    Usage:
        trainer = GradientEventTrainer(model, lr=1e-3, decision_interval=4)
        for x, y in loader:
            loss, stats = trainer.step(x, y)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        reuse_threshold: float = 0.10,
        approx_threshold: float = 0.25,
        history_size: int = 16,
        decision_interval: int = 4,
        min_full_layers: int = 4,
        approx_scale: float = 0.3,
    ):
        self.model = model
        self.lr = lr
        self.reuse_threshold = reuse_threshold
        self.approx_threshold = approx_threshold
        self.history_size = history_size
        self.decision_interval = decision_interval
        self.min_full_layers = min_full_layers
        self.approx_scale = approx_scale

        # Activation collector
        self.collector = ActivationCollector()
        self.awf_layers = self.collector.attach(model)
        self.n_layers = len(self.awf_layers)

        # Per-layer activation history (deque of normalized activation vectors)
        self.act_history: Dict[int, deque] = {i: deque(maxlen=history_size) for i in range(self.n_layers)}

        # Current per-layer decisions (0/1/2)
        self.current_levels: List[int] = [2] * self.n_layers

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Stats
        self.stats = {
            "L0_reuse": 0,
            "L1_approx": 0,
            "L2_full": 0,
            "total": 0,
            "steps": 0,
            "generator_skipped": 0,
            "scout_steps": 0,
        }
        self.step_count = 0
        # Save original requires_grad so we can restore
        self._orig_requires_grad = {}
        for layer in self.awf_layers:
            for n, p in layer.named_parameters():
                self._orig_requires_grad[(id(layer), n)] = p.requires_grad

    def _compute_novelty(self, lid: int, act: torch.Tensor) -> float:
        """Novelty = 1 - max_cosine_sim to recent activations."""
        history = self.act_history[lid]
        if len(history) < 2:
            return 1.0
        act_norm = F.normalize(act.unsqueeze(0), dim=-1)
        hist_stack = torch.stack(list(history))
        hist_norm = F.normalize(hist_stack, dim=-1)
        sims = (act_norm @ hist_norm.T).squeeze(0)
        max_sim = sims.max().item()
        return 1.0 - max_sim

    def _update_history(self, lid: int, act: torch.Tensor):
        act_norm = F.normalize(act.unsqueeze(0), dim=-1).squeeze(0)
        self.act_history[lid].append(act_norm)

    def _decide_levels(self) -> List[int]:
        """Decide training level (0/1/2) for each layer based on novelty."""
        levels = []
        novelties = []
        for lid in range(self.n_layers):
            act = self.collector.get_activation(lid)
            if act is None:
                levels.append(2)
                novelties.append(1.0)
                continue
            novelty = self._compute_novelty(lid, act)
            novelties.append(novelty)
            if novelty < self.reuse_threshold:
                levels.append(0)
            elif novelty < self.approx_threshold:
                levels.append(1)
            else:
                levels.append(2)

        # Safety: ensure at least min_full_layers are at Level 2
        n_full = sum(1 for l in levels if l == 2)
        if n_full < self.min_full_layers:
            # Promote the highest-novelty L0/L1 to L2
            to_promote = sorted(
                [(novelties[lid], lid) for lid, lvl in enumerate(levels) if lvl < 2],
                reverse=True
            )
            for _, lid in to_promote[:self.min_full_layers - n_full]:
                levels[lid] = 2

        return levels

    def _apply_freezing(self, levels: List[int]):
        """Set requires_grad on each layer's parameters based on level."""
        for lid, (level, layer) in enumerate(zip(levels, self.awf_layers)):
            # Level 0 = freeze (requires_grad=False) → skip backward
            # Level 1/2 = unfreeze (requires_grad=True)
            new_rg = level > 0
            for p in layer.parameters():
                if p.is_floating_point() and p.requires_grad != new_rg:
                    p.requires_grad_(new_rg)

        # If generator exists, freeze it when ALL layers are L0
        if hasattr(self.model, "generator"):
            all_l0 = all(l == 0 for l in levels)
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(not all_l0)

    def _restore_requires_grad(self):
        """Restore all parameters to requires_grad=True (for evaluation)."""
        for layer in self.awf_layers:
            for p in layer.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)
        if hasattr(self.model, "generator"):
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, dict]:
        """One training step with event-driven gradient computation."""
        self.step_count += 1

        # Every K steps, run scout to decide levels; reuse decision for K-1 steps after
        if self.step_count % self.decision_interval == 1 or self.decision_interval == 1:
            # Scout: forward with all layers trainable (to capture activations)
            self._apply_freezing([2] * self.n_layers)  # all trainable for scout
            self.collector.clear()
            with torch.no_grad():
                _ = self.model(x)  # forward only to populate activations
            self.stats["scout_steps"] += 1

            # Decide new levels
            self.current_levels = self._decide_levels()

            # Update activation history with new observations
            for lid in range(self.n_layers):
                act = self.collector.get_activation(lid)
                if act is not None:
                    self._update_history(lid, act)

        # Apply freezing for this step
        self._apply_freezing(self.current_levels)

        # Forward + backward with selective freezing
        self.optimizer.zero_grad()
        # Need to zero grads of frozen params too (they may have stale grads)
        for layer in self.awf_layers:
            for p in layer.parameters():
                if p.grad is not None:
                    p.grad.zero_()
        if hasattr(self.model, "generator"):
            for p in self.model.generator.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        logits = self.model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()

        # Apply approx scale to L1 layers
        for lid, (level, layer) in enumerate(zip(self.current_levels, self.awf_layers)):
            if level == 1:
                for p in layer.parameters():
                    if p.grad is not None:
                        p.grad.mul_(self.approx_scale)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Count actual stats
        n_l0 = sum(1 for l in self.current_levels if l == 0)
        n_l1 = sum(1 for l in self.current_levels if l == 1)
        n_l2 = sum(1 for l in self.current_levels if l == 2)
        gen_skipped = n_l0 == self.n_layers

        self.stats["L0_reuse"] += n_l0
        self.stats["L1_approx"] += n_l1
        self.stats["L2_full"] += n_l2
        self.stats["total"] += self.n_layers
        self.stats["steps"] += 1
        if gen_skipped:
            self.stats["generator_skipped"] += 1

        step_stats = {
            "L0": n_l0,
            "L1": n_l1,
            "L2": n_l2,
            "skipped_pct": n_l0 / self.n_layers,
            "approx_pct": n_l1 / self.n_layers,
            "full_pct": n_l2 / self.n_layers,
            "gen_skipped": gen_skipped,
            "loss": loss.item(),
        }
        return loss.item(), step_stats

    def get_cumulative_stats(self) -> dict:
        total = max(self.stats["total"], 1)
        steps = max(self.stats["steps"], 1)
        l2_pct = self.stats["L2_full"] / total
        return {
            "steps": self.stats["steps"],
            "scout_steps": self.stats["scout_steps"],
            "L0_reuse_pct": self.stats["L0_reuse"] / total,
            "L1_approx_pct": self.stats["L1_approx"] / total,
            "L2_full_pct": l2_pct,
            "generator_skipped": self.stats["generator_skipped"],
            "generator_skipped_pct": self.stats["generator_skipped"] / steps,
            "estimated_speedup": 1.0 / max(l2_pct, 0.05),  # speedup if backward ∝ L2%
        }

    def close(self):
        self._restore_requires_grad()
        self.collector.remove()


# ---------------------------------------------------------------------------
# Standard Trainer (for comparison)
# ---------------------------------------------------------------------------
class StandardTrainer:
    """Standard full-backward trainer for baseline comparison."""
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.step_count = 0

    def step(self, x, y):
        self.optimizer.zero_grad()
        logits = self.model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.step_count += 1
        return loss.item(), {"L2": 1, "L0": 0, "L1": 0}

    def get_cumulative_stats(self):
        return {"steps": self.step_count, "L2_full_pct": 1.0, "L0_reuse_pct": 0.0,
                "L1_approx_pct": 0.0, "generator_skipped": 0,
                "generator_skipped_pct": 0.0, "estimated_speedup": 1.0}

    def close(self):
        pass
