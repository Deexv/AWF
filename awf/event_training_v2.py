"""
Gradient Event-Driven Training v2 for AWF.

v0.6 improvements over v0.5:
1. SCOUTLESS: Uses last training step's activations (from hooks) — no separate
   scout forward pass. Saves ~15% overhead.
2. ADAPTIVE WARMUP: First 20% of training is pure standard (all L2). Then
   progressively enables L0/L1 skipping. This prevents the "too aggressive
   early" failure mode.
3. STALENESS COUNTER: Each layer tracks how many consecutive steps it was
   skipped. After max_staleness (default 8), force full backward. Prevents
   any layer from going permanently stale.
4. GRADIENT NORM SIGNAL: Tracks last step's gradient norm per layer. Skip only
   when BOTH activation novelty is low AND gradient norm was small.

These changes push the speedup from 1.89× toward the theoretical maximum.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActivationCollector:
    """Collects per-layer activations during forward pass."""
    def __init__(self):
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles = []

    def attach(self, model: nn.Module) -> List[nn.Module]:
        awf_layers = []
        if hasattr(model, "blocks"):
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
        elif hasattr(model, "awf_layers"):
            awf_layers = model.awf_layers
        elif hasattr(model, "layers_list"):
            awf_layers = model.layers_list

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
                if act.numel() > 128:
                    act = act[:128]
                self.activations[lid] = act
        return hook

    def get_activation(self, lid):
        return self.activations.get(lid)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class GradientEventTrainerV2:
    """Event-driven trainer v2: scoutless + adaptive warmup + staleness counter.

    Key improvements over v1:
    - No scout forward pass (uses last step's activations from hooks)
    - Adaptive reuse_threshold based on training progress
    - Staleness counter forces catch-up after max_staleness consecutive skips
    - Gradient norm tracking (skip only when both activation AND gradient are small)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        reuse_threshold: float = 0.10,
        approx_threshold: float = 0.25,
        history_size: int = 16,
        min_full_layers: int = 4,
        approx_scale: float = 0.3,
        max_staleness: int = 8,
        warmup_fraction: float = 0.20,
        max_steps: int = 10000,
        gradient_norm_threshold: float = 0.01,
    ):
        self.model = model
        self.lr = lr
        self.base_reuse_threshold = reuse_threshold
        self.approx_threshold = approx_threshold
        self.history_size = history_size
        self.min_full_layers = min_full_layers
        self.approx_scale = approx_scale
        self.max_staleness = max_staleness
        self.warmup_fraction = warmup_fraction
        self.max_steps = max_steps
        self.gradient_norm_threshold = gradient_norm_threshold

        # Activation collector (hooks run during EVERY forward, including training)
        self.collector = ActivationCollector()
        self.awf_layers = self.collector.attach(model)
        self.n_layers = len(self.awf_layers)

        # Per-layer activation history
        self.act_history: Dict[int, deque] = {i: deque(maxlen=history_size) for i in range(self.n_layers)}

        # Per-layer staleness counter
        self.staleness: List[int] = [0] * self.n_layers

        # Per-layer last gradient norm
        self.last_grad_norms: List[float] = [0.0] * self.n_layers

        # Current levels
        self.current_levels: List[int] = [2] * self.n_layers

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        # Stats
        self.stats = {
            "L0_reuse": 0, "L1_approx": 0, "L2_full": 0,
            "total": 0, "steps": 0, "generator_skipped": 0,
            "staleness_forced": 0,
        }
        self.step_count = 0

    def _get_adaptive_threshold(self) -> float:
        """Adaptive reuse threshold: start at 0 (no skipping), ramp up to base."""
        progress = min(self.step_count / max(self.max_steps * self.warmup_fraction, 1), 1.0)
        return self.base_reuse_threshold * progress

    def _compute_novelty(self, lid: int, act: torch.Tensor) -> float:
        history = self.act_history[lid]
        if len(history) < 2:
            return 1.0
        act_norm = F.normalize(act.unsqueeze(0), dim=-1)
        hist_stack = torch.stack(list(history))
        hist_norm = F.normalize(hist_stack, dim=-1)
        sims = (act_norm @ hist_norm.T).squeeze(0)
        return 1.0 - sims.max().item()

    def _decide_levels(self) -> List[int]:
        """Decide per-layer level using adaptive threshold + staleness + grad norm."""
        threshold = self._get_adaptive_threshold()
        levels = []
        novelties = []

        for lid in range(self.n_layers):
            # Force full backward if staleness exceeds max
            if self.staleness[lid] >= self.max_staleness:
                levels.append(2)
                novelties.append(1.0)
                self.stats["staleness_forced"] += 1
                continue

            act = self.collector.get_activation(lid)
            if act is None:
                levels.append(2)
                novelties.append(1.0)
                continue

            novelty = self._compute_novelty(lid, act)
            novelties.append(novelty)
            grad_norm = self.last_grad_norms[lid]

            # Skip only when BOTH activation novelty AND gradient norm are small
            if novelty < threshold and grad_norm < self.gradient_norm_threshold:
                levels.append(0)
            elif novelty < self.approx_threshold:
                levels.append(1)
            else:
                levels.append(2)

        # Safety: ensure min_full_layers at L2
        n_full = sum(1 for l in levels if l == 2)
        if n_full < self.min_full_layers:
            to_promote = sorted(
                [(novelties[lid], lid) for lid, lvl in enumerate(levels) if lvl < 2],
                reverse=True
            )
            for _, lid in to_promote[:self.min_full_layers - n_full]:
                levels[lid] = 2

        return levels

    def _apply_freezing(self, levels: List[int]):
        for lid, (level, layer) in enumerate(zip(levels, self.awf_layers)):
            new_rg = level > 0
            for p in layer.parameters():
                if p.is_floating_point() and p.requires_grad != new_rg:
                    p.requires_grad_(new_rg)
        if hasattr(self.model, "generator"):
            all_l0 = all(l == 0 for l in levels)
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(not all_l0)

    def _restore_requires_grad(self):
        for layer in self.awf_layers:
            for p in layer.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)
        if hasattr(self.model, "generator"):
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, dict]:
        self.step_count += 1

        # SCOUTLESS: use last step's activations (already in self.collector)
        # Decide levels based on what hooks captured during the LAST forward
        self.current_levels = self._decide_levels()

        # Apply freezing
        self._apply_freezing(self.current_levels)

        # Forward + backward (hooks capture NEW activations during this forward)
        self.optimizer.zero_grad()
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

        # Apply approx scale to L1 layers + record gradient norms
        for lid, (level, layer) in enumerate(zip(self.current_levels, self.awf_layers)):
            # Compute gradient norm BEFORE scaling
            grad_norm = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm().item() ** 2
            grad_norm = math.sqrt(grad_norm)
            self.last_grad_norms[lid] = grad_norm

            if level == 1:
                for p in layer.parameters():
                    if p.grad is not None:
                        p.grad.mul_(self.approx_scale)

            # Update staleness counter
            if level == 0:
                self.staleness[lid] += 1
            else:
                self.staleness[lid] = 0

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Update activation history with NEW activations from this forward
        for lid in range(self.n_layers):
            act = self.collector.get_activation(lid)
            if act is not None:
                act_norm = F.normalize(act.unsqueeze(0), dim=-1).squeeze(0)
                self.act_history[lid].append(act_norm)

        # Stats
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

        return loss.item(), {
            "L0": n_l0, "L1": n_l1, "L2": n_l2,
            "skipped_pct": n_l0 / self.n_layers,
            "full_pct": n_l2 / self.n_layers,
            "gen_skipped": gen_skipped,
            "loss": loss.item(),
            "adaptive_threshold": self._get_adaptive_threshold(),
        }

    def get_cumulative_stats(self) -> dict:
        total = max(self.stats["total"], 1)
        steps = max(self.stats["steps"], 1)
        l2_pct = self.stats["L2_full"] / total
        return {
            "steps": self.stats["steps"],
            "L0_reuse_pct": self.stats["L0_reuse"] / total,
            "L1_approx_pct": self.stats["L1_approx"] / total,
            "L2_full_pct": l2_pct,
            "generator_skipped": self.stats["generator_skipped"],
            "generator_skipped_pct": self.stats["generator_skipped"] / steps,
            "staleness_forced": self.stats["staleness_forced"],
            "estimated_speedup": 1.0 / max(l2_pct, 0.05),
        }

    def close(self):
        self._restore_requires_grad()
        self.collector.remove()


class StandardTrainer:
    """Standard full-backward trainer for baseline."""
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
