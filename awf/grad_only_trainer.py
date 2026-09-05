"""Gradient-norm-only event trainer — minimal overhead, maximum simplicity.

No hooks, no activation history, no novelty computation.
Decision signal: if a layer's gradient norm was small last step, skip it this step.

This is the theoretical minimum-overhead event trainer.
"""
import math, torch, torch.nn.functional as F
from typing import Dict, List, Tuple


class GradientOnlyTrainer:
    """Skip backward for layers whose last gradient norm was below threshold.

    Zero overhead beyond one float comparison per layer per step.
    """
    def __init__(self, model, lr=1e-3, skip_threshold=0.01,
                 max_staleness=6, min_full=6, warmup_steps=50):
        self.model = model
        self.lr = lr
        self.skip_threshold = skip_threshold
        self.max_staleness = max_staleness
        self.min_full = min_full
        self.warmup_steps = warmup_steps

        # Find AWF layers
        self.awf_layers = []
        if hasattr(model, "blocks"):
            for blk in model.blocks:
                if hasattr(blk, "attn"):
                    for n in ["q","k","v","o"]:
                        self.awf_layers.append(blk.attn.projs[n])
                if hasattr(blk, "up"): self.awf_layers.append(blk.up)
                if hasattr(blk, "down"): self.awf_layers.append(blk.down)
            if hasattr(model, "head"): self.awf_layers.append(model.head)
        self.n_layers = len(self.awf_layers)

        self.last_grad_norms = [float('inf')] * self.n_layers
        self.staleness = [0] * self.n_layers
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.step_count = 0
        self.stats = {"L0": 0, "L2": 0, "total": 0}

    def step(self, x, y) -> Tuple[float, dict]:
        self.step_count += 1

        # Decide which layers to freeze
        skip = [False] * self.n_layers
        if self.step_count > self.warmup_steps:
            for lid in range(self.n_layers):
                if self.staleness[lid] >= self.max_staleness:
                    skip[lid] = False  # force update
                elif self.last_grad_norms[lid] < self.skip_threshold:
                    skip[lid] = True

            # Ensure min_full layers are not skipped
            active = [lid for lid in range(self.n_layers) if not skip[lid]]
            if len(active) < self.min_full:
                # Unskip the stalest layers first
                stale_order = sorted(range(self.n_layers),
                                    key=lambda i: self.staleness[i], reverse=True)
                for lid in stale_order:
                    if skip[lid]:
                        skip[lid] = False
                        active.append(lid)
                        if len(active) >= self.min_full:
                            break

        # Apply freezing
        for lid, (s, layer) in enumerate(zip(skip, self.awf_layers)):
            rg = not s
            for p in layer.parameters():
                if p.is_floating_point() and p.requires_grad != rg:
                    p.requires_grad_(rg)
        # Generator: freeze if all layers frozen
        if hasattr(self.model, "generator"):
            all_frozen = all(skip)
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(not all_frozen)

        # Forward + backward
        self.optimizer.zero_grad()
        logits = self.model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()

        # Record gradient norms + update staleness
        n0, n2 = 0, 0
        for lid, (s, layer) in enumerate(zip(skip, self.awf_layers)):
            gn = 0.0
            for p in layer.parameters():
                if p.grad is not None:
                    gn += p.grad.data.norm().item() ** 2
            self.last_grad_norms[lid] = math.sqrt(gn)

            if s:
                self.staleness[lid] += 1
                n0 += 1
            else:
                self.staleness[lid] = 0
                n2 += 1

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.stats["L0"] += n0
        self.stats["L2"] += n2
        self.stats["total"] += self.n_layers

        return loss.item(), {"L0": n0, "L2": n2, "skip_pct": n0 / self.n_layers}

    def get_cumulative_stats(self):
        total = max(self.stats["total"], 1)
        l2_pct = self.stats["L2"] / total
        return {"steps": self.step_count, "L0_reuse_pct": self.stats["L0"] / total,
                "L2_full_pct": l2_pct, "estimated_speedup": 1.0 / max(l2_pct, 0.05)}

    def close(self):
        for layer in self.awf_layers:
            for p in layer.parameters():
                if p.is_floating_point(): p.requires_grad_(True)
        if hasattr(self.model, "generator"):
            for p in self.model.generator.parameters():
                if p.is_floating_point(): p.requires_grad_(True)
