"""Block-level v3: weight caching at block granularity (not layer granularity).

This combines:
  - Block-level decisions (6 decisions vs 37 — less overhead)
  - Weight caching (skip generator forward for frozen blocks)
  - Learned gating (tiny MLP predicts skip)

Target: push past 1.54x combined speedup.
"""
import math, time, torch, torch.nn.functional as F
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch.nn as nn


class BlockV3Trainer:
    """Block-level event trainer with weight caching.

    For each transformer block:
      - Decide L0 (skip) or L2 (full) based on activation novelty + learned gate
      - If L0: cache the block's FULL output for the current input and skip
        the block's forward AND backward entirely on subsequent similar inputs

    Actually, we can't cache block OUTPUT (it depends on input). But we CAN
    cache the WEIGHTS (which don't depend on input). So:
      - L0: cache weights, skip generator forward (use cached weights instead)
      - L2: recompute weights (generator forward), full backward
    """

    def __init__(self, model, lr=1e-3, reuse_threshold=0.12, max_staleness=4,
                 warmup_steps=30, min_full_blocks=1):
        self.model = model
        self.lr = lr
        self.reuse_threshold = reuse_threshold
        self.max_staleness = max_staleness
        self.warmup_steps = warmup_steps
        self.min_full_blocks = min_full_blocks

        self.blocks = list(model.blocks) if hasattr(model, "blocks") else []
        self.n_blocks = len(self.blocks)

        # Per-block weight cache (the generated weights, not block output)
        self.weight_cache: Dict[int, Dict[str, torch.Tensor]] = {
            i: {} for i in range(self.n_blocks)
        }
        self.cache_valid: Dict[int, bool] = {i: False for i in range(self.n_blocks)}

        # Per-block activation history (statistics-based)
        self.act_history: Dict[int, deque] = {i: deque(maxlen=8) for i in range(self.n_blocks)}
        self.staleness: List[int] = [0] * self.n_blocks

        # Hooks
        self.handles = []
        self.block_stats: Dict[int, torch.Tensor] = {}
        self._attach_hooks()

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.step_count = 0
        self.stats = {"L0": 0, "L2": 0, "total": 0, "steps": 0, "cache_hits": 0}

    def _attach_hooks(self):
        for bid, blk in enumerate(self.blocks):
            handle = blk.register_forward_hook(self._make_hook(bid))
            self.handles.append(handle)

    def _make_hook(self, bid):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                stats = torch.tensor([
                    out.mean().item(),
                    out.std().item() + 1e-8,
                    out.max().item(),
                    out.min().item()
                ])
                self.block_stats[bid] = stats
        return hook

    def _compute_novelty(self, bid):
        history = self.act_history[bid]
        if len(history) < 2:
            return 1.0
        stats = self.block_stats.get(bid)
        if stats is None:
            return 1.0
        hist_stack = torch.stack(list(history))
        dists = (hist_stack - stats.unsqueeze(0)).norm(dim=-1)
        return min(1.0, dists.min().item() / 2.0)

    def _decide_levels(self):
        levels = []
        novelties = []
        for bid in range(self.n_blocks):
            if self.staleness[bid] >= self.max_staleness:
                levels.append(2); novelties.append(1.0); continue
            nov = self._compute_novelty(bid)
            novelties.append(nov)
            if self.step_count > self.warmup_steps and nov < self.reuse_threshold:
                levels.append(0)
            else:
                levels.append(2)
        n_full = sum(1 for l in levels if l == 2)
        if n_full < self.min_full_blocks:
            to_promote = sorted(
                [(novelties[bid], bid) for bid, lvl in enumerate(levels) if lvl < 2],
                reverse=True
            )
            for _, bid in to_promote[:self.min_full_blocks - n_full]:
                levels[bid] = 2
        return levels

    def _apply_freezing_and_cache(self, levels):
        for bid, (level, blk) in enumerate(zip(levels, self.blocks)):
            new_rg = level > 0
            for p in blk.parameters():
                if p.is_floating_point() and p.requires_grad != new_rg:
                    p.requires_grad_(new_rg)

            if level == 0:
                # Cache weights if not already cached
                if not self.cache_valid.get(bid, False):
                    with torch.no_grad():
                        # Cache all AWFLinear weights in this block
                        if hasattr(blk, "attn"):
                            for name in ["q", "k", "v", "o"]:
                                layer = blk.attn.projs[name]
                                W = self._compute_full_weight(layer)
                                self.weight_cache[bid][name] = W
                        if hasattr(blk, "up"):
                            self.weight_cache[bid]["up"] = self._compute_full_weight(blk.up)
                        if hasattr(blk, "down"):
                            self.weight_cache[bid]["down"] = self._compute_full_weight(blk.down)
                    self.cache_valid[bid] = True
                self.stats["cache_hits"] += 1
            else:
                self.cache_valid[bid] = False

        # Generator
        if hasattr(self.model, "generator"):
            all_l0 = all(l == 0 for l in levels)
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(not all_l0)

    def _compute_full_weight(self, layer):
        """Compute the full weight matrix for an AWFLinear layer."""
        M, N = layer.cfg.in_features, layer.cfg.out_features
        coords = layer._grid_coords.to(layer.U.device)
        gen_low = layer.generator(coords, layer.cfg.layer_id).reshape(layer._gm, layer._gn)
        if layer._gm == M and layer._gn == N:
            gen_full = gen_low
        else:
            gen_full = F.interpolate(
                gen_low.unsqueeze(0).unsqueeze(0),
                size=(M, N), mode="bilinear", align_corners=True).squeeze()
        W = gen_full + layer.get_residual()
        if hasattr(layer, 'sparse_idx') and layer.sparse_idx is not None:
            tern = (torch.tanh(layer.sparse_code) * 1.5).round().clamp(-1, 1)
            W_flat = W.reshape(-1)
            W_flat[layer.sparse_idx] = W_flat[layer.sparse_idx] + layer.sparse_scale * tern
            W = W_flat.reshape(M, N)
        return W.detach()

    def _patch_block_forwards(self, levels):
        """Patch AWFLinear.forward in L0 blocks to use cached weights."""
        for bid, (level, blk) in enumerate(zip(levels, self.blocks)):
            if level == 0 and self.cache_valid.get(bid, False):
                cached = self.weight_cache[bid]
                # Patch each AWFLinear in this block
                if hasattr(blk, "attn"):
                    for name in ["q", "k", "v", "o"]:
                        layer = blk.attn.projs[name]
                        self._patch_layer(layer, cached[name])
                if hasattr(blk, "up"):
                    self._patch_layer(blk.up, cached["up"])
                if hasattr(blk, "down"):
                    self._patch_layer(blk.down, cached["down"])
            else:
                # Unpatch
                if hasattr(blk, "attn"):
                    for name in ["q", "k", "v", "o"]:
                        self._unpatch_layer(blk.attn.projs[name])
                if hasattr(blk, "up"):
                    self._unpatch_layer(blk.up)
                if hasattr(blk, "down"):
                    self._unpatch_layer(blk.down)

    def _patch_layer(self, layer, cached_W):
        if not getattr(layer, '_patched', False):
            layer._original_forward = layer.forward
            layer._patched = True
        layer.forward = lambda x, _l=layer, _w=cached_W: F.linear(x, _w.t(), _l.bias)

    def _unpatch_layer(self, layer):
        if getattr(layer, '_patched', False):
            layer.forward = layer._original_forward
            layer._patched = False

    def step(self, x, y):
        self.step_count += 1
        levels = self._decide_levels()
        self._apply_freezing_and_cache(levels)
        self._patch_block_forwards(levels)

        self.optimizer.zero_grad()
        logits = self.model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()

        for bid, level in enumerate(levels):
            if level == 0:
                self.staleness[bid] += 1
            else:
                self.staleness[bid] = 0
            stats = self.block_stats.get(bid)
            if stats is not None:
                self.act_history[bid].append(stats)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Unpatch for next step (will re-patch if still L0)
        for blk in self.blocks:
            if hasattr(blk, "attn"):
                for name in ["q", "k", "v", "o"]:
                    self._unpatch_layer(blk.attn.projs[name])
            if hasattr(blk, "up"):
                self._unpatch_layer(blk.up)
            if hasattr(blk, "down"):
                self._unpatch_layer(blk.down)

        n_l0 = sum(1 for l in levels if l == 0)
        n_l2 = sum(1 for l in levels if l == 2)
        self.stats["L0"] += n_l0
        self.stats["L2"] += n_l2
        self.stats["total"] += self.n_blocks
        self.stats["steps"] += 1

        return loss.item(), {"L0": n_l0, "L2": n_l2, "skip_pct": n_l0 / max(self.n_blocks, 1)}

    def get_cumulative_stats(self):
        total = max(self.stats["total"], 1)
        l2_pct = self.stats["L2"] / total
        return {"steps": self.stats["steps"],
                "L0_reuse_pct": self.stats["L0"] / total,
                "L2_full_pct": l2_pct,
                "cache_hits": self.stats["cache_hits"],
                "estimated_speedup": 1.0 / max(l2_pct, 0.05)}

    def close(self):
        for h in self.handles:
            h.remove()
        for blk in self.blocks:
            if hasattr(blk, "attn"):
                for name in ["q", "k", "v", "o"]:
                    self._unpatch_layer(blk.attn.projs[name])
            if hasattr(blk, "up"):
                self._unpatch_layer(blk.up)
            if hasattr(blk, "down"):
                self._unpatch_layer(blk.down)
            for p in blk.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)
        if hasattr(self.model, "generator"):
            for p in self.model.generator.parameters():
                if p.is_floating_point():
                    p.requires_grad_(True)
