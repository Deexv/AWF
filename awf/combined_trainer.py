"""
Combined trainer: GII (batch-level filtering) + Block-level event-driven (block-level).

This attacks BOTH sources of waste:
  1. Batch-level: skip entire batches that are redundant (GII)
  2. Block-level: skip backward through stable blocks (event-driven)

Theoretical combined speedup = 1/(batch_skip_rate) × 1/(block_l2_pct)
Example: 30% batches skipped × 50% blocks at L0 → 1.43× × 2.0× = 2.86× speedup
"""
import math, time, torch, torch.nn.functional as F
from typing import Tuple, Dict, List
from awf.gii_trainer import GradientInformationIndex
from awf.block_event_trainer import BlockLevelEventTrainer


class CombinedTrainer:
    """Combined GII + block-level event-driven trainer."""

    def __init__(self, model, lr=1e-3,
                 gii_skip_threshold=0.92, gii_warmup=50,
                 block_reuse_threshold=0.10, block_max_staleness=4,
                 block_warmup=50, use_learned_gate=True):
        self.model = model

        # GII for batch-level filtering
        self.gii = GradientInformationIndex(
            model, signature_dim=64, history_size=32,
            skip_threshold=gii_skip_threshold, warmup_steps=gii_warmup
        )

        # Block-level event trainer for the actual training
        # We'll use its infrastructure for block decisions but not its optimizer
        self.block_trainer = BlockLevelEventTrainer(
            model, lr=lr, reuse_threshold=block_reuse_threshold,
            max_staleness=block_max_staleness, warmup_steps=block_warmup,
            use_learned_gate=use_learned_gate
        )
        # Share the optimizer
        self.optimizer = self.block_trainer.optimizer

        self.step_count = 0
        self.stats = {
            "batches_seen": 0, "batches_skipped_gii": 0,
            "blocks_l0": 0, "blocks_l2": 0, "blocks_total": 0,
        }

    def step(self, x, y) -> Tuple[float, dict]:
        self.step_count += 1
        self.stats["batches_seen"] += 1

        # Forward pass (GII hook captures embedding activation)
        self.gii.reset_act_capture()
        # Block trainer's hooks also capture block stats during this forward
        logits = self.model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

        # GII decision: skip batch entirely? (based on activation novelty, no backward needed)
        should_skip, novelty = self.gii.should_skip()

        if should_skip:
            self.stats["batches_skipped_gii"] += 1
            return loss.item(), {"gii_skipped": True, "novelty": novelty,
                                  "block_skip_pct": 0}

        # Not skipped by GII → do block-level event training
        levels = self.block_trainer._decide_levels()
        self.block_trainer._apply_freezing(levels)

        self.optimizer.zero_grad()
        loss.backward()

        for bid, level in enumerate(levels):
            if level == 0:
                self.block_trainer.staleness[bid] += 1
            else:
                self.block_trainer.staleness[bid] = 0
            stats = self.block_trainer.block_stats.get(bid)
            if stats is not None:
                self.block_trainer.act_history[bid].append(stats)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.block_trainer._update_gate()

        n_l0 = sum(1 for l in levels if l == 0)
        n_l2 = sum(1 for l in levels if l == 2)
        self.stats["blocks_l0"] += n_l0
        self.stats["blocks_l2"] += n_l2
        self.stats["blocks_total"] += len(levels)

        return loss.item(), {
            "gii_skipped": False, "novelty": novelty,
            "block_skip_pct": n_l0 / max(len(levels), 1),
            "L0": n_l0, "L2": n_l2
        }

    def get_cumulative_stats(self) -> dict:
        batch_skip_rate = self.stats["batches_skipped_gii"] / max(self.stats["batches_seen"], 1)
        block_l2_pct = self.stats["blocks_l2"] / max(self.stats["blocks_total"], 1)
        return {
            "steps": self.step_count,
            "batches_seen": self.stats["batches_seen"],
            "batches_skipped_gii": self.stats["batches_skipped_gii"],
            "batch_skip_rate": batch_skip_rate,
            "block_l0_pct": self.stats["blocks_l0"] / max(self.stats["blocks_total"], 1),
            "block_l2_pct": block_l2_pct,
            "estimated_speedup": (1.0 / max(1 - batch_skip_rate, 0.01)) * (1.0 / max(block_l2_pct, 0.05)),
        }

    def close(self):
        self.gii.close()
        self.block_trainer.close()
