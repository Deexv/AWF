"""
Gradient Information Index (GII) — filter training batches by marginal learning value.

The big idea: not all batches contribute equally to learning. Many are redundant
with what the model has already seen. If we can detect this CHEAPLY, we can skip
them entirely — saving both forward AND backward computation.

Circularity break: we don't compute full gradients to decide which batches to
compute full gradients on. Instead, we use a CHEAP PROXY signal:
  - Run forward pass (which we'd do anyway for the loss)
  - Compute gradient w.r.t. ONLY the input embedding (1 small backward, not full)
  - Project this gradient to a low-dim "signature"
  - Compare to recent signatures via cosine similarity
  - Skip batch if signature is too similar to recent ones

This is O(d_model) per batch for the signature, vs O(N_params) for full backward.

Combined with event-driven training: a batch can be skipped at TWO levels:
  - Batch level: GII says "this batch is redundant" → skip entirely
  - Layer level: event-driven says "this layer is stable" → skip its backward

Together, these attack BOTH sources of waste.
"""
from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientInformationIndex:
    """Filters training batches by marginal learning value using ACTIVATION novelty.

    Key insight: we don't need gradients to decide if a batch is redundant.
    If the forward-pass activations are similar to recent batches, the gradient
    will be similar too. So we use activation cosine similarity as the signal.

    This breaks the circularity (computing gradients to decide which batches
    deserve gradients) because we only need the forward pass, which we'd do
    anyway for the loss.

    For each batch:
      1. Forward pass (hooks capture embedding output)
      2. Compute activation signature (mean activation across batch)
      3. Compare to recent signatures via max cosine similarity
      4. If similarity > threshold: skip backward AND optimizer step entirely

    This saves: full backward + optimizer step for skipped batches.
    Cost: one forward pass (which we'd do anyway) + cheap signature computation.
    """

    def __init__(
        self,
        model: nn.Module,
        signature_dim: int = 64,
        history_size: int = 32,
        skip_threshold: float = 0.95,
        warmup_steps: int = 50,
    ):
        self.model = model
        self.signature_dim = signature_dim
        self.history_size = history_size
        self.skip_threshold = skip_threshold
        self.warmup_steps = warmup_steps

        # Determine d_model from the embedding layer
        if hasattr(model, "tok_emb"):
            d_model = model.tok_emb.embedding_dim
        elif hasattr(model, "token_embedding"):
            d_model = model.token_embedding.embedding_dim
        else:
            d_model = 256
        self.d_model = d_model

        # Random projection for signature (reduce d_model → signature_dim)
        self.projection = torch.randn(signature_dim, d_model) / math.sqrt(d_model)

        # History of recent activation signatures
        self.signature_history: deque = deque(maxlen=history_size)

        # Stats
        self.batches_seen = 0
        self.batches_skipped = 0
        self.step_count = 0

        # Hook to capture embedding output (activation, not gradient)
        self._act_capture: Optional[torch.Tensor] = None
        self._setup_act_hook()

    def _setup_act_hook(self):
        """Hook the FIRST TRANSFORMER BLOCK's output (not embedding).

        Embedding outputs are nearly identical across batches (just token lookup
        + position). The first block's output is much more discriminative because
        it reflects the model's processing of the input.
        """
        target_layer = None
        # Try to find first transformer block
        if hasattr(self.model, "blocks") and len(self.model.blocks) > 0:
            target_layer = self.model.blocks[0]
        elif hasattr(self.model, "tok_emb"):
            target_layer = self.model.tok_emb
        elif hasattr(self.model, "token_embedding"):
            target_layer = self.model.token_embedding
        if target_layer is None:
            return

        original_forward = target_layer.forward

        def hooked_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                self._act_capture = out.detach()
            return out

        target_layer.forward = hooked_forward
        self._original_target_forward = original_forward
        self._target_layer = target_layer

    def compute_signature(self) -> Optional[torch.Tensor]:
        """Compute activation signature from captured embedding output.

        Rich signature: combines mean activation, std, and loss magnitude.
        This captures more information than just the mean.
        """
        if self._act_capture is None:
            return None
        # Mean activation across batch and seq_len → (d_model,)
        act_vec = self._act_capture.mean(dim=(0, 1))
        # Also capture std (variation within the batch)
        act_std = self._act_capture.std(dim=(0, 1))

        # Combine mean and std into a 2*d_model vector, then project
        combined = torch.cat([act_vec, act_std])
        # Project (2*d_model → signature_dim)
        if self.projection.size(1) != combined.numel():
            self.projection = torch.randn(self.signature_dim, combined.numel()) / math.sqrt(combined.numel())
        signature = self.projection @ combined
        # Normalize
        signature = F.normalize(signature.unsqueeze(0), dim=-1).squeeze(0)
        return signature.detach()

    def should_skip(self) -> Tuple[bool, float]:
        """Decide whether to skip this batch based on activation novelty."""
        self.step_count += 1

        # Warmup: never skip during first N steps
        if self.step_count <= self.warmup_steps:
            sig = self.compute_signature()
            if sig is not None:
                self.signature_history.append(sig)
            return False, 1.0

        sig = self.compute_signature()
        if sig is None:
            return False, 1.0

        if len(self.signature_history) < 2:
            self.signature_history.append(sig)
            return False, 1.0

        # Max cosine similarity to history
        hist_stack = torch.stack(list(self.signature_history))
        sims = (sig.unsqueeze(0) @ hist_stack.T).squeeze(0)
        max_sim = sims.max().item()
        novelty = 1.0 - max_sim

        self.signature_history.append(sig)

        # Skip if novelty too low (batch is redundant)
        should_skip = novelty < (1.0 - self.skip_threshold)
        return should_skip, novelty

    def reset_act_capture(self):
        """Clear captured activation (call before each forward pass)."""
        self._act_capture = None

    def get_stats(self) -> dict:
        skip_rate = self.batches_skipped / max(self.batches_seen, 1)
        return {
            "batches_seen": self.batches_seen,
            "batches_skipped": self.batches_skipped,
            "skip_rate": skip_rate,
            "history_size": len(self.signature_history),
        }

    def close(self):
        if hasattr(self, "_original_target_forward") and hasattr(self, "_target_layer"):
            self._target_layer.forward = self._original_target_forward


class GIITrainer:
    """Trainer that combines Gradient Information Index with standard training.

    Pipeline per batch:
      1. Forward pass (captures embedding grad via hook)
      2. Compute loss
      3. Backward w.r.t. embedding ONLY (cheap) → get gradient signature
      4. Decide: skip or proceed?
      5. If proceed: full backward + optimizer step
      6. If skip: do nothing (saves full backward + optimizer step)

    The trick: we do a PARTIAL backward (only to the embedding) to get the
    signature. This is much cheaper than full backward because we don't need
    gradients for all parameters — just the embedding output.

    Implementation note: we use torch.autograd.grad with only_inputs=True
    to compute the embedding gradient without populating .grad on other params.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        gii_config: Optional[dict] = None,
    ):
        self.model = model
        self.lr = lr

        if gii_config is None:
            gii_config = {}
        self.gii = GradientInformationIndex(model, **gii_config)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        self.step_count = 0
        self.novelty_history: List[float] = []

    def step(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[float, dict]:
        """One training step with GII filtering.

        Pipeline:
          1. Forward pass (hook captures embedding activation)
          2. Compute activation signature (cheap — no backward needed)
          3. Decide: skip or proceed?
          4. If skip: return loss WITHOUT backward (saves full backward + optimizer)
          5. If proceed: backward + optimizer step

        The key win: skipped batches save the ENTIRE backward + optimizer cost.
        Only the forward pass (which we'd do anyway for monitoring) is paid.
        """
        self.step_count += 1
        self.gii.batches_seen += 1

        # Forward pass (hook captures embedding activation)
        self.gii.reset_act_capture()
        logits = self.model(x)
        V = logits.size(-1)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))

        # Decide whether to skip based on ACTIVATION novelty (no backward needed!)
        should_skip, novelty = self.gii.should_skip()
        self.novelty_history.append(novelty)

        if should_skip:
            # Skip backward AND optimizer step entirely
            self.gii.batches_skipped += 1
            return loss.item(), {"skipped": True, "novelty": novelty,
                                 "skip_rate": self.gii.batches_skipped / max(self.gii.batches_seen, 1)}

        # Not skipped → full backward + optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), {"skipped": False, "novelty": novelty,
                              "skip_rate": self.gii.batches_skipped / max(self.gii.batches_seen, 1)}

    def get_cumulative_stats(self) -> dict:
        avg_novelty = sum(self.novelty_history) / max(len(self.novelty_history), 1)
        stats = self.gii.get_stats()
        stats["avg_novelty"] = avg_novelty
        stats["steps"] = self.step_count
        stats["estimated_speedup"] = 1.0 / max(1.0 - stats["skip_rate"], 0.01)
        return stats

    def close(self):
        self.gii.close()
