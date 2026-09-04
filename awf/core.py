"""
AWF Core — Algorithmic Weight Fabric primitives.

The fundamental object is a weight-generating function, not a weight tensor:
    W = G(coord, layer_id) + U @ V + sparse * scale

One generator is shared across ALL layers — the key amortization.

v0.2 improvements:
- Optional sparse ternary corrections (BitNet-style, {-1, 0, +1})
- int8 / int4 quantization helpers for inference compression
- Per-row int8 quantization (finer than per-tensor)
- Improved generator with more frequencies
"""
from __future__ import annotations
import math, struct, zlib
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


def num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quantization helpers (used at inference for additional compression)
# ---------------------------------------------------------------------------

def quantize_int8_per_row(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 quantization for 2D tensors.
    Returns (int8_codes, per_row_scales)."""
    assert weight.dim() == 2
    wmax = weight.abs().amax(dim=-1).clamp(min=1e-8)
    scales = wmax / 127.0
    codes = (weight / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return codes, scales


def quantize_int8_per_tensor(weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Per-tensor symmetric int8 quantization."""
    wmax = weight.abs().max().clamp(min=1e-8).item()
    scale = wmax / 127.0
    codes = (weight / scale).round().clamp(-128, 127).to(torch.int8)
    return codes, scale


def int8_per_row_bytes(codes: torch.Tensor) -> int:
    """Bytes for per-row int8 storage: codes + fp16 scales + magic."""
    M, N = codes.shape
    return M * N + M * 2 + 4   # int8 codes + fp16 scales + magic


def quantize_int4_per_row(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int4 quantization (range -8 to 7)."""
    assert weight.dim() == 2
    wmax = weight.abs().amax(dim=-1).clamp(min=1e-8)
    scales = wmax / 7.0
    codes = (weight / scales.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int8)
    return codes, scales


def int4_per_row_bytes(codes: torch.Tensor) -> int:
    """Bytes for per-row int4 storage: 2 codes per byte + fp16 scales."""
    M, N = codes.shape
    return (M * N + 1) // 2 + M * 2 + 4


def quantize_ternary(weight: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """BitNet-style ternary quantization: {-1, 0, +1}."""
    if weight.numel() == 0:
        return torch.zeros_like(weight, dtype=torch.int8), 1.0
    w_abs = weight.abs().flatten()
    k = max(1, int(0.005 * w_abs.numel()))
    top = torch.topk(w_abs, k=k, largest=True).values
    scale = float(top.mean().item())
    scale = max(scale, 1e-8)
    codes = torch.round(weight / scale).clamp(-1, 1).to(torch.int8)
    return codes, scale


# ---------------------------------------------------------------------------
# Coordinate generator: (i, j) -> scalar weight
# ---------------------------------------------------------------------------

class CoordGenerator(nn.Module):
    """CPPN-style weight generator with Fourier features. Shared across all layers.

    Key design choices:
    - Fourier features (NeRF-style) so output is NOT band-limited
    - Layer embedding so the same generator can produce different patterns per layer
    - Kaiming-style per-layer out_scale init (sqrt(2/fan_in))
    - Zero bias on final Linear (prevents collapse to uniform output)
    - GELU activations (gradient flows through negatives — ReLU kills small-weight nets)
    """
    def __init__(self, n_fourier: int = 12, hidden: int = 64, n_layers: int = 2,
                 n_layers_total: int = 16, layer_input_dims: Optional[List[int]] = None):
        super().__init__()
        self.n_fourier = n_fourier
        # Frequencies: powers of 2 (NeRF-style)
        freq_init = torch.tensor([[2.0**k, 2.0**k] for k in range(n_fourier)], dtype=torch.float32)
        self.log_freq = nn.Parameter(torch.log(freq_init + 1e-6))
        # Per-layer embedding
        self.layer_emb = nn.Parameter(torch.randn(n_layers_total, 8) * 0.1)
        # MLP
        in_dim = 2 * (2 * n_fourier) + 8
        layers = []
        d = in_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden)
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            layers += [lin, nn.GELU()]
            d = hidden
        final = nn.Linear(d, 1)
        nn.init.xavier_uniform_(final.weight); nn.init.zeros_(final.bias)
        layers.append(final)
        self.mlp = nn.Sequential(*layers)
        # Per-layer out_scale (Kaiming-style)
        if layer_input_dims is not None:
            scale = torch.ones(n_layers_total)
            for i, m in enumerate(layer_input_dims):
                if i < n_layers_total:
                    scale[i] = math.sqrt(2.0 / max(m, 1))
            self.out_scale = nn.Parameter(scale)
        else:
            self.out_scale = nn.Parameter(torch.ones(n_layers_total))
        self.out_bias = nn.Parameter(torch.zeros(n_layers_total))

    def forward_batched(self, coords_list: List[Tuple[torch.Tensor, int]]) -> List[torch.Tensor]:
        """Run generator across multiple (coords, layer_id) pairs in ONE batched call."""
        Ns = [c.shape[0] for c, _ in coords_list]
        if not Ns: return []
        total = sum(Ns)
        all_coords = torch.cat([c for c, _ in coords_list], dim=0)
        le_chunks, sc_chunks, bi_chunks = [], [], []
        for n, (_, lid) in zip(Ns, coords_list):
            le_chunks.append(self.layer_emb[lid].unsqueeze(0).expand(n, -1))
            sc_chunks.append(self.out_scale[lid].expand(n))
            bi_chunks.append(self.out_bias[lid].expand(n))
        all_le = torch.cat(le_chunks, dim=0)
        all_sc = torch.cat(sc_chunks, dim=0)
        all_bi = torch.cat(bi_chunks, dim=0)
        freq = torch.exp(self.log_freq)
        scaled = all_coords.unsqueeze(1) * freq.unsqueeze(0)
        sin = torch.sin(scaled).reshape(total, -1)
        cos = torch.cos(scaled).reshape(total, -1)
        h = torch.cat([sin, cos, all_le], dim=-1)
        out = self.mlp(h).squeeze(-1) * all_sc + all_bi
        return list(out.split(Ns))


# ---------------------------------------------------------------------------
# AWF Linear layer
# ---------------------------------------------------------------------------

@dataclass
class AWFLayerConfig:
    in_features: int
    out_features: int
    layer_id: int
    residual_rank: int = 4
    gen_grid: int = 16
    sparse_k: int = 0   # number of sparse ternary corrections (0 = disabled)


class AWFLinear(nn.Module):
    """Linear layer whose weight is RECONSTRUCTED, not stored.

    W = upsample(G(small_grid)) + U @ V + (sparse_ternary if sparse_k > 0)

    Generator runs on a small (m, m) grid (default 16x16=256 points) and is
    bilinearly upsampled to (M, N). This DECOUPLES generator cost from layer width.

    Sparse ternary corrections (BitNet-style {-1,0,+1}) are activated by
    `sparse_k > 0` and added at the top-k largest residual positions.
    """
    def __init__(self, cfg: AWFLayerConfig, generator: CoordGenerator):
        super().__init__()
        self.cfg = cfg
        self.generator = generator
        M, N = cfg.in_features, cfg.out_features
        # Low-rank residual (small init)
        self.U = nn.Parameter(torch.randn(M, cfg.residual_rank) * (0.01 / math.sqrt(M)))
        self.V = nn.Parameter(torch.randn(cfg.residual_rank, N) * (0.01 / math.sqrt(N)))
        self.bias = nn.Parameter(torch.zeros(N))
        # Sparse ternary corrections (optional)
        if cfg.sparse_k > 0:
            self.sparse_idx = nn.Parameter(
                torch.randint(0, max(M * N, 1), (cfg.sparse_k,), dtype=torch.long),
                requires_grad=False)
            # Use tanh * 1.5 -> round -> clamp for differentiable ternary
            self.sparse_code = nn.Parameter(torch.zeros(cfg.sparse_k))
            self.sparse_scale = nn.Parameter(torch.tensor(0.01))
        else:
            self.sparse_idx = None
            self.sparse_code = None
            self.sparse_scale = None
        # Small generator grid
        m = min(cfg.gen_grid, M)
        n = min(cfg.gen_grid, N)
        self._gm, self._gn = m, n
        i = torch.linspace(-1, 1, m); j = torch.linspace(-1, 1, n)
        gi, gj = torch.meshgrid(i, j, indexing="ij")
        self.register_buffer("_grid_coords", torch.stack([gi.flatten(), gj.flatten()], dim=-1), persistent=False)

    def get_residual(self):
        return self.U @ self.V

    def get_sparse_correction(self, W_flat):
        """Apply sparse ternary corrections if enabled."""
        if self.sparse_idx is None:
            return W_flat
        # tanh(code) * 1.5 -> round -> clamp(-1, 1) for differentiable ternary
        tern = (torch.tanh(self.sparse_code) * 1.5).round().clamp(-1, 1)
        W_flat[self.sparse_idx] = W_flat[self.sparse_idx] + self.sparse_scale * tern
        return W_flat

    def forward(self, x):
        # Generate low-res, upsample, add low-rank
        M, N = self.cfg.in_features, self.cfg.out_features
        gen_low = self.generator(self._grid_coords.to(self.U.device), self.cfg.layer_id).reshape(self._gm, self._gn)
        if self._gm == M and self._gn == N:
            gen_full = gen_low
        else:
            gen_full = F.interpolate(gen_low.unsqueeze(0).unsqueeze(0), size=(M, N),
                                      mode="bilinear", align_corners=True).squeeze()
        W = gen_full + self.get_residual()
        # Apply sparse ternary corrections if enabled
        if self.sparse_idx is not None:
            W_flat = W.reshape(-1)
            W_flat = self.get_sparse_correction(W_flat)
            W = W_flat.reshape(M, N)
        return F.linear(x, W.t(), self.bias)

    def personal_storage_bytes(self, fp=2):
        M, N = self.cfg.in_features, self.cfg.out_features
        # Low-rank: 2 * rank * (M + N) * fp + bias N * fp
        bytes_lr = 2 * self.cfg.residual_rank * (M + N) * fp + N * fp
        # Sparse ternary: 2 bits/code + 4 bytes/index + 4 bytes scale
        if self.cfg.sparse_k > 0:
            bytes_sparse = (self.cfg.sparse_k + 3) // 4 + self.cfg.sparse_k * 4 + 4
        else:
            bytes_sparse = 0
        return bytes_lr + bytes_sparse

    def activate_sparse_corrections(self):
        """After warmup, pick the top-k largest errors between current W and the
        generator-only output, and initialize the sparse corrections there.
        Call this after a few epochs of bootstrap training.
        """
        if self.sparse_idx is None or self.cfg.sparse_k == 0:
            return
        with torch.no_grad():
            M, N = self.cfg.in_features, self.cfg.out_features
            # Current full W
            W_current = self.materialize_weight_no_sparse()
            # Generator-only output (without low-rank)
            coords = self._grid_coords.to(self.U.device)
            gen_low = self.generator(coords, self.cfg.layer_id).reshape(self._gm, self._gn)
            if self._gm == M and self._gn == N:
                gen_full = gen_low
            else:
                gen_full = F.interpolate(gen_low.unsqueeze(0).unsqueeze(0),
                                          size=(M, N), mode="bilinear",
                                          align_corners=True).squeeze()
            # Residual = W - gen_only (includes low-rank part)
            residual_full = (W_current - gen_full).reshape(-1)
            # Pick top-k largest absolute residuals
            k = min(self.cfg.sparse_k, residual_full.numel())
            top_vals, top_idx = torch.topk(residual_full.abs(), k=k, largest=True)
            # Pad if buffer is larger than k
            if k < self.sparse_idx.numel():
                pad = self.sparse_idx.numel() - k
                top_idx = torch.cat([top_idx, torch.zeros(pad, dtype=torch.long, device=top_idx.device)])
                top_vals = torch.cat([top_vals, torch.zeros(pad, device=top_vals.device)])
            self.sparse_idx.copy_(top_idx)
            scale = top_vals[:k].mean().clamp(min=1e-6)
            self.sparse_scale.copy_(scale)
            # Initialize code: tanh(0.5) ~ 0.46, * 1.5 ~ 0.69, round ~ 1 (matches sign)
            sign = torch.sign(residual_full[top_idx[:k]])
            init = sign * 0.5
            if k < self.sparse_code.numel():
                init = torch.cat([init, torch.zeros(self.sparse_code.numel() - k, device=init.device)])
            self.sparse_code.copy_(init)

    def materialize_weight_no_sparse(self) -> torch.Tensor:
        """Reconstruct W without sparse corrections (for sparse activation)."""
        M, N = self.cfg.in_features, self.cfg.out_features
        gen_low = self.generator(self._grid_coords.to(self.U.device), self.cfg.layer_id).reshape(self._gm, self._gn)
        if self._gm == M and self._gn == N:
            gen_full = gen_low
        else:
            gen_full = F.interpolate(gen_low.unsqueeze(0).unsqueeze(0), size=(M, N),
                                      mode="bilinear", align_corners=True).squeeze()
        return gen_full + self.get_residual()


# Patch CoordGenerator to also have a non-batched forward (used by AWFLinear above)
def _gen_forward(self, coords, layer_id):
    le = self.layer_emb[layer_id].unsqueeze(0).expand(coords.shape[0], -1)
    freq = torch.exp(self.log_freq)
    scaled = coords.unsqueeze(1) * freq.unsqueeze(0)
    sin = torch.sin(scaled).reshape(coords.shape[0], -1)
    cos = torch.cos(scaled).reshape(coords.shape[0], -1)
    h = torch.cat([sin, cos, le], dim=-1)
    out = self.mlp(h).squeeze(-1)
    return out * self.out_scale[layer_id] + self.out_bias[layer_id]
CoordGenerator.forward = _gen_forward


# ---------------------------------------------------------------------------
# AWF Transformer (the LLM)
# ---------------------------------------------------------------------------

class AWFMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, generator, layer_id, residual_rank=4, gen_grid=16, sparse_k=0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.projs = nn.ModuleDict()
        for name in ["q", "k", "v", "o"]:
            cfg = AWFLayerConfig(d_model, d_model, layer_id, residual_rank, gen_grid, sparse_k)
            self.projs[name] = AWFLinear(cfg, generator)

    def forward(self, x):
        B, T, C = x.shape
        q = self.projs["q"](x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.projs["k"](x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.projs["v"](x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        return self.projs["o"](a)


class AWFTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, generator, layer_id, ff_mult=4,
                 residual_rank=4, gen_grid=16, sparse_k=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = AWFMultiHeadAttention(d_model, n_heads, generator, layer_id,
                                          residual_rank, gen_grid, sparse_k)
        self.ln2 = nn.LayerNorm(d_model)
        up_cfg = AWFLayerConfig(d_model, ff_mult * d_model, layer_id + 1, residual_rank, gen_grid, sparse_k)
        down_cfg = AWFLayerConfig(ff_mult * d_model, d_model, layer_id + 2, residual_rank, gen_grid, sparse_k)
        self.up = AWFLinear(up_cfg, generator)
        self.down = AWFLinear(down_cfg, generator)

    def forward(self, x):
        h = self.ln1(x); x = x + self.attn(h)
        h = self.ln2(x); x = x + self.down(F.gelu(self.up(h)))
        return x


class AWFTransformer(nn.Module):
    """A small GPT-style LLM where every Linear is replaced by AWFLinear.
    One CoordGenerator is shared across ALL layers (the amortization).
    """
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128,
                 ff_mult=4, residual_rank=8, gen_kwargs=None, gen_grid=16, sparse_k=0):
        super().__init__()
        if gen_kwargs is None:
            gen_kwargs = dict(n_fourier=16, hidden=96, n_layers=3)
        # Each block uses 6 AWF layers (q,k,v,o,up,down). Plus head = 6*n_layers + 1
        total_awf = 6 * n_layers + 1
        layer_input_dims = []
        for _ in range(n_layers):
            layer_input_dims.extend([d_model, d_model, d_model, d_model, d_model, ff_mult * d_model])
        layer_input_dims.append(d_model)
        self.generator = CoordGenerator(n_layers_total=total_awf + 4,
                                         layer_input_dims=layer_input_dims, **gen_kwargs)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList()
        lid = 0
        for _ in range(n_layers):
            self.blocks.append(AWFTransformerBlock(d_model, n_heads, self.generator, lid,
                                                    ff_mult, residual_rank, gen_grid, sparse_k))
            lid += 3
        self.ln_f = nn.LayerNorm(d_model)
        head_cfg = AWFLayerConfig(d_model, vocab_size, lid, residual_rank, gen_grid, sparse_k)
        self.head = AWFLinear(head_cfg, self.generator)
        self.sparse_k = sparse_k

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

    def activate_sparse_corrections(self):
        """Activate sparse corrections on all AWFLinear layers (call after warmup)."""
        for blk in self.blocks:
            for name in ["q", "k", "v", "o"]:
                blk.attn.projs[name].activate_sparse_corrections()
            blk.up.activate_sparse_corrections()
            blk.down.activate_sparse_corrections()
        self.head.activate_sparse_corrections()

    def total_storage_bytes(self, fp=2):
        gen = num_params(self.generator) * fp
        emb = sum(p.numel() for p in self.tok_emb.parameters()) * fp + self.pos_emb.numel() * fp
        for blk in self.blocks:
            emb += sum(p.numel() for p in blk.ln1.parameters()) * fp
            emb += sum(p.numel() for p in blk.ln2.parameters()) * fp
            for n in ["q","k","v","o"]:
                emb += blk.attn.projs[n].personal_storage_bytes(fp)
            emb += blk.up.personal_storage_bytes(fp) + blk.down.personal_storage_bytes(fp)
        emb += sum(p.numel() for p in self.ln_f.parameters()) * fp
        emb += self.head.personal_storage_bytes(fp)
        return gen + emb

    def quantize_for_inference(self, method="int8_per_row"):
        """Apply quantization to all low-rank U, V matrices (in-place).
        Returns total storage bytes after quantization.

        method: 'int8_per_row', 'int8_per_tensor', 'int4_per_row'
        """
        total = 0
        # Generator: keep at fp16 (small, high-precision-sensitive)
        gen_bytes = num_params(self.generator) * 2
        total += gen_bytes
        # Embeddings and LayerNorms: keep at fp16
        emb_bytes = sum(p.numel() for p in self.tok_emb.parameters()) * 2 + self.pos_emb.numel() * 2
        for blk in self.blocks:
            emb_bytes += sum(p.numel() for p in blk.ln1.parameters()) * 2
            emb_bytes += sum(p.numel() for p in blk.ln2.parameters()) * 2
        emb_bytes += sum(p.numel() for p in self.ln_f.parameters()) * 2
        total += emb_bytes

        def quant_and_replace(layer):
            nonlocal total
            if method == "int8_per_row":
                U_codes, U_scales = quantize_int8_per_row(layer.U.detach().cpu())
                V_codes, V_scales = quantize_int8_per_row(layer.V.detach().cpu())
                with torch.no_grad():
                    layer.U.copy_((U_codes.float() * U_scales.unsqueeze(-1)).to(layer.U.device))
                    layer.V.copy_((V_codes.float() * V_scales.unsqueeze(-1)).to(layer.V.device))
                total += int8_per_row_bytes(U_codes) + int8_per_row_bytes(V_codes)
            elif method == "int8_per_tensor":
                U_codes, U_scale = quantize_int8_per_tensor(layer.U.detach().cpu())
                V_codes, V_scale = quantize_int8_per_tensor(layer.V.detach().cpu())
                with torch.no_grad():
                    layer.U.copy_((U_codes.float() * U_scale).to(layer.U.device))
                    layer.V.copy_((V_codes.float() * V_scale).to(layer.V.device))
                total += U_codes.numel() + 4 + V_codes.numel() + 4
            elif method == "int4_per_row":
                U_codes, U_scales = quantize_int4_per_row(layer.U.detach().cpu())
                V_codes, V_scales = quantize_int4_per_row(layer.V.detach().cpu())
                with torch.no_grad():
                    layer.U.copy_((U_codes.float() * U_scales.unsqueeze(-1)).to(layer.U.device))
                    layer.V.copy_((V_codes.float() * V_scales.unsqueeze(-1)).to(layer.V.device))
                total += int4_per_row_bytes(U_codes) + int4_per_row_bytes(V_codes)
            # Bias: fp16
            if layer.bias is not None:
                total += layer.bias.numel() * 2
            # Sparse ternary (already 2-bit codes)
            if layer.sparse_idx is not None and layer.cfg.sparse_k > 0:
                k = layer.sparse_idx.numel()
                total += (k + 3) // 4 + k * 4 + 4

        for blk in self.blocks:
            for n in ["q", "k", "v", "o"]:
                quant_and_replace(blk.attn.projs[n])
            quant_and_replace(blk.up)
            quant_and_replace(blk.down)
        quant_and_replace(self.head)

        return total


# ---------------------------------------------------------------------------
# Dense Transformer (baseline)
# ---------------------------------------------------------------------------

class DenseTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x); a, _ = self.attn(h, h, h, need_weights=False); x = x + a
        h = self.ln2(x); x = x + self.ff(h)
        return x


class DenseTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4, block_size=128, ff_mult=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList([DenseTransformerBlock(d_model, n_heads, ff_mult)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    awf = AWFTransformer(vocab_size=100, d_model=128, n_layers=4, n_heads=4, block_size=128)
    x = torch.randint(0, 100, (2, 128))
    y = awf(x)
    print(f"AWF Transformer: {x.shape} -> {y.shape}, params={num_params(awf):,}, gen={num_params(awf.generator):,}, storage={awf.total_storage_bytes(2)/1024:.1f}KB")
    dense = DenseTransformer(vocab_size=100, d_model=128, n_layers=4, n_heads=4, block_size=128)
    y2 = dense(x)
    print(f"Dense Transformer: {x.shape} -> {y2.shape}, params={num_params(dense):,}, storage={num_params(dense)*4/1024:.1f}KB")
