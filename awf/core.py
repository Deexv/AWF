"""
AWF Core — Algorithmic Weight Fabric primitives.

The fundamental object is a weight-generating function, not a weight tensor:
    W = G(coord, layer_id) + U @ V + sparse * scale

One generator is shared across ALL layers — the key amortization.
"""
from __future__ import annotations
import math, struct, zlib
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def num_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


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


class AWFLinear(nn.Module):
    """Linear layer whose weight is RECONSTRUCTED, not stored.

    W = upsample(G(small_grid)) + U @ V
    Generator runs on a small (m, m) grid (default 16x16=256 points) and is
    bilinearly upsampled to (M, N). This DECOUPLES generator cost from layer width.
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
        # Small generator grid
        m = min(cfg.gen_grid, M)
        n = min(cfg.gen_grid, N)
        self._gm, self._gn = m, n
        i = torch.linspace(-1, 1, m); j = torch.linspace(-1, 1, n)
        gi, gj = torch.meshgrid(i, j, indexing="ij")
        self.register_buffer("_grid_coords", torch.stack([gi.flatten(), gj.flatten()], dim=-1), persistent=False)

    def get_residual(self):
        return self.U @ self.V

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
        return F.linear(x, W.t(), self.bias)

    def personal_storage_bytes(self, fp=2):
        M, N = self.cfg.in_features, self.cfg.out_features
        return 2 * self.cfg.residual_rank * (M + N) * fp + N * fp


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
    def __init__(self, d_model, n_heads, generator, layer_id, residual_rank=4, gen_grid=16):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.head_dim = d_model, n_heads, d_model // n_heads
        self.projs = nn.ModuleDict()
        for name in ["q", "k", "v", "o"]:
            cfg = AWFLayerConfig(d_model, d_model, layer_id, residual_rank, gen_grid)
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
                 residual_rank=4, gen_grid=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = AWFMultiHeadAttention(d_model, n_heads, generator, layer_id,
                                          residual_rank, gen_grid)
        self.ln2 = nn.LayerNorm(d_model)
        up_cfg = AWFLayerConfig(d_model, ff_mult * d_model, layer_id + 1, residual_rank, gen_grid)
        down_cfg = AWFLayerConfig(ff_mult * d_model, d_model, layer_id + 2, residual_rank, gen_grid)
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
                 ff_mult=4, residual_rank=8, gen_kwargs=None, gen_grid=16):
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
                                                    ff_mult, residual_rank, gen_grid))
            lid += 3
        self.ln_f = nn.LayerNorm(d_model)
        head_cfg = AWFLayerConfig(d_model, vocab_size, lid, residual_rank, gen_grid)
        self.head = AWFLinear(head_cfg, self.generator)

    def forward(self, x):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)

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
