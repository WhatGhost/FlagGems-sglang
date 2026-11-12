# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chunked gated delta-rule forward (vLLM-Ascend chunk.py semantics).

The shipped vLLM-Ascend version is a chunked schedule (chunk_local_cumsum +
chunk_scaled_dot_kkt + solve_tril + recompute_w_u + an AscendC state kernel), but
chunking is only a scheduling decision: the chunked and fused-recurrent forms
compute the same gated delta-rule recurrence.

Architecture
------------
1. `_qk_prep_kernel`, used only when use_qk_l2norm_in_kernel is set: the
   per-(b, t, h) L2 normalization of q and k are row operations that do not touch
   the recurrent state, so they run in one fully data-parallel pass instead of
   inside the sequential loop.  They are pure normalizations: the `scale`
   multiply stays in the recurrence kernel and is applied exactly once.

2. `_gdn_fwd_kernel`: the recurrence.  One program owns one (sequence, value
   head) pair and one contiguous block of the value axis, and walks the whole
   sequence from its initial state.  The recurrence is column-separable in V,
   so V is the only intra-head parallel axis; K cannot be split because the
   prediction needs the full K reduction at every step.

Every step is the exact update.  Starting part-way through a sequence from a
zero state is not an option: in real models g = -exp(A_log) * softplus(.) gets
arbitrarily close to 0 (Qwen3.5 / Qwen3.6 have heads with A_log near -5.4,
about 0.5% decay per step), so the carried-in state never becomes negligible.

Grouped value heads: v, g and beta may have HV heads for H query/key heads
(HV a multiple of H); value head hv reads query/key head hv // (HV // H).

State layout: initial_state / final_state are [N, HV, K, V] (key axis first).
The state is held as [BLOCK_V, BLOCK_K] or [BLOCK_K, BLOCK_V] in registers
(KV_LAYOUT); which is faster depends on how many programs there are.
"""

import torch
import triton
import triton.language as tl

# Measured on gfx1100: about 1024 one-warp programs keep the part busy without
# re-reading q and k from too many V blocks of the same head.
_TARGET_PROGS = 1024


@triton.jit
def _qk_prep_kernel(
    x_ptr,
    y_ptr,
    TH,
    H,
    K,
    s_x0,
    s_x1,
    s_x2,
    s_y0,
    s_y1,
    s_y2,
    ROWS,
    RPB: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program normalizes RPB (b, t, h) rows of x into y, in float32."""
    rr = tl.program_id(0) * RPB + tl.arange(0, RPB)
    ok = rr < ROWS
    b = rr // TH
    rem = rr % TH
    t = rem // H
    h = rem % H

    offs = tl.arange(0, BLOCK_K)
    m = ok[:, None] & (offs < K)[None, :]
    src = (
        x_ptr
        + b[:, None] * s_x0
        + t[:, None] * s_x1
        + h[:, None] * s_x2
        + offs[None, :]
    )
    x = tl.load(src, mask=m, other=0.0).to(tl.float32)
    n = x * tl.rsqrt(tl.sum(x * x, axis=1) + 1e-6)[:, None]
    dst = (
        y_ptr
        + b[:, None] * s_y0
        + t[:, None] * s_y1
        + h[:, None] * s_y2
        + offs[None, :]
    )
    tl.store(dst, n, mask=m)


@triton.jit
def _load_step(
    k_base,
    q_base,
    v_base,
    g_base,
    b_base,
    t,
    seq_end,
    s_qt,
    s_vt,
    s_gt,
    s_bt,
    offs_k,
    mk,
    offs_v,
    mv,
):
    # A step past the end loads g = beta = 0, which leaves the state as is.
    m = t < seq_end
    kt = tl.load(k_base + t * s_qt + offs_k, mask=mk & m, other=0.0)
    qt = tl.load(q_base + t * s_qt + offs_k, mask=mk & m, other=0.0)
    vt = tl.load(v_base + t * s_vt + offs_v, mask=mv & m, other=0.0)
    gt = tl.load(g_base + t * s_gt, mask=m, other=0.0)
    bt = tl.load(b_base + t * s_bt, mask=m, other=0.0)
    return (
        kt.to(tl.float32),
        qt.to(tl.float32),
        vt.to(tl.float32),
        gt.to(tl.float32),
        bt.to(tl.float32),
    )


@triton.jit
def _gdn_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    bt_ptr,
    hi_ptr,
    ho_ptr,
    cu_ptr,
    o_ptr,
    T,
    HV,
    R,
    K,
    V,
    scale,
    s_qb,
    s_qt,
    s_qh,
    s_vb,
    s_vt,
    s_vh,
    s_gb,
    s_gt,
    s_gh,
    s_bb,
    s_bt,
    s_bh,
    s_in,
    s_ih,
    s_ik,
    s_fn,
    s_fh,
    s_fk,
    s_ob,
    s_ot,
    s_oh,
    HAS_INIT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    OUT_FINAL: tl.constexpr,
    KV_LAYOUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_v = tl.program_id(1)
    hv = pid % HV
    n = pid // HV
    h = hv // R

    offs_k = tl.arange(0, BLOCK_K)
    mk = offs_k < K
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mv = offs_v < V
    if KV_LAYOUT:
        k_idx = offs_k[:, None]
        v_idx = offs_v[None, :]
        m_s = mk[:, None] & mv[None, :]
    else:
        k_idx = offs_k[None, :]
        v_idx = offs_v[:, None]
        m_s = mv[:, None] & mk[None, :]

    if IS_VARLEN:
        seq_begin = tl.load(cu_ptr + n).to(tl.int32)
        seq_end = tl.load(cu_ptr + n + 1).to(tl.int32)
        b = 0
    else:
        seq_begin = 0
        seq_end = T
        b = n

    q_base = q_ptr + b * s_qb + h * s_qh
    k_base = k_ptr + b * s_qb + h * s_qh
    v_base = v_ptr + b * s_vb + hv * s_vh
    g_base = g_ptr + b * s_gb + hv * s_gh
    b_base = bt_ptr + b * s_bb + hv * s_bh
    o_base = o_ptr + b * s_ob + hv * s_oh

    if HAS_INIT:
        S = tl.load(
            hi_ptr + n * s_in + hv * s_ih + k_idx * s_ik + v_idx,
            mask=m_s,
            other=0.0,
        ).to(tl.float32)
    elif KV_LAYOUT:
        S = tl.zeros((BLOCK_K, BLOCK_V), dtype=tl.float32)
    else:
        S = tl.zeros((BLOCK_V, BLOCK_K), dtype=tl.float32)

    for t0 in range(seq_begin, seq_end, UNROLL):
        # Issue the loads of UNROLL steps before the dependent updates.
        steps = ()
        for j in tl.static_range(UNROLL):
            steps += (
                _load_step(
                    k_base,
                    q_base,
                    v_base,
                    g_base,
                    b_base,
                    t0 + j,
                    seq_end,
                    s_qt,
                    s_vt,
                    s_gt,
                    s_bt,
                    offs_k,
                    mk,
                    offs_v,
                    mv,
                ),
            )
        for j in tl.static_range(UNROLL):
            kt, qt, vt, gt, bt = steps[j]
            qt = qt * scale

            # decay, predict k, delta-rule correction, rank-1 update, then
            # read out with q *after* the update
            S = S * tl.exp(gt)
            if KV_LAYOUT:
                pred = tl.sum(S * kt[:, None], axis=0)
                c = bt * (vt - pred)
                S = S + kt[:, None] * c[None, :]
                o = tl.sum(S * qt[:, None], axis=0)
            else:
                pred = tl.sum(S * kt[None, :], axis=1)
                c = bt * (vt - pred)
                S = S + c[:, None] * kt[None, :]
                o = tl.sum(S * qt[None, :], axis=1)
            tl.store(
                o_base + (t0 + j) * s_ot + offs_v,
                o.to(o_ptr.dtype.element_ty),
                mask=mv & (t0 + j < seq_end),
            )

    if OUT_FINAL:
        tl.store(
            ho_ptr + n * s_fn + hv * s_fh + k_idx * s_fk + v_idx,
            S.to(ho_ptr.dtype.element_ty),
            mask=m_s,
        )


def _pow2(x):
    return max(16, triton.next_power_of_2(int(x)))


def _pick_layout(chains, v_dim):
    """(KV_LAYOUT, BLOCK_V) for the V split, measured on gfx1100."""
    if chains <= 16:
        return False, 2
    v = triton.next_power_of_2(max(1, chains * v_dim // _TARGET_PROGS))
    return True, max(4, min(8, v, triton.next_power_of_2(max(1, v_dim))))


def chunk_gated_delta_rule_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AssertionError("q, k and v must share a dtype")
    if q.dtype == torch.float32:
        raise AssertionError(
            "chunk_gated_delta_rule does not support float32; use bfloat16"
        )
    if beta.dim() != 3:
        raise AssertionError("beta must be of shape [B, T, HV]")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    batch, t_len, heads, k_dim = q.shape
    v_heads, v_dim = v.shape[2], v.shape[3]
    if k.shape != q.shape:
        raise ValueError("q and k must have the same shape")
    if v.shape[:2] != (batch, t_len) or heads == 0 or v_heads % heads != 0:
        raise ValueError(
            "v must be [B, T, HV, V] with HV a multiple of the "
            f"{heads} query/key heads; got {tuple(v.shape)}"
        )
    if g.shape != (batch, t_len, v_heads) or beta.shape != g.shape:
        raise ValueError(
            f"g and beta must be of shape {(batch, t_len, v_heads)}; got "
            f"{tuple(g.shape)} and {tuple(beta.shape)}"
        )

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError(
                "batch size is expected to be 1 when using cu_seqlens; "
                f"got {batch}"
            )
        n_seq = int(cu_seqlens.shape[0]) - 1
        has_cu = True
    else:
        n_seq = batch
        has_cu = False

    state_shape = (n_seq, v_heads, k_dim, v_dim)
    if initial_state is not None:
        if initial_state.shape[0] != n_seq:
            raise ValueError(
                "the number of initial states must equal the number of "
                f"sequences, i.e. {n_seq} rather than {initial_state.shape[0]}"
            )
        if tuple(initial_state.shape) != state_shape:
            raise ValueError(
                f"initial_state must be of shape {state_shape}; got "
                f"{tuple(initial_state.shape)}"
            )
        if initial_state.stride(-1) != 1:
            initial_state = initial_state.contiguous()

    o = torch.empty(
        (batch, t_len, v_heads, v_dim), dtype=q.dtype, device=q.device
    )
    if output_final_state:
        final_state = torch.empty(
            state_shape, dtype=torch.float32, device=q.device
        )
    else:
        final_state = None
    if n_seq * v_heads == 0:
        return o, final_state

    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()

    BLOCK_K = _pow2(k_dim)
    l2norm = bool(use_qk_l2norm_in_kernel)

    qk_q, qk_k = q, k
    if l2norm and t_len:
        rows = batch * t_len * heads
        rpb = max(1, 256 // BLOCK_K)
        prep_grid = (triton.cdiv(rows, rpb),)
        qn = torch.empty(
            (batch, t_len, heads, k_dim), dtype=torch.float32, device=q.device
        )
        kn = torch.empty(
            (batch, t_len, heads, k_dim), dtype=torch.float32, device=q.device
        )
        _qk_prep_kernel[prep_grid](
            q,
            qn,
            t_len * heads,
            heads,
            k_dim,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            qn.stride(0),
            qn.stride(1),
            qn.stride(2),
            rows,
            RPB=rpb,
            BLOCK_K=BLOCK_K,
            num_warps=1,
        )
        _qk_prep_kernel[prep_grid](
            k,
            kn,
            t_len * heads,
            heads,
            k_dim,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            kn.stride(0),
            kn.stride(1),
            kn.stride(2),
            rows,
            RPB=rpb,
            BLOCK_K=BLOCK_K,
            num_warps=1,
        )
        qk_q, qk_k = qn, kn
    elif k.stride() != q.stride():
        # the scan reads q and k with one set of strides
        qk_q, qk_k = q.contiguous(), k.contiguous()

    kv_layout, BLOCK_V = _pick_layout(n_seq * v_heads, v_dim)
    NV = (v_dim + BLOCK_V - 1) // BLOCK_V

    hi = initial_state if initial_state is not None else o
    ho = final_state if final_state is not None else o
    s_i = initial_state.stride()[:3] if initial_state is not None else (0,) * 3
    s_f = final_state.stride()[:3] if final_state is not None else (0,) * 3

    _gdn_fwd_kernel[(n_seq * v_heads, NV)](
        qk_q,
        qk_k,
        v,
        g,
        beta,
        hi,
        ho,
        cu_seqlens if cu_seqlens is not None else o,
        o,
        t_len,
        v_heads,
        v_heads // heads,
        k_dim,
        v_dim,
        scale,
        qk_q.stride(0),
        qk_q.stride(1),
        qk_q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        beta.stride(0),
        beta.stride(1),
        beta.stride(2),
        *s_i,
        *s_f,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        HAS_INIT=initial_state is not None,
        IS_VARLEN=has_cu,
        OUT_FINAL=output_final_state,
        KV_LAYOUT=kv_layout,
        BLOCK_K=BLOCK_K,
        BLOCK_V=BLOCK_V,
        UNROLL=8,
        num_warps=1,
        # On gfx1100 (Triton 3.6) the software-pipelined loop gave wrong
        # outputs on the packed path with BLOCK_V=8; UNROLL issues the loads
        # ahead instead.
        num_stages=1,
    )

    return o, final_state


__all__ = ["chunk_gated_delta_rule_fwd"]
