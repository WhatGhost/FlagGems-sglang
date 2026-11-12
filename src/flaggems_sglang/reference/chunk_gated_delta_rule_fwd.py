# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference for the chunked gated delta-rule forward (vLLM-Ascend chunk.py).

Chunking is only a scheduling decision: the chunked and fused-recurrent forms
compute the same gated delta-rule recurrence, so the reference is the plain
sequential recurrence. For each head ``h`` and step ``t``::

    decay  = exp(g_t)
    pred   = decay * (S . k_t)              # S is [V, K]
    vcorr  = (v_t - pred) * beta_t
    S      = S * decay + outer(vcorr, k_t)
    o_t    = S . (q_t * scale)

Shapes: ``q, k`` are ``[B, T, H, K]``; ``v`` is ``[B, T, HV, V]``; ``g`` and
``beta`` are ``[B, T, HV]``, where HV is a multiple of H and value head ``hv``
reads query/key head ``hv // (HV // H)`` (HV == H is the plain case).
``scale`` defaults to ``K ** -0.5``. With ``use_qk_l2norm_in_kernel`` the
per-step ``q``/``k`` rows are L2-normalized before the ``scale`` multiply.
With ``cu_seqlens`` (B == 1) the T axis packs N sequences, each starting from
its own ``initial_state[n]``; states are ``[N, HV, K, V]``. Returns
``(output, final_state)`` where ``final_state`` is ``None`` unless
``output_final_state`` is set.
"""

import torch


def reference(
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
    B, T, H, K = q.shape
    HV, V = v.shape[-2], v.shape[-1]
    group = HV // H
    if scale is None:
        scale = K**-0.5

    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = g.float()
    bf = beta.float()
    if use_qk_l2norm_in_kernel:
        qf = qf / qf.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()
        kf = kf / kf.pow(2).sum(-1, keepdim=True).add(1e-6).sqrt()

    if cu_seqlens is None:
        sequences = [(b, 0, T) for b in range(B)]
    else:
        bounds = cu_seqlens.tolist()
        sequences = [(0, s, e) for s, e in zip(bounds[:-1], bounds[1:])]
    N = len(sequences)

    out = torch.zeros(B, T, HV, V, device=q.device, dtype=torch.float32)
    if initial_state is None:
        state = torch.zeros(N, HV, K, V, device=q.device, dtype=torch.float32)
    else:
        state = initial_state.clone().float()

    for n, (b, start, end) in enumerate(sequences):
        for hv in range(HV):
            h = hv // group
            # S kept as [V, K] internally; initial_state stores [K, V].
            S = state[n, hv].transpose(0, 1).contiguous()
            for t in range(start, end):
                kt = kf[b, t, h]
                qt = qf[b, t, h] * scale
                vt = vf[b, t, hv]
                decay = torch.exp(gf[b, t, hv])
                pred = decay * (S * kt[None, :]).sum(-1)
                vcorr = (vt - pred) * bf[b, t, hv]
                S = S * decay + vcorr[:, None] * kt[None, :]
                out[b, t, hv] = (S * qt[None, :]).sum(-1)
            state[n, hv] = S.transpose(0, 1).contiguous()

    final_state = state if output_final_state else None
    return out.to(v.dtype), final_state
