# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import torch
import triton
import triton.language as tl

_BLOCK = 8192


@triton.jit
def _qkvzba_contiguous_kernel(
    qkvz_ptr,
    ba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    s_qkvz,
    s_ba,
    QKV: tl.constexpr,
    Z: tl.constexpr,
    NV: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_BA: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0)
    pid = tl.program_id(1)
    if WIDE:
        row = row.to(tl.int64)

    # No reordering: [q | k | v] is a prefix of each [q | k | v | z] row.
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(qkvz_ptr + row * s_qkvz + cols, mask=cols < QKV + Z)
    tl.store(qkv_ptr + row * QKV + cols, x, mask=cols < QKV)
    tl.store(
        z_ptr + row * Z + (cols - QKV),
        x,
        mask=(cols >= QKV) & (cols < QKV + Z),
    )

    if pid == 0:
        j = tl.arange(0, BLOCK_BA)
        y = tl.load(ba_ptr + row * s_ba + j, mask=j < 2 * NV)
        tl.store(b_ptr + row * NV + j, y, mask=j < NV)
        tl.store(a_ptr + row * NV + (j - NV), y, mask=(j >= NV) & (j < 2 * NV))


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    num_heads_qk = int(num_heads_qk)
    num_heads_v = int(num_heads_v)
    head_qk = int(head_qk)
    head_v = int(head_v)
    b = mixed_qkvz.shape[0]
    qkv_dim = 2 * num_heads_qk * head_qk + num_heads_v * head_v
    z_dim = num_heads_v * head_v
    if mixed_qkvz.shape[-1] != qkv_dim + z_dim:
        raise ValueError(
            f"mixed_qkvz must have {qkv_dim + z_dim} columns, "
            f"got {mixed_qkvz.shape[-1]}"
        )
    if mixed_ba.shape[-1] != 2 * num_heads_v:
        raise ValueError(
            f"mixed_ba must have {2 * num_heads_v} columns, "
            f"got {mixed_ba.shape[-1]}"
        )

    mixed_qkv = torch.empty(
        (b, qkv_dim), dtype=mixed_qkvz.dtype, device=mixed_qkvz.device
    )
    z = torch.empty(
        (b, num_heads_v, head_v),
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b_out = torch.empty(
        (b, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device
    )
    a_out = torch.empty(
        (b, num_heads_v), dtype=mixed_ba.dtype, device=mixed_ba.device
    )
    if b == 0:
        return mixed_qkv, z, b_out, a_out

    if mixed_qkvz.stride(-1) != 1:
        mixed_qkvz = mixed_qkvz.contiguous()
    if mixed_ba.stride(-1) != 1:
        mixed_ba = mixed_ba.contiguous()
    wide = (
        b * mixed_qkvz.stride(0) >= (1 << 31)
        or b * mixed_ba.stride(0) >= (1 << 31)
        or mixed_qkv.numel() >= (1 << 31)
        or z.numel() >= (1 << 31)
    )
    block = min(_BLOCK, triton.next_power_of_2(qkv_dim + z_dim))
    _qkvzba_contiguous_kernel[(b, triton.cdiv(qkv_dim + z_dim, block))](
        mixed_qkvz,
        mixed_ba,
        mixed_qkv,
        z,
        b_out,
        a_out,
        mixed_qkvz.stride(0),
        mixed_ba.stride(0),
        QKV=qkv_dim,
        Z=z_dim,
        NV=num_heads_v,
        BLOCK=block,
        BLOCK_BA=triton.next_power_of_2(2 * num_heads_v),
        WIDE=wide,
        num_warps=16,
    )
    return mixed_qkv, z, b_out, a_out


__all__ = ["fused_qkvzba_split_reshape_cat_contiguous"]
