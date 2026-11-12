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

import torch


def reference(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    # Qwen3.5 / Qwen3.6 store the projections as contiguous blocks,
    # [q | k | v | z] and [b | a], instead of Qwen3-Next's per-key-head
    # interleaving; the outputs match fused_qkvzba_split_reshape_cat's.
    b = mixed_qkvz.shape[0]
    qk_dim = num_heads_qk * head_qk
    v_dim = num_heads_v * head_v
    q, k, v, z = torch.split(
        mixed_qkvz, [qk_dim, qk_dim, v_dim, v_dim], dim=-1
    )
    b_out, a_out = torch.split(mixed_ba, [num_heads_v, num_heads_v], dim=-1)
    return (
        torch.cat([q, k, v], dim=-1).contiguous(),
        z.reshape(b, num_heads_v, head_v).contiguous(),
        b_out.contiguous(),
        a_out.contiguous(),
    )
