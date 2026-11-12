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


def reference(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_pad,
    cumsum_buffer,
    pad_sorted_token_ids,
):
    # ``num_experts`` counts the trailing "filtered expert" slot, which never
    # receives tokens and therefore gets no blocks; only 0..num_experts-2 do.
    # As in sgl_kernel, the results are written into the three output buffers
    # (blocks past the padded count keep their expert ids), which are returned.
    num_routed = num_experts - 1
    flat = topk_ids.flatten()
    numel = flat.numel()
    sorted_token_ids.fill_(numel)

    offset = 0
    for e in range(num_routed):
        idx = (flat == e).nonzero().flatten()
        n = idx.numel()
        aligned = ((n + block_size - 1) // block_size) * block_size
        if n:
            sorted_token_ids[offset : offset + n] = idx.to(
                sorted_token_ids.dtype
            )
        nblocks = aligned // block_size
        if nblocks:
            beg = offset // block_size
            expert_ids[beg : beg + nblocks] = e
        offset += aligned

    num_tokens_post_pad.fill_(offset)
    return sorted_token_ids, expert_ids, num_tokens_post_pad
