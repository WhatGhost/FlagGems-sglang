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

"""Correctness test for attention/fused_qkvzba_split_reshape_cat_contiguous."""

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _check(actual, expected):
    # Pure data movement: every output is an exact copy.
    for a, e in zip(actual, expected):
        assert a.shape == e.shape and a.dtype == e.dtype
        assert torch.equal(a, e)


def _case(batch, nqk, nv, head_qk, head_v, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=cfg.device).manual_seed(seed)
    qkvz_dim = 2 * nqk * head_qk + 2 * nv * head_v
    mixed_qkvz = torch.randn(
        batch, qkvz_dim, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    mixed_ba = torch.randn(
        batch, 2 * nv, dtype=torch.float32, device=cfg.device, generator=g
    ).to(dtype)
    return dict(
        mixed_qkvz=mixed_qkvz,
        mixed_ba=mixed_ba,
        num_heads_qk=nqk,
        num_heads_v=nv,
        head_qk=head_qk,
        head_v=head_v,
        check=_check,
    )


# Per TP=4 rank, Qwen3.6-27B has 4 key heads and 12 value heads (ratio 3)
# and Qwen3.6-35B-A3B has 4 and 8.
_CASE_FACTORIES = [
    lambda: _case(1, 4, 12, 128, 128),
    lambda: _case(37, 4, 8, 128, 128, seed=1),
    lambda: _case(64, 16, 32, 128, 128, seed=2),
    lambda: _case(333, 4, 4, 128, 64, seed=3),
    lambda: _case(5, 2, 6, 64, 96, dtype=torch.float16, seed=4),
]

CORRECTNESS_CASES = list(range(len(_CASE_FACTORIES)))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_idx", range(len(CORRECTNESS_CASES)))
@pytest.mark.fused_qkvzba_split_reshape_cat_contiguous
def test_fused_qkvzba_split_reshape_cat_contiguous(case_idx):
    # Build only this case; a module-level list would allocate
    # every case's tensors at import time.
    case = _CASE_FACTORIES[case_idx]()
    reference = get_reference("fused_qkvzba_split_reshape_cat_contiguous")
    check = (
        case.pop("check", None)
        if isinstance(case, dict) and "check" in case
        else None
    )
    kwargs = case if isinstance(case, dict) else {}

    # Reference
    expected = reference(**kwargs)

    # Operator under test
    try:
        candidate = get_candidate(
            "fused_qkvzba_split_reshape_cat_contiguous", reference=reference
        )
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "attention/fused_qkvzba_split_reshape_cat_contiguous ops module "
            "not found"
        )
        return

    try:
        actual = candidate(**kwargs)
    except NotImplementedError:
        pytest.skip(
            "attention/fused_qkvzba_split_reshape_cat_contiguous not yet "
            "implemented"
        )
        return

    # Compare
    if check is not None:
        check(actual, expected)
    elif isinstance(expected, (tuple, list)):
        for a, e in zip(actual, expected):
            assert torch.equal(a, e)
    # Restore check for reuse
    if check is not None:
        case["check"] = check
