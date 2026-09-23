# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""The aspect-ratio scale has to survive `muon_update` being recompiled for a new shape.

The first call compiles for static sizes. The next shape makes dynamo recompile with symbolic
ones, and inductor then cast both sides of `max(1, rows / cols)` to int64, so a 32 x 12 update
got sqrt(2) instead of sqrt(32 / 12). On CPU that hit every branch. On CUDA it hit the expert and
per-head branches with the standard iteration, where the scale is computed before a graph break.

Each case runs an integer ratio first to force that recompile, then compares against the same
function run eagerly.
"""

import pytest
import torch

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.muon.original_muon import muon_update

# The first shape has an integer aspect ratio, so the statically compiled call is right either
# way. The next two are only right if the ratio is not truncated, and the last one is wide.
CASES = {
    "expert": ([(2, 64, 32), (2, 32, 12), (2, 80, 24), (2, 24, 80)], dict(is_expert_group=True)),
    "per_head": ([(192, 32), (96, 12), (240, 24), (96, 256)], dict(num_heads=3)),
}

DEVICES = ["cpu"]
if get_accelerator().device_name() != "cpu":
    DEVICES.append(get_accelerator().device_name())


@pytest.fixture(autouse=True)
def _fresh_compile_cache():
    # Earlier tests may have compiled `muon_update` already, or used up its recompiles, and then
    # these calls fall back to eager and pass without testing anything.
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ns_method", ["gram", "standard"])
@pytest.mark.parametrize("branch", CASES)
def test_scale_matches_eager_after_recompiling(branch, ns_method, device):
    shapes, kwargs = CASES[branch]
    eager = muon_update._torchdynamo_orig_callable
    for shape in shapes:
        torch.manual_seed(0)
        grad = torch.randn(*shape, device=device)
        compiled = muon_update(grad.clone(), torch.zeros_like(grad), ns_method=ns_method, **kwargs)
        expected = eager(grad.clone(), torch.zeros_like(grad), ns_method=ns_method, **kwargs)
        # A truncated scale is off by 13% for 32 x 12 and 5% for 80 x 24. Newton-Schulz rounding,
        # compiled against eager, stays well under 1%.
        ratio = (compiled.norm() / expected.norm()).item()
        assert abs(ratio - 1) < 1e-2, f"{branch} {shape}: compiled/eager norm ratio {ratio:.4f}"
