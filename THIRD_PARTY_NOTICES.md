# Third-Party Notices

This file records third-party source notices for code incorporated into
DeepSpeed source files.

## TorchTitan

The following files contain portions derived from TorchTitan:

- `deepspeed/module_inject/auto_ep_layer.py`
- `deepspeed/moe/ep_experts.py`
- `deepspeed/moe/ep_kernels.py`
- `deepspeed/moe/ep_router.py`

Source project: https://github.com/pytorch/torchtitan

TorchTitan is licensed under the BSD 3-Clause License:

```text
BSD 3-Clause License

(c) Meta Platforms, Inc. and affiliates.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## TileLang

The following files contain portions derived from the TileLang DeepSeek-V3.2 examples, taken through the GLM-5
plugin of THUDM/slime (Apache-2.0, commit `4c193f1`, `slime_plugins/models/glm5/ops/`):

| DeepSpeed file | TileLang source (revision) | slime file |
|---|---|---|
| `deepspeed/models/glm_moe_dsa/kernels/indexer_fwd.py` | `examples/deepseek_v32/fp8_lighting_indexer.py` (`4956b5835fa554af6c03d4a6289cad44bf310869`) | `tilelang_indexer_fwd.py` |
| `deepspeed/models/glm_moe_dsa/kernels/sparse_mla_fwd.py` | `examples/deepseek_v32/sparse_mla_fwd.py` (`e666d2d3cc483829c57618c9ebf2e4f4ada0819d`) | `tilelang_sparse_mla_fwd.py` |
| `deepspeed/models/glm_moe_dsa/kernels/sparse_mla_bwd.py` | `examples/deepseek_v32/sparse_mla_bwd.py` (`4ff81c7d40803d269569e157e847623e84553f78`) | `tilelang_sparse_mla_bwd.py` |

Source projects: https://github.com/tile-ai/tilelang and https://github.com/THUDM/slime

TileLang is licensed under the MIT License (the same text at all three revisions):

```text
MIT License

Copyright (c) Tile-AI.
**During the period from December 1, 2024, to Mar 14, 2025, this project is
subject to additional collaboration terms with Microsoft Corporation.**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```
