# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression test for https://github.com/deepspeedai/DeepSpeed/issues/7942

When torch._dynamo skips a frame (e.g. because of a graph break inside a
for/while loop), the frame runs in eager mode.  DeepCompile removes the
ZeRO-3 parameter-gathering hooks, so parameters accessed in the skipped
frame remain partitioned (shape ``[0]``).  For an embedding layer this
causes ``RuntimeError: 'weight' must be 2-D``.

This test creates a model whose forward contains an embedding lookup
followed by a loop with a deliberate graph break, reproducing the pattern
that triggers the bug.
"""

import argparse
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed import comm

torch._dynamo.config.cache_size_limit = 100


class SkippedFrameModel(torch.nn.Module):
    """Model that triggers a dynamo frame skip.

    ``forward`` contains an embedding lookup followed by a loop whose body
    calls ``print`` (an opaque side-effect), which causes a graph break
    inside the loop.  Dynamo skips the entire frame, so the embedding lookup
    runs in eager mode with ZeRO-3 partitioned weights.
    """

    def __init__(self, vocab_size=128, hidden=64, n_layers=2):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
        self.head = torch.nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
            # graph break inside a loop body — dynamo skips the entire frame
            if torch.compiler.is_compiling():
                torch._dynamo.graph_break()
        return self.head(h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config_z3.json")
    args = parser.parse_args()

    model = SkippedFrameModel()
    engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())
    engine.compile()

    device = get_accelerator().current_device_name()
    input_ids = torch.randint(0, 128, (1, 16), device=device)

    for step in range(3):
        loss = engine(input_ids).sum()
        engine.backward(loss)
        engine.step()
        if comm.get_rank() == 0:
            print(f"step={step} loss={loss.item():.4f}")

    if comm.get_rank() == 0:
        print("PASS")


if __name__ == "__main__":
    main()
