# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Regression test for skipped eager frames under DeepCompile ZeRO-3."""

import argparse
import logging

import torch

import deepspeed
from deepspeed import comm
from deepspeed.accelerator import get_accelerator

torch._dynamo.config.cache_size_limit = 100


def configure_dynamo_logging():
    try:
        import torch._logging

        torch._logging.set_logs(dynamo=logging.INFO, graph_breaks=True)
        torch._dynamo.config.verbose = True
    except Exception:
        pass


def dynamo_counter_text():
    counters = torch._dynamo.utils.counters
    return repr({str(key): dict(value) for key, value in counters.items()})


class SkippedFrameModel(torch.nn.Module):

    def __init__(self, vocab_size=384, hidden=384, n_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
        self.head = torch.nn.Linear(hidden, vocab_size, bias=False)

    @torch.compiler.disable
    def _compiler_disabled_forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
            h = torch.relu(h)
        return self.head(h)

    def forward(self, input_ids):
        return self._compiler_disabled_forward(input_ids)


def main():
    configure_dynamo_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default="ds_config_z3_deepcompile_no_persist.json")
    args = parser.parse_args()

    model = SkippedFrameModel()
    assert all(p.numel() > 100000 for p in model.parameters())

    engine, _, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters())
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    engine.compile()

    device = get_accelerator().current_device_name()
    input_ids = torch.randint(0, model.vocab_size, (1, 16), device=device)

    for step in range(3):
        loss = engine(input_ids).sum()
        engine.backward(loss)
        engine.step()
        if comm.get_rank() == 0:
            print(f"step={step} loss={loss.item():.4f}")

    counters = dynamo_counter_text()
    fallback = getattr(engine, "_deepcompile_z3_eager_fallback", None)
    fallback_stats = fallback.stats() if fallback is not None else {}
    if comm.get_rank() == 0:
        print(f"dynamo_counters={counters}")
        print(f"fallback_stats={fallback_stats}")
    assert "compiler.disable" in counters or "Skip inlining" in counters
    assert fallback_stats.get("total_gathered_params", 0) > 0

    if comm.get_rank() == 0:
        print("PASS")


if __name__ == "__main__":
    main()
