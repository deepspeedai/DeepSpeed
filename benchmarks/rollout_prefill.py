# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Calibrate prefill costs separately from a small, reproducible route sweep."""

import argparse
from dataclasses import asdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from deepspeed.runtime.rollout.base import RolloutRequest, SamplingConfig
from deepspeed.runtime.rollout.hybrid_engine_rollout import HybridEngineRollout, HybridEngineRolloutConfig
import deepspeed.runtime.rollout.hybrid_engine_rollout as rollout_backend
from deepspeed.accelerator import get_accelerator


def nonnegative_fit(matrix, targets):
    """Solve three-parameter NNLS by checking its eight possible active sets."""
    matrix = np.asarray(matrix, dtype=float)
    targets = np.asarray(targets, dtype=float)
    scale = np.maximum(np.abs(matrix).max(axis=0), 1e-12)
    scaled = matrix / scale
    best = np.zeros(matrix.shape[1])
    error = float(targets @ targets)
    for count in range(1, matrix.shape[1] + 1):
        for columns in combinations(range(matrix.shape[1]), count):
            values = np.linalg.lstsq(scaled[:, columns], targets, rcond=None)[0]
            if np.any(values < 0):
                continue
            candidate = np.zeros(matrix.shape[1])
            candidate[list(columns)] = values
            residual = scaled @ candidate - targets
            if residual @ residual < error:
                best, error = candidate, float(residual @ residual)
    return (best / scale).tolist()


def make_request(lengths, token, pad, device):
    width = max(lengths)
    ids = torch.full((len(lengths), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        ids[row, -length:] = token
        mask[row, -length:] = 1
    return RolloutRequest(ids, mask)


def measure(rollout, request, steps, continuous, repeats):
    sampling = SamplingConfig(max_new_tokens=steps,
                              temperature=0,
                              continuous_batch_size=request.prompt_ids.shape[0] if continuous else None)
    samples = []
    for iteration in range(-1, repeats):
        get_accelerator().synchronize()
        start = time.perf_counter()
        output = rollout.generate(request, sampling)
        get_accelerator().synchronize()
        if iteration >= 0:
            samples.append((time.perf_counter() - start) * 1000)
    return samples, output


def calibrate(engine, tokenizer, token, pad, device, repeats):
    model = engine.module
    config = model.config.get_text_config()
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    attention_scale = 4 * config.num_hidden_layers * config.num_attention_heads * head_dim / 1e9
    kv_scale = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * next(
        model.parameters()).element_size() / 2**20
    matrix, times, copy_rates, records = [], [], [], []
    # These calibration shapes differ from every held-out sweep workload below.
    shapes = [(1, 32), (1, 256), (4, 64), (4, 384), (8, 768), (16, 96)]
    for batch_size, width in shapes:
        request = make_request([width] * batch_size, token, pad, device)
        cfg = HybridEngineRolloutConfig(adaptive_prefill=True,
                                        enable_profiling=True,
                                        continuous_cache_capacity=width + 1,
                                        prefill_fixed_cost_ms=1e9,
                                        prefill_token_cost_ms=0,
                                        prefill_attention_cost_ms=0,
                                        prefill_kv_cost_ms=0)
        rollout = HybridEngineRollout(engine, tokenizer, cfg)
        profiles = []
        for iteration in range(-1, repeats):
            rollout.generate(request, SamplingConfig(max_new_tokens=1, temperature=0,
                                                     continuous_batch_size=batch_size))
            if iteration >= 0:
                profiles.append(rollout.get_last_profile())
        prefill = statistics.mean(p['prefill_forward_ms'] for p in profiles)
        copying = statistics.mean(p['cache_management_overhead_ms'] for p in profiles)
        assert all(p['num_prefill_forwards'] == 1 for p in profiles)
        matrix.append([1, batch_size * width, attention_scale * batch_size * width**2])
        times.append(prefill)
        copy_rates.append(copying / (kv_scale * batch_size * width))
        records.append({
            'batch_size': batch_size,
            'width': width,
            'prefill_ms': prefill,
            'cache_management_ms': copying
        })
        print('CALIBRATE', batch_size, width, round(prefill, 3), 'ms', flush=True)
    fixed, linear, attention = nonnegative_fit(matrix, times)
    # Differences across decode budgets cancel most prefill and allocation costs.
    request = make_request([80] * 16, token, pad, device)
    differences = {}
    for steps in (1, 8):
        cb_cfg = HybridEngineRolloutConfig(adaptive_prefill=True,
                                           continuous_cache_capacity=80 + steps,
                                           prefill_fixed_cost_ms=1e9,
                                           prefill_token_cost_ms=0,
                                           prefill_attention_cost_ms=0,
                                           prefill_kv_cost_ms=0)
        cb = HybridEngineRollout(engine, tokenizer, cb_cfg)
        ordinary = HybridEngineRollout(engine, tokenizer)
        cb_times, _ = measure(cb, request, steps, True, repeats)
        ordinary_times, _ = measure(ordinary, request, steps, False, repeats)
        differences[steps] = statistics.mean(cb_times) - statistics.mean(ordinary_times)
    costs = {
        'prefill_fixed_cost_ms': fixed,
        'prefill_token_cost_ms': linear,
        'prefill_attention_cost_ms': attention,
        'prefill_kv_cost_ms': statistics.median(copy_rates),
        'continuous_decode_cost_ms': max(0, (differences[8] - differences[1]) / 7)
    }
    return costs, {
        'prefill_shapes': records,
        'decode_cb_minus_no_cb_ms': differences,
        'note': 'KV rate includes cache management overhead; coefficients are configuration-specific estimates.'
    }


def sweep(engine, tokenizer, token, pad, device, steps, repeats, costs):
    workloads = {
        'short128': [16] * 128,
        'moderate_tail': [128] + [4] * 31,
        'spread': [16 * (i + 1) for i in range(32)],
        'long_tail': [512] + [16] * 31
    }
    report = {}
    modes = ['no_cb', 'equal_width', 'aligned', 'auto']
    for name, lengths in workloads.items():
        request = make_request(lengths, token, pad, device)
        cfgs = {
            mode:
            HybridEngineRolloutConfig(adaptive_prefill=mode == 'auto',
                                      align_decode_fronts=mode == 'aligned',
                                      **(costs if mode == 'auto' else {}))
            for mode in modes
        }
        rollouts = {mode: HybridEngineRollout(engine, tokenizer, cfg) for mode, cfg in cfgs.items()}
        samples = {mode: [] for mode in modes}
        outputs = {}
        for iteration in range(-1, repeats):
            shift = iteration % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                rollout = rollouts[mode]
                sampling = SamplingConfig(max_new_tokens=steps,
                                          temperature=0,
                                          continuous_batch_size=None if mode == 'no_cb' else len(lengths))
                get_accelerator().synchronize()
                start = time.perf_counter()
                output = rollout.generate(request, sampling)
                get_accelerator().synchronize()
                if iteration >= 0:
                    samples[mode].append((time.perf_counter() - start) * 1000)
                outputs[mode] = output.input_ids[:, max(lengths):].cpu().tolist()
        means = {mode: statistics.mean(values) for mode, values in samples.items()}
        best = min(modes[:-1], key=means.get)
        stats = rollouts['auto'].get_last_continuous_stats()
        report[name] = {
            'lengths': lengths,
            'max_new_tokens': steps,
            'mean_ms': means,
            'raw_ms': samples,
            'best_baseline': best,
            'auto_vs_best': means['auto'] / means[best],
            'auto_strategy': 'batched' if stats is None else 'bucketed',
            'output_ids_match': all(value == outputs['no_cb'] for value in outputs.values()),
            'output_match_by_path': {
                mode: value == outputs['no_cb']
                for mode, value in outputs.items()
            },
            'outputs': outputs
        }
        rollouts['auto'].enable_profiling = True
        rollouts['auto'].generate(
            request, SamplingConfig(max_new_tokens=steps, temperature=0, continuous_batch_size=len(lengths)))
        report[name]['auto_profile'] = rollouts['auto'].get_last_profile()
        print('SWEEP',
              name,
              means,
              'auto/best',
              round(report[name]['auto_vs_best'], 4),
              report[name]['auto_strategy'],
              flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='tiny-qwen2')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--new-tokens', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--calibrate',
                        type=Path,
                        help='Write coefficients using separate calibration shapes, then exit.')
    parser.add_argument('--cost-config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('prefill-sweep.json'))
    args = parser.parse_args()
    if args.repeats <= 0 or args.new_tokens <= 0:
        parser.error('repeats and new-tokens must be positive')
    torch.manual_seed(8497)
    torch.set_num_threads(4)
    device = get_accelerator().device_name()
    dtype = getattr(torch, args.dtype)
    if args.model == 'tiny-qwen2':
        config = Qwen2Config(vocab_size=128,
                             hidden_size=64,
                             intermediate_size=128,
                             num_hidden_layers=2,
                             num_attention_heads=4,
                             num_key_value_heads=2,
                             max_position_embeddings=4096,
                             pad_token_id=0,
                             eos_token_id=None,
                             attn_implementation='sdpa')
        model = Qwen2ForCausalLM(config).to(device=device, dtype=dtype).eval()
        token, pad = 3, 0
    else:
        native_tokenizer = AutoTokenizer.from_pretrained(args.model)
        token = native_tokenizer.encode('hello', add_special_tokens=False)[0]
        pad = native_tokenizer.pad_token_id
        if pad is None:
            pad = native_tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                                     attn_implementation='sdpa').to(device).eval()
    tokenizer = SimpleNamespace(pad_token_id=pad, eos_token_id=None)
    engine = SimpleNamespace(module=model)
    source = Path(rollout_backend.__file__)
    metadata = {
        'model': args.model,
        'model_config': json.loads(json.dumps(model.config.to_dict())),
        'dtype': args.dtype,
        'gpu': torch.cuda.get_device_name(0),  #ignore-cuda
        'sm_count': torch.cuda.get_device_properties(0).multi_processor_count,  #ignore-cuda
        'torch': torch.__version__,
        'transformers': __import__('transformers').__version__,
        'backend': 'sdpa',
        'seed': 8497,
        'cpu_threads': 4,
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'scope':
        'Controlled EOS-disabled generation; includes routing and cache work; excludes loading and tokenization.'
    }
    with torch.inference_mode():
        if args.calibrate:
            costs, measurements = calibrate(engine, tokenizer, token, pad, device, args.repeats)
            args.calibrate.parent.mkdir(parents=True, exist_ok=True)
            args.calibrate.write_text(
                json.dumps({
                    'metadata': metadata,
                    'costs': costs,
                    'calibration': measurements
                }, indent=2) + '\n')
            print('COSTS', json.dumps(costs), flush=True)
            return
        calibration = json.loads(args.cost_config.read_text()) if args.cost_config else {}
        if 'metadata' in calibration:
            for key in ('model', 'model_config', 'dtype', 'gpu', 'sm_count', 'backend', 'torch', 'transformers',
                        'allow_tf32'):
                if calibration['metadata'][key] != metadata[key]:
                    parser.error(f'calibration differs in {key}; recalibrate for this configuration')
        costs = calibration.get('costs', {})
        # A supplied JSON uses the same validated public configuration fields as an application.
        cfg = HybridEngineRolloutConfig(adaptive_prefill=True, **costs)
        report = {
            'metadata': metadata,
            'auto_config': asdict(cfg),
            'repeats': args.repeats,
            'workloads': sweep(engine, tokenizer, token, pad, device, args.new_tokens, args.repeats, costs)
        }
    worst = max(report['workloads'].items(), key=lambda pair: pair[1]['auto_vs_best'])
    report['worst_case'] = {
        'workload': worst[0],
        'auto_vs_best': worst[1]['auto_vs_best'],
        'regression_percent': (worst[1]['auto_vs_best'] - 1) * 100
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('WORST', report['worst_case'], flush=True)


if __name__ == '__main__':
    main()
