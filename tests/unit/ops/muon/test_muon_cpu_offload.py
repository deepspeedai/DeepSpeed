# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import deepspeed
import torch
import pytest

from unit.common import DistributedTest
from unit.simple_model import SimpleModel
from deepspeed.accelerator import get_accelerator

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported", allow_module_level=True)


@pytest.mark.parametrize('zero_stage', [2])
class TestMuonCPUOffload(DistributedTest):

    def test_momentum_buffer_on_cpu(self, zero_stage):
        """Verify Muon CPU offload creates momentum buffer on CPU.

        This is the key invariant: after a training step with CPU offload,
        the Muon momentum buffer must reside on CPU (not GPU), confirming
        that muon_update ran on CPU and no GPU memory is wasted.
        """
        hidden_dim = 32
        batch_size = 8
        config_dict = {
            "train_batch_size": batch_size,
            "optimizer": {
                "type": "muon",
                "params": {
                    "lr": 0.01
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "reduce_scatter": False,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
            },
        }

        model = SimpleModel(hidden_dim=hidden_dim, nlayers=5)
        engine, optimizer, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
            dist_init_required=False,
        )

        x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
        y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
        loss = engine(x, y)
        engine.backward(loss)
        engine.step()

        # Muon momentum buffer must exist and be on CPU.
        # If muon_update was silently skipped, momentum_buffer would not be created.
        flatten_copy = optimizer.optimizer.param_groups[0]['params'][0]
        state = optimizer.optimizer.state[flatten_copy]
        assert 'momentum_buffer' in state, ("momentum_buffer not found in optimizer state. "
                                            "muon_update was not called in the CPU offload path.")
        assert state['momentum_buffer'].device.type == 'cpu', (
            f"Momentum buffer is on {state['momentum_buffer'].device}, expected CPU")


@pytest.mark.parametrize('zero_stage', [2])
class TestMuonCPUOffloadCosim(DistributedTest):

    def test_cosim_offload_vs_no_offload(self, zero_stage):
        """Verify CPU offload produces results consistent with GPU path.

        With the same random seed, offload and non-offload should produce
        close parameters. If muon_update is skipped or wrong in either path,
        the results diverge significantly.
        """
        hidden_dim = 32
        batch_size = 8

        def train(offload):
            torch.manual_seed(42)
            config_dict = {
                "train_batch_size": batch_size,
                "optimizer": {
                    "type": "muon",
                    "params": {
                        "lr": 0.01
                    }
                },
                "fp16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": zero_stage,
                    "reduce_scatter": False,
                },
            }
            if offload:
                config_dict["zero_optimization"]["offload_optimizer"] = {
                    "device": "cpu",
                    "pin_memory": True,
                }

            model = SimpleModel(hidden_dim=hidden_dim, nlayers=5)
            engine, _, _, _ = deepspeed.initialize(
                config=config_dict,
                model=model,
                model_parameters=model.parameters(),
                dist_init_required=False,
            )

            for _ in range(3):
                x = torch.randn(batch_size, hidden_dim, device=engine.device, dtype=torch.half)
                y = torch.randint(0, hidden_dim, (batch_size, ), device=engine.device)
                loss = engine(x, y)
                engine.backward(loss)
                engine.step()

            return {n: p.clone().detach().float().cpu() for n, p in model.named_parameters()}

        params_offload = train(offload=True)
        params_no_offload = train(offload=False)

        for name in params_offload:
            p_off = params_offload[name]
            p_no = params_no_offload[name]
            # Both paths should produce the same NaN pattern
            nan_mask = p_off.isnan() | p_no.isnan()
            assert nan_mask.equal(p_off.isnan()), (f"{name}: NaN pattern differs between offload and non-offload. "
                                                   "muon_update produced different results.")
            # On non-NaN elements, cosine similarity should be very high
            valid = ~nan_mask
            if valid.sum() > 0:
                cos_sim = torch.nn.functional.cosine_similarity(p_off[valid].unsqueeze(0),
                                                                p_no[valid].unsqueeze(0)).item()
                assert cos_sim > 0.99, (f"{name}: cosine similarity {cos_sim:.4f} between offload and "
                                        f"non-offload is too low, indicating muon_update results diverge.")
