# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch
import deepspeed.comm as dist
import deepspeed
import pytest
from deepspeed.ops.adam import FusedAdam
from unit.common import DistributedTest
from unit.simple_model import SimpleModel, SimpleOptimizer, random_dataloader, SimpleMoEModel, sequence_dataloader
from deepspeed.utils.torch import required_torch_version
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.op_builder import CPUAdamBuilder, FusedLambBuilder
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

if torch.half not in get_accelerator().supported_dtypes():
    pytest.skip(f"fp16 not supported, valid dtype: {get_accelerator().supported_dtypes()}", allow_module_level=True)


class TestLambFP32GradClip(DistributedTest):
    world_size = 2

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestLambFP16(DistributedTest):
    world_size = 2

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test__basic(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test_empty_grad(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=True)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestAdamFP32EmptyGrad(DistributedTest):
    world_size = 2

    def test(self):
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": False
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim, empty_grad=True)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestAdamwFP16Basic(DistributedTest):
    world_size = 1

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {"train_batch_size": 1, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestMixedPrecisionFusedAdam(DistributedTest):
    world_size = 1

    @staticmethod
    def _initialize_engine(model, dtype, adam_w_mode=True, clip_grad=0.0):
        optimizer = FusedAdam(model.parameters(), lr=2e-3, weight_decay=0.01, adam_w_mode=adam_w_mode)
        precision_config = {"enabled": True}
        if dtype == torch.float16:
            precision_config["loss_scale"] = 128.0
        config = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "gradient_clipping": clip_grad,
            "fp16" if dtype == torch.float16 else "bf16": precision_config,
        }
        engine, optimizer, _, _ = deepspeed.initialize(config=config, model=model, optimizer=optimizer)
        return engine, optimizer

    @staticmethod
    def _run_step(engine, inputs, labels, missing_grad_index=None):
        loss = engine(inputs, labels)
        engine.backward(loss)
        if missing_grad_index is not None:
            list(engine.module.parameters())[missing_grad_index].grad = None
        engine.step()

    @staticmethod
    def _assert_optimizer_matches(candidate_engine, candidate_optimizer, reference_engine, reference_optimizer, dtype):
        for candidate, reference in zip(candidate_engine.module.parameters(), reference_engine.module.parameters()):
            torch.testing.assert_close(candidate, reference, rtol=0, atol=torch.finfo(dtype).eps)
        for candidate, reference in zip(candidate_optimizer.fp32_groups_flat, reference_optimizer.fp32_groups_flat):
            torch.testing.assert_close(candidate, reference, rtol=1e-6, atol=1e-7)
            candidate_state = candidate_optimizer.optimizer.state[candidate]
            reference_state = reference_optimizer.optimizer.state[reference]
            assert candidate_state["step"] == reference_state["step"]
            torch.testing.assert_close(candidate_state["exp_avg"], reference_state["exp_avg"], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(candidate_state["exp_avg_sq"],
                                       reference_state["exp_avg_sq"],
                                       rtol=1e-6,
                                       atol=1e-7)
        assert candidate_optimizer._global_grad_norm == pytest.approx(reference_optimizer._global_grad_norm)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    @pytest.mark.parametrize("adam_w_mode", [False, True], ids=["adam", "adamw"])
    @pytest.mark.parametrize("clip_grad", [0.0, 0.5])
    def test_matches_fallback(self, monkeypatch, dtype, adam_w_mode, clip_grad):
        if dtype not in get_accelerator().supported_dtypes():
            pytest.skip(f"{dtype} is not supported")

        hidden_dim = 8
        torch.manual_seed(1234)
        candidate_model = SimpleModel(hidden_dim)
        reference_model = SimpleModel(hidden_dim)
        reference_model.load_state_dict(candidate_model.state_dict())
        candidate_engine, candidate_optimizer = self._initialize_engine(candidate_model, dtype, adam_w_mode, clip_grad)
        reference_engine, reference_optimizer = self._initialize_engine(reference_model, dtype, adam_w_mode, clip_grad)
        reference_optimizer.optimizer.multi_tensor_adam_mixed_precision = None

        calls = 0
        mixed_precision_step = candidate_optimizer.optimizer._step_with_mixed_precision_grads

        def counted_mixed_precision_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            return mixed_precision_step(*args, **kwargs)

        monkeypatch.setattr(candidate_optimizer.optimizer, "_step_with_mixed_precision_grads",
                            counted_mixed_precision_step)

        torch.manual_seed(5678)
        for step in range(5):
            inputs = torch.randn(2, hidden_dim, device=candidate_engine.device, dtype=dtype)
            labels = torch.randint(hidden_dim, (2, ), device=candidate_engine.device)
            for engine in (candidate_engine, reference_engine):
                self._run_step(engine, inputs, labels)

            assert calls == step + 1
            self._assert_optimizer_matches(candidate_engine, candidate_optimizer, reference_engine,
                                           reference_optimizer, dtype)

    def test_missing_gradient_matches_fallback_and_decays_moments(self):
        dtype = torch.float16
        hidden_dim = 8
        torch.manual_seed(1234)
        candidate_model = SimpleModel(hidden_dim)
        reference_model = SimpleModel(hidden_dim)
        reference_model.load_state_dict(candidate_model.state_dict())
        candidate_engine, candidate_optimizer = self._initialize_engine(candidate_model, dtype)
        reference_engine, reference_optimizer = self._initialize_engine(reference_model, dtype)
        reference_optimizer.optimizer.multi_tensor_adam_mixed_precision = None

        torch.manual_seed(5678)
        inputs = torch.randn(2, hidden_dim, device=candidate_engine.device, dtype=dtype)
        labels = torch.randint(hidden_dim, (2, ), device=candidate_engine.device)
        for engine in (candidate_engine, reference_engine):
            self._run_step(engine, inputs, labels)

        candidate_master = candidate_optimizer.fp32_groups_flat[0]
        state = candidate_optimizer.optimizer.state[candidate_master]
        missing_grad_index = 1
        missing_param = list(candidate_engine.module.parameters())[missing_grad_index]
        missing_start = list(candidate_engine.module.parameters())[0].numel()
        missing_end = missing_start + missing_param.numel()
        exp_avg_before = state["exp_avg"][missing_start:missing_end].clone()
        exp_avg_sq_before = state["exp_avg_sq"][missing_start:missing_end].clone()

        inputs = torch.randn(2, hidden_dim, device=candidate_engine.device, dtype=dtype)
        labels = torch.randint(hidden_dim, (2, ), device=candidate_engine.device)
        for engine in (candidate_engine, reference_engine):
            self._run_step(engine, inputs, labels, missing_grad_index=missing_grad_index)

        self._assert_optimizer_matches(candidate_engine, candidate_optimizer, reference_engine, reference_optimizer,
                                       dtype)
        beta1, beta2 = candidate_optimizer.optimizer.param_groups[0]["betas"]
        torch.testing.assert_close(state["exp_avg"][missing_start:missing_end], exp_avg_before * beta1)
        torch.testing.assert_close(state["exp_avg_sq"][missing_start:missing_end], exp_avg_sq_before * beta2)

    def test_missing_capability_uses_existing_path(self, monkeypatch):
        dtype = torch.float16
        hidden_dim = 8
        torch.manual_seed(1234)
        candidate_model = SimpleModel(hidden_dim)
        reference_model = SimpleModel(hidden_dim)
        reference_model.load_state_dict(candidate_model.state_dict())
        candidate_engine, candidate_optimizer = self._initialize_engine(candidate_model, dtype)
        reference_engine, reference_optimizer = self._initialize_engine(reference_model, dtype)
        candidate_optimizer.optimizer.multi_tensor_adam_mixed_precision = None
        reference_optimizer.optimizer.multi_tensor_adam_mixed_precision = None

        def unexpected_mixed_precision_step(*args, **kwargs):
            pytest.fail("mixed-precision step must not run without kernel capability")

        monkeypatch.setattr(candidate_optimizer.optimizer, "_step_with_mixed_precision_grads",
                            unexpected_mixed_precision_step)
        torch.manual_seed(5678)
        inputs = torch.randn(2, hidden_dim, device=candidate_engine.device, dtype=dtype)
        labels = torch.randint(hidden_dim, (2, ), device=candidate_engine.device)
        for engine in (candidate_engine, reference_engine):
            self._run_step(engine, inputs, labels)
        self._assert_optimizer_matches(candidate_engine, candidate_optimizer, reference_engine, reference_optimizer,
                                       dtype)

    def test_checkpoint_state_schema_and_resume(self):
        dtype = torch.float16
        hidden_dim = 8
        torch.manual_seed(1234)
        uninterrupted_model = SimpleModel(hidden_dim)
        fallback_model = SimpleModel(hidden_dim)
        fallback_model.load_state_dict(uninterrupted_model.state_dict())
        uninterrupted_engine, uninterrupted_optimizer = self._initialize_engine(uninterrupted_model, dtype)
        fallback_engine, fallback_optimizer = self._initialize_engine(fallback_model, dtype)
        fallback_optimizer.optimizer.multi_tensor_adam_mixed_precision = None

        torch.manual_seed(5678)
        batches = [(torch.randn(2, hidden_dim, device=uninterrupted_engine.device,
                                dtype=dtype), torch.randint(hidden_dim, (2, ), device=uninterrupted_engine.device))
                   for _ in range(4)]
        for inputs, labels in batches[:3]:
            self._run_step(uninterrupted_engine, inputs, labels)
            self._run_step(fallback_engine, inputs, labels)

        def key_schema(value):
            if isinstance(value, dict):
                return {key: key_schema(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [key_schema(child) for child in value]
            return None

        checkpoint_model = copy.deepcopy(uninterrupted_engine.module.state_dict())
        checkpoint_optimizer = copy.deepcopy(uninterrupted_optimizer.state_dict())
        assert key_schema(checkpoint_optimizer) == key_schema(fallback_optimizer.state_dict())

        resumed_model = SimpleModel(hidden_dim)
        resumed_engine, resumed_optimizer = self._initialize_engine(resumed_model, dtype)
        resumed_engine.module.load_state_dict(checkpoint_model)
        resumed_optimizer.load_state_dict(checkpoint_optimizer)

        inputs, labels = batches[3]
        self._run_step(uninterrupted_engine, inputs, labels)
        self._run_step(resumed_engine, inputs, labels)
        self._assert_optimizer_matches(uninterrupted_engine, uninterrupted_optimizer, resumed_engine,
                                       resumed_optimizer, dtype)


class TestFP16OptimizerForMoE(DistributedTest):
    world_size = 2

    def test_unfused_gradnorm(self, monkeypatch):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        def mock_unscale_and_clip_grads(total_norm, apply_scale=True):
            torch_norm_tensor = get_accelerator().FloatTensor([total_norm])
            all_gather_results = [torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gather_results, torch_norm_tensor)
            assert len(set([x.item() for x in all_gather_results])) == 1
            return 1.0

        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        optimizer = torch.optim.AdamW(params=model.parameters())
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       optimizer=optimizer,
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device,
                                          dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    def test_fused_gradnorm(self, monkeypatch):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        def mock_unscale_and_clip_grads(grads_groups_flat, total_norm, apply_scale=True):
            torch_norm_tensor = get_accelerator().FloatTensor([total_norm])
            all_gather_results = [torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gather_results, torch_norm_tensor)
            assert len(set([x.item() for x in all_gather_results])) == 1
            return 1.0

        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        param_group = {'params': [p for p in model.parameters()], 'name': 'random-unique-name'}
        params = split_params_into_different_moe_groups_for_optimizer(param_group)
        # optimizer = torch.optim.AdamW(params=model.parameters())
        optimizer = FusedAdam(params=params)
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       optimizer=optimizer,
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device,
                                          dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()

    def test_fused_adam_uses_existing_path(self, monkeypatch):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {"train_batch_size": 2, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        param_group = {'params': list(model.parameters()), 'name': 'random-unique-name'}
        optimizer = FusedAdam(params=split_params_into_different_moe_groups_for_optimizer(param_group))
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       optimizer=optimizer,
                                                       dist_init_required=False)

        def unexpected_mixed_precision_step(*args, **kwargs):
            pytest.fail("MoE must use the existing optimizer path")

        monkeypatch.setattr(optimizer.optimizer, "_step_with_mixed_precision_grads", unexpected_mixed_precision_step)
        batch = next(
            iter(
                sequence_dataloader(model=engine,
                                    total_samples=2,
                                    hidden_dim=hidden_dim,
                                    device=engine.device,
                                    dtype=torch.float16)))
        loss = engine(batch[0], batch[1])
        engine.backward(loss)
        engine.step()

    @pytest.mark.parametrize("fused_lamb_legacy", [(False), (True)])
    @pytest.mark.skipif(not deepspeed.ops.__compatible_ops__[FusedLambBuilder.NAME],
                        reason="FusedLambBuilder has not been implemented on this system.")
    def test_lamb_gradnorm(self, monkeypatch, fused_lamb_legacy: bool):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if not required_torch_version(min_version=1.8):
            pytest.skip("DeepSpeed MoE tests need torch 1.8 or higher to run correctly")

        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10

        def mock_unscale_and_clip_grads(total_norm, apply_scale=True):
            torch_norm_tensor = get_accelerator().FloatTensor([total_norm])
            all_gather_results = [torch.zeros_like(torch_norm_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(all_gather_results, torch_norm_tensor)
            assert len(set([x.item() for x in all_gather_results])) == 1
            return 1.0

        # initialize MoE
        model = SimpleMoEModel(hidden_dim, ep_size=2)
        engine, optimizer, _, _ = deepspeed.initialize(config=config_dict,
                                                       model=model,
                                                       model_parameters=model.parameters(),
                                                       dist_init_required=False)
        monkeypatch.setattr(optimizer, 'unscale_and_clip_grads', mock_unscale_and_clip_grads)
        optimizer.fused_lamb_legacy = fused_lamb_legacy
        data_loader = sequence_dataloader(model=engine,
                                          total_samples=50,
                                          hidden_dim=hidden_dim,
                                          device=engine.device,
                                          dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = engine(batch[0], batch[1])
            engine.backward(loss)
            engine.step()


class TestAdamwFP16EmptyGrad(DistributedTest):
    world_size = 1

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {"train_batch_size": 1, "steps_per_print": 1, "fp16": {"enabled": True}}
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.AdamW(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestAdamFP16ZeroOneCycleCompatibility(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "scheduler": {
                "type": "OneCycle",
                "params": {
                    "cycle_first_step_size": 16000,
                    "cycle_first_stair_count": 8000,
                    "decay_step_size": 16000,
                    "cycle_min_lr": 1e-06,
                    "cycle_max_lr": 3e-05,
                    "decay_lr_rate": 1e-07,
                    "cycle_min_mom": 0.85,
                    "cycle_max_mom": 0.99,
                    "decay_mom_rate": 0.0
                }
            },
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroStaticScale(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload, hidden_dim=4):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 138.
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            }
        }

        model = SimpleModel(hidden_dim)
        model, optim, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        # Ensure the static scaler is configured.
        assert optim.dynamic_loss_scale == False
        assert optim.loss_scaler.loss_scale == 138.

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroAllowUntestedOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage, use_cpu_offload):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        config_dict = {
            "train_batch_size": 4,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload
            },
            "zero_allow_untested_optimizer": False,
            "zero_force_ds_cpu_optimizer": False
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = SimpleOptimizer(model.parameters())
        with pytest.raises(AssertionError):
            model, optim, _, _ = deepspeed.initialize(config=config_dict,
                                                      model=model,
                                                      optimizer=optimizer,
                                                      model_parameters=model.parameters())
            model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("use_cpu_offload", [True, False])
class TestZeroEmptyPartition(DistributedTest):
    world_size = 3

    def test(self, zero_stage, use_cpu_offload):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        if use_cpu_offload and not deepspeed.ops.__compatible_ops__[CPUAdamBuilder.NAME]:
            pytest.skip("cpu-adam is not compatible")

        if zero_stage == 3:
            pytest.skip("skip for now")

        config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 8
            },
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": zero_stage,
                "cpu_offload": use_cpu_offload,
                "reduce_bucket_size": 100,
                "allgather_bucket_size": 100
            }
        }
        hidden_dim = 1
        model = SimpleModel(hidden_dim)

        # Ensure model has 2 parameters, to cause empty partition with DP=3
        assert len(list(model.parameters())) == 2
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        # Now make sure things work..
        data_loader = random_dataloader(model=model,
                                        total_samples=1,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize("zero_stage", [1, 2, 3])
@pytest.mark.parametrize("optimizer_constructor", [FusedAdam, torch.optim.Adam])
class TestZeroSupportedClientOptimizer(DistributedTest):
    world_size = 1

    def test(self, zero_stage, optimizer_constructor):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": zero_stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        client_optimizer = optimizer_constructor(params=model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=client_optimizer)
        model.destroy()


class TestZero2ReduceScatterOff(DistributedTest):
    world_size = 2

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 2,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
            },
            "gradient_clipping": 1.0,
            "zero_optimization": {
                "stage": 2,
                "contiguous_gradients": True,
                "allgather_bucket_size": 2000000000,
                "reduce_bucket_size": 200000000,
                "overlap_comm": False,
                "reduce_scatter": False
            },
            "fp16": {
                "enabled": True
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


@pytest.mark.parametrize("adam_type", ["Adam", "AdamW"])
@pytest.mark.parametrize("torch_impl", [True, False])
class TestFP16AdamTypes(DistributedTest):
    world_size = 1

    def test(self, adam_type, torch_impl):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 10
            },
            "optimizer": {
                "type": adam_type,
                "torch_adam": torch_impl,
                "params": {
                    "lr": 0.00015
                }
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, model_parameters=model.parameters())

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()


class TestZero3LazyScatter(DistributedTest):
    world_size = 1

    def test(self):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True,
                "initial_scale_power": 10
            },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 0.00015
                }
            },
            "zero_optimization": {
                "stage": 3
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        model, _, _, _ = deepspeed.initialize(
            config=config_dict,
            model=model,
            model_parameters=model.parameters(),
        )

        data_loader = random_dataloader(model=model,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)

        for _, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()


@pytest.mark.parametrize('stage', [1, 2, 3])
class TestZeroEmptyGrad(DistributedTest):
    world_size = 1

    def test(self, stage):
        if not get_accelerator().is_fp16_supported():
            pytest.skip("fp16 is not supported")
        config_dict = {
            "train_batch_size": 1,
            "steps_per_print": 1,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": stage
            }
        }
        hidden_dim = 10

        model = SimpleModel(hidden_dim)
        optimizer = torch.optim.Adam(model.parameters())
        model, _, _, _ = deepspeed.initialize(config=config_dict, model=model, optimizer=optimizer)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float16)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

        model.destroy()
