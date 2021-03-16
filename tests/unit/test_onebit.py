import torch
import torch.distributed as dist
import deepspeed
import argparse
import pytest
import json
import os
import numpy as np
import time
from common import distributed_test
from simple_model import SimpleModel, SimpleOptimizer, random_dataloader, args_from_dict, create_deepspeed_args

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR < 1 or TORCH_MINOR < 8:
    pytest.skip("NCCL-based 1-bit compression requires torch 1.8 or higher",
                allow_module_level=True)


def test_onebitadam_fp16_basic(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 0.00015,
                "weight_decay": 0.01,
                "freeze_step": 2,
                "cuda_aware": False,
                "comm_backend_name": "nccl"
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_onebitadam_fp16_basic(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_onebitadam_fp16_basic(args=args, model=model, hidden_dim=hidden_dim)


def test_onebitadam_fp32_basic(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 0.00015,
                "weight_decay": 0.01,
                "freeze_step": 2,
                "cuda_aware": False,
                "comm_backend_name": "nccl"
            }
        },
        "gradient_clipping": 1.0,
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)

    @distributed_test(world_size=[1, 2])
    def _test_onebitadam_fp32_basic(args, model, hidden_dim):
        model, _, _, _ = deepspeed.initialize(args=args,
                                              model=model,
                                              model_parameters=model.parameters())
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device,
                                        dtype=torch.float)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()

    _test_onebitadam_fp32_basic(args=args, model=model, hidden_dim=hidden_dim)


def test_onebitadam_exp_avg_mask(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 0.00015,
                "weight_decay": 0.01,
                "freeze_step": 2,
                "cuda_aware": False,
                "comm_backend_name": "nccl"
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)
    param_optimizer = list(model.named_parameters())
    mask1 = torch.zeros_like(param_optimizer[0][1].data)
    for col in range(mask1.size()[1]):
        mask1[0][col] += 1
    mask1 = torch.flatten(mask1)
    optimizer_grouped_parameters = [{
        'params': [param_optimizer[0][1]],
        'weight_decay': 0.01,
        'exp_avg_mask': mask1
    },
                                    {
                                        'params': [param_optimizer[1][1]],
                                        'weight_decay': 0.01
                                    }]

    @distributed_test(world_size=[2])
    def _test_onebitadam_exp_avg_mask(args, model, hidden_dim):
        model, optimizer, _, _ = deepspeed.initialize(args=args,
                                                      model=model,
                                                      model_parameters=optimizer_grouped_parameters)
        data_loader = random_dataloader(model=model,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model.device)
        for n, batch in enumerate(data_loader):
            loss = model(batch[0], batch[1])
            model.backward(loss)
            model.step()
        # Test whether the momentum mask works
        for v in optimizer.state.values():
            if v['exp_avg'].size() == mask1.size():
                assert torch.allclose(v['exp_avg'], v['exp_avg'].mul_(mask1.to(device=v['exp_avg'].device)), atol=1e-07), f"Momentum mask is not working properly"

    _test_onebitadam_exp_avg_mask(args=args, model=model, hidden_dim=hidden_dim)


def test_onebitadam_checkpointing(tmpdir):
    config_dict = {
        "train_batch_size": 2,
        "steps_per_print": 1,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 0.00015,
                "weight_decay": 0.01,
                "freeze_step": 2,
                "cuda_aware": False,
                "comm_backend_name": "nccl"
            }
        },
        "gradient_clipping": 1.0,
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        }
    }
    args = args_from_dict(tmpdir, config_dict)
    hidden_dim = 10

    model = SimpleModel(hidden_dim)
    param_optimizer = list(model.named_parameters())
    mask1 = torch.zeros_like(param_optimizer[0][1].data)
    mask2 = torch.zeros_like(param_optimizer[0][1].data)
    for col in range(mask1.size()[1]):
        mask1[0][col] += 1
        mask2[1][col] += 1
    mask1 = torch.flatten(mask1)
    mask2 = torch.flatten(mask2)

    optimizer_grouped_parameters_1 = [{
        'params': [param_optimizer[0][1]],
        'weight_decay': 0.01,
        'exp_avg_mask': mask1
    },
                                      {
                                          'params': [param_optimizer[1][1]],
                                          'weight_decay': 0.01
                                      }]

    optimizer_grouped_parameters_2 = [{
        'params': [param_optimizer[0][1]],
        'weight_decay': 0.01,
        'exp_avg_mask': mask2
    },
                                      {
                                          'params': [param_optimizer[1][1]],
                                          'weight_decay': 0.01
                                      }]

    optimizer_grouped_parameters_3 = [{
        'params': [param_optimizer[0][1]],
        'weight_decay': 0.01
    },
                                      {
                                          'params': [param_optimizer[1][1]],
                                          'weight_decay': 0.01
                                      }]

    @distributed_test(world_size=[2])
    def _test_onebitadam_checkpointing(mask1, mask2, args, model, hidden_dim):
        model_1, optimizer_1, _, _ = deepspeed.initialize(args=args,
                                                          model=model,
                                                          model_parameters=optimizer_grouped_parameters_1)
        data_loader = random_dataloader(model=model_1,
                                        total_samples=10,
                                        hidden_dim=hidden_dim,
                                        device=model_1.device)
        for n, batch in enumerate(data_loader):
            loss = model_1(batch[0], batch[1])
            model_1.backward(loss)
            model_1.step()
        # Test whether momentum mask still exist after saving checkpoint
        assert optimizer_1.optimizer.adam_freeze_key is True
        mask1 = mask1.to(device=optimizer_1.param_groups[0]['exp_avg_mask'].device)
        assert torch.allclose(optimizer_1.param_groups[0]['exp_avg_mask'], mask1, atol=1e-07), f"Incorrect momentum mask"
        save_folder = os.path.join(tmpdir, 'saved_checkpoint')
        # optimizer_1.optimizer.gather_compression_errors()
        model_1.save_checkpoint(save_folder, tag=None)
        time.sleep(5)
        assert torch.allclose(optimizer_1.param_groups[0]['exp_avg_mask'], mask1, atol=1e-07), f"Momentum mask should not change after saving checkpoint"


        model_2, optimizer_2, _, _ = deepspeed.initialize(args=args,
                                                          model=model,
                                                          model_parameters=optimizer_grouped_parameters_2)
        # Test whether momentum mask stays the same after loading checkpoint
        mask2 = mask2.to(device=optimizer_2.param_groups[0]['exp_avg_mask'].device)
        assert torch.allclose(optimizer_2.param_groups[0]['exp_avg_mask'], mask2, atol=1e-07), f"Incorrect momentum mask"
        model_2.load_checkpoint(save_folder,
                                tag=None,
                                load_optimizer_states=True,
                                load_lr_scheduler_states=True)
        assert torch.allclose(optimizer_2.param_groups[0]['exp_avg_mask'], mask2, atol=1e-07), f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is resetted
        for v in optimizer_2.state.values():
            assert 'worker_error' not in v, f"Incorrect worker error"
            assert 'server_error' not in v, f"Incorrect server error"
        assert optimizer_2.optimizer.adam_freeze_key is True

        model_3, optimizer_3, _, _ = deepspeed.initialize(args=args,
                                                          model=model,
                                                          model_parameters=optimizer_grouped_parameters_3)
        optimizer_3.optimizer.freeze_step = 20
        data_loader = random_dataloader(model=model_3,
                                        total_samples=50,
                                        hidden_dim=hidden_dim,
                                        device=model_3.device)
        for n, batch in enumerate(data_loader):
            loss = model_3(batch[0], batch[1])
            model_3.backward(loss)
            model_3.step()
        assert optimizer_3.optimizer.adam_freeze_key is True
        # Test whether momentum mask stays the same after loading checkpoint
        assert 'exp_avg_mask' not in optimizer_3.param_groups[0], f"Incorrect momentum mask"
        model_3.load_checkpoint(save_folder,
                                tag=None,
                                load_optimizer_states=True,
                                load_lr_scheduler_states=True)
        assert 'exp_avg_mask' not in optimizer_3.param_groups[0], f"Momentum mask should not change after loading checkpoint"
        # Test whether worker&server error is resetted
        for v in optimizer_3.state.values():
            assert 'worker_error' not in v, f"Incorrect worker error"
            assert 'server_error' not in v, f"Incorrect server error"
        assert optimizer_3.optimizer.adam_freeze_key is False

    _test_onebitadam_checkpointing(mask1,
                                   mask2,
                                   args=args,
                                   model=model,
                                   hidden_dim=hidden_dim)


def test_compressed_allreduce_basic(tmpdir):
    @distributed_test(world_size=[1, 2])
    def _test_compressed_allreduce_basic():
        from deepspeed.runtime.comm.nccl import NcclBackend
        size = dist.get_world_size()
        rank = dist.get_rank()
        backend = NcclBackend()
        local_rank = dist.get_rank()
        device = torch.device("cuda", dist.get_rank())

        # A simulated compression function using torch.distributed
        def torch_sim(a):
            a_sign = a.sign().add_(1).bool().float().add_(-0.5).mul_(2.0)
            scale = a.norm() / np.sqrt(a.numel())
            a_compressed = scale * a_sign
            a_sign = None
            worker_error = a - a_compressed
            dist.all_reduce(a_compressed)
            a_compressed.mul_(1 / dist.get_world_size())
            a_server_sign = a_compressed.sign().add_(1).bool().float().add_(-0.5).mul_(
                2.0)
            a_list = torch.chunk(a_compressed, chunks=dist.get_world_size())
            server_scale = [
                chunk_a.norm() / np.sqrt(chunk_a.numel()) for chunk_a in a_list
            ]
            a_sign_list = torch.chunk(a_server_sign, dist.get_world_size())
            a_server_compressed = torch.cat(
                [server_scale[i] * a_sign_list[i] for i in range(dist.get_world_size())])
            rank = dist.get_rank()
            server_error = a_list[rank] - server_scale[rank] * a_sign_list[rank]
            torch.cuda.synchronize()
            torch.distributed.barrier()
            return a_server_compressed, worker_error, server_error

        tensor_size = 300 * 2**20
        server_size = int(tensor_size / size)
        if tensor_size % (8 * size) != 0:
            right_tensor_size = tensor_size + (8 * size - (tensor_size % (8 * size)))
        else:
            right_tensor_size = tensor_size
        right_server_size = right_tensor_size // size

        # Adding bias to the initialization of the gradient we are communicating
        # In order to get rid of the case where some elements in the gradient are too small
        a = (torch.rand(tensor_size, device=device) - 0.5) + 0.01 * rank

        worker_error = torch.zeros(right_tensor_size, device=device)
        server_error = torch.zeros(right_server_size, device=device)

        a_torch, worker_error_torch, server_error_torch = torch_sim(a)
        torch.cuda.empty_cache()

        a_after = backend.compressed_allreduce(a, worker_error, server_error, local_rank)

        threshold = 1e-6
        magnitude_threshold = 1e-6
        diff_mask = (a_after - a_torch) > threshold
        diff_server_mask = torch.chunk(diff_mask, size)[rank]
        mpi_server = torch.chunk(a_after, size)[rank] + server_error
        torch_server = torch.chunk(a_torch, size)[rank] + server_error_torch

        # If the number in the compensated_server_m is too small (e.g 1e-8), then calling sign() might be problematic
        # The test would skip those numbers that are too small in compensated_server_m
        check_mag_mask = mpi_server[diff_server_mask] > magnitude_threshold
        if torch.sum(check_mag_mask) != 0:
            print('Fails at {} of positions'.format(torch.sum(check_mag_mask)))
        assert torch.sum(diff_server_mask) == 0 or torch.sum(check_mag_mask) == 0

    _test_compressed_allreduce_basic()
