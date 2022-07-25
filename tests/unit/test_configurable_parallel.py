import torch
import deepspeed
import pytest
import random
import numpy as np
import torch.multiprocessing as mp
import deepspeed.comm as dist
from .common import distributed_test
from .megatron_model import get_gpt2_model, get_megatron_version
from .megatron_model import MockGPT2ModelPipe as GPT2ModelPipe
from deepspeed.utils import RepeatingLoader

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
pytestmark = pytest.mark.skipif(
    TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 5),
    reason='Megatron-LM package requires Pytorch version 1.5 or above')


def reset_random(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TestConfigurableMP:
    def setup_method(self, method):
        reset_random()

    def get_inputs(self, bs=1, seq_len=20):
        input_ids = torch.randint(low=0, high=1000, size=(bs, seq_len))
        position_ids = torch.randint(low=0, high=2, size=(bs, seq_len))
        attention_mask = torch.randint(low=0,
                                       high=2,
                                       size=(bs,
                                             seq_len),
                                       dtype=torch.bool)
        return [input_ids, position_ids, attention_mask]

    def get_deepspeed_model(self, model, tmpdir):
        ds_config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
        }

        from megatron import mpu
        model, _, _,_ = deepspeed.initialize(model=model,
                                             mpu=mpu,
                                             model_parameters=model.parameters(),
                                             config=ds_config_dict)
        return model

    def test_gpt2_basic(self, tmpdir):
        # basic test case, mp_size=1, verify ckpt saving/loading.

        @distributed_test(world_size=1)
        def _run():
            inputs = self.get_inputs()
            args_defaults = {
                'num_layers': 2,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            model = get_gpt2_model(args_defaults)
            model = self.get_deepspeed_model(model, tmpdir)

            model.eval()
            baseline = model(inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda())

            tag = 'mp_1'
            state_dict = {}
            state_dict['checkpoint_version'] = get_megatron_version()
            model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)
            dist.barrier()
            model.load_checkpoint(tmpdir,
                                  tag=tag,
                                  load_optimizer_states=False,
                                  load_lr_scheduler_states=False)

            test = model(inputs[0], inputs[1], inputs[2])
            assert torch.allclose(baseline, test, atol=1e-07), f"Baseline output {baseline} is not equal to save-then-load output {test}"

        _run()

    def test_gpt2_mp2_no_resize(self, tmpdir):
        # test mp_size=2 case, verify ckpt saving/loading without resize.

        @distributed_test(world_size=2)
        def _run(inputs):
            args_defaults = {
                'num_layers': 2,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            model = get_gpt2_model(args_defaults, mp_size=2)
            model = self.get_deepspeed_model(model, tmpdir)

            model.eval()

            baseline = model(inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda())

            tag = 'mp_2'
            state_dict = {}
            state_dict['checkpoint_version'] = get_megatron_version()
            model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)
            dist.barrier()
            model.load_checkpoint(tmpdir,
                                  tag=tag,
                                  load_optimizer_states=False,
                                  load_lr_scheduler_states=False)

            test = model(inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda())
            assert torch.allclose(baseline, test, rtol=1.0, atol=1e-07), f"Baseline output {baseline} is not equal to save-then-load output {test}"

        inputs = self.get_inputs()
        _run(inputs)

    def _test_gpt2_config_mp(self, tmpdir, mp_size, resize):
        # test mp_size=2 case, verify resize=1 case for ckpt merging.

        @distributed_test(world_size=mp_size)
        def _run_baseline(inputs, tag, output, quit_event):
            reset_random()
            args_defaults = {
                'num_layers': 2,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            model = get_gpt2_model(args_defaults, mp_size=mp_size)
            model = self.get_deepspeed_model(model, tmpdir)

            model.eval()

            with torch.no_grad():
                baseline = model(inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda())
                if dist.get_rank() == 0:
                    output.put(baseline.cpu())

                state_dict = {}
                state_dict['checkpoint_version'] = get_megatron_version()
                model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)
                quit_event.wait()

        @distributed_test(world_size=resize)
        def _run_resize(inputs, tag, output, quit_event):
            reset_random()
            args_defaults = {
                'num_layers': 2,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            model = get_gpt2_model(args_defaults, mp_size=resize)
            model = self.get_deepspeed_model(model, tmpdir)

            model.eval()

            with torch.no_grad():
                model.load_checkpoint(tmpdir,
                                      tag=tag,
                                      load_optimizer_states=False,
                                      load_lr_scheduler_states=False)
                test = model(inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda())
                if dist.get_rank() == 0:
                    output.put(test.cpu())
            quit_event.wait()

        def _verify(b_queue, t_queue, baseline_event, test_event):
            baseline = b_queue.get()
            baseline_event.set()

            test = t_queue.get()
            test_event.set()

            assert torch.allclose(baseline, test, atol=1e-03), f"Baseline output {baseline} is not equal to save-then-load output {test}"

        tag = f'mp_{mp_size}_resize_{resize}'
        inputs = self.get_inputs()

        baseline = mp.Queue()
        test = mp.Queue()
        baseline_event = mp.Event()
        test_event = mp.Event()

        verify_process = mp.Process(target=_verify,
                                    args=(baseline,
                                          test,
                                          baseline_event,
                                          test_event))
        verify_process.start()

        _run_baseline(inputs, tag, baseline, baseline_event)
        _run_resize(inputs, tag, test, test_event)

        verify_process.join()

    def test_gpt2_mp_2to1(self, tmpdir):
        # test mp_size=2 case, verify resize=1 case for ckpt merging.
        self._test_gpt2_config_mp(tmpdir, mp_size=2, resize=1)

    def test_gpt2_mp_2to4(self, tmpdir):
        # test mp_size=2 case, verify resize=4 case for ckpt splitting.
        self._test_gpt2_config_mp(tmpdir, mp_size=2, resize=4)


class TestConfigurablePP:
    def setup_method(self, method):
        reset_random()

    def get_inputs(self, bs=1, seq_len=1, hidden_size=128):
        hidden_states = torch.randn(bs, seq_len, hidden_size)
        attention_mask = torch.randint(low=0,
                                       high=2,
                                       size=(bs,
                                             seq_len),
                                       dtype=torch.bool)
        return (hidden_states, attention_mask)

    def get_deepspeed_model(self, model, tmpdir):
        ds_config_dict = {
            "train_micro_batch_size_per_gpu": 1,
            "optimizer": {
                "type": "Lamb",
                "params": {
                    "lr": 0.00015
                }
            },
        }
        dist.barrier()

        model, _, _,_ = deepspeed.initialize(model=model,
                                             model_parameters=model.parameters(),
                                             config=ds_config_dict)
        return model.cuda()

    def get_topology(self, mp, pp, world_size):
        assert world_size % (pp * mp) == 0
        dp = world_size // (pp * mp)

        from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology
        topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=mp, num_dp=dp)

        return topo

    def test_pp_basic(self, tmpdir):
        # basic test case, mp_size=2, pp_size=2, verify ckpt saving/loading.

        mp_size = 2
        pp_size = 2
        world_size = mp_size * pp_size

        @distributed_test(world_size=world_size)
        def _run():
            args_defaults = {
                'num_layers': 8,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            topo = self.get_topology(mp_size, pp_size, world_size)
            gpt2_pipe_model = GPT2ModelPipe(num_layers=8,
                                            num_stages=pp_size,
                                            mp_size=mp_size,
                                            args_others=args_defaults,
                                            topo=topo)
            model = self.get_deepspeed_model(gpt2_pipe_model, tmpdir)

            tag = 'pp_basic'
            state_dict = {}
            state_dict['checkpoint_version'] = get_megatron_version()
            model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)

            if model.is_first_stage() or model.is_last_stage():
                inputs = self.get_inputs()
                loader = RepeatingLoader([(inputs[0], 0)])
                data_iter = iter(loader)
            else:
                data_iter = None

            baseline = model.eval_batch(data_iter=data_iter,
                                        compute_loss=False,
                                        reduce_output=None)

            dist.barrier()
            model.load_checkpoint(tmpdir,
                                  tag=tag,
                                  load_optimizer_states=False,
                                  load_lr_scheduler_states=False)
            dist.barrier()

            test = model.eval_batch(data_iter=data_iter,
                                    compute_loss=False,
                                    reduce_output=None)

            if test is not None:
                assert len(baseline) == len(test)
                # Compare outputs of each microbatch
                for mb in range(len(baseline)):
                    for b, t in zip(baseline[mb], test[mb]):
                        if b.is_floating_point():  # don't compare masks
                            assert torch.allclose(b, t, atol=1e-07), f"Baseline output {baseline} is not equal to save-then-load output {test}"

        _run()

    def _test_gpt2_config_pp(self, tmpdir, mp_size, pp_size, mp_resize, pp_resize):
        @distributed_test(world_size=pp_size * mp_size)
        def _run_baseline(inputs, tag, output, quit_event):
            reset_random()
            args_defaults = {
                'num_layers': 8,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            topo = self.get_topology(mp_size, pp_size, mp_size * pp_size)
            gpt2_pipe_model = GPT2ModelPipe(num_layers=8,
                                            num_stages=pp_size,
                                            mp_size=mp_size,
                                            args_others=args_defaults,
                                            topo=topo)
            model = self.get_deepspeed_model(gpt2_pipe_model, tmpdir)

            with torch.no_grad():
                inputs = [x.cuda() for x in inputs]
                if model.is_first_stage() or model.is_last_stage():
                    loader = RepeatingLoader([(inputs[0], 0)])
                    data_iter = iter(loader)
                else:
                    data_iter = None

                baseline = model.eval_batch(data_iter=data_iter,
                                            compute_loss=False,
                                            reduce_output=None)

                if baseline is not None:
                    # baseline should be [[hidden, True]]]
                    assert len(baseline) == 1
                    assert len(baseline[0]) == 1
                    assert torch.is_tensor(baseline[0][0])
                    output.put(baseline[0][0].cpu())

                state_dict = {}
                state_dict['checkpoint_version'] = get_megatron_version()
                model.save_checkpoint(tmpdir, tag=tag, client_state=state_dict)
                quit_event.wait()

        @distributed_test(world_size=mp_resize * pp_resize)
        def _run_resize(inputs, tag, output, quit_event):
            reset_random()
            args_defaults = {
                'num_layers': 8,
                'hidden_size': 128,
                'num_attention_heads': 8,
                'max_position_embeddings': 128,
            }

            topo = self.get_topology(mp_resize, pp_resize, mp_resize * pp_resize)
            gpt2_pipe_model = GPT2ModelPipe(num_layers=8,
                                            num_stages=pp_resize,
                                            mp_size=mp_resize,
                                            args_others=args_defaults,
                                            topo=topo)
            model = self.get_deepspeed_model(gpt2_pipe_model, tmpdir)

            with torch.no_grad():
                model.load_checkpoint(tmpdir,
                                      tag=tag,
                                      load_optimizer_states=False,
                                      load_lr_scheduler_states=False)
                inputs = [x.cuda() for x in inputs]
                if model.is_first_stage() or model.is_last_stage():
                    loader = RepeatingLoader([(inputs[0], 0)])
                    data_iter = iter(loader)
                else:
                    data_iter = None

                test = model.eval_batch(data_iter=data_iter,
                                        compute_loss=False,
                                        reduce_output=None)

                if test is not None:
                    # test should be [[hidden, True]]]
                    assert len(test) == 1
                    assert len(test[0]) == 1
                    assert torch.is_tensor(test[0][0])
                    output.put(test[0][0].cpu())

            quit_event.wait()

        def _verify(b_queue, t_queue, baseline_event, test_event):
            baseline = b_queue.get()
            baseline_event.set()

            test = t_queue.get()
            test_event.set()

            assert torch.allclose(baseline, test, atol=1e-03), f"Baseline output {baseline} is not equal to save-then-load output {test}"

        tag = f'mp_{mp_size}to{mp_resize}_pp_{pp_size}to{pp_resize}'

        baseline = mp.Queue()
        test = mp.Queue()
        baseline_event = mp.Event()
        test_event = mp.Event()

        verify_process = mp.Process(target=_verify,
                                    args=(baseline,
                                          test,
                                          baseline_event,
                                          test_event))
        verify_process.start()

        inputs = self.get_inputs()
        _run_baseline(inputs, tag, baseline, baseline_event)
        _run_resize(inputs, tag, test, test_event)

        verify_process.join()

    def test_gpt2_mp1_pp_2to1(self, tmpdir):
        self._test_gpt2_config_pp(tmpdir, mp_size=1, pp_size=2, mp_resize=1, pp_resize=1)

    def test_gpt2_mp1_pp_2to4(self, tmpdir):
        self._test_gpt2_config_pp(tmpdir, mp_size=1, pp_size=2, mp_resize=1, pp_resize=4)

    def test_gpt2_mp2_pp_2to1(self, tmpdir):
        self._test_gpt2_config_pp(tmpdir, mp_size=2, pp_size=2, mp_resize=2, pp_resize=1)

    def test_gpt2_mp2_pp_1to2(self, tmpdir):
        self._test_gpt2_config_pp(tmpdir, mp_size=2, pp_size=1, mp_resize=2, pp_resize=2)

    def test_gpt2_pp_2to1_mp_2to1(self, tmpdir):
        self._test_gpt2_config_pp(tmpdir, mp_size=2, pp_size=2, mp_resize=1, pp_resize=1)

    def test_gpt2_pp_1to2_mp_1to2(self, tmpdir):
        self._test_gpt2_config_pp(tmpdir, mp_size=1, pp_size=1, mp_resize=2, pp_resize=2)
