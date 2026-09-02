# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import ast
import inspect
import textwrap
import sys
import unittest

import torch
from unittest import mock

from deepspeed.module_inject.auto_ep_comm import (COMM_BACKEND, DEEPEP_BACKEND, SUPPORTED_DTYPES, _conform_rows,
                                                  _DeepEPCombine, _DeepEPDispatch, _import_deep_ep, _qps_for_sms,
                                                  assert_dtype_supported, destroy_exchanges)
from deepspeed.module_inject import auto_ep_layer
from deepspeed.module_inject.auto_ep_config import parse_autoep_config, validate_autoep_config


class TestAutoEPCommBackendSelection(unittest.TestCase):
    """Backend choice lives in the config, alongside the rest of AutoEP."""

    @staticmethod
    def validate(config):
        validate_autoep_config(config, world_size=1, pp_size=1, tp_size=1, sp_size=1)

    def test_defaults_to_the_collective_path(self):
        # A config that asks for nothing must leave existing jobs unchanged.
        config = parse_autoep_config({"enabled": True})

        self.assertEqual(config.comm_backend, COMM_BACKEND)
        self.assertEqual(config.comm_num_sm, 12)
        self.assertEqual(config.comm_qp_margin, 4)

    def test_selects_deepep(self):
        config = parse_autoep_config({"enabled": True, "comm_backend": "deepep"})

        self.assertEqual(config.comm_backend, DEEPEP_BACKEND)

    def test_deepep_requires_an_explicit_capacity(self):
        config = parse_autoep_config({"enabled": True, "comm_backend": "deepep"})

        with self.assertRaises(ValueError) as caught:
            self.validate(config)

        message = str(caught.exception)
        self.assertIn("comm_max_tokens_per_rank", message)
        self.assertIn("train_micro_batch_size_per_gpu", message)

    def test_deepep_accepts_a_configured_capacity(self):
        config = parse_autoep_config({
            "enabled": True,
            "comm_backend": "deepep",
            "comm_max_tokens_per_rank": 4096,
        })

        self.validate(config)

    def test_sm_budget_and_qp_margin_are_configurable(self):
        config = parse_autoep_config({"enabled": True, "comm_num_sm": 24, "comm_qp_margin": 8})

        self.assertEqual(config.comm_num_sm, 24)
        self.assertEqual(config.comm_qp_margin, 8)

    def test_rejects_unknown_backend(self):
        # Failing loudly beats silently running the wrong transport.
        config = parse_autoep_config({"enabled": True, "comm_backend": "moonep"})

        with self.assertRaises(ValueError) as caught:
            self.validate(config)
        self.assertIn("moonep", str(caught.exception))

    def test_rejects_a_zero_sm_budget(self):
        # Zero would hand the whole GPU to the collective.
        config = parse_autoep_config({"enabled": True, "comm_num_sm": 0})

        with self.assertRaises(ValueError):
            self.validate(config)

    def test_rejects_a_negative_qp_margin(self):
        config = parse_autoep_config({"enabled": True, "comm_qp_margin": -1})

        with self.assertRaises(ValueError):
            self.validate(config)

    def test_queue_pairs_leave_room_for_the_control_path(self):
        # DeepEP's own default exhausts the QPs that ZeRO and the
        # data-parallel groups have already claimed in a training step.
        self.assertEqual(_qps_for_sms(12, 4), 16)


class TestDtypeGuard(unittest.TestCase):
    """DeepEP's dispatch kernel takes bfloat16 rows and nothing else."""

    def test_rejects_fp16(self):
        with self.assertRaises(TypeError) as caught:
            assert_dtype_supported(torch.float16)
        self.assertIn("bfloat16", str(caught.exception))

    def test_rejects_fp32(self):
        # Not a near miss: the kernel asserts on it, and the buffer is sized in
        # bfloat16 elements, so fp32 rows would not fit the capacity either.
        with self.assertRaises(TypeError) as caught:
            assert_dtype_supported(torch.float32)
        self.assertIn("bfloat16", str(caught.exception))

    def test_accepts_bfloat16(self):
        self.assertIsNone(assert_dtype_supported(torch.bfloat16))

    def test_only_bfloat16_is_in_the_supported_set(self):
        self.assertEqual(SUPPORTED_DTYPES, (torch.bfloat16, ))


class TestTeardownScope(unittest.TestCase):
    """Teardown belongs to one engine's module, not to the whole process."""

    def test_only_the_given_module_tree_is_released(self):
        # Several engines can exist at once, and buffers are built with
        # explicitly_destroy: freeing another engine's would leave it dispatching
        # into memory DeepEP has already reclaimed.
        mine, theirs = mock.Mock(), mock.Mock()
        owned, foreign = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        owned._deepep_exchange = mine
        foreign._deepep_exchange = theirs

        destroy_exchanges(torch.nn.Sequential(owned))

        mine.destroy.assert_called_once_with()
        theirs.destroy.assert_not_called()
        self.assertIsNone(owned._deepep_exchange)
        self.assertIs(foreign._deepep_exchange, theirs)

    def test_a_module_without_layers_is_a_no_op(self):
        destroy_exchanges(torch.nn.Linear(2, 2))


class TestDeepEPPreflight(unittest.TestCase):
    """Opting in on an unsuitable machine should say what is missing."""

    def test_missing_package_names_its_requirements(self):
        with mock.patch.dict(sys.modules, {"deep_ep": None}):
            with self.assertRaises(ImportError) as caught:
                _import_deep_ep()
        message = str(caught.exception)
        self.assertIn("2.30.4", message)
        self.assertIn("comm_backend", message)

    def test_old_torch_nccl_warns_but_does_not_block(self):
        # torch reports the NCCL it bundles, which DeepEP need not be using;
        # DeepEP has been measured working while torch reported 2.28.9, so this
        # signal must not stop a run.
        module = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"deep_ep": module}):
            with mock.patch("deepspeed.module_inject.auto_ep_comm._nccl_version", return_value=(2, 28, 9)):
                with mock.patch("deepspeed.module_inject.auto_ep_comm.logger") as log:
                    self.assertIs(_import_deep_ep(), module)
        message = log.warning.call_args[0][0]
        self.assertIn("2.28.9", message)
        self.assertIn("2.30.4", message)

    def test_new_enough_nccl_passes(self):
        module = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"deep_ep": module}):
            with mock.patch("deepspeed.module_inject.auto_ep_comm._nccl_version", return_value=(2, 30, 4)):
                self.assertIs(_import_deep_ep(), module)

    def test_unknown_nccl_version_does_not_block(self):
        # Failing to detect a version is not evidence of an unusable one.
        module = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"deep_ep": module}):
            with mock.patch("deepspeed.module_inject.auto_ep_comm._nccl_version", return_value=None):
                self.assertIs(_import_deep_ep(), module)


class TestGradientConformance(unittest.TestCase):
    """Autograd checks a gradient against the exact input it belongs to."""

    def test_trims_a_longer_buffer(self):
        # DeepEP returns buffers sized for the worst case; the rows past the
        # ones that carried tokens hold no gradient.
        grad = torch.ones((10, 4))

        conformed = _conform_rows(grad, (6, 4))

        self.assertEqual(tuple(conformed.shape), (6, 4))
        self.assertTrue(torch.equal(conformed, torch.ones((6, 4))))

    def test_extends_a_shorter_buffer_with_zeros(self):
        grad = torch.ones((3, 4))

        conformed = _conform_rows(grad, (5, 4))

        self.assertEqual(tuple(conformed.shape), (5, 4))
        self.assertTrue(torch.equal(conformed[:3], torch.ones((3, 4))))
        self.assertTrue(torch.equal(conformed[3:], torch.zeros((2, 4))))

    def test_matching_shape_is_passed_through_untouched(self):
        grad = torch.randn((7, 4))

        self.assertIs(_conform_rows(grad, (7, 4)), grad)

    def test_conforms_a_one_dimensional_weight_buffer(self):
        # Routing weights arrive one per row rather than one per hidden unit,
        # so conforming has to work without a trailing dimension.
        grad = torch.ones(9)

        self.assertEqual(tuple(_conform_rows(grad, (4, )).shape), (4, ))
        self.assertEqual(tuple(_conform_rows(grad, (12, )).shape), (12, ))


class TestRoutingWeightsAreApplied(unittest.TestCase):
    """The layer must apply the routing weights itself.

    DeepEP's combine does not multiply rows by the topk_weights it is given.
    It transports and reduces them separately and returns them as a second
    output, so handing the weights to combine drops them from the result: the
    expert outputs come back summed but unweighted, which trains quietly and
    wrongly.
    """

    BUFFER_ROWS = 32
    ARRIVED_ROWS = 12
    HIDDEN = 8
    LOCAL_EXPERTS = 4
    WEIGHT = 0.25

    def route(self, score_apply):
        """Drive _deepep_route with a stub exchange, recording what each stage saw."""
        prefix = torch.tensor([3, 6, 9, self.ARRIVED_ROWS], dtype=torch.int64)
        handle = mock.Mock(psum_num_recv_tokens_per_expert=prefix, num_expanded_tokens=self.ARRIVED_ROWS)
        exchange = mock.Mock(last_handle=handle, num_max_tokens_per_rank=1024)

        buffer_rows = torch.ones((self.BUFFER_ROWS, self.HIDDEN))
        buffer_weights = torch.full((self.BUFFER_ROWS, ), self.WEIGHT)
        seen = {}

        def fake_dispatch(_exchange, *_args):
            return buffer_rows, buffer_weights, exchange

        def fake_combine(_exchange, rows, _handle, **kwargs):
            seen["combine_rows"] = rows
            seen["combine_kwargs"] = kwargs
            return rows

        def fake_experts(rows, counts):
            seen["expert_input"] = rows
            seen["counts"] = counts
            return rows

        layer = mock.Mock(
            _deepep_exchange=exchange,
            num_local_experts=self.LOCAL_EXPERTS,
            score_apply=score_apply,
            comm_num_sm=12,
            comm_qp_margin=4,
            experts=fake_experts,
        )

        with mock.patch.object(auto_ep_layer, "deepep_dispatch", fake_dispatch), \
                mock.patch.object(auto_ep_layer, "deepep_combine", fake_combine), \
                mock.patch.object(auto_ep_layer.dist, "all_reduce", lambda *a, **k: None):
            router_output = auto_ep_layer.RouterOutput(
                top_scores=torch.ones((4, 2)),
                selected_experts=torch.zeros((4, 2), dtype=torch.long),
                num_tokens_per_expert=torch.zeros(self.LOCAL_EXPERTS, dtype=torch.long),
            )
            tokens = torch.ones((4, self.HIDDEN), dtype=torch.bfloat16)
            seen["result"] = auto_ep_layer.AutoEPMoELayer._deepep_route(layer, tokens, router_output)
        return seen

    def test_combine_is_never_given_the_weights(self):
        for score_apply in ("pre", "post"):
            with self.subTest(score_apply=score_apply):
                seen = self.route(score_apply)

                self.assertNotIn("topk_weights", seen["combine_kwargs"])

    def test_post_mode_weights_the_expert_output(self):
        seen = self.route("post")

        # The experts saw unscaled rows, and the scaling landed after them.
        self.assertTrue(torch.allclose(seen["expert_input"].float(), torch.ones(1)))
        self.assertTrue(torch.allclose(seen["combine_rows"].float(), torch.full((1, ), self.WEIGHT)))

    def test_pre_mode_weights_the_expert_input(self):
        seen = self.route("pre")

        # Scaling landed before the experts, and is not applied a second time.
        self.assertTrue(torch.allclose(seen["expert_input"].float(), torch.full((1, ), self.WEIGHT)))
        self.assertTrue(torch.allclose(seen["combine_rows"].float(), torch.full((1, ), self.WEIGHT)))

    def test_combine_receives_exactly_the_rows_that_arrived(self):
        # Dispatch returns a worst-case buffer while combine reads the rows the
        # handle recorded. Handing it the whole buffer reads past what the
        # handle describes, which does not raise: it faults.
        seen = self.route("post")

        self.assertEqual(seen["expert_input"].shape[0], self.ARRIVED_ROWS)
        self.assertEqual(seen["combine_rows"].shape[0], self.ARRIVED_ROWS)
        self.assertTrue(torch.equal(seen["counts"], torch.tensor([3, 3, 3, 3], dtype=torch.int32)))

    def test_row_count_comes_from_the_handle_not_the_device(self):
        # Reading it off the prefix sum needs a device-to-host synchronisation
        # in front of every layer's GEMM; the handle already holds it as an int.
        source = inspect.getsource(auto_ep_layer.AutoEPMoELayer._deepep_route)

        self.assertIn("arrived = handle.num_expanded_tokens", source)
        self.assertNotIn("psum_num_recv_tokens_per_expert[-1]", source)


class TestBufferLifecycle(unittest.TestCase):
    """The statically sized buffer never makes a rank-local resize decision."""

    def test_the_route_does_not_synchronize_capacity(self):
        source = inspect.getsource(auto_ep_layer.AutoEPMoELayer._deepep_route)

        self.assertNotIn("all_reduce", source)
        self.assertNotIn("_agree_deepep_capacity", source)

    def test_outgrowing_the_buffer_names_the_remedy(self):
        exchange = mock.Mock(num_max_tokens_per_rank=512)
        layer = mock.Mock(_deepep_exchange=exchange)

        with mock.patch.object(auto_ep_layer.dist, "all_reduce", lambda *a, **k: None):
            router_output = auto_ep_layer.RouterOutput(
                top_scores=torch.ones((600, 1)),
                selected_experts=torch.zeros((600, 1), dtype=torch.long),
                num_tokens_per_expert=torch.zeros(4, dtype=torch.long),
            )
            with self.assertRaises(RuntimeError) as caught:
                auto_ep_layer.AutoEPMoELayer._deepep_route(layer, torch.ones((600, 8), dtype=torch.bfloat16),
                                                           router_output)

        message = str(caught.exception)
        self.assertIn("comm_max_tokens_per_rank", message)
        self.assertIn("600", message)

    def test_the_configured_capacity_sizes_the_buffer(self):
        layer = mock.Mock(_deepep_exchange=None, comm_max_tokens_per_rank=4096, comm_num_sm=12, comm_qp_margin=4)

        built = mock.Mock(return_value=mock.Mock(num_max_tokens_per_rank=4096))
        with mock.patch.object(auto_ep_layer, "DeepEPExchange", built), \
                mock.patch.object(auto_ep_layer, "deepep_dispatch", side_effect=RuntimeError("stop here")), \
                mock.patch.object(auto_ep_layer.dist, "all_reduce", lambda *a, **k: None):
            router_output = auto_ep_layer.RouterOutput(
                top_scores=torch.ones((8, 1)),
                selected_experts=torch.zeros((8, 1), dtype=torch.long),
                num_tokens_per_expert=torch.zeros(4, dtype=torch.long),
            )
            with self.assertRaises(RuntimeError):
                auto_ep_layer.AutoEPMoELayer._deepep_route(layer, torch.ones((8, 8), dtype=torch.bfloat16),
                                                           router_output)

        self.assertEqual(built.call_args.kwargs["num_max_tokens_per_rank"], 4096)


class TestAutogradSignatures(unittest.TestCase):
    """Both directions must return a gradient for every differentiable input.

    A missing router-weight gradient does not fail loudly: training runs, the
    loss falls, and the gate simply never learns. These check the arity that
    carries it rather than leaving it to a live run to reveal.
    """

    @staticmethod
    def gradient_count(function) -> int:
        """How many values the backward returns, parsed rather than counted.

        Counting commas in the source would also count the ones inside calls
        like ``_conform_rows(grad, shape)``.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
        assert returns, "backward has no return statement"
        value = returns[-1].value
        return len(value.elts) if isinstance(value, ast.Tuple) else 1

    def test_dispatch_backward_returns_a_gradient_per_input(self):
        # ctx is not an input autograd returns a gradient for.
        inputs = len(inspect.signature(_DeepEPDispatch.forward).parameters) - 1

        self.assertEqual(inputs, 4)
        self.assertEqual(self.gradient_count(_DeepEPDispatch.backward), inputs)

    def test_combine_backward_returns_a_gradient_per_input(self):
        # Combine takes no weights: DeepEP would not apply them, so the layer
        # folds them into the rows before calling it.
        inputs = len(inspect.signature(_DeepEPCombine.forward).parameters) - 1

        self.assertEqual(inputs, 3)
        self.assertEqual(self.gradient_count(_DeepEPCombine.backward), inputs)

    def test_dispatch_forward_returns_weights_so_they_stay_differentiable(self):
        # The received weights have to leave the custom function as an output;
        # reading them off the exchange afterwards puts them outside the graph.
        source = inspect.getsource(_DeepEPDispatch.forward)

        self.assertIn("return received, recv_weights", source)


if __name__ == "__main__":
    unittest.main()
