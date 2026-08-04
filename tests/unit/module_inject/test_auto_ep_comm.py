# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import ast
import inspect
import os
import textwrap
import sys
import unittest

import torch
from unittest import mock

from deepspeed.module_inject.auto_ep_comm import (AVAILABLE_BACKENDS, DEEPEP_BACKEND, NCCL_BACKEND, _conform_rows,
                                                  _DeepEPCombine, _DeepEPDispatch, _import_deep_ep, configured_backend,
                                                  configured_num_sms)


class TestAutoEPCommBackendSelection(unittest.TestCase):

    def test_defaults_to_nccl(self):
        # An unset variable must leave existing jobs on the shipped path.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(configured_backend(), NCCL_BACKEND)
            self.assertEqual(configured_num_sms(), 0)

    def test_selects_deepep(self):
        with mock.patch.dict(os.environ, {"DEEPSPEED_AUTOEP_COMM_BACKEND": "DeepEP"}):
            self.assertEqual(configured_backend(), DEEPEP_BACKEND)

    def test_rejects_unknown_backend(self):
        # Failing loudly beats silently running the wrong transport.
        with mock.patch.dict(os.environ, {"DEEPSPEED_AUTOEP_COMM_BACKEND": "moonep"}):
            with self.assertRaises(ValueError) as caught:
                configured_backend()
            self.assertIn(str(AVAILABLE_BACKENDS), str(caught.exception))

    def test_sm_budget_is_configurable(self):
        with mock.patch.dict(os.environ, {"DEEPSPEED_AUTOEP_COMM_SMS": "24"}):
            self.assertEqual(configured_num_sms(), 24)

    def test_blank_value_is_treated_as_unset(self):
        # An empty variable is what an unset shell variable expands to.
        with mock.patch.dict(os.environ, {"DEEPSPEED_AUTOEP_COMM_BACKEND": "  "}):
            self.assertEqual(configured_backend(), NCCL_BACKEND)
        with mock.patch.dict(os.environ, {"DEEPSPEED_AUTOEP_COMM_SMS": ""}):
            self.assertEqual(configured_num_sms(), 0)

    def test_non_numeric_sm_budget_is_rejected(self):
        with mock.patch.dict(os.environ, {"DEEPSPEED_AUTOEP_COMM_SMS": "many"}):
            with self.assertRaises(ValueError):
                configured_num_sms()


class TestDeepEPPreflight(unittest.TestCase):
    """Opting in on an unsuitable machine should say what is missing."""

    def test_missing_package_names_its_requirements(self):
        with mock.patch.dict(sys.modules, {"deep_ep": None}):
            with self.assertRaises(ImportError) as caught:
                _import_deep_ep()
        message = str(caught.exception)
        self.assertIn("2.30.4", message)
        self.assertIn("DEEPSPEED_AUTOEP_COMM_BACKEND", message)

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
        inputs = len(inspect.signature(_DeepEPCombine.forward).parameters) - 1

        self.assertEqual(inputs, 4)
        self.assertEqual(self.gradient_count(_DeepEPCombine.backward), inputs)

    def test_dispatch_forward_returns_weights_so_they_stay_differentiable(self):
        # The received weights have to leave the custom function as an output;
        # reading them off the exchange afterwards puts them outside the graph.
        source = inspect.getsource(_DeepEPDispatch.forward)

        self.assertIn("return received, recv_weights", source)


if __name__ == "__main__":
    unittest.main()
