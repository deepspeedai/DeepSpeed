# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import importlib.util
import os
import sys
import unittest

# Load module directly
file_path = os.path.join(
    os.path.dirname(__file__),
    "../../../deepspeed/runtime/production_debt.py",
)
spec = importlib.util.spec_from_file_location("deepspeed_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["deepspeed_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtZeROGate = production_debt_mod.ProductionDebtZeROGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtZeROGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtZeROGate(
            never_equate_intent_to_approval=True,
            max_acceptable_ddi=12.0,
        )

    def test_clean_training_step_passes_readiness(self) -> None:
        report = self.gate.evaluate_training_step(
            run_id="deepspeed_zero3_llama_70b_cluster",
            allocated_gpu_memory_bytes=70000000000,
            peak_offload_memory_bytes=72500000000,
            step_latency_seconds=0.38,
            communication_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.ddi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_training_step_fails_debt(self) -> None:
        report = self.gate.evaluate_training_step(
            run_id="uncalibrated_zero_offload_run",
            allocated_gpu_memory_bytes=70000000000,
            peak_offload_memory_bytes=190000000000,  # 2.71x offload memory sprawl
            step_latency_seconds=3.5,  # High step latency
            communication_stalls=3,  # 3 All-Gather communication stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.ddi_score, 50.0)
        self.assertIn("HIGH_ZERO_OFFLOAD_MEMORY_SPRAWL_2.71X", report.critical_smells)
        self.assertIn("HIGH_DISTRIBUTED_STEP_LATENCY_3.50S", report.critical_smells)
        self.assertIn("DETECTED_3_ALLGATHER_COMMUNICATION_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_CHECKPOINT_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_training_step("run-1")
        self.gate.evaluate_training_step("run-2")
        self.gate.evaluate_training_step("run-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
