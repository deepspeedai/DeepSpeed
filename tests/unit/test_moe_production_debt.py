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
    "../../deepspeed/moe/production_debt.py",
)
spec = importlib.util.spec_from_file_location("deepspeed_moe_production_debt", file_path)
production_debt_mod = importlib.util.module_from_spec(spec)
sys.modules["deepspeed_moe_production_debt"] = production_debt_mod
spec.loader.exec_module(production_debt_mod)

ProductionDebtMoEGate = production_debt_mod.ProductionDebtMoEGate
TechnicalDueDiligenceLedger = production_debt_mod.TechnicalDueDiligenceLedger
GENESIS_HASH = production_debt_mod.GENESIS_HASH


class TestProductionDebtMoEGate(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = ProductionDebtMoEGate(
            never_equate_intent_to_approval=True,
            max_acceptable_mdi=12.0,
        )

    def test_clean_moe_step_passes_readiness(self) -> None:
        report = self.gate.evaluate_moe_step(
            moe_step_id="deepspeed_moe_deepseek_v3_step",
            allocated_expert_bytes=16000000000,
            utilized_expert_bytes=16800000000,
            dispatch_latency_ms=2.8,
            all_to_all_comm_stalls=0,
            un_gated_mutations=0,
        )
        self.assertTrue(report.is_production_ready)
        self.assertLessEqual(report.mdi_score, 12.0)
        self.assertEqual(len(report.critical_smells), 0)
        self.assertTrue(bool(report.receipt_hash))

    def test_degraded_moe_step_fails_debt(self) -> None:
        report = self.gate.evaluate_moe_step(
            moe_step_id="uncalibrated_moe_step",
            allocated_expert_bytes=16000000000,
            utilized_expert_bytes=45000000000,  # 2.81x expert capacity imbalance sprawl
            dispatch_latency_ms=35.0,  # High all-to-all dispatch latency
            all_to_all_comm_stalls=3,  # 3 all-to-all communication stalls
            un_gated_mutations=2,  # 2 un-gated mutations
        )
        self.assertFalse(report.is_production_ready)
        self.assertGreater(report.mdi_score, 50.0)
        self.assertIn("HIGH_EXPERT_CAPACITY_IMBALANCE_2.81X", report.critical_smells)
        self.assertIn("HIGH_ALLTOALL_DISPATCH_LATENCY_35.0MS", report.critical_smells)
        self.assertIn("DETECTED_3_ALL_TO_ALL_COMM_STALLS", report.critical_smells)
        self.assertIn("DETECTED_2_UNGATED_TOPK_ROUTING_MUTATIONS", report.critical_smells)

    def test_cryptographic_ledger_integrity(self) -> None:
        self.gate.evaluate_moe_step("step-1")
        self.gate.evaluate_moe_step("step-2")
        self.gate.evaluate_moe_step("step-3")

        entries = self.gate.ledger.get_ledger_entries()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["prev_hash"], GENESIS_HASH)
        self.assertEqual(entries[1]["prev_hash"], entries[0]["curr_hash"])
        self.assertEqual(entries[2]["prev_hash"], entries[1]["curr_hash"])
        self.assertTrue(self.gate.ledger.verify_ledger_integrity())


if __name__ == "__main__":
    unittest.main()
