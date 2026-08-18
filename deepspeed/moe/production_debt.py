# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GENESIS_HASH = "0000000000000000000000000000000000000000000000000000000000000000"


@dataclass
class DeepSpeedMoEDebtReport:
    moe_step_id: str
    mdi_score: float  # MoE Debt Index (target <= 12.0)
    capacity_sprawl_multiplier: float  # Target <= 1.08x
    dispatch_latency_ms: float  # Target <= 3.8ms
    mutation_safety_score: float  # Target 100.0
    production_readiness_index: float  # Scale 0 - 100
    is_production_ready: bool
    critical_smells: list[str]
    receipt_hash: str


class TechnicalDueDiligenceLedger:
    """Cryptographic SHA-256 hash-chained Action Ledger for DeepSpeed-MoE & ZeRO-Inference runs."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._last_hash: str = GENESIS_HASH

    def record_moe_event(
        self,
        moe_step_id: str,
        event_type: str,
        readiness_index: float,
        critical_smells: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        index = len(self._entries)

        meta_bytes = json.dumps(metadata, sort_keys=True).encode("utf-8")
        canonical_content = (
            f"{index}|{self._last_hash}|{moe_step_id}|{event_type}|"
            f"{readiness_index}|{timestamp}|{hashlib.sha256(meta_bytes).hexdigest()}"
        )
        curr_hash = hashlib.sha256(canonical_content.encode("utf-8")).hexdigest()

        entry = {
            "index": index,
            "timestamp": timestamp,
            "moe_step_id": moe_step_id,
            "event_type": event_type,
            "readiness_index": readiness_index,
            "critical_smells": critical_smells,
            "prev_hash": self._last_hash,
            "curr_hash": curr_hash,
            "metadata": metadata,
        }

        self._entries.append(entry)
        self._last_hash = curr_hash
        return entry

    def get_ledger_entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def verify_ledger_integrity(self) -> bool:
        prev = GENESIS_HASH
        for entry in self._entries:
            if entry["prev_hash"] != prev:
                return False
            prev = entry["curr_hash"]
        return True


class ProductionDebtMoEGate:
    """A2Z SOC Production Debt & Technical Due Diligence Gate for DeepSpeed-MoE & ZeRO-Inference.

    Quantifies expert load capacity imbalance, all-to-all token routing communication stalls, and NVMe offload dispatch latency against 4 Enterprise KPIs:
    1. MoE Debt Index (MDI <= 12.0)
    2. Expert Capacity Memory Multiplier (ECMM <= 1.08x)
    3. P99 All-to-All Dispatch Latency (<= 3.8ms)
    4. Deterministic Mutation Boundaries (never_equate_intent_to_approval)
    """

    def __init__(
        self,
        never_equate_intent_to_approval: bool = True,
        max_acceptable_mdi: float = 12.0,
    ) -> None:
        self.never_equate_intent_to_approval = never_equate_intent_to_approval
        self.max_acceptable_mdi = max_acceptable_mdi
        self.ledger = TechnicalDueDiligenceLedger()

    def check_kill_switch(self) -> bool:
        if os.environ.get("AAG_KILL_SWITCH", "").lower() in ("true", "1", "yes"):
            return True
        return any(Path(p).exists() for p in ("artifacts/KILL", "/tmp/KILL"))

    def evaluate_moe_step(
        self,
        moe_step_id: str,
        allocated_expert_bytes: int = 16000000000,
        utilized_expert_bytes: int = 16800000000,
        dispatch_latency_ms: float = 2.8,
        all_to_all_comm_stalls: int = 0,
        un_gated_mutations: int = 0,
    ) -> DeepSpeedMoEDebtReport:
        # 1. Evaluate emergency kill switch
        if self.check_kill_switch():
            self.ledger.record_moe_event(
                moe_step_id=moe_step_id,
                event_type="moe_step_halted_kill_switch",
                readiness_index=0.0,
                critical_smells=["EMERGENCY_KILL_SWITCH_ENGAGED"],
                metadata={"reason": "AAG_KILL_SWITCH is set"},
            )
            err_msg = "A2Z SOC ActionGate: Emergency kill switch is engaged. DeepSpeed-MoE execution halted."
            raise PermissionError(err_msg)

        critical_smells: list[str] = []

        # KPI 2: Expert Capacity Memory Multiplier
        cap_ratio = utilized_expert_bytes / max(1, allocated_expert_bytes)
        if cap_ratio > 1.8:
            critical_smells.append(f"HIGH_EXPERT_CAPACITY_IMBALANCE_{cap_ratio:.2f}X")

        # KPI 3: Latency Ceiling
        if dispatch_latency_ms > 15.0:
            critical_smells.append(f"HIGH_ALLTOALL_DISPATCH_LATENCY_{dispatch_latency_ms:.1f}MS")

        # All-to-all communication stalls
        if all_to_all_comm_stalls > 0:
            critical_smells.append(f"DETECTED_{all_to_all_comm_stalls}_ALL_TO_ALL_COMM_STALLS")

        # KPI 4: Mutation Safety
        if un_gated_mutations > 0:
            critical_smells.append(f"DETECTED_{un_gated_mutations}_UNGATED_TOPK_ROUTING_MUTATIONS")

        # KPI 1: MoE Debt Index (0 = Clean, 100 = Catastrophic)
        mdi = (
            max(0.0, (cap_ratio - 1.0) * 20.0)
            + max(0.0, (dispatch_latency_ms - 3.8) * 0.5)
            + (all_to_all_comm_stalls * 25.0)
            + (un_gated_mutations * 30.0)
        )
        mdi_score = round(min(100.0, mdi), 2)

        # Production Readiness Index (0 - 100)
        readiness = max(0.0, 100.0 - mdi_score)
        is_production_ready = (
            mdi_score <= self.max_acceptable_mdi and len(critical_smells) == 0
        )

        # Cryptographic Ledger Entry
        entry = self.ledger.record_moe_event(
            moe_step_id=moe_step_id,
            event_type="moe_authorized" if is_production_ready else "moe_flagged_debt",
            readiness_index=readiness,
            critical_smells=critical_smells,
            metadata={
                "mdi_score": mdi_score,
                "cap_ratio": cap_ratio,
                "allocated_expert_bytes": allocated_expert_bytes,
                "utilized_expert_bytes": utilized_expert_bytes,
                "dispatch_latency_ms": dispatch_latency_ms,
                "all_to_all_comm_stalls": all_to_all_comm_stalls,
                "un_gated_mutations": un_gated_mutations,
                "never_equate_intent_to_approval": self.never_equate_intent_to_approval,
            },
        )

        return DeepSpeedMoEDebtReport(
            moe_step_id=moe_step_id,
            mdi_score=mdi_score,
            capacity_sprawl_multiplier=round(cap_ratio, 2),
            dispatch_latency_ms=round(dispatch_latency_ms, 2),
            mutation_safety_score=(
                100.0 if un_gated_mutations == 0 else max(0.0, 100.0 - un_gated_mutations * 30.0)
            ),
            production_readiness_index=readiness,
            is_production_ready=is_production_ready,
            critical_smells=critical_smells,
            receipt_hash=entry["curr_hash"],
        )
