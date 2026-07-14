# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

# Capability tags produced and consumed by the built-in DeepCompile passes. Keeping the tags
# in one place lets passes declare dependencies on each other without hard-coding pass names.
CAP_Z3_GATHER_RELEASE = "z3_gather_release"


@dataclass(frozen=True)
class PassContract:
    """Lightweight metadata describing what an optimization pass expects and produces.

    Contracts let DeepCompile validate a pass schedule before it runs. A pass may only appear
    after every capability it ``requires`` has been ``provides``-d by an earlier pass, and two
    passes that name each other in ``conflicts_with`` may not share a schedule. ``phase`` is
    informational for now and records whether a pass rewrites the forward graph, the backward
    graph, or both.
    """

    name: str
    provides: FrozenSet[str] = frozenset()
    requires: FrozenSet[str] = frozenset()
    conflicts_with: FrozenSet[str] = frozenset()
    phase: str = "both"


class PassContractError(ValueError):
    """Raised when a pass schedule violates the registered pass contracts."""


_pass_contracts: Dict[str, PassContract] = {}


def register_pass_contract(contract: PassContract) -> None:
    _pass_contracts[contract.name] = contract


def get_pass_contract(name: str) -> Optional[PassContract]:
    return _pass_contracts.get(name)


def _resolve_pass_name(pass_ref, fn_to_name: Optional[Dict]) -> Optional[str]:
    # Schedules may reference a pass either by its registered name or by its callable. We only
    # know how to look up a contract by name, so translate callables back to their name here.
    if isinstance(pass_ref, str):
        return pass_ref
    if fn_to_name is not None:
        return fn_to_name.get(pass_ref)
    return None


def validate_schedule(schedule: List[Tuple[int, List]], fn_to_name: Optional[Dict] = None) -> None:
    """Validate that a DeepCompile pass schedule satisfies the registered pass contracts.

    ``schedule`` uses the ``[(step, passes), ...]`` format consumed by ``init_schedule``. Each
    entry in ``passes`` may be a registered pass name or a pass callable; pass ``fn_to_name`` to
    resolve callables (typically ``{fn: name for name, fn in opt_passes.items()}``). Passes that
    have no registered contract are treated as unconstrained and skipped, so mixed schedules of
    contracted and ad-hoc passes remain valid. Raises :class:`PassContractError` on the first
    unmet requirement or conflict.

    Each step is validated independently: DeepCompile resets Dynamo and recompiles from the
    original graph at every launched step (see ``launch_compile_passes``), so capabilities a pass
    provides in one step do not carry over to later steps. A pass must therefore find every
    capability it requires among the passes scheduled earlier within the same step.
    """
    for step, passes in schedule:
        provided: set = set()
        applied: List[str] = []

        for pass_ref in passes:
            name = _resolve_pass_name(pass_ref, fn_to_name)
            if name is None:
                continue

            contract = _pass_contracts.get(name)
            if contract is None:
                continue

            missing = contract.requires - provided
            if missing:
                raise PassContractError(f"Pass '{name}' (step {step}) requires {sorted(missing)}, which no earlier "
                                        f"pass provides. Passes scheduled so far: {applied}.")

            # Conflicts are treated symmetrically: either pass may declare the incompatibility.
            conflicts = set(contract.conflicts_with.intersection(applied))
            for prev_name in applied:
                prev_contract = _pass_contracts.get(prev_name)
                if prev_contract is not None and name in prev_contract.conflicts_with:
                    conflicts.add(prev_name)
            if conflicts:
                raise PassContractError(f"Pass '{name}' (step {step}) conflicts with already-scheduled pass(es) "
                                        f"{sorted(conflicts)}.")

            provided |= contract.provides
            applied.append(name)
