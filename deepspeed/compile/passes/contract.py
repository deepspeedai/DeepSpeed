# Copyright (c) DeepSpeed Team.
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

    A contract does not carry the pass name: the registry key set by ``register_compile_pass`` is
    the single source of truth for a pass's identity.
    """

    provides: FrozenSet[str] = frozenset()
    requires: FrozenSet[str] = frozenset()
    conflicts_with: FrozenSet[str] = frozenset()
    phase: str = "both"


class PassContractError(ValueError):
    """Raised when a pass schedule violates the registered pass contracts."""


_pass_contracts: Dict[str, PassContract] = {}


def register_pass_contract(name: str, contract: Optional[PassContract]) -> None:
    # ``contract=None`` clears any contract previously registered under ``name`` so a pass that is
    # re-registered through the two-argument API does not keep a stale contract.
    if contract is None:
        _pass_contracts.pop(name, None)
    else:
        _pass_contracts[name] = contract


def get_pass_contract(name: str) -> Optional[PassContract]:
    return _pass_contracts.get(name)


def _resolve_pass_name(pass_ref, name_registry: Optional[Dict]) -> Optional[str]:
    # Schedules usually reference passes by their registered name, but a user may also drop a raw
    # callable into a schedule. Contracts are keyed by name, so resolve a callable back to its
    # name by object identity (avoiding any reliance on the callable being hashable or comparable).
    if isinstance(pass_ref, str):
        return pass_ref
    if name_registry is None:
        return None
    matches = [name for name, fn in name_registry.items() if fn is pass_ref]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise PassContractError(f"Pass callable is registered under multiple names {sorted(matches)}; reference it by "
                                f"name in the schedule so the intended contract can be selected.")
    return None


def validate_schedule(schedule: List[Tuple[int, List]], name_registry: Optional[Dict] = None) -> None:
    """Validate that a DeepCompile pass schedule satisfies the registered pass contracts.

    ``schedule`` uses the ``[(step, passes), ...]`` format consumed by ``init_schedule``. Validate
    it before names are converted to callables so the pass identity the user selected is preserved.
    Each entry in ``passes`` is normally a registered pass name; a raw callable is resolved back to
    its name through ``name_registry`` (the ``{name: fn}`` registry) by object identity, and a
    callable registered under more than one name must instead be referenced by name. Passes with no
    registered contract are treated as unconstrained and skipped, so mixed schedules of contracted
    and ad-hoc passes remain valid. Raises :class:`PassContractError` on the first unmet requirement
    or conflict.

    Each step is validated independently: DeepCompile resets Dynamo and recompiles from the
    original graph at every launched step (see ``launch_compile_passes``), so capabilities a pass
    provides in one step do not carry over to later steps. A pass must therefore find every
    capability it requires among the passes scheduled earlier within the same step.
    """
    for step, passes in schedule:
        provided: set = set()
        applied: List[str] = []

        for pass_ref in passes:
            name = _resolve_pass_name(pass_ref, name_registry)
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
