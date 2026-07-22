# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest

from deepspeed.compile.passes import contract as contract_mod
from deepspeed.compile.passes.contract import (PassContract, PassContractError, register_pass_contract,
                                               get_pass_contract, validate_schedule)


@pytest.fixture
def clean_registry():
    # validate_schedule reads a module-level registry, so isolate each test from the built-in
    # contracts and from one another by snapshotting and restoring it.
    saved = dict(contract_mod._pass_contracts)
    contract_mod._pass_contracts.clear()
    yield
    contract_mod._pass_contracts.clear()
    contract_mod._pass_contracts.update(saved)


def _register_zero3_and_prefetch():
    register_pass_contract("zero3", PassContract(provides=frozenset({"z3"})))
    register_pass_contract("prefetch", PassContract(requires=frozenset({"z3"})))


def test_register_and_get(clean_registry):
    contract = PassContract(provides=frozenset({"z3"}))
    register_pass_contract("zero3", contract)
    assert get_pass_contract("zero3") is contract
    assert get_pass_contract("missing") is None


def test_none_contract_clears_previous(clean_registry):
    # Re-registering a pass without a contract must drop the stale one, not keep it.
    register_pass_contract("prefetch", PassContract(requires=frozenset({"z3"})))
    assert get_pass_contract("prefetch") is not None
    register_pass_contract("prefetch", None)
    assert get_pass_contract("prefetch") is None
    # With the contract cleared, the pass is unconstrained again.
    validate_schedule([(0, ["prefetch"])])


def test_valid_order_passes(clean_registry):
    _register_zero3_and_prefetch()
    validate_schedule([(0, ["zero3"]), (10, ["zero3", "prefetch"])])


def test_missing_requirement_raises(clean_registry):
    _register_zero3_and_prefetch()
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, ["prefetch"])])


def test_requirement_within_same_step(clean_registry):
    # Passes in the same step run left to right, so a provider earlier in the list satisfies a
    # later consumer.
    _register_zero3_and_prefetch()
    validate_schedule([(0, ["zero3", "prefetch"])])


def test_requirement_does_not_carry_across_steps(clean_registry):
    # DeepCompile recompiles from the original graph at each step, so a provider in an earlier
    # step does not satisfy a consumer in a later step; the requirement must be met per step.
    _register_zero3_and_prefetch()
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, ["zero3"]), (10, ["prefetch"])])


def test_conflict_is_symmetric(clean_registry):
    register_pass_contract("a", PassContract(conflicts_with=frozenset({"b"})))
    register_pass_contract("b", PassContract())
    # "b" declares nothing, but "a" lists it as a conflict; either ordering must be rejected.
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["a", "b"])])
    with pytest.raises(PassContractError, match="conflicts"):
        validate_schedule([(0, ["b", "a"])])


def test_uncontracted_passes_are_skipped(clean_registry):
    _register_zero3_and_prefetch()
    # An ad-hoc pass with no registered contract must not break validation of the rest.
    validate_schedule([(0, ["zero3", "ad_hoc_pass", "prefetch"])])


def test_callables_resolved_by_identity(clean_registry):
    _register_zero3_and_prefetch()

    def zero3_fn():
        pass

    def prefetch_fn():
        pass

    name_registry = {"zero3": zero3_fn, "prefetch": prefetch_fn}
    validate_schedule([(0, [zero3_fn]), (10, [zero3_fn, prefetch_fn])], name_registry)
    with pytest.raises(PassContractError, match="requires"):
        validate_schedule([(0, [prefetch_fn])], name_registry)


def test_ambiguous_callable_requires_explicit_name(clean_registry):
    register_pass_contract("zero3", PassContract(provides=frozenset({"z3"})))

    def shared_fn():
        pass

    # The same callable registered under two names cannot be resolved to a single contract.
    name_registry = {"zero3": shared_fn, "zero3_alias": shared_fn}
    with pytest.raises(PassContractError, match="multiple names"):
        validate_schedule([(0, [shared_fn])], name_registry)
