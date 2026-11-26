# Copyright (c) DeepSpeed Team.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Check ``affine_transfer_cost`` against bytes over capacity worked out by hand.

Each expected time below is computed in the test from the bytes a schedule moves and the capacities
the model is given, not by calling the code under test. The GH200 case uses the capacities of RFC
#9026, so its direct and sharded estimates are the RFC's own model rows (502 and 367 ms).
"""

import pytest

from deepspeed.checkpoint.affine import contiguous_split_map, replicated_map
from deepspeed.checkpoint.affine_replica_policy import local_first_least_loaded
from deepspeed.checkpoint.affine_transfer import AffineTransferError, plan_transfer, plan_transfers
from deepspeed.checkpoint.affine_transfer_cost import (TransferCostModel, TransferStage, cheapest, estimate_seconds,
                                                       node_route)
from deepspeed.checkpoint.affine_transfer_executor import StagingBudget, TransferEndpoints

GiB = 1 << 30


def _one_sender_two_receivers():
    # 16 float32 elements to each of two receivers on another node, 8 elements per message: two rounds.
    plan = plan_transfer(replicated_map((4, 4), 2), replicated_map((4, 4), 1))
    return TransferStage([plan], TransferEndpoints({0: 0}, {0: 1, 1: 2}), 4, StagingBudget(32))


def test_a_round_takes_its_busiest_resource():
    stage = _one_sender_two_receivers()
    route = node_route({0: 0, 1: 1, 2: 1})
    # Each round, the sender sends 2 x 32 bytes and each receiver receives 32 bytes.
    receive_bound = {'network_send': 100.0, 'network_receive': 40.0}
    model = TransferCostModel(route, receive_bound, seconds_per_call=2.0, seconds_per_round=0.5)
    assert estimate_seconds([stage], model) == pytest.approx(2.0 + 2 * (0.5 + 32 / 40))
    send_bound = {'network_send': 50.0, 'network_receive': 1000.0}
    model = TransferCostModel(route, send_bound, seconds_per_call=2.0, seconds_per_round=0.5)
    assert estimate_seconds([stage], model) == pytest.approx(2.0 + 2 * (0.5 + 64 / 50))


def test_resources_bind_on_their_own():
    # Process 2 receives one row from process 0 on the other node and one from process 3 on its own,
    # in the same round. The two arrive over different resources, so the round takes the slower one,
    # not the sum: the mixed incast of RFC #9026 (45.8 ms predicted, 46.0 ms measured on GH200).
    plan = plan_transfer(replicated_map((2, 8), 1), contiguous_split_map((2, 8), [1, 1], 0))
    stage = TransferStage([plan], TransferEndpoints({0: 0, 1: 3}, {0: 2}), 4)
    route = node_route({0: 0, 2: 1, 3: 1}, intra_node=('device_pair', ))
    capacity = {'network_send': 1e9, 'network_receive': 16.0, 'device_pair': 32.0}
    assert estimate_seconds([stage], TransferCostModel(route, capacity)) == pytest.approx(max(32 / 16, 32 / 32))


def test_host_launches_overlap_the_bytes():
    # Process 0 sends each of two strided column blocks to its own receiver, packing each into its
    # arena: two regions. The receivers take theirs in place, so they launch no copy.
    plan = plan_transfer(contiguous_split_map((2, 4), [2, 2], 1), replicated_map((2, 4), 1))
    stage = TransferStage([plan], TransferEndpoints({0: 0}, {0: 1, 1: 2}), 4)
    route = node_route({0: 0, 1: 1, 2: 1})
    capacity = {'network_send': 32.0, 'network_receive': 1e9}  # the sender's 32 bytes take 1 s
    for seconds_per_region in (0.1, 2.0):
        model = TransferCostModel(route, capacity, seconds_per_call=0.5, seconds_per_region=seconds_per_region)
        assert estimate_seconds([stage], model) == pytest.approx(0.5 + max(1.0, 2 * seconds_per_region))

    # With both target locations on process 0, the two blocks are local copies: two regions, no message.
    local = TransferStage([plan], TransferEndpoints({0: 0}, {0: 0, 1: 0}), 4)
    model = TransferCostModel(route, capacity, seconds_per_call=0.5, seconds_per_region=0.25)
    assert estimate_seconds([local], model) == pytest.approx(0.5 + 2 * 0.25)

    # Every round of this batch is one contiguous region on both sides, sent and received in place.
    in_place = _one_sender_two_receivers()
    model = TransferCostModel(route, {'network_send': 64.0, 'network_receive': 1e9}, seconds_per_region=10.0)
    assert estimate_seconds([in_place], model) == pytest.approx(2 * 64 / 64)


def test_gh200_relay_beats_direct_and_sharded():
    # 8 GiB of bf16 from trainers on node 0 into four engines on node 1, with the GH200 capacities of
    # RFC #9026: 68.4 GB/s network send and 23.4 GB/s network receive per GPU, 52.1 GB/s per NVLink pair.
    shape, element_size, budget = (4, GiB), 2, StagingBudget(64 << 20)
    trainers = {i: i for i in range(4)}
    engines = {i: 4 + i for i in range(4)}
    nodes = {process: process // 4 for process in range(8)}
    capacity = {'network_send': 68.4e9, 'network_receive': 23.4e9, 'device_pair': 52.1e9}
    model = TransferCostModel(node_route(nodes, intra_node=('device_pair', )), capacity)
    every_engine = replicated_map(shape, 4)
    rows = contiguous_split_map(shape, [1] * 4, 0)

    # Direct: every engine reads the whole weight from trainer 0, whose network send carries it four times.
    direct_plan = plan_transfer(every_engine, replicated_map(shape, 1))
    direct = [TransferStage([direct_plan], TransferEndpoints({0: 0}, engines), element_size, budget)]

    # Sharded: each trainer holds one row, so every engine receives the whole weight from four senders.
    sharded_plan = plan_transfer(every_engine, rows)
    sharded = [TransferStage([sharded_plan], TransferEndpoints(trainers, engines), element_size, budget)]

    # Relay: each engine fetches one row from its own trainer replica, then the engines exchange the
    # rows inside node 1. An engine's own row stays on its GPU in stage two, so it moves no bytes.
    fetch_plan = plan_transfers({'w': (rows, replicated_map(shape, 4))},
                                source_processes=trainers,
                                target_processes=engines,
                                element_size={'w': element_size},
                                replica_policy=local_first_least_loaded)['w']
    spread_plan = plan_transfer(every_engine, rows)
    relay = [
        TransferStage([fetch_plan], TransferEndpoints(trainers, engines), element_size, budget),
        TransferStage([spread_plan], TransferEndpoints(engines, engines), element_size, budget),
    ]

    choice = cheapest({'direct': direct, 'sharded': sharded, 'relay': relay}, model)
    weight = 8 * GiB
    assert choice.seconds['direct'] == pytest.approx(4 * weight / 68.4e9)
    assert choice.seconds['sharded'] == pytest.approx(weight / 23.4e9)
    assert choice.seconds['relay'] == pytest.approx(weight / 4 / 23.4e9 + weight / 4 / 52.1e9)
    assert choice.name == 'relay'


def test_a_tie_goes_to_the_first_name():
    stage = _one_sender_two_receivers()
    capacity = {'network_send': 1.0, 'network_receive': 1.0}
    choice = cheapest({'b': [stage], 'a': [stage]}, TransferCostModel(node_route({0: 0, 1: 1, 2: 1}), capacity))
    assert choice.name == 'a'
    assert choice.seconds['a'] == choice.seconds['b']


def test_inputs_it_cannot_price_are_refused():
    stage = _one_sender_two_receivers()
    both = {'network_send': 1.0, 'network_receive': 1.0}
    with pytest.raises(AffineTransferError, match='no capacity for'):
        estimate_seconds([stage], TransferCostModel(node_route({0: 0, 1: 1, 2: 1}), {'network_send': 1.0}))
    with pytest.raises(AffineTransferError, match='has no node'):
        estimate_seconds([stage], TransferCostModel(node_route({0: 0, 1: 1}), both))
    with pytest.raises(AffineTransferError, match='positive finite'):
        TransferCostModel(node_route({}), {'network_send': 0.0})
    with pytest.raises(AffineTransferError, match='unknown intra-node resource'):
        node_route({}, intra_node=('nvlink', ))
    with pytest.raises(AffineTransferError, match='no candidate'):
        cheapest({}, TransferCostModel(node_route({}), {}))
