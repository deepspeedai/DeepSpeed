Affine Shard Transfers
======================

AutoTP and AutoEP both describe the shards they create with affine maps from
``deepspeed.checkpoint.affine``. ``deepspeed.checkpoint.affine_transfer``
plans how to move a parameter from one such sharding to another, and
``deepspeed.checkpoint.affine_transfer_executor`` executes that plan directly
into existing packed target shards without reconstructing the full logical
tensor. Neither depends on which parallelism produced the maps.

Executing affine transfers
--------------------------

Endpoint mappings use transport-group ranks and distinguish source locations
from target locations::

    from deepspeed.checkpoint.affine_transfer import plan_transfer
    from deepspeed.checkpoint.affine_transfer_executor import (
        StagingBudget, TransferEndpoints, execute_transfer)

    plan = plan_transfer(target_map, source_map)
    execute_transfer(plan, local_sources, local_targets,
                     TransferEndpoints(source_bindings, target_bindings),
                     dtype, StagingBudget(bytes_per_peer=8 << 20))

Use ``execute_transfers`` with uniquely keyed ``TransferRequest`` objects to pack
multiple parameters into peer messages. Each process supplies only its own
contiguous buffers; strided plan regions and buffer storage offsets are retained.
All participants must agree on requests, endpoints, dtype and staging budget.
Preflight refusal leaves targets untouched. Return guarantees target readiness;
a transport failure after payload begins can leave partially written targets.
The budget applies to each peer direction, not total device memory. Weights must
remain frozen during transfer; optimizer migration and live AutoEP swaps are outside
this API. Maintenance contact: ``@0z5a``.

Planning a sync with replica policies
-------------------------------------

``plan_transfers`` plans every parameter of one sync and hands one replica policy every request at
once, so a cost-aware policy can balance load over all of it::

    from deepspeed.checkpoint.affine_replica_policy import local_first_least_loaded
    from deepspeed.checkpoint.affine_transfer import plan_transfers

    plans = plan_transfers(pairs,
                           source_processes=source_processes,
                           target_processes=target_processes,
                           element_size=element_size,
                           min_split_bytes=1 << 20,
                           replica_policy=local_first_least_loaded)

``pairs`` maps a parameter name to its ``(target_map, source_map)``. Every pair is validated as in
``plan_transfer``, and ``limits`` applies to each parameter on its own.

The policy receives one ``ReplicaRequest`` per intersection of a target piece with a source
region, in a fixed order, and returns one answer per request. A request names the parameter, the
target location, the region's ``logical_origin``, ``shape``, target offset and strides, and every
validated holder as a ``ReplicaCandidate``. ``source_processes`` and ``target_processes`` map plan
locations to processes, so a policy can tell a local copy from a transfer; without them the
process fields are ``None``. ``element_size`` gives bytes per element for every parameter in
``pairs``, which fills ``bytes`` and is required by ``min_split_bytes``; a missing or non-positive
entry is refused before the policy runs.

An answer is the location of one validated holder, or ``ReplicaPart`` objects that cut the region
along axis 0 and name a holder for each cut. The planner refuses an answer unless there is one per
request, every location is a validated holder of that request, and the parts tile the request
exactly along axis 0; a split is also refused when a part is under ``min_split_bytes``, and a
parameter is refused when its parts exceed ``PlanBudget.max_pair_checks``. Such refusals raise
``InvalidReplicaPolicyError`` before any segment is emitted.

``deepspeed.checkpoint.affine_replica_policy`` provides ``first_holder`` (``plan_transfer``'s
default), ``local_first_least_loaded`` (the default here) and ``per_request``, which runs a
per-request ``replica_selector`` as a batch policy. Every rank must return the same answers, so
cost inputs must come from data every rank shares, such as a link calibration gathered across
ranks, never from a rank-local measurement.

Transport and staging
---------------------

The staging arena and ``StagingBudget`` belong to the current transport, not to
affine transfer itself. Torch point-to-point operations move one contiguous
tensor per peer per message, so a round whose regions are strided or
come from several segments is packed through a per-peer arena; a round that is
one region already contiguous in its buffer is sent or received in place. A
zero-copy transport such as GPUDirect RDMA would instead take each round's
``(offset, length)`` pairs as a scatter/gather list over registered memory. That
removes the arena copy but needs a registration lifetime this API has no place
for, so the wire contract (one packed message per peer per round) is expected to
change when such a path lands.

Inspecting a schedule
---------------------

``schedule_transfers`` returns the schedule ``execute_transfers`` would run for
a batch, without buffers or communication, so a replica policy or cost model can
price a plan before running it::

    from deepspeed.checkpoint.affine_transfer_executor import schedule_transfers

    schedules = schedule_transfers([request.plan for request in requests],
                                   endpoints, budget, element_size)

Pass the plans in request order, since that order fixes how chunks are packed
into rounds. Each ``ProcessSchedule`` lists, per peer and direction, the rounds
with their element and chunk counts and whether the round is sent or received in
place, the staging arena each peer direction allocates, the local copies, and the
round count that ``TransferStats.rounds`` reports. ``execute_transfers`` builds
its rounds and arenas from the same computation, so the two cannot drift apart.
The schedule follows the current transport described above and changes with it.
