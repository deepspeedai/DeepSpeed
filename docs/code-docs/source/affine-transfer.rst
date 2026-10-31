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

Calibrating links
-----------------

``calibrate_links`` measures what moving bytes costs between the processes of
the default group, so that a cost model or replica policy can price a plan's
messages with numbers from the machine it runs on. It only measures; it does not
plan, choose replicas, or fit anything by itself. Every process calls it and
receives the same raw samples::

    from deepspeed.checkpoint.affine_transfer_calibration import (calibrate_links, fit_links,
                                                                   node_path_of)

    calibration = calibrate_links(path_of=node_path_of(processes_per_node=8))
    fits = fit_links(calibration)
    fits['fan_in']['inter_node']  # {'alpha_seconds', 'bytes_per_second', ...}

Five patterns are sampled at every size, 4 KiB to 256 MiB by default:

- ``local``: a device-local copy on each process.
- ``peer``: each ordered process pair alone (1 to 1).
- ``concurrent``: every process sends to ``process + shift`` at once, so links
  are measured while shared.
- ``fan_out``: one root sends to ``fan_width`` peers at once (1 to N).
- ``fan_in``: ``fan_width`` peers send to one root at once (N to 1), measuring
  receiver contention.

Each message is labelled with ``path_of(from_process, to_process)``. The default
labels only ``local`` and ``peer``; ``node_path_of`` splits ``peer`` into
``intra_node`` and ``inter_node``, and any callable can be used, for example one
that tells NVLink from PCIe pairs. ``fit_links`` pools the samples per label and
fits ``alpha + bytes / bandwidth`` to the median at each size.

Processes here are default-group ranks. A plan's source and target locations are
mapped to processes by ``TransferEndpoints``, so translate through it before
joining a plan's messages with these labels. The pair sweep makes
``world_size * (world_size - 1)`` barriered samples per size and repeat, and each
process holds up to ``(2 + fan_width) * max(sizes)`` bytes, so pass shorter
``sizes`` or a smaller ``fan_width`` on large jobs. The numbers describe the
transport used for the measurement (torch point-to-point) on that topology.

Analyzing a plan
----------------

``analyze_transfers`` reports what an ``execute_transfers`` batch will do,
without running it or communicating. It takes the same plans, endpoints,
element size and ``StagingBudget`` and replays the executor's own operation
building and round packing, so it can be called on one process, for example to
compare two candidate plans or budgets before choosing one::

    from deepspeed.checkpoint.affine_transfer_analysis import (analyze_transfers, expected_stats,
                                                                report_to_dict)
    from deepspeed.checkpoint.affine_transfer_calibration import node_path_of
    from deepspeed.checkpoint.affine_transfer_executor import StagingBudget

    report = analyze_transfers(plans, endpoints, element_size=2, budget=StagingBudget(256 << 20),
                               path_of=node_path_of(processes_per_node=8))
    summary = report_to_dict(report)

The report covers volume (logical, moved, local-copy and remote bytes), the
distribution of segment, chunk and message sizes, the number of rounds, every
process's sent and received bytes, peers, messages, pack and unpack bytes and
staging arena size, and bytes per process pair and per path label. Each message
is labelled with ``path_of(from_process, to_process)``, the same hook
``calibrate_links`` uses, so per-path totals can be priced with ``fit_links``
results. ``expected_stats(report, process)`` gives the ``TransferStats`` that
process will return, which makes the prediction checkable after a run.

The report only describes a plan. It does not choose replicas, rank plans or
estimate time; those decisions belong to the caller.
