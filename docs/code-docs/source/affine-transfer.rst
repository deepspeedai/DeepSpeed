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
a process group, the default group unless ``group`` is given, so that a cost
model or replica policy can price a plan's
messages with numbers from the machine it runs on. It only measures; it does not
plan, choose replicas, or fit anything by itself. Every process calls it and
receives the same raw samples::

    from deepspeed.checkpoint.affine_transfer_calibration import (calibrate_links, fit_links,
                                                                   node_path_of)

    calibration = calibrate_links(path_of=node_path_of(processes_per_node=8), fan_width=[1, 2, 4, 7])
    fits = fit_links(calibration)
    fits['peer_pairs'][(0, 1)]       # the link from process 0 to process 1 alone
    fits['fan_in'][(0, (1, 2, 3, 4))]['inter_node']  # per-link cost while 1-4 send to 0 at once
    fits['receive'][(0, (1, 2, 3, 4))]               # process 0's aggregate rate receiving from 1-4

Five patterns are sampled at every size, 4 KiB to 256 MiB by default:

- ``local``: a device-local copy on each process.
- ``peer``: each ordered process pair alone (1 to 1).
- ``concurrent``: every process sends to ``process + shift`` at once, so links
  are measured while shared.
- ``fan_out``: one root sends to ``width`` peers at once (1 to N).
- ``fan_in``: ``width`` peers send to one root at once (N to 1), measuring
  receiver contention.

``fan_width`` takes one width or a list of them; each fan sample records its
``width``. It defaults to one less than the fewest processes on any node, and at
least 1, so the default fans fit in a node however large the job is. A root's
default peers are taken from ``root + 1`` onwards, those on its node (by host
name) first. ``fan_peers(root, width)`` picks other peers, for example the
senders behind one PCIe switch or across a socket, and must return the same
peers on every process.

Each message is labelled with ``path_of(from_process, to_process)``. The default
labels only ``local`` and ``peer``; ``node_path_of`` splits ``peer`` into
``intra_node`` and ``inter_node``, and any callable can be used, for example one
that tells NVLink from PCIe pairs. ``fit_links`` pools the samples per label and
fits ``alpha + bytes / bandwidth`` to the median at each size, and
``peer_pairs`` fits each ordered pair alone. Fan fits are keyed by
``(root, peers)``, with the peers sorted; the width is ``len(peers)``. ``send``
and ``receive`` fit each root's own time against all the bytes it moved, giving
the root's aggregate rate with those peers, which is what a plan's per-process
sent and received bytes are priced against.

A fan finishes when the busiest resource it crosses does: a peer's link, the
root, or a link the peers share. Behind PCIe switches that depends on *which*
peers share a process more than on how many, which is why fans are not keyed by
width alone. On 8 B70 cards (xccl, 256 MiB messages) the pairs ranged from 3.6
to 54 GB/s, depending on the sender and direction, and a root's fan-in
bandwidth at one width varied with its senders::

    width   min ~ max over roots (GB/s)   spread
    1        7.0 ~ 53.9                   7.7x
    2       12.6 ~ 55.2                   4.4x
    4       18.7 ~ 37.3                   2.0x
    7       16.4 ~ 23.6                   1.4x

There, use ``peer_pairs``, label paths by topology, and choose the sender sets
to measure with ``fan_peers``; to compare several sets of one width, calibrate
once per set and concatenate the ``fan_out`` and ``fan_in`` samples before
fitting. On a uniform fabric, fits of one width can be pooled by
``len(peers)``.

Processes here are ranks of ``group``; pass the group the transfer runs on. A
plan's source and target locations are
mapped to processes by ``TransferEndpoints``, so translate through it before
joining a plan's messages with these labels. The pair sweep makes
``world_size * (world_size - 1)`` barriered samples per size and repeat, and each
process holds up to ``(2 + max(widths)) * max(sizes)`` bytes, 2.25 GiB with the
default width at 8 processes per node and 256 MiB, so pass shorter
``sizes`` or smaller fan widths on large jobs. The numbers describe the
transport used for the measurement (torch point-to-point) on that topology.
