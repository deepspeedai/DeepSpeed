# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

# ---------------------------------------------------------------------------
# Pure helper functions — no Ray import required.
# These are testable without a Ray cluster and are used internally by RayTopology.
# ---------------------------------------------------------------------------

# Reasonable upper bound for pipeline stages and micro-batches.
# Anything beyond this is almost certainly a configuration error
# rather than a legitimate deep pipeline.
_MAX_PIPELINE_STAGES = 1_000_000
_MAX_MICRO_BATCHES = 1_000_000


def create_default_bundles(num_stages, num_gpus=1, num_cpus=1):
    """Create a list of default resource bundles for a pipeline.

    Each bundle requests the same amount of GPU and CPU resources.
    For heterogeneous placement, pass a custom bundle list directly.

    Args:
        num_stages (int): Number of pipeline stages.
        num_gpus (int): GPUs per stage (default 1).
        num_cpus (int): CPUs per stage (default 1).

    Returns:
        list[dict]: One resource dict per stage.

    Raises:
        ValueError: If num_stages is not a positive integer.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    return [{"GPU": num_gpus, "CPU": num_cpus} for _ in range(num_stages)]


def validate_bundles(bundles, num_stages):
    """Validate that a bundle list matches the expected number of stages.

    Each bundle must be a dict with at least one resource key.

    Args:
        bundles (list[dict]): Per-stage resource dictionaries.
        num_stages (int): Expected number of stages.

    Raises:
        ValueError: If bundle count != num_stages or any bundle is invalid.
        TypeError: If bundles is not a list.
    """
    if not isinstance(bundles, list):
        raise TypeError(f"bundles must be a list, got {type(bundles).__name__}")

    if len(bundles) != num_stages:
        raise ValueError(f"Expected {num_stages} bundles, got {len(bundles)}")

    for idx, bundle in enumerate(bundles):
        if not isinstance(bundle, dict):
            raise TypeError(f"Bundle {idx} must be a dict, got {type(bundle).__name__}")
        if len(bundle) == 0:
            raise ValueError(f"Bundle {idx} is empty — must contain at least one resource key")


def get_adjacent_stages(stage_id, num_stages):
    """Return the previous and next stage IDs for a given stage.

    Boundary stages return ``None`` for the out-of-range neighbor.

    Args:
        stage_id (int): The current stage index (0-based).
        num_stages (int): Total number of pipeline stages.

    Returns:
        tuple: ``(prev_stage, next_stage)`` where each is ``int`` or ``None``.

    Raises:
        IndexError: If stage_id is out of range.
    """
    if not (0 <= stage_id < num_stages):
        raise IndexError(f"stage_id {stage_id} out of range [0, {num_stages})")

    prev_stage = stage_id - 1 if stage_id > 0 else None
    next_stage = stage_id + 1 if stage_id < num_stages - 1 else None
    return (prev_stage, next_stage)


def compute_pipe_buffers(stage_id, num_stages, micro_batches):
    """Compute the number of pipeline buffers needed for a stage.

    Earlier stages need more buffers because they have more in-flight
    micro-batches during 1F1B scheduling.

    Args:
        stage_id (int): The current stage index (0-based).
        num_stages (int): Total number of pipeline stages.
        micro_batches (int): Number of micro-batches per batch.

    Returns:
        int: Minimum number of pipeline buffers (at least 2).

    Raises:
        ValueError: If num_stages or micro_batches is not a positive integer,
            or exceeds the maximum allowed value.
        TypeError: If any argument is not an integer.
    """
    if not isinstance(num_stages, int):
        raise TypeError(f"num_stages must be int, got {type(num_stages).__name__} ({num_stages})")
    if not isinstance(micro_batches, int):
        raise TypeError(f"micro_batches must be int, got {type(micro_batches).__name__} ({micro_batches})")
    if not isinstance(stage_id, int):
        raise TypeError(f"stage_id must be int, got {type(stage_id).__name__} ({stage_id})")

    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")
    if num_stages > _MAX_PIPELINE_STAGES:
        raise ValueError(f"num_stages {num_stages} exceeds maximum allowed ({_MAX_PIPELINE_STAGES}). "
                         f"Pipeline stage count is unreasonably large.")
    if micro_batches < 1:
        raise ValueError(f"micro_batches must be >= 1, got {micro_batches}")
    if micro_batches > _MAX_MICRO_BATCHES:
        raise ValueError(f"micro_batches {micro_batches} exceeds maximum allowed ({_MAX_MICRO_BATCHES}). "
                         f"Micro-batch count is unreasonably large.")
    buffers = min(num_stages - stage_id, micro_batches)
    return max(2, buffers)


def validate_strategy(strategy):
    """Normalize and validate a Ray placement group strategy name.

    Accepts common variations and returns the canonical form accepted
    by ``ray.util.placement_group()``.

    Args:
        strategy (str): Strategy name (case-insensitive).

    Returns:
        str: Canonical strategy name.

    Raises:
        ValueError: If strategy is not recognized.
    """
    VALID_STRATEGIES = {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}
    upper = strategy.upper()
    if upper not in VALID_STRATEGIES:
        raise ValueError(f"Unknown placement strategy: {strategy}. Must be one of {sorted(VALID_STRATEGIES)}")
    return upper


class RayTopology:
    """Maps pipeline stage IDs to Ray placement group bundles.

    Each pipeline stage is allocated a dedicated resource bundle within a
    Ray placement group. This enables heterogeneous resource allocation
    where different stages can request different GPU types, CPU counts,
    or custom resources.

    The placement group is created once during executor initialization
    and removed during shutdown.

    Args:
        num_stages (int): Number of pipeline stages.
        bundles (list, optional): Per-stage resource bundles. Each entry
            is a dict of Ray resource labels (e.g. ``{"GPU": 1, "CPU": 4}``).
            Defaults to one GPU per stage.
        strategy (str): Placement group strategy. Default ``"STRICT_SPREAD"``
            places each bundle on a different node when possible.
        name (str): Placement group name for Ray dashboard visibility.
    """

    def __init__(self, num_stages, bundles=None, strategy="STRICT_SPREAD", name="deepspeed-pp"):
        if not HAS_RAY:
            raise ImportError("RayTopology requires Ray. Install with: pip install ray")

        self._num_stages = num_stages
        self._strategy = validate_strategy(strategy)
        self._name = name
        self._pg = None

        if bundles is None:
            bundles = create_default_bundles(num_stages)
        else:
            validate_bundles(bundles, num_stages)
        self._bundles = list(bundles)  # defensive copy

    def initialize(self):
        """Create the Ray placement group and wait until it is ready.

        Returns:
            The Ray PlacementGroup handle.
        """
        if self._pg is not None:
            return self._pg

        self._pg = ray.util.placement_group(self._bundles, strategy=self._strategy, name=self._name)
        ray.get(self._pg.ready())
        return self._pg

    def get_stage_options(self, stage_id):
        """Return Ray actor options for scheduling a stage within the placement group.

        Args:
            stage_id (int): Pipeline stage index (0-based).

        Returns:
            dict: Options dict with ``scheduling_strategy`` for use with
            ``ActorClass.options(**options).remote(...)``.
        """
        if self._pg is None:
            raise RuntimeError("Placement group not initialized. Call initialize() first.")

        return {
            "scheduling_strategy":
            ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=self._pg,
                placement_group_bundle_index=stage_id,
            ),
        }

    def shutdown(self):
        """Remove the placement group and release resources."""
        if self._pg is not None:
            ray.util.remove_placement_group(self._pg)
            self._pg = None

    @property
    def num_stages(self):
        """int: Number of pipeline stages."""
        return self._num_stages

    @property
    def placement_group(self):
        """The Ray PlacementGroup handle, or ``None`` if not initialized."""
        return self._pg

    def adjacent_stages(self, stage_id):
        """Return prev/next stage IDs for a given stage.

        Convenience wrapper around :func:`get_adjacent_stages`.

        Args:
            stage_id (int): The pipeline stage index.

        Returns:
            tuple: ``(prev_stage, next_stage)``.
        """
        return get_adjacent_stages(stage_id, self._num_stages)
