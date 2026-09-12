# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Tests for Ray topology helper functions and RayTopology class.

Helper functions (create_default_bundles, validate_bundles, get_adjacent_stages,
compute_pipe_buffers, validate_strategy) are pure and testable without Ray.

RayTopology integration tests use the ray_isolated fixture from conftest.
"""

import pytest
import re

# =========================================================================
# Pure helper function tests — no Ray required
# =========================================================================


class TestCreateDefaultBundles:
    """Tests for create_default_bundles()."""

    def test_default_two_gpu(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        bundles = create_default_bundles(num_stages=3, num_gpus=1, num_cpus=1)
        assert len(bundles) == 3
        for b in bundles:
            assert b == {"GPU": 1, "CPU": 1}

    def test_custom_resources(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        bundles = create_default_bundles(num_stages=2, num_gpus=2, num_cpus=8)
        assert bundles == [{"GPU": 2, "CPU": 8}, {"GPU": 2, "CPU": 8}]

    def test_cpu_only(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        bundles = create_default_bundles(num_stages=4, num_gpus=0, num_cpus=4)
        assert len(bundles) == 4
        for b in bundles:
            assert b == {"GPU": 0, "CPU": 4}

    def test_single_stage(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        bundles = create_default_bundles(num_stages=1)
        assert bundles == [{"GPU": 1, "CPU": 1}]

    def test_zero_stages_raises(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            create_default_bundles(num_stages=0)

    def test_negative_stages_raises(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            create_default_bundles(num_stages=-1)

    def test_large_pipeline(self):
        from deepspeed.runtime.pipe.ray.placement import create_default_bundles

        bundles = create_default_bundles(num_stages=128)
        assert len(bundles) == 128
        for b in bundles:
            assert b == {"GPU": 1, "CPU": 1}


class TestValidateBundles:
    """Tests for validate_bundles()."""

    def test_valid_bundles(self):
        from deepspeed.runtime.pipe.ray.placement import validate_bundles

        # Should not raise
        validate_bundles([{"GPU": 1}, {"CPU": 4}], num_stages=2)

    def test_count_mismatch_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_bundles

        with pytest.raises(ValueError, match=re.escape("Expected 3 bundles, got 2")):
            validate_bundles([{"GPU": 1}, {"GPU": 1}], num_stages=3)

    def test_not_a_list_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_bundles

        with pytest.raises(TypeError, match="bundles must be a list"):
            validate_bundles({"GPU": 1}, num_stages=1)

    def test_empty_bundle_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_bundles

        with pytest.raises(ValueError, match="Bundle 0 is empty"):
            validate_bundles([{}], num_stages=1)

    def test_empty_bundle_in_middle_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_bundles

        with pytest.raises(ValueError, match="Bundle 1 is empty"):
            validate_bundles([{"GPU": 1}, {}, {"GPU": 1}], num_stages=3)

    def test_bundle_not_dict_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_bundles

        with pytest.raises(TypeError, match="Bundle 0 must be a dict"):
            validate_bundles(["gpu"], num_stages=1)


class TestGetAdjacentStages:
    """Tests for get_adjacent_stages()."""

    def test_middle_stage(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        prev_s, next_s = get_adjacent_stages(stage_id=1, num_stages=4)
        assert prev_s == 0
        assert next_s == 2

    def test_first_stage(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        prev_s, next_s = get_adjacent_stages(stage_id=0, num_stages=4)
        assert prev_s is None
        assert next_s == 1

    def test_last_stage(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        prev_s, next_s = get_adjacent_stages(stage_id=3, num_stages=4)
        assert prev_s == 2
        assert next_s is None

    def test_single_stage(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        prev_s, next_s = get_adjacent_stages(stage_id=0, num_stages=1)
        assert prev_s is None
        assert next_s is None

    def test_two_stages(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        # Stage 0
        prev_s, next_s = get_adjacent_stages(stage_id=0, num_stages=2)
        assert prev_s is None
        assert next_s == 1

        # Stage 1
        prev_s, next_s = get_adjacent_stages(stage_id=1, num_stages=2)
        assert prev_s == 0
        assert next_s is None

    def test_out_of_range_raises(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        with pytest.raises(IndexError, match="stage_id 4 out of range"):
            get_adjacent_stages(stage_id=4, num_stages=4)

    def test_negative_stage_raises(self):
        from deepspeed.runtime.pipe.ray.placement import get_adjacent_stages

        with pytest.raises(IndexError, match="stage_id -1 out of range"):
            get_adjacent_stages(stage_id=-1, num_stages=2)


class TestComputePipeBuffersInvariants:
    """Property-based invariants for compute_pipe_buffers().

    These tests verify mathematical properties that must hold for ALL valid
    inputs, using parameterized sweeps across the (stage_id, num_stages,
    micro_batches) space.
    """

    @pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 8, 16, 32])
    @pytest.mark.parametrize("micro_batches", [1, 2, 4, 8, 16, 32, 128])
    def test_floor_is_always_two(self, num_stages, micro_batches):
        """Result is never less than 2 for any valid inputs."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        for stage_id in range(num_stages):
            buffers = compute_pipe_buffers(stage_id, num_stages, micro_batches)
            assert buffers >= 2, (f"Buffers < 2: stage={stage_id}, stages={num_stages}, "
                                  f"micro_batches={micro_batches} → {buffers}")

    @pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 8, 16, 32])
    @pytest.mark.parametrize("micro_batches", [1, 2, 4, 8, 16, 32, 128])
    def test_never_exceeds_num_stages(self, num_stages, micro_batches):
        """Buffers never exceed the total number of stages."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        for stage_id in range(num_stages):
            buffers = compute_pipe_buffers(stage_id, num_stages, micro_batches)
            assert buffers <= num_stages, (f"Buffers > stages: stage={stage_id}, stages={num_stages}, "
                                           f"micro_batches={micro_batches} → {buffers}")

    @pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 8, 16, 32])
    @pytest.mark.parametrize("micro_batches", [1, 2, 4, 8, 16, 32, 128])
    def test_never_exceeds_micro_batches(self, num_stages, micro_batches):
        """Buffers never exceed the total number of micro-batches."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        for stage_id in range(num_stages):
            buffers = compute_pipe_buffers(stage_id, num_stages, micro_batches)
            assert buffers <= micro_batches, (f"Buffers > micro_batches: stage={stage_id}, stages={num_stages}, "
                                              f"micro_batches={micro_batches} → {buffers}")

    @pytest.mark.parametrize("num_stages", [2, 3, 4, 8, 16, 32])
    @pytest.mark.parametrize("micro_batches", [2, 4, 8, 16, 32, 128])
    def test_monotonic_decreasing(self, num_stages, micro_batches):
        """Buffers decrease (or stay same) as stage_id increases."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        prev = float("inf")
        for stage_id in range(num_stages):
            buffers = compute_pipe_buffers(stage_id, num_stages, micro_batches)
            assert buffers <= prev, (f"Non-monotonic: stage={stage_id}, stages={num_stages}, "
                                     f"micro_batches={micro_batches}, prev={prev}, current={buffers}")
            prev = buffers

    @pytest.mark.parametrize("num_stages", [1, 2, 3, 4, 8, 16, 32])
    @pytest.mark.parametrize("micro_batches", [1, 2, 4, 8, 16, 32, 128])
    def test_first_stage_maximum(self, num_stages, micro_batches):
        """Stage 0 always has the most buffers (or is tied)."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        first = compute_pipe_buffers(0, num_stages, micro_batches)
        for stage_id in range(1, num_stages):
            assert first >= compute_pipe_buffers(stage_id, num_stages, micro_batches)


class TestComputePipeBuffersParameterized:
    """Parameterized exact-value tests for compute_pipe_buffers().

    Covers the full cross-product of stage positions, pipeline depths,
    and micro-batch counts to catch off-by-one errors in the formula.
    """

    # (stage_id, num_stages, micro_batches, expected)
    _CASES = [
        # ---- single-stage pipeline: always clamped to 2 ----
        (0, 1, 1, 2),
        (0, 1, 2, 2),
        (0, 1, 4, 2),
        (0, 1, 128, 2),
        # ---- two-stage: micro_batches=1 ----
        (0, 2, 1, 2),
        (1, 2, 1, 2),
        # ---- two-stage: micro_batches=4 ----
        (0, 2, 4, 2),
        (1, 2, 4, 2),
        # ---- two-stage: micro_batches=2 ----
        (0, 2, 2, 2),
        (1, 2, 2, 2),
        # ---- four-stage: micro_batches=8 (unlimited) ----
        (0, 4, 8, 4),  # min(4, 8) = 4
        (1, 4, 8, 3),  # min(3, 8) = 3
        (2, 4, 8, 2),  # min(2, 8) → max(2, 2) = 2
        (3, 4, 8, 2),  # min(1, 8) → max(2, 1) = 2
        # ---- four-stage: micro_batches=4 (exact match) ----
        (0, 4, 4, 4),  # min(4, 4) = 4
        (1, 4, 4, 3),
        (2, 4, 4, 2),
        (3, 4, 4, 2),
        # ---- four-stage: micro_batches=2 (limited) ----
        (0, 4, 2, 2),  # min(4, 2) = 2
        (1, 4, 2, 2),  # min(3, 2) → max(2, 2) = 2
        (2, 4, 2, 2),
        (3, 4, 2, 2),
        # ---- eight-stage: micro_batches=32 (unlimited) ----
        (0, 8, 32, 8),
        (1, 8, 32, 7),
        (2, 8, 32, 6),
        (3, 8, 32, 5),
        (4, 8, 32, 4),
        (5, 8, 32, 3),
        (6, 8, 32, 2),
        (7, 8, 32, 2),
        # ---- eight-stage: micro_batches=4 (limited) ----
        (0, 8, 4, 4),  # min(8, 4) = 4
        (1, 8, 4, 4),  # min(7, 4) = 4
        (2, 8, 4, 4),  # min(6, 4) = 4
        (3, 8, 4, 4),  # min(5, 4) = 4
        (4, 8, 4, 4),  # min(4, 4) = 4
        (5, 8, 4, 3),  # min(3, 4) = 3
        (6, 8, 4, 2),  # min(2, 4) → max(2, 2) = 2
        (7, 8, 4, 2),  # min(1, 4) → max(2, 1) = 2
        # ---- deep pipeline: 32 stages, 128 micro_batches ----
        (0, 32, 128, 32),
        (15, 32, 128, 17),
        (30, 32, 128, 2),
        (31, 32, 128, 2),
    ]

    @pytest.mark.parametrize("stage_id, num_stages, micro_batches, expected", _CASES)
    def test_exact_values(self, stage_id, num_stages, micro_batches, expected):
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        result = compute_pipe_buffers(stage_id, num_stages, micro_batches)
        assert result == expected, (f"stage={stage_id}, stages={num_stages}, "
                                    f"micro_batches={micro_batches}: expected {expected}, got {result}")


class TestComputePipeBuffersEdgeCases:
    """Edge case tests for compute_pipe_buffers()."""

    def test_all_stages_clamped_to_two_when_single_micro_batch(self):
        """With only 1 micro-batch, every stage gets exactly 2 buffers."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        for num_stages in [1, 2, 4, 8, 32, 128]:
            for stage_id in range(num_stages):
                assert compute_pipe_buffers(stage_id, num_stages,
                                            1) == 2, (f"Expected 2 for stage={stage_id}, stages={num_stages}, "
                                                      f"micro_batches=1")

    def test_stages_after_limit_get_two_buffers(self):
        """Any stage where (num_stages - stage_id) <= 2 gets exactly 2 buffers."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        # Stage 6 of 8: 8-6=2 → max(2, min(2, 32)) = 2
        assert compute_pipe_buffers(6, 8, 32) == 2
        # Stage 7 of 8: 8-7=1 → max(2, min(1, 32)) = 2
        assert compute_pipe_buffers(7, 8, 32) == 2
        # Stage 0 of 2 with 1 micro_batch: 2-0=2 → max(2, min(2,1)) = 2
        assert compute_pipe_buffers(0, 2, 1) == 2

    def test_micro_batch_bound_masks_stage_bound(self):
        """When micro_batches < (num_stages - stage_id), it's the real limit."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        # stage 0, 8 stages, 4 micro_batches: min(8-0, 4) = 4
        assert compute_pipe_buffers(0, 8, 4) == 4
        # stage 3, 8 stages, 4 micro_batches: min(8-3, 4) = 4
        assert compute_pipe_buffers(3, 8, 4) == 4
        # stage 4, 8 stages, 4 micro_batches: min(8-4, 4) = 4
        assert compute_pipe_buffers(4, 8, 4) == 4
        # stage 5, 8 stages, 4 micro_batches: min(8-5, 4) = 3
        assert compute_pipe_buffers(5, 8, 4) == 3

    def test_strictly_decreasing_until_floor(self):
        """Buffers strictly decrease until hitting the floor of 2."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        # With unlimited micro_batches, each stage decreases by exactly 1
        prev = 128
        found_floor = False
        for stage_id in range(64):
            buffers = compute_pipe_buffers(stage_id, 64, 256)
            if buffers == 2:
                found_floor = True
            elif found_floor:
                # Once we hit floor, all subsequent must also be floor
                assert buffers == 2, f"Bounced off floor at stage {stage_id}"
            else:
                assert buffers == prev - 1, (f"Expected {prev - 1} at stage {stage_id}, got {buffers}")
            prev = buffers

    def test_formula_respects_integer_arithmetic(self):
        """Verify integer math: no float rounding surprises."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        assert isinstance(compute_pipe_buffers(0, 4, 8), int)
        assert isinstance(compute_pipe_buffers(1, 3, 5), int)
        assert isinstance(compute_pipe_buffers(2, 8, 1), int)


class TestComputePipeBuffersInvalidInputs:
    """Tests that compute_pipe_buffers() rejects invalid inputs cleanly."""

    # ---- num_stages < 1 ----

    @pytest.mark.parametrize("bad_value, label", [
        (0, "zero"),
        (-1, "negative"),
        (-8, "negative_large"),
        (-1024, "very_negative"),
    ])
    def test_invalid_num_stages_zero_or_negative(self, bad_value, label):
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            compute_pipe_buffers(stage_id=0, num_stages=bad_value, micro_batches=4)

    # ---- num_stages beyond max tolerance ----

    @pytest.mark.parametrize("bad_value, label", [
        (1_000_001, "just_over_limit"),
        (2_000_000, "2M"),
        (10_000_000, "10M"),
        (10**9, "1B"),
    ])
    def test_invalid_num_stages_beyond_max_tolerance(self, bad_value, label):
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(ValueError, match="num_stages.*exceeds maximum allowed"):
            compute_pipe_buffers(stage_id=0, num_stages=bad_value, micro_batches=4)

    # ---- num_stages is float ----

    @pytest.mark.parametrize("bad_value", [
        1.0,
        2.5,
        0.1,
        1e6,
    ])
    def test_invalid_num_stages_float_type(self, bad_value):
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(TypeError, match="num_stages must be int"):
            compute_pipe_buffers(stage_id=0, num_stages=bad_value, micro_batches=4)

    # ---- micro_batches is float ----

    @pytest.mark.parametrize("bad_value", [
        1.0,
        2.5,
        0.1,
        1e6,
    ])
    def test_invalid_micro_batches_float_type(self, bad_value):
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(TypeError, match="micro_batches must be int"):
            compute_pipe_buffers(stage_id=0, num_stages=4, micro_batches=bad_value)

    # ---- micro_batches < 1 ----

    @pytest.mark.parametrize("bad_value, label", [
        (0, "zero_micro"),
        (-1, "negative_micro"),
        (-4, "negative_small"),
        (-64, "negative_large"),
    ])
    def test_micro_batches_zero_or_negative_raises(self, bad_value, label):
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(ValueError, match="micro_batches must be >= 1"):
            compute_pipe_buffers(stage_id=0, num_stages=4, micro_batches=bad_value)

    # ---- both invalid ----

    def test_both_negative_raises_stages_first(self):
        """num_stages is checked before micro_batches."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            compute_pipe_buffers(stage_id=0, num_stages=-1, micro_batches=-1)

    def test_zero_stages_zero_micro_batches(self):
        """num_stages=0 is checked first."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(ValueError, match="num_stages must be >= 1"):
            compute_pipe_buffers(stage_id=0, num_stages=0, micro_batches=0)

    # ---- float / non-integer inputs ----

    def test_all_args_float_raises_type_error(self):
        """All float arguments are rejected by the type guard."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        with pytest.raises(TypeError, match="stage_id must be int"):
            compute_pipe_buffers(stage_id=0.0, num_stages=2.0, micro_batches=4.0)


class TestComputePipeBuffersValidInputs:
    """Tests that compute_pipe_buffers() accepts valid inputs and returns correct values.

    Covers the boundary of minimum valid values and large-scale pipelines
    to ensure the validation guards do not reject legitimate calls.
    """

    # (stage_id, num_stages, micro_batches, expected)
    _POSITIVE_CASES = [
        # ---- minimum valid input (boundary) ----
        (0, 1, 1, 2),  # smallest possible pipeline
        # ---- one large, one at minimum ----
        (0, 128, 1, 2),  # deep pipeline, single micro-batch
        (0, 1, 256, 2),  # single stage, lots of micro-batches
        # ---- equal values ----
        (0, 8, 8, 8),  # num_stages == micro_batches
        (4, 8, 8, 4),  # middle stage, equal params
        (7, 8, 8, 2),  # last stage, equal params (clamped)
        # ---- micro_batches > num_stages ----
        (0, 4, 16, 4),  # first stage capped by num_stages
        (2, 4, 16, 2),  # clamped to floor
        # ---- micro_batches == num_stages // 2 ----
        (0, 16, 8, 8),  # micro_batches half of stages
        (7, 16, 8, 8),  # still above floor at stage 7
        (8, 16, 8, 8),  # exactly at micro_batches limit
        (9, 16, 8, 7),  # starts decreasing
        # ---- large pipeline with plentiful micro_batches ----
        (0, 256, 1024, 256),
        (128, 256, 1024, 128),
        (254, 256, 1024, 2),
        (255, 256, 1024, 2),
        # ---- asymmetric: num_stages > micro_batches heavily ----
        (0, 1024, 4, 4),  # limited by micro_batches
        (511, 1024, 4, 4),
        (512, 1024, 4, 4),
        (1020, 1024, 4, 2),  # finally drops below floor
        (1021, 1024, 4, 2),
        (1022, 1024, 4, 2),
        (1023, 1024, 4, 2),
    ]

    @pytest.mark.parametrize("stage_id, num_stages, micro_batches, expected", _POSITIVE_CASES)
    def test_valid_inputs_return_correct_value(self, stage_id, num_stages, micro_batches, expected):
        """Valid inputs should not raise and should return the expected value."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        result = compute_pipe_buffers(stage_id, num_stages, micro_batches)
        assert result == expected, (f"stage={stage_id}, stages={num_stages}, "
                                    f"micro_batches={micro_batches}: expected {expected}, got {result}")

    @pytest.mark.parametrize("num_stages, micro_batches", [
        (1, 1),
        (2, 1),
        (1, 2),
        (4, 8),
        (8, 4),
        (16, 16),
        (128, 256),
        (256, 128),
        (1024, 65536),
    ])
    def test_valid_inputs_do_not_raise(self, num_stages, micro_batches):
        """All positive pairs should pass validation without raising."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        for stage_id in range(num_stages):
            try:
                result = compute_pipe_buffers(stage_id, num_stages, micro_batches)
            except Exception as e:
                raise AssertionError(f"Unexpected {type(e).__name__} for "
                                     f"stage={stage_id}, stages={num_stages}, "
                                     f"micro_batches={micro_batches}: {e}") from e
            assert isinstance(result, int)
            assert result >= 2

    def test_stage_id_zero_across_range(self):
        """Stage 0 is valid for all positive (num_stages, micro_batches)."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        for num_stages in range(1, 33):
            for micro_batches in range(1, 33):
                result = compute_pipe_buffers(0, num_stages, micro_batches)
                assert result >= 2

    def test_max_tolerance_boundary_accepted(self):
        """The maximum allowed num_stages (1,000,000) is valid and returns >= 2."""
        from deepspeed.runtime.pipe.ray.placement import compute_pipe_buffers

        result = compute_pipe_buffers(stage_id=0, num_stages=1_000_000, micro_batches=4)
        assert result == 4
        assert result >= 2


class TestValidateStrategy:
    """Tests for validate_strategy()."""

    def test_valid_strategies(self):
        from deepspeed.runtime.pipe.ray.placement import validate_strategy

        assert validate_strategy("PACK") == "PACK"
        assert validate_strategy("SPREAD") == "SPREAD"
        assert validate_strategy("STRICT_PACK") == "STRICT_PACK"
        assert validate_strategy("STRICT_SPREAD") == "STRICT_SPREAD"

    def test_lower_case(self):
        from deepspeed.runtime.pipe.ray.placement import validate_strategy

        assert validate_strategy("pack") == "PACK"
        assert validate_strategy("Strict_Spread") == "STRICT_SPREAD"

    def test_invalid_strategy_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_strategy

        with pytest.raises(ValueError, match="Unknown placement strategy"):
            validate_strategy("INVALID")

    def test_empty_string_raises(self):
        from deepspeed.runtime.pipe.ray.placement import validate_strategy

        with pytest.raises(ValueError, match="Unknown placement strategy"):
            validate_strategy("")


# =========================================================================
# RayTopology integration tests — require Ray
# =========================================================================

pytest.importorskip("ray", reason="Ray is not installed")


class TestRayTopologyUnit:
    """Unit tests for RayTopology placement group creation and shutdown."""

    def test_create_default_bundles(self):
        """RayTopology creates correct number of bundles by default."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=4)
        assert topo.num_stages == 4
        assert topo.placement_group is None

    def test_create_custom_bundles(self):
        """RayTopology accepts custom resource bundles."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        bundles = [{"GPU": 1, "CPU": 2}, {"GPU": 1, "CPU": 8}]
        topo = RayTopology(num_stages=2, bundles=bundles)
        assert topo.num_stages == 2

    def test_bundles_mismatch_raises(self):
        """ValueError when bundle count doesn't match num_stages."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        with pytest.raises(ValueError, match="Expected 3 bundles"):
            RayTopology(num_stages=3, bundles=[{"GPU": 1}, {"GPU": 1}])

    def test_invalid_strategy_raises(self):
        """ValueError on invalid placement strategy."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        with pytest.raises(ValueError, match="Unknown placement strategy"):
            RayTopology(num_stages=2, strategy="NONEXISTENT")

    def test_strategy_case_insensitive(self):
        """Strategy names are case-insensitive."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=2, strategy="strict_spread")
        assert topo._strategy == "STRICT_SPREAD"

    def test_adjacent_stages_delegates(self):
        """adjacent_stages method delegates to get_adjacent_stages."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=4)
        prev_s, next_s = topo.adjacent_stages(stage_id=2)
        assert prev_s == 1
        assert next_s == 3

    def test_initialize_and_shutdown(self, ray_isolated):
        """Placement group is created on initialize and removed on shutdown."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=2)
        topo.initialize()
        assert topo.placement_group is not None
        topo.shutdown()
        assert topo.placement_group is None

    def test_get_stage_options(self, ray_isolated):
        """get_stage_options returns scheduling strategy for each stage."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=2)
        topo.initialize()

        for stage_id in range(2):
            options = topo.get_stage_options(stage_id)
            assert "scheduling_strategy" in options

        topo.shutdown()

    def test_get_options_before_init_raises(self):
        """RuntimeError when get_stage_options called before initialize."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=2)
        with pytest.raises(RuntimeError, match="not initialized"):
            topo.get_stage_options(0)


class TestRayTopologyIntegration:
    """Integration tests for RayTopology with Ray placement groups."""

    def test_placement_group_created(self, ray_isolated):
        """Placement group is created and ready after initialize."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology
        import ray

        topo = RayTopology(num_stages=2, bundles=[{"CPU": 1}, {"CPU": 1}])
        topo.initialize()

        pg = topo.placement_group
        assert pg is not None
        state = ray._private.state.state.placement_group_table(pg.id)
        assert state["state"] == "CREATED"

        topo.shutdown()

    def test_double_initialize_idempotent(self, ray_isolated):
        """Calling initialize twice returns the same placement group."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology

        topo = RayTopology(num_stages=2)
        pg1 = topo.initialize()
        pg2 = topo.initialize()
        assert pg1 is pg2
        topo.shutdown()

    def test_heterogeneous_bundles(self, ray_isolated):
        """Placement group with heterogeneous resource bundles."""
        from deepspeed.runtime.pipe.ray.placement import RayTopology
        import ray

        bundles = [{"CPU": 1}, {"CPU": 2}, {"CPU": 4}]
        topo = RayTopology(num_stages=3, bundles=bundles)
        topo.initialize()

        pg = topo.placement_group
        assert pg is not None
        state = ray._private.state.state.placement_group_table(pg.id)

        # Each bundle should be in the placement group
        assert len(state["bundles"]) == 3
        assert state["bundles"]["0"]["CPU"] == 1
        assert state["bundles"]["1"]["CPU"] == 2
        assert state["bundles"]["2"]["CPU"] == 4

        topo.shutdown()
