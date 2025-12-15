# SPDX-License-Identifier: Apache-2.0
# Copyright (c) DeepSpeed Team

# DeepSpeed Team
"""
Automated testing of parallel strategy combinations using random configurations.

This test automatically generates random parallel configurations and tests
both parallel_state_refactored and DeepSpeed to see if they produce compatible results.
"""

import pytest
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Try to import both libraries
try:
    from deepspeed.utils.parallel_state import RankGenerator
    PARALLEL_STATE_AVAILABLE = True
except ImportError as e:
    PARALLEL_STATE_AVAILABLE = False
    print(f"Warning: Could not import Megatron parallel_state_refactored: {e}")

try:
    from deepspeed.utils import groups as ds_groups
    from deepspeed.runtime.sequence_parallel import parallel_state_sp as ds_sp
    DEEPSPEED_AVAILABLE = True
except ImportError as e:
    DEEPSPEED_AVAILABLE = False
    print(f"Warning: Could not import DeepSpeed: {e}")


class ParallelConfigGenerator:
    """Generate random parallel configurations for testing."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.tested_configs = []
        self.failed_configs = []

    def generate_random_config(self, max_size=1024, min_parallel_size=1, max_parallel_size=32):
        """Generate a random parallel configuration.

        Args:
            max_size: Maximum world size to consider
            min_parallel_size: Minimum parallel size for each dimension
            max_parallel_size: Maximum parallel size for each dimension

        Returns:
            Dict with tp, dp, pp, cp, ep values and order
        """
        # Generate random sizes for each dimension
        # Don't filter invalid configurations - we want to test and report all cases
        tp = random.randint(min_parallel_size, max_parallel_size)
        dp = random.randint(min_parallel_size, max_parallel_size)
        pp = random.randint(min_parallel_size, max_parallel_size)
        cp = random.randint(min_parallel_size, max_parallel_size)
        ep = random.randint(min_parallel_size, max_parallel_size)

        # Calculate world size
        world_size = tp * dp * pp * cp * ep

        # If world size is too large, scale down proportionally
        # But try to keep at least one dimension > 1
        if world_size > max_size:
            # Scale down proportionally
            scale_factor = (max_size / world_size)**0.25
            tp = max(1, int(tp * scale_factor))
            dp = max(1, int(dp * scale_factor))
            pp = max(1, int(pp * scale_factor))
            cp = max(1, int(cp * scale_factor))
            ep = max(1, int(ep * scale_factor))
            world_size = tp * dp * pp * cp * ep

            # Ensure at least one dimension is > 1
            if world_size == 1:
                tp = 2
                world_size = 2

        # Generate random order (but must include all non-1 dimensions)
        dimensions = []
        if tp > 1:
            dimensions.append('tp')
        if dp > 1:
            dimensions.append('dp')
        if pp > 1:
            dimensions.append('pp')
        if cp > 1:
            dimensions.append('cp')
        if ep > 1:
            dimensions.append('ep')

        # Shuffle to get random order
        random.shuffle(dimensions)
        order = '-'.join(dimensions) if dimensions else 'tp'

        # If no dimensions > 1, use default
        if not dimensions:
            order = 'tp-dp'
            tp = 2
            dp = 2

        config = {
            "tp": tp,
            "dp": dp,
            "pp": pp,
            "cp": cp,
            "ep": ep,
            "order": order,
            "world_size": tp * dp * pp * cp * ep,
        }

        return config

    def generate_systematic_configs(self, max_world_size=512):
        """Generate systematic configurations covering common cases.

        Args:
            max_world_size: Maximum world size to consider

        Returns:
            List of configurations
        """
        configs = []

        # Single parallelism - test larger sizes
        for size in [2, 4, 8, 16, 32, 64, 128, 256]:
            if size <= max_world_size:
                configs.append({"tp": size, "dp": 1, "pp": 1, "cp": 1, "ep": 1, "order": "tp", "world_size": size})
                configs.append({"tp": 1, "dp": size, "pp": 1, "cp": 1, "ep": 1, "order": "dp", "world_size": size})
                configs.append({"tp": 1, "dp": 1, "pp": size, "cp": 1, "ep": 1, "order": "pp", "world_size": size})

        # Two-way combinations - more variations
        for tp, dp in [(2, 2), (2, 4), (4, 2), (2, 8), (8, 2), (4, 4), (2, 16), (16, 2), (4, 8), (8, 4)]:
            if tp * dp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": 1,
                    "cp": 1,
                    "ep": 1,
                    "order": "tp-dp",
                    "world_size": tp * dp
                })
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": 1,
                    "cp": 1,
                    "ep": 1,
                    "order": "dp-tp",
                    "world_size": tp * dp
                })

        for tp, pp in [(2, 2), (2, 4), (4, 2), (2, 8), (8, 2), (4, 4)]:
            if tp * pp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": 1,
                    "pp": pp,
                    "cp": 1,
                    "ep": 1,
                    "order": "tp-pp",
                    "world_size": tp * pp
                })

        for tp, cp in [(2, 2), (2, 4), (4, 2), (2, 8)]:
            if tp * cp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": 1,
                    "pp": 1,
                    "cp": cp,
                    "ep": 1,
                    "order": "tp-cp",
                    "world_size": tp * cp
                })

        for tp, ep in [(2, 2), (2, 4), (4, 2), (2, 8)]:
            if tp * ep <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": 1,
                    "pp": 1,
                    "cp": 1,
                    "ep": ep,
                    "order": "tp-ep",
                    "world_size": tp * ep
                })

        # Three-way combinations - more variations
        for tp, pp, dp in [(2, 2, 2), (2, 2, 4), (2, 4, 2), (4, 2, 2), (2, 2, 8), (2, 4, 4), (4, 4, 2)]:
            if tp * pp * dp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": pp,
                    "cp": 1,
                    "ep": 1,
                    "order": "tp-pp-dp",
                    "world_size": tp * pp * dp
                })
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": pp,
                    "cp": 1,
                    "ep": 1,
                    "order": "tp-dp-pp",
                    "world_size": tp * pp * dp
                })

        for tp, cp, dp in [(2, 2, 2), (2, 2, 4), (2, 4, 2)]:
            if tp * cp * dp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": 1,
                    "cp": cp,
                    "ep": 1,
                    "order": "tp-cp-dp",
                    "world_size": tp * cp * dp
                })

        for tp, ep, dp in [(2, 2, 2), (2, 2, 4), (2, 4, 2)]:
            if tp * ep * dp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": 1,
                    "cp": 1,
                    "ep": ep,
                    "order": "tp-ep-dp",
                    "world_size": tp * ep * dp
                })

        # Four-way combinations - more variations
        for tp, pp, dp, cp in [(2, 2, 2, 2), (2, 2, 2, 4), (2, 2, 4, 2), (2, 4, 2, 2)]:
            if tp * pp * dp * cp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": pp,
                    "cp": cp,
                    "ep": 1,
                    "order": "tp-pp-dp-cp",
                    "world_size": tp * pp * dp * cp
                })

        for tp, ep, pp, dp in [(2, 2, 2, 2), (2, 2, 2, 4), (2, 2, 4, 2)]:
            if tp * ep * pp * dp <= max_world_size:
                configs.append({
                    "tp": tp,
                    "dp": dp,
                    "pp": pp,
                    "cp": 1,
                    "ep": ep,
                    "order": "tp-ep-pp-dp",
                    "world_size": tp * ep * pp * dp
                })

        return configs

    def generate_random_configs(self, count=1000, max_size=1024):
        """Generate multiple random configurations.

        Args:
            count: Number of random configurations to generate
            max_size: Maximum world size

        Returns:
            List of configurations
        """
        configs = []
        seen = set()

        for _ in range(count):
            config = self.generate_random_config(max_size=max_size)
            # Create a unique key for this configuration
            key = (config["tp"], config["dp"], config["pp"], config["cp"], config["ep"], config["order"])
            if key not in seen:
                seen.add(key)
                configs.append(config)

        return configs

    def generate_random_config_by_dimension(self,
                                            dimension_count: int,
                                            max_size=1024,
                                            min_parallel_size=2,
                                            max_parallel_size=32):
        """Generate a random configuration with exactly the specified number of dimensions > 1.

        Args:
            dimension_count: Number of dimensions that should be > 1 (1-5)
            max_size: Maximum world size
            min_parallel_size: Minimum parallel size for each dimension
            max_parallel_size: Maximum parallel size for each dimension

        Returns:
            Dict with tp, dp, pp, cp, ep values and order
        """
        # All possible dimensions
        all_dims = ['tp', 'dp', 'pp', 'cp', 'ep']

        # Randomly select which dimensions to activate
        active_dims = random.sample(all_dims, min(dimension_count, len(all_dims)))

        # Initialize all dimensions to 1
        config = {
            "tp": 1,
            "dp": 1,
            "pp": 1,
            "cp": 1,
            "ep": 1,
        }

        # Set active dimensions to random values
        for dim in active_dims:
            config[dim] = random.randint(min_parallel_size, max_parallel_size)

        # Calculate world size
        world_size = config["tp"] * config["dp"] * config["pp"] * config["cp"] * config["ep"]

        # If world size is too large, scale down proportionally
        if world_size > max_size:
            scale_factor = (max_size / world_size)**(1.0 / dimension_count)
            for dim in active_dims:
                config[dim] = max(min_parallel_size, int(config[dim] * scale_factor))
            world_size = config["tp"] * config["dp"] * config["pp"] * config["cp"] * config["ep"]

        # Generate random order from active dimensions
        random.shuffle(active_dims)
        order = '-'.join(active_dims)

        config["order"] = order
        config["world_size"] = world_size

        return config

    def generate_random_configs_by_dimension(self,
                                             counts_by_dimension: Dict[int, int],
                                             max_size=1024,
                                             min_parallel_size=2,
                                             max_parallel_size=32):
        """Generate random configurations for each dimension separately.

        Args:
            counts_by_dimension: Dict mapping dimension count (1-5) to number of configs to generate
                                 e.g., {1: 100, 2: 200, 3: 150, 4: 100, 5: 50}
            max_size: Maximum world size
            min_parallel_size: Minimum parallel size for each dimension
            max_parallel_size: Maximum parallel size for each dimension

        Returns:
            List of configurations grouped by dimension count
        """
        all_configs = []
        seen = set()

        for dim_count, count in counts_by_dimension.items():
            if dim_count < 1 or dim_count > 5:
                continue

            dim_configs = []
            attempts = 0
            # Increased max_attempts for larger test sets (20x more configs)
            max_attempts = count * 20  # Prevent infinite loops, allow more attempts for uniqueness

            while len(dim_configs) < count and attempts < max_attempts:
                attempts += 1
                config = self.generate_random_config_by_dimension(dim_count, max_size, min_parallel_size,
                                                                  max_parallel_size)

                # Create a unique key for this configuration
                key = (config["tp"], config["dp"], config["pp"], config["cp"], config["ep"], config["order"])

                if key not in seen:
                    seen.add(key)
                    dim_configs.append(config)
                    all_configs.append(config)

            if len(dim_configs) < count:
                print(
                    f"Warning: Only generated {len(dim_configs)}/{count} configs for {dim_count}D combinations (attempted {attempts} times)"
                )

        return all_configs


class ErrorCategorizer:
    """Categorize and aggregate errors by type."""

    def __init__(self):
        self.error_categories = defaultdict(list)
        self.combination_stats = defaultdict(int)

    def categorize_error(self, error_msg: str, config: Dict) -> str:
        """Categorize an error message into a category."""
        error_lower = error_msg.lower()

        if "ep and cp cannot both be > 1" in error_lower:
            return "EP_CP_CONFLICT"
        elif "cp not supported" in error_lower:
            return "CP_NOT_SUPPORTED"
        elif "pp requires" in error_lower or "pipeline" in error_lower:
            return "PP_REQUIRES_MPU"
        elif "not divisible" in error_lower:
            return "DIVISIBILITY_ERROR"
        elif "order" in error_lower and "specified" in error_lower:
            return "ORDER_MISMATCH"
        elif "not available" in error_lower:
            return "FEATURE_NOT_AVAILABLE"
        else:
            return "OTHER_ERROR"

    def get_combination_type(self, config: Dict) -> str:
        """Get the combination type string for a configuration."""
        dims = []
        if config["tp"] > 1:
            dims.append("TP")
        if config["dp"] > 1:
            dims.append("DP")
        if config["pp"] > 1:
            dims.append("PP")
        if config["cp"] > 1:
            dims.append("CP")
        if config["ep"] > 1:
            dims.append("EP")

        if not dims:
            return "NONE"

        return "+".join(sorted(dims))

    def record_error(self, error_msg: str, config: Dict, library: str):
        """Record an error with categorization."""
        category = self.categorize_error(error_msg, config)
        combo_type = self.get_combination_type(config)

        self.error_categories[category].append({
            "error": error_msg,
            "config": config,
            "library": library,
            "combination": combo_type,
        })

        self.combination_stats[combo_type] += 1

    def get_error_summary(self) -> Dict:
        """Get summary of errors by category."""
        summary = {}
        for category, errors in self.error_categories.items():
            summary[category] = {
                "count": len(errors),
                "examples": errors[:5],  # First 5 examples
                "unique_combinations": len(set(e["combination"] for e in errors)),
            }
        return summary


class ParallelCompatibilityTester:
    """Test compatibility between Megatron and DeepSpeed for parallel configurations."""

    def __init__(self):
        self.results = {
            "megatron_success": [],
            "megatron_failures": [],
            "deepspeed_success": [],
            "deepspeed_failures": [],
            "compatible": [],
            "incompatible": [],
            "megatron_only": [],
            "deepspeed_only": [],
        }
        self.error_categorizer = ErrorCategorizer()
        self.combination_stats = defaultdict(
            lambda: {
                "total": 0,
                "megatron_success": 0,
                "megatron_failures": 0,
                "deepspeed_success": 0,
                "deepspeed_failures": 0,
                "compatible": 0,
                "megatron_only": 0,
                "deepspeed_only": 0,
                "incompatible": 0,
            })

    def test_megatron_config(self, config: Dict) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Test if a configuration works with Megatron.

        Returns:
            (success, error_message, result_data)
        """
        if not PARALLEL_STATE_AVAILABLE:
            return False, "Megatron not available", None

        try:
            # Check EP and CP constraint
            if config["ep"] > 1 and config["cp"] > 1:
                return False, "EP and CP cannot both be > 1 in Megatron", None

            # Create RankGenerator
            rg = RankGenerator(tp=config["tp"],
                               ep=config["ep"],
                               dp=config["dp"],
                               pp=config["pp"],
                               cp=config["cp"],
                               order=config["order"])

            # Test getting ranks for each dimension
            result_data = {
                "world_size": rg.world_size,
                "tp_groups": rg.get_ranks("tp") if config["tp"] > 1 else [],
                "dp_groups": rg.get_ranks("dp") if config["dp"] > 1 else [],
                "pp_groups": rg.get_ranks("pp") if config["pp"] > 1 else [],
                "cp_groups": rg.get_ranks("cp") if config["cp"] > 1 else [],
                "ep_groups": rg.get_ranks("ep") if config["ep"] > 1 else [],
            }

            # Test combined groups
            if len([d for d in ["tp", "dp", "pp", "cp", "ep"] if config[d] > 1]) > 1:
                combined_token = config["order"]
                result_data["combined_groups"] = rg.get_ranks(combined_token)

            return True, None, result_data

        except Exception as e:
            return False, str(e), None

    def test_deepspeed_config(self, config: Dict) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """Test if a configuration is supported by DeepSpeed.

        Returns:
            (supported, error_message, support_info)
        """
        if not DEEPSPEED_AVAILABLE:
            return False, "DeepSpeed not available", None

        support_info = {
            "tp_supported": False,
            "dp_supported": False,
            "pp_supported": False,
            "cp_supported": False,
            "ep_supported": False,
            "sp_supported": False,
            "notes": [],
        }

        # Check TP support
        if config["tp"] > 1:
            support_info["tp_supported"] = hasattr(ds_groups, 'get_tensor_model_parallel_group')

        # Check DP support
        if config["dp"] > 1:
            support_info["dp_supported"] = hasattr(ds_groups, 'get_data_parallel_group')

        # Check PP support
        if config["pp"] > 1:
            # DeepSpeed supports PP via mpu or pipe module
            support_info["pp_supported"] = (hasattr(ds_groups, 'bwc_pipeline_parallel_world_size')
                                            or self._check_module_exists('deepspeed.pipe'))
            if not support_info["pp_supported"]:
                support_info["notes"].append("PP requires mpu object or deepspeed.pipe module")

        # Check CP support
        if config["cp"] > 1:
            support_info["cp_supported"] = hasattr(ds_groups, 'get_context_parallel_group')
            if not support_info["cp_supported"]:
                support_info["notes"].append("CP not supported in DeepSpeed")

        # Check EP support
        if config["ep"] > 1:
            support_info["ep_supported"] = (hasattr(ds_groups, '_create_expert_and_data_parallel')
                                            or hasattr(ds_groups, '_create_expert_data_and_model_parallel'))

        # Check SP support (DeepSpeed-specific)
        support_info["sp_supported"] = hasattr(ds_sp, 'initialize_sequence_parallel')

        # Determine overall support
        required_dims = [d for d in ["tp", "dp", "pp", "cp", "ep"] if config[d] > 1]
        supported_dims = []
        if config["tp"] > 1 and support_info["tp_supported"]:
            supported_dims.append("tp")
        if config["dp"] > 1 and support_info["dp_supported"]:
            supported_dims.append("dp")
        if config["pp"] > 1 and support_info["pp_supported"]:
            supported_dims.append("pp")
        if config["cp"] > 1 and support_info["cp_supported"]:
            supported_dims.append("cp")
        if config["ep"] > 1 and support_info["ep_supported"]:
            supported_dims.append("ep")

        fully_supported = len(supported_dims) == len(required_dims)

        return fully_supported, None, support_info

    def _check_module_exists(self, module_name):
        """Check if a module exists."""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def _simulate_deepspeed_rank_generation(self, config: Dict) -> Optional[Dict]:
        """Simulate DeepSpeed's rank generation logic based on code analysis.

        This attempts to replicate DeepSpeed's rank assignment logic for comparison.
        """
        if not DEEPSPEED_AVAILABLE:
            return None

        try:
            world_size = config["world_size"]
            result = {}

            # For TP+DP: DeepSpeed uses mesh_device which creates groups in a specific way
            if config["tp"] > 1 and config["dp"] > 1 and config["pp"] == 1 and config["cp"] == 1 and config["ep"] == 1:
                # DeepSpeed's _init_tp_mesh_device creates:
                # TP groups: [0,1], [2,3], [4,5], ... (consecutive)
                # DP groups: [0,2,4,...], [1,3,5,...] (strided)
                tp_size = config["tp"]
                dp_size = config["dp"]

                tp_groups = []
                for i in range(world_size // tp_size):
                    group = list(range(i * tp_size, (i + 1) * tp_size))
                    tp_groups.append(group)

                dp_groups = []
                for i in range(tp_size):
                    group = list(range(i, world_size, tp_size))
                    dp_groups.append(group)

                result["tp_groups"] = tp_groups
                result["dp_groups"] = dp_groups
                result["world_size"] = world_size
                return result

            # For other combinations, we can't easily simulate without actual distributed setup
            # But we can note that DeepSpeed supports it
            return {"supported": True, "note": "Rank generation requires actual distributed setup"}

        except Exception as e:
            return {"error": str(e)}

    def _compare_rank_groups(self, megatron_groups: List[List[int]], deepspeed_groups: List[List[int]]) -> Dict:
        """Compare rank groups from Megatron and DeepSpeed.

        Returns:
            Dict with comparison results
        """
        comparison = {"same_structure": False, "same_ranks": False, "differences": []}

        if not megatron_groups or not deepspeed_groups:
            return comparison

        # Check if same number of groups
        if len(megatron_groups) != len(deepspeed_groups):
            comparison["differences"].append(
                f"Group count mismatch: Megatron={len(megatron_groups)}, DeepSpeed={len(deepspeed_groups)}")
            return comparison

        # Check if same group sizes
        megatron_sizes = [len(g) for g in megatron_groups]
        deepspeed_sizes = [len(g) for g in deepspeed_groups]
        if megatron_sizes != deepspeed_sizes:
            comparison["differences"].append(
                f"Group size mismatch: Megatron={megatron_sizes}, DeepSpeed={deepspeed_sizes}")
            return comparison

        # Check if same ranks (order may differ)
        megatron_ranks = set()
        for group in megatron_groups:
            megatron_ranks.update(group)

        deepspeed_ranks = set()
        for group in deepspeed_groups:
            deepspeed_ranks.update(group)

        if megatron_ranks != deepspeed_ranks:
            comparison["differences"].append(
                f"Rank set mismatch: Megatron={sorted(megatron_ranks)}, DeepSpeed={sorted(deepspeed_ranks)}")
            return comparison

        # Check if same structure (same groups, possibly different order)
        megatron_sets = [set(g) for g in megatron_groups]
        deepspeed_sets = [set(g) for g in deepspeed_groups]

        if sorted(megatron_sets, key=lambda x: min(x)) == sorted(deepspeed_sets, key=lambda x: min(x)):
            comparison["same_structure"] = True
            comparison["same_ranks"] = True
        else:
            comparison["differences"].append("Group structure differs (same ranks but different grouping)")

        return comparison

    def test_config_compatibility(self, config: Dict):
        """Test compatibility of a configuration between both libraries."""
        # Get combination type for statistics
        combo_type = self.error_categorizer.get_combination_type(config)
        self.combination_stats[combo_type]["total"] += 1

        # Test Megatron
        megatron_success, megatron_error, megatron_result = self.test_megatron_config(config)

        # Test DeepSpeed
        deepspeed_success, deepspeed_error, deepspeed_support = self.test_deepspeed_config(config)

        # Record errors in categorizer
        if not megatron_success and megatron_error:
            self.error_categorizer.record_error(megatron_error, config, "Megatron")
            self.combination_stats[combo_type]["megatron_failures"] += 1
        else:
            self.combination_stats[combo_type]["megatron_success"] += 1

        if not deepspeed_success:
            # Get error message from support_info notes
            error_msg = deepspeed_support.get("notes", ["Not supported"])[0] if deepspeed_support else "Not supported"
            self.error_categorizer.record_error(error_msg, config, "DeepSpeed")
            self.combination_stats[combo_type]["deepspeed_failures"] += 1
        else:
            self.combination_stats[combo_type]["deepspeed_success"] += 1

        # Try to simulate DeepSpeed rank generation for comparison
        deepspeed_simulated = None
        if megatron_success and deepspeed_success:
            deepspeed_simulated = self._simulate_deepspeed_rank_generation(config)

        # Compare rank generation if both succeeded and we have simulated results
        rank_comparison = None
        if megatron_success and deepspeed_success and deepspeed_simulated and "tp_groups" in deepspeed_simulated:
            # Compare TP groups
            if config["tp"] > 1 and "tp_groups" in megatron_result:
                rank_comparison = self._compare_rank_groups(megatron_result["tp_groups"],
                                                            deepspeed_simulated.get("tp_groups", []))
            # Compare DP groups
            if config["dp"] > 1 and "dp_groups" in megatron_result and not rank_comparison:
                rank_comparison = self._compare_rank_groups(megatron_result["dp_groups"],
                                                            deepspeed_simulated.get("dp_groups", []))

        # Record results
        config_key = f"tp={config['tp']},dp={config['dp']},pp={config['pp']},cp={config['cp']},ep={config['ep']},order={config['order']}"

        if megatron_success:
            self.results["megatron_success"].append(config_key)
        else:
            self.results["megatron_failures"].append({
                "config": config_key,
                "error": megatron_error,
                "combination": combo_type,
            })

        if deepspeed_success:
            self.results["deepspeed_success"].append(config_key)
        else:
            self.results["deepspeed_failures"].append({
                "config": config_key,
                "error": deepspeed_error,
                "support_info": deepspeed_support,
                "combination": combo_type,
            })

        # Determine compatibility and update stats
        if megatron_success and deepspeed_success:
            compat_entry = {
                "config": config_key,
                "megatron_result": megatron_result,
                "deepspeed_support": deepspeed_support,
                "combination": combo_type,
            }
            if rank_comparison:
                compat_entry["rank_comparison"] = rank_comparison
                if rank_comparison.get("same_structure"):
                    compat_entry["rank_match"] = True
                else:
                    compat_entry["rank_match"] = False
                    compat_entry["rank_differences"] = rank_comparison.get("differences", [])
            self.results["compatible"].append(compat_entry)
            self.combination_stats[combo_type]["compatible"] += 1
        elif megatron_success and not deepspeed_success:
            self.results["megatron_only"].append({
                "config":
                config_key,
                "megatron_result":
                megatron_result,
                "deepspeed_issue":
                deepspeed_support.get("notes", []) if deepspeed_support else [],
                "combination":
                combo_type,
            })
            self.combination_stats[combo_type]["megatron_only"] += 1
        elif not megatron_success and deepspeed_success:
            self.results["deepspeed_only"].append({
                "config": config_key,
                "megatron_error": megatron_error,
                "deepspeed_support": deepspeed_support,
                "combination": combo_type,
            })
            self.combination_stats[combo_type]["deepspeed_only"] += 1
        else:
            self.results["incompatible"].append({
                "config":
                config_key,
                "megatron_error":
                megatron_error,
                "deepspeed_issue":
                deepspeed_support.get("notes", []) if deepspeed_support else [],
                "combination":
                combo_type,
            })
            self.combination_stats[combo_type]["incompatible"] += 1


class TestAutomatedParallelCombinations:
    """Automated tests for parallel strategy combinations."""

    def test_systematic_configurations(self):
        """Test systematic configurations covering common cases."""
        generator = ParallelConfigGenerator(seed=42)
        tester = ParallelCompatibilityTester()

        configs = generator.generate_systematic_configs(max_world_size=16)

        print("\n" + "=" * 80)
        print("SYSTEMATIC CONFIGURATION TESTING")
        print("=" * 80)
        print(f"\nTesting {len(configs)} systematic configurations...")

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing: {config}")
            tester.test_config_compatibility(config)

        self._print_results(tester, "Systematic")
        self._generate_comprehensive_report(tester, "Systematic")

    def test_random_configurations(self):
        """Test random configurations."""
        generator = ParallelConfigGenerator(seed=123)
        tester = ParallelCompatibilityTester()

        configs = generator.generate_random_configs(count=1000, max_size=1024)

        print("\n" + "=" * 80)
        print("RANDOM CONFIGURATION TESTING")
        print("=" * 80)
        print(f"\nTesting {len(configs)} random configurations...")
        print(f"Max world size: 1024, Max parallel size per dimension: 32")

        for i, config in enumerate(configs, 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(configs)} ({(i/len(configs)*100):.1f}%)")
            tester.test_config_compatibility(config)

        self._print_results(tester, "Random")
        self._generate_comprehensive_report(tester, "Random")

    def test_random_configurations_by_dimension(self):
        """Test random configurations generated separately for each dimension."""
        generator = ParallelConfigGenerator(seed=789)
        tester = ParallelCompatibilityTester()

        # Generate configs for each dimension separately
        # This ensures balanced coverage across all dimensions
        # Increased by 20x for comprehensive testing
        counts_by_dimension = {
            1: 4000,  # 1D: 4000 configs (200 * 20)
            2: 6000,  # 2D: 6000 configs (300 * 20) - more because there are more 2D combinations
            3: 5000,  # 3D: 5000 configs (250 * 20)
            4: 3000,  # 4D: 3000 configs (150 * 20)
            5: 2000,  # 5D: 2000 configs (100 * 20)
        }

        print("\n" + "=" * 80)
        print("RANDOM CONFIGURATION TESTING BY DIMENSION")
        print("=" * 80)
        print(f"\nGenerating configurations by dimension:")
        for dim, count in counts_by_dimension.items():
            print(f"  {dim}D: {count} configurations")

        configs = generator.generate_random_configs_by_dimension(counts_by_dimension=counts_by_dimension,
                                                                 max_size=1024,
                                                                 min_parallel_size=2,
                                                                 max_parallel_size=32)

        print(f"\nTotal unique configurations generated: {len(configs)}")
        print(f"Max world size: 1024, Parallel size range: 2-32")

        # Count configs by dimension
        dim_counts = defaultdict(int)
        for config in configs:
            dim_count = len([d for d in ["tp", "dp", "pp", "cp", "ep"] if config[d] > 1])
            dim_counts[dim_count] += 1

        print("\nActual distribution:")
        for dim in sorted(dim_counts.keys()):
            print(f"  {dim}D: {dim_counts[dim]} configurations")

        print(f"\nTesting {len(configs)} configurations...")

        for i, config in enumerate(configs, 1):
            # Update progress more frequently for large test sets
            if i % 1000 == 0 or i == len(configs):
                print(f"Progress: {i}/{len(configs)} ({(i/len(configs)*100):.1f}%)")
            tester.test_config_compatibility(config)

        self._print_results(tester, "Random by Dimension")
        self._generate_comprehensive_report(tester, "Random by Dimension")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        generator = ParallelConfigGenerator(seed=456)
        tester = ParallelCompatibilityTester()

        # Edge cases - including larger sizes
        edge_configs = [
            # Maximum dimensions - larger sizes
            {
                "tp": 8,
                "dp": 8,
                "pp": 8,
                "cp": 1,
                "ep": 1,
                "order": "tp-dp-pp",
                "world_size": 512
            },
            {
                "tp": 16,
                "dp": 16,
                "pp": 4,
                "cp": 1,
                "ep": 1,
                "order": "tp-dp-pp",
                "world_size": 1024
            },
            # EP and CP conflict
            {
                "tp": 2,
                "dp": 2,
                "pp": 1,
                "cp": 2,
                "ep": 2,
                "order": "tp-ep-dp",
                "world_size": 8
            },
            {
                "tp": 4,
                "dp": 4,
                "pp": 1,
                "cp": 4,
                "ep": 4,
                "order": "tp-ep-dp",
                "world_size": 64
            },
            # Single dimension - larger sizes
            {
                "tp": 1,
                "dp": 1,
                "pp": 64,
                "cp": 1,
                "ep": 1,
                "order": "pp",
                "world_size": 64
            },
            {
                "tp": 128,
                "dp": 1,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "tp",
                "world_size": 128
            },
            {
                "tp": 1,
                "dp": 256,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "dp",
                "world_size": 256
            },
            # All dimensions - larger sizes
            {
                "tp": 2,
                "dp": 2,
                "pp": 2,
                "cp": 2,
                "ep": 1,
                "order": "tp-pp-dp-cp",
                "world_size": 16
            },
            {
                "tp": 4,
                "dp": 4,
                "pp": 4,
                "cp": 4,
                "ep": 1,
                "order": "tp-pp-dp-cp",
                "world_size": 256
            },
            # Different orders
            {
                "tp": 2,
                "dp": 4,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "dp-tp",
                "world_size": 8
            },
            {
                "tp": 2,
                "dp": 4,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "tp-dp",
                "world_size": 8
            },
            {
                "tp": 8,
                "dp": 16,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "dp-tp",
                "world_size": 128
            },
            {
                "tp": 8,
                "dp": 16,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "tp-dp",
                "world_size": 128
            },
            # Large multi-dimensional
            {
                "tp": 8,
                "dp": 8,
                "pp": 4,
                "cp": 1,
                "ep": 1,
                "order": "tp-pp-dp",
                "world_size": 256
            },
            {
                "tp": 4,
                "dp": 8,
                "pp": 8,
                "cp": 1,
                "ep": 1,
                "order": "tp-pp-dp",
                "world_size": 256
            },
        ]

        print("\n" + "=" * 80)
        print("EDGE CASE TESTING")
        print("=" * 80)
        print(f"\nTesting {len(edge_configs)} edge case configurations...")

        for i, config in enumerate(edge_configs, 1):
            print(f"\n[{i}/{len(edge_configs)}] Testing: {config}")
            tester.test_config_compatibility(config)

        self._print_results(tester, "Edge Cases")
        self._generate_comprehensive_report(tester, "Edge Cases")

    def _print_results(self, tester: ParallelCompatibilityTester, test_type: str):
        """Print test results."""
        results = tester.results

        print("\n" + "=" * 80)
        print(f"{test_type} TEST RESULTS")
        print("=" * 80)

        print(f"\n✓ Megatron Success: {len(results['megatron_success'])}")
        print(f"✗ Megatron Failures: {len(results['megatron_failures'])}")
        if results['megatron_failures']:
            print("\nMegatron Failures:")
            for failure in results['megatron_failures'][:10]:  # Show first 10
                print(f"  - {failure['config']}: {failure['error']}")
            if len(results['megatron_failures']) > 10:
                print(f"  ... and {len(results['megatron_failures']) - 10} more")

        print(f"\n✓ DeepSpeed Success: {len(results['deepspeed_success'])}")
        print(f"✗ DeepSpeed Failures: {len(results['deepspeed_failures'])}")
        if results['deepspeed_failures']:
            print("\nDeepSpeed Failures:")
            for failure in results['deepspeed_failures'][:10]:  # Show first 10
                print(f"  - {failure['config']}")
                if failure.get('support_info'):
                    notes = failure['support_info'].get('notes', [])
                    if notes:
                        print(f"    Notes: {', '.join(notes)}")
            if len(results['deepspeed_failures']) > 10:
                print(f"  ... and {len(results['deepspeed_failures']) - 10} more")

        print(f"\n✓ Compatible (Both Support): {len(results['compatible'])}")
        if results['compatible']:
            print("  Examples:")
            rank_matches = 0
            rank_mismatches = 0
            for item in results['compatible'][:10]:
                if isinstance(item, dict):
                    config = item.get('config', 'Unknown')
                    rank_comp = item.get('rank_comparison')
                    if rank_comp:
                        if rank_comp.get('same_structure'):
                            print(f"    - {config} ✓ Rank groups match")
                            rank_matches += 1
                        else:
                            print(f"    - {config} ⚠ Rank groups differ")
                            rank_mismatches += 1
                            if rank_comp.get('differences'):
                                for diff in rank_comp['differences'][:2]:
                                    print(f"      {diff}")
                    else:
                        print(f"    - {config}")
                else:
                    print(f"    - {item}")
            if len(results['compatible']) > 10:
                print(f"    ... and {len(results['compatible']) - 10} more")

            if rank_matches > 0 or rank_mismatches > 0:
                print(f"\n  Rank Comparison Summary:")
                print(f"    Matches: {rank_matches}")
                print(f"    Mismatches: {rank_mismatches}")
                print(f"    (Note: Comparison only available for TP+DP combinations)")

        print(f"\n⚠ Megatron Only: {len(results['megatron_only'])}")
        if results['megatron_only']:
            print("  Examples:")
            for item in results['megatron_only'][:5]:
                print(f"    - {item['config']}")
                if item.get('deepspeed_issue'):
                    print(f"      DeepSpeed issue: {', '.join(item['deepspeed_issue'])}")
            if len(results['megatron_only']) > 5:
                print(f"    ... and {len(results['megatron_only']) - 5} more")

        print(f"\n→ DeepSpeed Only: {len(results['deepspeed_only'])}")
        if results['deepspeed_only']:
            print("  Examples:")
            for item in results['deepspeed_only'][:5]:
                print(f"    - {item['config']}")
                print(f"      Megatron error: {item['megatron_error']}")
            if len(results['deepspeed_only']) > 5:
                print(f"    ... and {len(results['deepspeed_only']) - 5} more")

        print(f"\n✗ Incompatible (Neither Support): {len(results['incompatible'])}")
        if results['incompatible']:
            print("  Examples:")
            for item in results['incompatible'][:5]:
                print(f"    - {item['config']}")
                print(f"      Megatron: {item['megatron_error']}")
            if len(results['incompatible']) > 5:
                print(f"    ... and {len(results['incompatible']) - 5} more")

        print("\n" + "=" * 80)

    def _generate_comprehensive_report(self, tester: ParallelCompatibilityTester, test_type: str):
        """Generate comprehensive test report with error categorization and combination statistics."""
        results = tester.results
        error_summary = tester.error_categorizer.get_error_summary()
        combo_stats = tester.combination_stats

        print("\n" + "=" * 80)
        print(f"{test_type} COMPREHENSIVE TEST REPORT")
        print("=" * 80)

        # Overall statistics
        print("\n" + "-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        total_tested = (len(results['megatron_success']) + len(results['megatron_failures']) +
                        len(results['deepspeed_success']) + len(results['deepspeed_failures']))
        print(f"Total Configurations Tested: {total_tested}")
        print(
            f"  Megatron Success: {len(results['megatron_success'])} ({len(results['megatron_success'])/total_tested*100:.1f}%)"
        )
        print(
            f"  Megatron Failures: {len(results['megatron_failures'])} ({len(results['megatron_failures'])/total_tested*100:.1f}%)"
        )
        print(
            f"  DeepSpeed Success: {len(results['deepspeed_success'])} ({len(results['deepspeed_success'])/total_tested*100:.1f}%)"
        )
        print(
            f"  DeepSpeed Failures: {len(results['deepspeed_failures'])} ({len(results['deepspeed_failures'])/total_tested*100:.1f}%)"
        )
        print(f"  Compatible: {len(results['compatible'])} ({len(results['compatible'])/total_tested*100:.1f}%)")
        print(
            f"  Megatron Only: {len(results['megatron_only'])} ({len(results['megatron_only'])/total_tested*100:.1f}%)"
        )
        print(
            f"  DeepSpeed Only: {len(results['deepspeed_only'])} ({len(results['deepspeed_only'])/total_tested*100:.1f}%)"
        )
        print(f"  Incompatible: {len(results['incompatible'])} ({len(results['incompatible'])/total_tested*100:.1f}%)")

        # Error categorization
        print("\n" + "-" * 80)
        print("ERROR CATEGORIZATION (Aggregated by Type)")
        print("-" * 80)
        for category, summary in sorted(error_summary.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"\n{category}: {summary['count']} occurrences")
            print(f"  Affects {summary['unique_combinations']} unique combination types")
            print(f"  Examples:")
            for example in summary['examples'][:3]:
                combo = example.get('combination', 'Unknown')
                lib = example.get('library', 'Unknown')
                print(f"    - {combo} ({lib}): {example['error'][:80]}")
            if len(summary['examples']) > 3:
                print(f"    ... and {len(summary['examples']) - 3} more examples")

        # Combination type statistics
        print("\n" + "-" * 80)
        print("COMBINATION TYPE STATISTICS")
        print("-" * 80)
        print(
            f"{'Combination':<20} {'Total':<8} {'M-Succ':<8} {'M-Fail':<8} {'DS-Succ':<8} {'DS-Fail':<8} {'Compat':<8} {'M-Only':<8} {'DS-Only':<8} {'Incomp':<8}"
        )
        print("-" * 100)

        # Sort by total count
        sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        for combo_type, stats in sorted_combos:
            if stats['total'] > 0:
                print(f"{combo_type:<20} {stats['total']:<8} {stats['megatron_success']:<8} "
                      f"{stats['megatron_failures']:<8} {stats['deepspeed_success']:<8} "
                      f"{stats['deepspeed_failures']:<8} {stats['compatible']:<8} "
                      f"{stats['megatron_only']:<8} {stats['deepspeed_only']:<8} "
                      f"{stats['incompatible']:<8}")

        # Detailed combination analysis
        print("\n" + "-" * 80)
        print("DETAILED COMBINATION ANALYSIS")
        print("-" * 80)

        # Group by number of dimensions
        by_dimension_count = defaultdict(list)
        for combo_type, stats in combo_stats.items():
            dim_count = len([c for c in combo_type.split('+') if c != 'NONE'])
            by_dimension_count[dim_count].append((combo_type, stats))

        for dim_count in sorted(by_dimension_count.keys()):
            print(f"\n{dim_count}-Dimensional Combinations:")
            combos = sorted(by_dimension_count[dim_count], key=lambda x: x[1]['total'], reverse=True)
            for combo_type, stats in combos[:10]:  # Show top 10
                if stats['total'] > 0:
                    compat_rate = (stats['compatible'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    print(f"  {combo_type}:")
                    print(f"    Total: {stats['total']}, Compatible: {stats['compatible']} ({compat_rate:.1f}%)")
                    print(f"    Megatron: {stats['megatron_success']} success, {stats['megatron_failures']} failures")
                    print(
                        f"    DeepSpeed: {stats['deepspeed_success']} success, {stats['deepspeed_failures']} failures")
            if len(combos) > 10:
                print(f"    ... and {len(combos) - 10} more {dim_count}-dimensional combinations")

        print("\n" + "=" * 80)

    def test_cp_vs_sp_compatibility_by_dimension(self):
        """Test CP vs SP compatibility using the same config generation as test_random_configurations_by_dimension.

        This test:
        1. Uses parallel_state_refactored with CP
        2. Uses DeepSpeed with SP
        3. Compares CP rank groups with SP rank groups to see if they match
        """
        generator = ParallelConfigGenerator(seed=789)

        # Use the same configuration generation as test_random_configurations_by_dimension
        counts_by_dimension = {
            1: 4000,  # 1D: 4000 configs
            2: 6000,  # 2D: 6000 configs
            3: 5000,  # 3D: 5000 configs
            4: 3000,  # 4D: 3000 configs
            5: 2000,  # 5D: 2000 configs
        }

        print("\n" + "=" * 80)
        print("CP vs SP COMPATIBILITY TESTING BY DIMENSION")
        print("=" * 80)
        print(f"\nGenerating configurations by dimension:")
        for dim, count in counts_by_dimension.items():
            print(f"  {dim}D: {count} configurations")

        configs = generator.generate_random_configs_by_dimension(counts_by_dimension=counts_by_dimension,
                                                                 max_size=1024,
                                                                 min_parallel_size=2,
                                                                 max_parallel_size=32)

        # Filter to only include configs with CP > 1 and EP == 1 (EP and CP cannot both be > 1)
        configs_with_cp = [c for c in configs if c["cp"] > 1 and c["ep"] == 1]

        print(f"\nTotal unique configurations generated: {len(configs)}")
        print(f"Configurations with CP > 1 and EP == 1: {len(configs_with_cp)}")
        print(f"Max world size: 1024, Parallel size range: 2-32")

        # Test CP vs SP compatibility
        results = {
            "total_tested": 0,
            "cp_groups_generated": 0,
            "sp_groups_generated": 0,
            "rank_groups_match": 0,
            "rank_groups_differ": 0,
            "errors": 0,
            "match_details": [],
            "differ_details": [],
        }

        combination_stats = defaultdict(lambda: {
            "total": 0,
            "match": 0,
            "differ": 0,
            "errors": 0,
        })

        print(f"\nTesting {len(configs_with_cp)} configurations for CP vs SP compatibility...")

        for i, config in enumerate(configs_with_cp, 1):
            if i % 1000 == 0 or i == len(configs_with_cp):
                print(f"Progress: {i}/{len(configs_with_cp)} ({(i/len(configs_with_cp)*100):.1f}%)")

            results["total_tested"] += 1

            # Get combination type
            combo_type = self._get_combination_type_for_cp_sp(config)
            combination_stats[combo_type]["total"] += 1

            try:
                # Get CP rank groups from Megatron
                if not PARALLEL_STATE_AVAILABLE:
                    results["errors"] += 1
                    combination_stats[combo_type]["errors"] += 1
                    continue

                rg = RankGenerator(tp=config["tp"],
                                   ep=config["ep"],
                                   dp=config["dp"],
                                   pp=config["pp"],
                                   cp=config["cp"],
                                   order=config["order"])

                cp_groups = rg.get_ranks("cp")
                if cp_groups:
                    results["cp_groups_generated"] += 1

                # Simulate SP rank groups from DeepSpeed
                # DeepSpeed SP creates consecutive rank groups
                sp_groups = self._simulate_deepspeed_sp_groups(config["world_size"], config["cp"])
                if sp_groups:
                    results["sp_groups_generated"] += 1

                # Compare CP and SP groups
                if self._compare_cp_sp_groups(cp_groups, sp_groups):
                    results["rank_groups_match"] += 1
                    combination_stats[combo_type]["match"] += 1
                    results["match_details"].append(config)
                else:
                    results["rank_groups_differ"] += 1
                    combination_stats[combo_type]["differ"] += 1
                    results["differ_details"].append({
                        "config": config,
                        "cp_groups": cp_groups,
                        "sp_groups": sp_groups,
                    })

            except Exception as e:
                results["errors"] += 1
                combination_stats[combo_type]["errors"] += 1

        # Generate report
        self._generate_cp_vs_sp_report(results, combination_stats)

    def _simulate_deepspeed_sp_groups(self, world_size: int, sp_size: int) -> List[List[int]]:
        """Simulate DeepSpeed's SP rank group generation.

        DeepSpeed SP creates groups as consecutive ranks:
        - Group 0: [0, 1, ..., sp_size-1]
        - Group 1: [sp_size, sp_size+1, ..., 2*sp_size-1]
        - etc.
        """
        if sp_size <= 1 or world_size % sp_size != 0:
            return []

        num_groups = world_size // sp_size
        groups = []
        for i in range(num_groups):
            group = list(range(i * sp_size, (i + 1) * sp_size))
            groups.append(group)

        return groups

    def _compare_cp_sp_groups(self, cp_groups: List[List[int]], sp_groups: List[List[int]]) -> bool:
        """Compare CP and SP rank groups to see if they match."""
        if not cp_groups and not sp_groups:
            return True

        if not cp_groups or not sp_groups:
            return False

        if len(cp_groups) != len(sp_groups):
            return False

        # Check if all CP groups have a matching SP group (order may differ)
        cp_sets = [set(g) for g in cp_groups]
        sp_sets = [set(g) for g in sp_groups]

        # Check if all CP groups match SP groups
        for cp_set in cp_sets:
            found = False
            for sp_set in sp_sets:
                if cp_set == sp_set:
                    found = True
                    break
            if not found:
                return False

        # Check if all SP groups match CP groups
        for sp_set in sp_sets:
            found = False
            for cp_set in cp_sets:
                if sp_set == cp_set:
                    found = True
                    break
            if not found:
                return False

        return True

    def _get_combination_type_for_cp_sp(self, config: Dict) -> str:
        """Get combination type string for CP vs SP testing."""
        dims = []
        if config["tp"] > 1:
            dims.append("TP")
        if config["dp"] > 1:
            dims.append("DP")
        if config["pp"] > 1:
            dims.append("PP")
        if config["cp"] > 1:
            dims.append("CP")
        # Note: EP is always 1 in this test

        if not dims:
            return "NONE"

        return "+".join(sorted(dims))

    def _generate_cp_vs_sp_report(self, results: Dict, combination_stats: Dict):
        """Generate comprehensive CP vs SP compatibility report."""
        print("\n" + "=" * 80)
        print("CP vs SP COMPATIBILITY TEST REPORT")
        print("=" * 80)

        # Overall statistics
        print("\n" + "-" * 80)
        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total Configurations Tested: {results['total_tested']}")
        print(f"  CP Groups Generated: {results['cp_groups_generated']}")
        print(f"  SP Groups Generated: {results['sp_groups_generated']}")
        print(f"  Rank Groups Match: {results['rank_groups_match']}")
        print(f"  Rank Groups Differ: {results['rank_groups_differ']}")
        print(f"  Errors: {results['errors']}")

        if results['total_tested'] > 0:
            match_rate = (results['rank_groups_match'] / results['total_tested']) * 100
            print(f"\n  Match Rate: {match_rate:.2f}%")
            print(f"  CP can replace SP in {match_rate:.2f}% of tested configurations")

        # Combination type statistics
        print("\n" + "-" * 80)
        print("COMBINATION TYPE STATISTICS")
        print("-" * 80)
        print(f"{'Combination':<20} {'Total':<8} {'Match':<8} {'Differ':<8} {'Errors':<8} {'Match Rate':<12}")
        print("-" * 80)

        sorted_combos = sorted(combination_stats.items(), key=lambda x: x[1]['total'], reverse=True)
        for combo_type, stats in sorted_combos:
            if stats['total'] > 0:
                match_rate = (stats['match'] / stats['total'] * 100) if stats['total'] > 0 else 0
                print(f"{combo_type:<20} {stats['total']:<8} {stats['match']:<8} "
                      f"{stats['differ']:<8} {stats['errors']:<8} {match_rate:.1f}%")

        # Examples of matching configurations
        print("\n" + "-" * 80)
        print("EXAMPLES OF MATCHING CONFIGURATIONS (CP can replace SP)")
        print("-" * 80)
        for i, config in enumerate(results['match_details'][:10], 1):
            print(f"{i}. {config}")
            print(f"   CP size: {config['cp']}, Order: {config['order']}")

        if len(results['match_details']) > 10:
            print(f"\n... and {len(results['match_details']) - 10} more matching configurations")

        # Examples of differing configurations
        if results['differ_details']:
            print("\n" + "-" * 80)
            print("EXAMPLES OF DIFFERING CONFIGURATIONS (CP cannot replace SP)")
            print("-" * 80)
            for i, item in enumerate(results['differ_details'][:10], 1):
                config = item['config']
                cp_groups = item['cp_groups']
                sp_groups = item['sp_groups']
                print(f"{i}. {config}")
                print(f"   CP size: {config['cp']}, Order: {config['order']}")
                print(f"   CP groups count: {len(cp_groups)}, SP groups count: {len(sp_groups)}")
                if cp_groups and sp_groups:
                    print(f"   CP first group: {cp_groups[0]}")
                    print(f"   SP first group: {sp_groups[0]}")

            if len(results['differ_details']) > 10:
                print(f"\n... and {len(results['differ_details']) - 10} more differing configurations")

        # Conclusion
        print("\n" + "=" * 80)
        print("CONCLUSION")
        print("=" * 80)
        if results['rank_groups_match'] > 0:
            match_rate = (results['rank_groups_match'] / results['total_tested']) * 100
            print(f"\n✓ CP can replace SP in {match_rate:.2f}% of tested configurations")
            print(
                f"  - {results['rank_groups_match']} out of {results['total_tested']} configurations have matching rank groups"
            )
        else:
            print("\n✗ CP cannot replace SP in any of the tested configurations")

        if results['rank_groups_differ'] > 0:
            print(f"\n⚠ {results['rank_groups_differ']} configurations have different rank groups")
            print("  - These configurations may require special handling when migrating from CP to SP")

        print("\n" + "=" * 80)

    def test_comprehensive_automated_testing(self):
        """Comprehensive automated testing with all test types."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE AUTOMATED PARALLEL COMBINATION TESTING")
        print("=" * 80)

        # Create a combined tester for overall report
        combined_tester = ParallelCompatibilityTester()

        # Run all test types and accumulate results
        print("\n[1/3] Running systematic configurations...")
        generator1 = ParallelConfigGenerator(seed=42)
        configs1 = generator1.generate_systematic_configs(max_world_size=512)
        print(f"Testing {len(configs1)} systematic configurations...")
        for i, config in enumerate(configs1, 1):
            if i % 50 == 0 or i == len(configs1):
                print(f"  Progress: {i}/{len(configs1)}")
            combined_tester.test_config_compatibility(config)

        print("\n[2/4] Running random configurations by dimension...")
        generator2 = ParallelConfigGenerator(seed=789)
        # Increased by 20x for comprehensive testing
        counts_by_dimension = {
            1: 4000,  # 1D: 4000 configs (200 * 20)
            2: 6000,  # 2D: 6000 configs (300 * 20)
            3: 5000,  # 3D: 5000 configs (250 * 20)
            4: 3000,  # 4D: 3000 configs (150 * 20)
            5: 2000,  # 5D: 2000 configs (100 * 20)
        }
        configs2 = generator2.generate_random_configs_by_dimension(counts_by_dimension=counts_by_dimension,
                                                                   max_size=1024,
                                                                   min_parallel_size=2,
                                                                   max_parallel_size=32)
        print(f"Testing {len(configs2)} random configurations (balanced by dimension)...")
        print(f"Max world size: 1024, Parallel size range: 2-32")
        for i, config in enumerate(configs2, 1):
            # Update progress more frequently for large test sets
            if i % 1000 == 0 or i == len(configs2):
                print(f"  Progress: {i}/{len(configs2)} ({(i/len(configs2)*100):.1f}%)")
            combined_tester.test_config_compatibility(config)

        print("\n[3/4] Running additional random configurations...")
        generator3 = ParallelConfigGenerator(seed=123)
        # Increased by 20x: 500 * 20 = 10000
        configs3 = generator3.generate_random_configs(count=10000, max_size=1024)
        print(f"Testing {len(configs3)} additional random configurations...")
        for i, config in enumerate(configs3, 1):
            # Update progress more frequently for large test sets
            if i % 1000 == 0 or i == len(configs3):
                print(f"  Progress: {i}/{len(configs3)} ({(i/len(configs3)*100):.1f}%)")
            combined_tester.test_config_compatibility(config)

        print("\n[4/4] Running edge cases...")
        edge_configs = [
            {
                "tp": 8,
                "dp": 8,
                "pp": 8,
                "cp": 1,
                "ep": 1,
                "order": "tp-dp-pp",
                "world_size": 512
            },
            {
                "tp": 16,
                "dp": 16,
                "pp": 4,
                "cp": 1,
                "ep": 1,
                "order": "tp-dp-pp",
                "world_size": 1024
            },
            {
                "tp": 2,
                "dp": 2,
                "pp": 1,
                "cp": 2,
                "ep": 2,
                "order": "tp-ep-dp",
                "world_size": 8
            },
            {
                "tp": 4,
                "dp": 4,
                "pp": 1,
                "cp": 4,
                "ep": 4,
                "order": "tp-ep-dp",
                "world_size": 64
            },
            {
                "tp": 1,
                "dp": 1,
                "pp": 64,
                "cp": 1,
                "ep": 1,
                "order": "pp",
                "world_size": 64
            },
            {
                "tp": 128,
                "dp": 1,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "tp",
                "world_size": 128
            },
            {
                "tp": 1,
                "dp": 256,
                "pp": 1,
                "cp": 1,
                "ep": 1,
                "order": "dp",
                "world_size": 256
            },
            {
                "tp": 2,
                "dp": 2,
                "pp": 2,
                "cp": 2,
                "ep": 1,
                "order": "tp-pp-dp-cp",
                "world_size": 16
            },
            {
                "tp": 4,
                "dp": 4,
                "pp": 4,
                "cp": 4,
                "ep": 1,
                "order": "tp-pp-dp-cp",
                "world_size": 256
            },
        ]
        print(f"Testing {len(edge_configs)} edge case configurations...")
        for config in edge_configs:
            combined_tester.test_config_compatibility(config)

        # Generate comprehensive report
        print("\n" + "=" * 80)
        print("COMPREHENSIVE FINAL REPORT")
        print("=" * 80)
        self._generate_comprehensive_report(combined_tester, "COMPREHENSIVE")

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
