# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import shutil
import subprocess
import time
import datetime
import math
import hjson

from ..runtime.config_utils import dict_raise_error_on_duplicate_keys
from ..runtime.constants import *

from ..runtime.zero.config import ZERO_OPTIMIZATION, ZeroStageEnum
from ..utils import logger
from .config import DeepSpeedAutotuningConfig
from .constants import *
from .scheduler import ResourceManager
from .tuner import GridSearchTuner, RandomTuner, ModelBasedTuner
from .utils import *
from deepspeed.accelerator import get_accelerator

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

try:
    import mlflow
    has_mlflow = True
except Exception as e:
    has_mlflow = False

ZERO_OPTIMIZATION_STAGE = "stage"
OFFLOAD_OPTIMIZER = "offload_optimizer"
OFFLOAD_PARAM = "offload_param"
ZERO_OPTIMIZATION_STAGE_DEFAULT = ZeroStageEnum.disabled


class Autotuner:
    """The DeepSpeed Autotuner automatically discovers the optimal DeepSpeed configuration that delivers good training speed. The Autotuner uses model information, system information, and heuristics to efficiently tune system knobs that affect compute and memory efficiencies, such as ZeRO optimization stages, micro-batch sizes, and many other ZeRO optimization configurations. It not only reduces the time and resources user spend on tuning, but also can discover configurations better than hand-tuned methods.
    Autotuning with DeepSpeed requires no code change from DeepSpeed users. Please refer to the README for usage details.
    """

    def __init__(self, args, active_resources):
        self.args = args
        self.selected_exp_dir = None

        assert tabulate is not None, "Missing required package `tabulate`, please install with `pip install deepspeed[autotuning]`."

        logger.debug(f"autotuning args={args}")

        self.user_config = self._get_user_config(args.user_args)
        assert self.user_config is not None, "DeepSpeed configuration is not provided"

        self.autotuning_config = DeepSpeedAutotuningConfig(self.user_config)
        if self.user_config[AUTOTUNING]:
            if AUTOTUNING_EXPS_DIR in self.user_config[AUTOTUNING].keys():
                del self.user_config[AUTOTUNING][AUTOTUNING_EXPS_DIR]
            if AUTOTUNING_RESULTS_DIR in self.user_config[AUTOTUNING].keys():
                del self.user_config[AUTOTUNING][AUTOTUNING_RESULTS_DIR]

        self.exps_dir = self.autotuning_config.exps_dir
        if self.autotuning_config.overwrite and os.path.exists(self.exps_dir):
            shutil.rmtree(self.exps_dir, ignore_errors=True)
        if not os.path.exists(self.exps_dir):
            try:
                os.makedirs(self.exps_dir, exist_ok=True)
                logger.info(f"Created autotuning experiments directory: {self.exps_dir}")
            except:
                logger.error(
                    f"Failed to create {self.exps_dir}, please check `exps_dir` in the autotuning config file is accessible by all the nodes in the job."
                )
                exit(-1)

        self.results_dir = self.autotuning_config.results_dir
        if self.autotuning_config.overwrite and os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir, ignore_errors=True)
        if not os.path.exists(self.results_dir):
            try:
                os.makedirs(self.results_dir, exist_ok=True)
                logger.info(f"Created autotuning results directory: {self.exps_dir}")
            except:
                logger.error(
                    f"Failed to create {self.results_dir}, please check `results_dir` in the autotuning config file is accessible by all the nodes in the job."
                )
                exit(-1)

        # set the active resource for the autotuner resource manager
        self.rm = self._get_resource_manager(active_resources)

        # get resource requirement for each autotuning experiment
        self.exp_num_nodes, self.exp_num_gpus = self._get_exp_resources(args)

        assert self.exp_num_gpus <= self.rm.num_gpus_per_node, "num_gpus in the autotuning configuration must not be less than the --num_gpus value in the train script if any"
        assert self.exp_num_nodes <= len(
            self.rm.nodes
        ), "num_nodes in the autotuning configuration must not be less than the --num_nodes value in the train script if any"

        self.records = {}
        self.optimal_cmd = None
        self.optimal_ds_config = None

        self.mlflow_parent_id = None

    def print_tuning_results(self):
        """Print the autotuning results in tabular format.
        """
        best_space_records = self.get_best_space_records()
        tab = []
        if best_space_records:
            for key, val in best_space_records.items():
                if not val:
                    continue
                row = []
                row.append(key)
                num_exps = 0
                if key == GLOBAL_TUNING_SPACE:
                    cnt = 0
                    for k, v in best_space_records.items():
                        if k != GLOBAL_TUNING_SPACE:
                            cnt += v[2]
                    num_exps = cnt
                else:
                    num_exps = val[2]
                row.append(num_exps)
                row.append(val[1])
                row.append(val[0]['name'])
                tab.append(row)
            summary = tabulate(tab,
                               headers=["tuning_space", "num_experiments", "best_metric_val", "best_exp_name"],
                               tablefmt="pipe")
            print(summary)
            with open(os.path.join(self.results_dir, 'summary.txt'), 'w', buffering=BUFSIZE) as fd:
                fd.write(summary)
                fd.flush()
                os.fsync(fd)

        if GLOBAL_TUNING_SPACE in best_space_records:
            best_exp, best_metric_val, total_num_exps = best_space_records[GLOBAL_TUNING_SPACE]
            if best_exp:
                logger.info(
                    f"{best_exp['name']} is the optimal setup after tuning. The exp result is at {best_exp['result_dir']}."
                )
            else:
                logger.info(f"No optimal setup is found. Please check that experiments were run successfully.")
            tuning_duration = datetime.timedelta(seconds=(time.time() - self.start_time))

            logger.info(f"Tuning completed in {tuning_duration}")
            with open(os.path.join(self.results_dir, 'summary.txt'), 'a') as f:
                f.write(
                    f"\n\nTuning completed in {tuning_duration}. Total number of experiments: {self.rm.experiment_count - 1}."
                )
                f.flush()

    def _get_user_config(self, user_args):
        """Get DeepSpeed configuration from the user arguments passed to the launcher.

        Args:
            user_args ([list]): user arguments passed to the DeepSpeed launcher

        Returns:
            [dict]: DeepSpeed configuration dictionary
        """
        user_config_file = None
        if "--deepspeed_config" in user_args:
            idx = user_args.index("--deepspeed_config")
            assert ".json" in user_args[
                idx + 1], "DeepSpeed --deepspeed_config requires a json file to specify the configuration"

            user_config_file = user_args[idx + 1]
        elif "--deepspeed" in user_args:
            idx = user_args.index("--deepspeed")
            if ".json" in user_args[idx + 1]:
                user_config_file = user_args[idx + 1]

        logger.debug(f"user_config_file = {user_config_file}")
        if user_config_file is not None:
            assert os.path.isfile(user_config_file), "DeepSpeed configuration file: {} is not an existing file".format(
                user_config_file)
            if os.path.exists(user_config_file):
                return json.load(open(user_config_file, "r"), object_pairs_hook=dict_raise_error_on_duplicate_keys)

        return None

    def _get_resource_manager(self, active_resources):
        """Initialize and return a resource manager

        Args:
            active_resources ([dict]): A dictionary of hostname and its slots (GPUs), e.g. {"worker-0": "0,1,2,3,4,5,6,7,8"}

        Raises:
            RuntimeError: raises the error if no GPU is available

        Returns:
            [ResourceManager]: A resource manager that schedules and runs autotuning experiments.
        """
        logger.info(f"active_resources = {active_resources}")

        hosts = []
        ngpus_per_node = 100
        for hostname, slots in active_resources.items():
            hosts.append(hostname)
            ngpus_per_node = min(len(slots), ngpus_per_node)

        assert ngpus_per_node > 0, "no gpu is available"

        return ResourceManager(args=self.args,
                               hosts=hosts,
                               num_gpus_per_node=ngpus_per_node,
                               results_dir=self.results_dir,
                               exps_dir=self.exps_dir,
                               arg_mappings=self.autotuning_config.arg_mappings)

    def _get_exp_resources(self, args):
        """Get resource requirement for each autotuning experiment

        Args:
            args (dict): user args

        Returns:
            num_nodes, num_gpus: the number of gpus and number of nodes used in the autotuning experiments
        """
        if args.num_nodes > 0:
            num_nodes = args.num_nodes
        else:
            num_nodes = len(self.rm.nodes)

        if args.num_gpus > 0:
            num_gpus = args.num_gpus
        else:
            num_gpus = self.rm.num_gpus_per_node

        return num_nodes, num_gpus

    def metric(self):
        return self.autotuning_config.metric

    def fast_enabled(self):
        return self.autotuning_config.fast

    def max_train_batch_size(self):
        return self.autotuning_config.max_train_batch_size

    def mp_size(self):
        return self.autotuning_config.mp_size

    def max_train_micro_batch_size_per_gpu(self):
        if self.max_train_batch_size(
        ) and self.max_train_batch_size() > 0:  # if the user specifies a max_train_batch_size
            max_train_micro_batch_size = self.max_train_batch_size() * self.mp_size() // (
                self.exp_num_gpus * self.exp_num_nodes)  # gradient accumulation steps >=1
            return min(self.autotuning_config.max_train_micro_batch_size_per_gpu, max_train_micro_batch_size)
        else:
            return self.autotuning_config.max_train_micro_batch_size_per_gpu

    def min_train_micro_batch_size_per_gpu(self):
        return self.autotuning_config.min_train_micro_batch_size_per_gpu

    def num_tuning_micro_batch_sizes(self):
        return self.autotuning_config.num_tuning_micro_batch_sizes

    def fp16_enabled(self):
        if FP16 in self.user_config.keys():
            return self.user_config[FP16].get(FP16_ENABLED, FP16_ENABLED_DEFAULT)
        else:
            return False

    def get_gpu_memory_info(self):
        return get_accelerator().total_memory()

    def get_activation_memory_per_gpu(self):
        if self.model_info and "activation_mem_per_gpu" in self.model_info:
            return self.model_info["activation_mem_per_gpu"]

    def get_instantiation_memory_required_per_gpu(self, zero_stage):
        num_params = self.get_model_num_params()
        total_gpus = self.exp_num_nodes * self.exp_num_gpus
        fp16_enabled = self.fp16_enabled()

        if not num_params:
            return 0
        # assume the model uses Adam optimizer
        # ZeroStageEnum.disabled:
        params_mem = num_params * (2 if fp16_enabled else 4)
        gradients_mem = num_params * (2 if fp16_enabled else 4)
        optimizer_mem = num_params * (16 if fp16_enabled else 8)

        if zero_stage >= ZeroStageEnum.optimizer_states:
            optimizer_mem = optimizer_mem / total_gpus

        if zero_stage >= ZeroStageEnum.gradients:
            gradients_mem = gradients_mem / total_gpus

        if zero_stage >= ZeroStageEnum.weights:
            params_mem = params_mem / total_gpus

        mem_per_gpu = (params_mem + gradients_mem + optimizer_mem) / self.mp_size()

        return mem_per_gpu

    def _generate_experiments(self, tuning_space, max_train_batch_size_per_gpu):
        """Generates a list of autotuning experiments given a tuning_space.
            The corresponding parameter values are replaced by user-defined values in the DeepSpeed configuration file.
        Args:
            tuning_space ([dict]): A DeepSpeed configuration dictionary where a value can be a list (called a tuning parameter). For example,
                {
                    "zero_optimization": {
                        "stage": 1,
                        "reduce_bucket_size": [5e7,
                                            5e8,
                                            1e9],
                        "allgather_bucket_size": [5e7,
                                                5e8,
                                                1e9],
                    }
                }
                reduce_bucket_size and allgather_bucket_size are the tuning parameters in this tuning space.
        Returns:
            [list]: a list of experiments generated by taking combinations of values of the tuning space. The above tuning space generates 3*3 = 9 experiments if the user DeepSpeed configuration file does not overwrite the two tuning parameters or define more tuning parameters.
        """
        exps = []

        # each zero stage uses a different template configuration file
        config_zero = tuning_space.get(ZERO_OPTIMIZATION, {})
        stage = config_zero.get(ZERO_OPTIMIZATION_STAGE, ZERO_OPTIMIZATION_STAGE_DEFAULT)
        template_config = {}
        if stage == 0:
            template_path = DEFAULT_TEMPLATE_PATH_ZERO_0
            template_config = hjson.load(open(template_path, 'r'))
            prefix = "z0_"

        elif stage == 1:
            template_path = DEFAULT_TEMPLATE_PATH_ZERO_1
            template_config = hjson.load(open(template_path, 'r'))
            prefix = "z1_"

        elif stage == 2:
            template_path = DEFAULT_TEMPLATE_PATH_ZERO_2
            template_config = hjson.load(open(template_path, 'r'))
            prefix = "z2_"

        elif stage == 3:
            template_path = DEFAULT_TEMPLATE_PATH_ZERO_3
            template_config = hjson.load(open(template_path, 'r'))
            model_info = self.model_info
            if model_info and "hidden_size" in model_info:
                hs = model_info["hidden_size"]
                template_config[ZERO_OPTIMIZATION]['reduce_bucket_size'] = hs * hs
                template_config[ZERO_OPTIMIZATION]['stage3_prefetch_bucket_size'] = 0.9 * hs * hs
                template_config[ZERO_OPTIMIZATION]['stage3_param_persistence_threshold'] = 10 * hs
            prefix = "z3_"
        else:
            return exps

        # replace the corresponding parameter values if the user specifies them in the DeepSpeed configuration file
        replace_dict(tuning_space, self.user_config, [ZERO_OPTIMIZATION, TRAIN_MICRO_BATCH_SIZE_PER_GPU])

        logger.debug(f"tuning_space = {json.dumps(tuning_space)}")

        all_configs = get_all_configs(tuning_space, ignore_keys=["optimizer"])

        tuning_keys = get_tuning_keys(tuning_space)

        logger.debug(f"tuning_keys = {tuning_keys}")

        logger.debug(f"before pruning total configs = {len(all_configs)}")

        pruned_list = prune_configs(all_configs)

        logger.debug(f"after pruning total configs = {len(pruned_list)}")

        for config in pruned_list:
            exp_config = copy.deepcopy(template_config)
            # fill the template with the expr config
            replace_dict(exp_config, config)

            # if the config does not use offloading, remove the offloading section
            config_zero = config.get(ZERO_OPTIMIZATION, None)
            if config_zero:
                if OFFLOAD_OPTIMIZER not in config_zero and OFFLOAD_OPTIMIZER in exp_config[ZERO_OPTIMIZATION]:
                    del exp_config[ZERO_OPTIMIZATION][OFFLOAD_OPTIMIZER]
                if OFFLOAD_PARAM not in config_zero and OFFLOAD_PARAM in exp_config[ZERO_OPTIMIZATION]:
                    del exp_config[ZERO_OPTIMIZATION][OFFLOAD_PARAM]
            # set gradient accumulation steps according to max_train_batch_size_per_gpu
            mbs = exp_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU]
            gas = max_train_batch_size_per_gpu // mbs
            exp_config[GRADIENT_ACCUMULATION_STEPS] = gas
            exp_config[TRAIN_BATCH_SIZE] = mbs * gas * \
                self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
            exp = {}
            # generate the expr name
            exp_name = canonical_name(exp_config, tuning_keys, prefix)
            exp['name'] = exp_name
            exp[DS_CONFIG] = exp_config
            exp['num_gpus'] = self.exp_num_gpus
            exp['num_nodes'] = self.exp_num_nodes
            exps.append(exp)

        return exps

    def tune(self):
        """ Tunes Zero stages, micro batch size per GPU, and other Zero configurations. Performance metrics of different tuning spaces are recorded in self.records.
        """
        if has_mlflow:
            self.mlflow_parent_id = os.environ['MLFLOW_RUN_ID']
            mlflow.start_run(run_id=self.mlflow_parent_id)

        self.start_time = time.time()
        if self.fast_enabled():
            logger.info(f"Fast mode is enabled. Tuning micro batch size only.")

        # model info profile run with DEFAULT_MIN_MEM_CONFIG
        model_info = self.model_info_profile_run()
        if model_info:
            self.model_info = model_info
        else:
            return

        logger.info(f"The model has {number_to_string(self.get_model_num_params())} parameters.")

        self.gpu_mem = self.get_gpu_memory_info()
        logger.info(f"Memory per GPU in the system is {memory_to_string(self.gpu_mem, postfix='B')}.")

        self.activation_mem = self.get_activation_memory_per_gpu()
        logger.info(
            f"The model requires at least {memory_to_string(self.activation_mem, postfix='B')} activation memory for micro batch size 1."
        )

        stage = self.user_config.get(ZERO_OPTIMIZATION, {}).get(ZERO_OPTIMIZATION_STAGE, 0)

        user_zero_stages = [stage] if not isinstance(stage, list) else stage
        logger.info(f"User-defined zero stages are {stage}.")

        mbs = 0
        max_mbs = 0
        metric_val = 0

        required_gpu_mem = self.get_instantiation_memory_required_per_gpu(ZeroStageEnum.disabled) + self.activation_mem
        if self.gpu_mem > required_gpu_mem:
            if "all" in user_zero_stages or ZeroStageEnum.disabled in user_zero_stages:
                logger.info(
                    f"The model might be runable with ZERO 0 (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory with mbs = 1), adding DEFAULT_TUNING_SPACE_ZERO_0 to the global tuning space"
                )
                next_max_mbs, next_mbs, next_metric_val = self.tune_space(DEFAULT_TUNING_SPACE_ZERO_0)
                if next_mbs > mbs:
                    mbs = next_mbs
                    max_mbs = next_max_mbs
                    metric_val = next_metric_val
                if has_mlflow:
                    mlflow.log_metric(f"z0{self.metric()}", next_metric_val)
        else:
            logger.info(
                f"The model is not runable with ZERO stage {ZeroStageEnum.disabled} (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory with mbs = 1)"
            )

        required_gpu_mem = self.get_instantiation_memory_required_per_gpu(
            ZeroStageEnum.optimizer_states) + self.activation_mem
        if self.gpu_mem > required_gpu_mem:
            if "all" in user_zero_stages or ZeroStageEnum.optimizer_states in user_zero_stages:
                logger.info(
                    f"The model might be runable with ZERO 1 (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory), adding DEFAULT_TUNING_SPACE_ZERO_1 to the global tuning space"
                )
                next_max_mbs, next_mbs, next_metric_val = self.tune_space(DEFAULT_TUNING_SPACE_ZERO_1,
                                                                          prev_max_mbs=max_mbs,
                                                                          prev_best_mbs=mbs,
                                                                          prev_best_metric_val=metric_val)
                if next_mbs > mbs:
                    mbs = next_mbs
                    max_mbs = next_max_mbs
                    metric_val = next_metric_val
                if has_mlflow:
                    mlflow.log_metric(f"z1{self.metric()}", next_metric_val)
        else:
            logger.info(
                f"The model is not runable with ZERO stage {ZeroStageEnum.optimizer_states} (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory with mbs = 1)"
            )

        required_gpu_mem = self.get_instantiation_memory_required_per_gpu(
            ZeroStageEnum.gradients) + self.activation_mem
        if self.gpu_mem > required_gpu_mem:
            if "all" in user_zero_stages or ZeroStageEnum.gradients in user_zero_stages:
                logger.info(
                    f"The model might be runable with ZERO 2 (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory), adding DEFAULT_TUNING_SPACE_ZERO_2 to the global tuning space"
                )
                next_max_mbs, next_mbs, next_metric_val = self.tune_space(DEFAULT_TUNING_SPACE_ZERO_2,
                                                                          prev_max_mbs=max_mbs,
                                                                          prev_best_mbs=mbs,
                                                                          prev_best_metric_val=metric_val)
                if next_mbs > mbs:
                    mbs = next_mbs
                    max_mbs = next_max_mbs
                    metric_val = next_metric_val
                if has_mlflow:
                    mlflow.log_metric(f"z2{self.metric()}", next_metric_val)
        else:
            logger.info(
                f"The model is not runable with ZERO stage {ZeroStageEnum.gradients} (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory with mbs = 1)"
            )

        required_gpu_mem = self.get_instantiation_memory_required_per_gpu(ZeroStageEnum.weights) + self.activation_mem
        if self.gpu_mem > required_gpu_mem:
            if "all" in user_zero_stages or ZeroStageEnum.weights in user_zero_stages:
                logger.info(
                    f"The model might be runable with ZERO 3 (which requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory), adding DEFAULT_TUNING_SPACE_ZERO_3 to the global tuning space"
                )
                _, _, next_metric_val = self.tune_space(DEFAULT_TUNING_SPACE_ZERO_3,
                                                        prev_max_mbs=max_mbs,
                                                        prev_best_mbs=mbs,
                                                        prev_best_metric_val=metric_val)
                if has_mlflow:
                    mlflow.log_metric(f"z3{self.metric()}", next_metric_val)
        else:
            logger.info(
                f"The model has {self.get_model_num_params()} parameters and requires at least {memory_to_string(required_gpu_mem, postfix='B')} memory per GPU with DeepSpeed Zero stage {ZeroStageEnum.weights} optimization. Memory per GPU in system is {memory_to_string(self.gpu_mem)}. No tuning is performed."
            )
            return
        if has_mlflow:
            mlflow.end_run()

    def tune_space(self, tuning_space, prev_max_mbs=0, prev_best_mbs=0, prev_best_metric_val=0):
        config_zero = tuning_space.get(ZERO_OPTIMIZATION, {})
        stage = config_zero.get(ZERO_OPTIMIZATION_STAGE, None)
        tuning_space_name = TUNING_MICRO_BATCH_SIZE_PREFIX + str(stage)
        tuning_micro_batch_sizes = []
        max_train_batch_size_per_gpu = 0
        tuning_micro_batch_sizes_overwritten = False

        # calculate max micro batch size using gpu memory, model instantiation memory and activation memory
        # calculated_max_micro_batch_size = (memory_per_gpu - instantiation_memory) // activation_memory_micro_batch_size_1
        calculated_max_micro_batch_size = int(
            self.gpu_mem - self.get_instantiation_memory_required_per_gpu(stage)) // self.activation_mem
        logger.info(
            f"Start tuning for space {tuning_space_name}, calculated_max_micro_batch_size = {calculated_max_micro_batch_size}"
        )

        if calculated_max_micro_batch_size < prev_max_mbs:
            logger.info(f"No need to tune Zero stage {stage}. End tuning for space {tuning_space_name}")
            return 0, 0, 0

        if TRAIN_MICRO_BATCH_SIZE_PER_GPU in self.user_config and isinstance(
                self.user_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU], list):
            # user-specified micro batch size per gpu is a list which overwrites the default tuning behavior
            tuning_micro_batch_sizes = [
                s for s in self.user_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] if isinstance(s, int)
            ]
            gas = self.get_gas_from_user_config()
            min_micro_batch_size = min(tuning_micro_batch_sizes)
            max_micro_batch_size = max(tuning_micro_batch_sizes)
            max_train_batch_size_per_gpu = max_micro_batch_size * gas
            tuning_micro_batch_sizes_overwritten = True
        else:
            # auto-detects the list of micro batch sizes to tune
            min_micro_batch_size, max_micro_batch_size = self.get_min_max_micro_batch_size(
                stage, prev_max_mbs, calculated_max_micro_batch_size)

            if max_micro_batch_size < prev_max_mbs:
                logger.info(f"No need to tune Zero stage {stage}. End tuning for space {tuning_space_name}")
                return 0, 0, 0

            tuning_micro_batch_sizes, max_train_batch_size_per_gpu = self.get_tuning_micro_batch_size_list(
                min_micro_batch_size,
                max_micro_batch_size,
                num_tuning_micro_batch_sizes=self.num_tuning_micro_batch_sizes())

        logger.info(
            f"tuning_micro_batch_sizes = {tuning_micro_batch_sizes}, max_train_batch_size_per_gpu = {max_train_batch_size_per_gpu}"
        )

        # return if the tuning_micro_batch_sizes list is empty
        if not tuning_micro_batch_sizes:
            logger.info(f"End tuning for space {tuning_space_name}")
            return 0, 0, 0

        # tune micro batch sizes and gradient accumulation steps given max_train_batch_size_per_gpu
        tuning_micro_batch_sizes = self.run_tuning_micro_batch_sizes(tuning_micro_batch_sizes,
                                                                     max_train_batch_size_per_gpu,
                                                                     min_micro_batch_size, stage,
                                                                     tuning_micro_batch_sizes_overwritten)

        fast_best_record = self.get_best_space_record(tuning_space_name)
        fast_best_metric_val = fast_best_record[1] if fast_best_record else 0
        fast_best_mbs = fast_best_record[0][DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU] if fast_best_record else 0
        logger.info(f"fast_best_mbs = {fast_best_mbs}, name = {fast_best_record[0]['name']}")

        if self.fast_enabled() or stage == 0:
            logger.info(f"End tuning for space: {tuning_space_name}")
            return max_micro_batch_size, fast_best_mbs, fast_best_metric_val

        # if the best metric or the micro batch size for that best metric in the current Zero stage after tuning micro batch size is less than the corresponding value in the previous Zero stage, return, do not tune other Zero configuration parameters
        if stage > 0:
            if fast_best_mbs <= prev_best_mbs or fast_best_metric_val < prev_best_metric_val:
                logger.info(
                    f"End tuning for space: {tuning_space_name}. No need to tune other Zero configuration parameters.")
                return max_micro_batch_size, fast_best_mbs, fast_best_metric_val

        tuning_space[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = tuning_micro_batch_sizes
        tuning_space_name = canonical_name(tuning_space,
                                           tuning_keys=get_tuning_keys(tuning_space),
                                           prefix="z" + str(stage) + "_",
                                           omit_val=True)

        logger.info(f'Tuning space is {tuning_space}')
        logger.info(f'Tuning space name is {tuning_space_name}')

        exps = self._generate_experiments(tuning_space, max_train_batch_size_per_gpu)

        logger.info(f'Tuner type is {self.autotuning_config.tuner_type}')
        if self.autotuning_config.tuner_type == AUTOTUNING_TUNER_MODELBASED:
            t = ModelBasedTuner(exps, self.rm, self.metric(), tuning_space)
        elif self.autotuning_config.tuner_type == AUTOTUNING_TUNER_RANDOM:
            t = RandomTuner(exps, self.rm, self.metric())
        else:
            t = GridSearchTuner(exps, self.rm, self.metric())

        sample_size = len(self.rm.nodes) * self.rm.num_gpus_per_node // (self.exp_num_gpus * self.exp_num_nodes)
        num_exps = t.tune(sample_size=sample_size,
                          n_trials=self.autotuning_config.tuner_num_trials,
                          early_stopping=self.autotuning_config.tuner_early_stopping)
        exp = t.best_exp
        metric_val = t.best_metric_val
        if exp:
            self.update_records(tuning_space_name, exp, metric_val, num_exps)

        full_best_record = self.get_best_space_record(tuning_space_name)
        full_best_metric_val = full_best_record[1] if full_best_record else -1

        if full_best_metric_val > fast_best_metric_val:
            best_metric_val = full_best_metric_val
            best_mbs = full_best_record[0][DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU] if full_best_record else -1
        else:
            best_metric_val = fast_best_metric_val
            best_mbs = fast_best_mbs

        logger.info(f"End tuning for space: {tuning_space_name}")
        return max_micro_batch_size, best_mbs, best_metric_val

    def get_plateau_mbs(self, tuning_space_name):
        if tuning_space_name not in self.records:
            return 0
        space_records = self.records[tuning_space_name]
        sorted_space_records = sorted(space_records, key=lambda x: x[0][DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU])
        prev_metric_val = None
        prev_micro_batch_size = 0
        for (exp, metric_val, _) in sorted_space_records:
            if prev_metric_val:
                if metric_val < prev_metric_val:
                    break
                if (metric_val >= prev_metric_val
                        and (metric_val - prev_metric_val) / prev_metric_val < METRIC_PERCENT_DIFF_CONST):
                    break
            prev_metric_val = metric_val
            prev_micro_batch_size = exp[DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU]
        plateau_mbs = prev_micro_batch_size
        return plateau_mbs

    def get_model_num_params(self):
        if self.model_info and "num_params" in self.model_info:
            return self.model_info["num_params"]

    def model_info_profile_run(self):
        """Does a model information profiling experiment that collects the number of model parameters and activation memory.\
            The experiment produces a "profile_model_info" folder under self.results_dir.
        Returns:
            [dict]: a model information dictionary, e.g., {"num_params": 335144976, "trainable_num_params": 335144976, "activation_mem_per_gpu": 324358144, "rank": 0}
        """
        logger.info("Starting model info profile run.")
        model_info = self.autotuning_config.model_info
        if model_info and MODEL_INFO_NUM_PARAMS in model_info:
            return model_info

        ds_config = copy.deepcopy(self.user_config)
        replace_dict(ds_config, DEFAULT_MIN_MEM_CONFIG)

        model_info_path = os.path.join(self.results_dir, "profile_model_info", "model_info.json")
        ds_config[AUTOTUNING] = {"enabled": True, "model_info_path": model_info_path, "model_info": {"profile": True}}

        exp_config = {}
        exp_name = "profile_model_info"
        exp_config['name'] = exp_name
        exp_config[DS_CONFIG] = ds_config
        exp_config['num_gpus'] = self.exp_num_gpus
        exp_config['num_nodes'] = self.exp_num_nodes
        exp_config['hostfile'] = self.args.hostfile
        exp_path = os.path.join(self.exps_dir, f'{exp_name}.json')

        with open(exp_path, 'w', buffering=BUFSIZE) as fd:
            json.dump(exp_config, fd)
            fd.flush()
            os.fsync(fd)

        self.rm.schedule_experiments([exp_path])
        self.rm.run()

        for exp_id, (exp_json, err) in self.rm.finished_experiments.items():
            self.rm.clear()
            if err:
                logger.error(f"The model is not runnable with DeepSpeed with error = {err}")
                return None

        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                model_info = hjson.load(f)
                return model_info

    def update_records(self, space_name, exp, metric_val, num_exps):
        if space_name not in self.records:
            self.records[space_name] = [(exp, metric_val, num_exps)]
        else:
            self.records[space_name].append((exp, metric_val, num_exps))

    def get_best_space_record(self, space_name):
        if space_name not in self.records:
            return None
        space_records = self.records[space_name]
        best_space_record = None
        space_num_exps = 0
        for (exp, metric_val, num_exps) in space_records:
            space_num_exps += num_exps
            if best_space_record is None or metric_val > best_space_record[1]:
                best_space_record = (exp, metric_val)
        if best_space_record:
            best_space_record = best_space_record + (space_num_exps, )
        return best_space_record

    def get_best_space_records(self):
        best_space_records = {}
        global_best_record = None
        for space_name, space_records in self.records.items():
            best_space_record = self.get_best_space_record(space_name)
            if best_space_record:
                best_space_records[space_name] = best_space_record
                if not global_best_record or best_space_record[1] > global_best_record[1]:
                    global_best_record = best_space_record
        if global_best_record:
            best_space_records[GLOBAL_TUNING_SPACE] = global_best_record
        return best_space_records

    def run_tuning_micro_batch_sizes(self, tuning_micro_batch_sizes, max_train_batch_size_per_gpu,
                                     min_micro_batch_size, stage, tuning_micro_batch_sizes_overwritten):
        assert tuning_micro_batch_sizes, "the tuning micro batch size list is empty"
        tuning_micro_batch_sizes.sort()
        max_micro_batch_size = tuning_micro_batch_sizes[-1]
        max_micro_batch_size_metric_val = 0

        ds_config = get_first_config(self.user_config)
        ds_config[ZERO_OPTIMIZATION] = {ZERO_OPTIMIZATION_STAGE: stage}
        tuning_space_name = TUNING_MICRO_BATCH_SIZE_PREFIX + str(stage)

        exp_paths = []
        for mbs in tuning_micro_batch_sizes:
            ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = mbs
            gas = max_train_batch_size_per_gpu // mbs
            ds_config[GRADIENT_ACCUMULATION_STEPS] = gas
            ds_config[TRAIN_BATCH_SIZE] = mbs * gas * \
                self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
            exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(mbs)
            exp_config = {}
            exp_config['name'] = exp_name
            exp_config[DS_CONFIG] = ds_config
            exp_config['num_gpus'] = self.exp_num_gpus
            exp_config['num_nodes'] = self.exp_num_nodes
            exp_config['hostfile'] = self.args.hostfile
            exp_path = os.path.join(self.exps_dir, f'{exp_name}.json')

            with open(exp_path, 'w', buffering=BUFSIZE) as fd:
                json.dump(exp_config, fd)
                fd.flush()
                os.fsync(fd)
            exp_paths.append(exp_path)

        self.rm.schedule_experiments(exp_paths)
        self.rm.run()

        for exp_id, (exp, err) in self.rm.finished_experiments.items():
            if exp:
                metric_file = exp[DS_CONFIG][AUTOTUNING][AUTOTUNING_METRIC_PATH]
                if os.path.exists(metric_file):

                    with open(metric_file, 'r') as f:
                        results = hjson.load(f)
                        metric_val = results[self.metric()]
                        self.update_records(tuning_space_name, exp, metric_val, 1)
                        if max_micro_batch_size == exp[DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU]:
                            max_micro_batch_size_metric_val = metric_val
                        if has_mlflow:
                            os.environ.pop('MLFLOW_RUN_ID')
                            mlflow.start_run(nested=True, run_name=exp['name'])
                            for metric in results:
                                mlflow.log_metric(metric, results[metric])
                            mlflow.end_run()
                            os.environ['MLFLOW_RUN_ID'] = self.mlflow_parent_id
                else:
                    self.update_records(tuning_space_name, exp, 0, 1)
            else:
                mbs = exp[DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU]
                logger.info(f"micro batch size = {mbs} was not run successfully")

        self.rm.clear()

        if tuning_micro_batch_sizes_overwritten:
            return tuning_micro_batch_sizes

        # in a auto-detected tuning_micro_batch_sizes list, max_micro_batch_size might not be performant as the memory consumption is close to max
        # try smaller values while gas stays the same
        # if finding a more performant mbs value, use it to replace max_micro_batch_size in the list
        min_micro_batch_size_with_same_gas = (tuning_micro_batch_sizes[-2] +
                                              1) if len(tuning_micro_batch_sizes) > 1 else min_micro_batch_size

        prev_best_metric_val = max_micro_batch_size_metric_val
        prev_best_mbs = max_micro_batch_size

        stride = (max_micro_batch_size - min_micro_batch_size_with_same_gas) // 3
        if stride == 0:
            stride = 1
        for mbs in reversed(range(min_micro_batch_size_with_same_gas, max_micro_batch_size, stride)):
            ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = mbs
            gas = max_train_batch_size_per_gpu // mbs
            ds_config[GRADIENT_ACCUMULATION_STEPS] = gas
            ds_config[TRAIN_BATCH_SIZE] = mbs * gas * \
                self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
            exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(mbs)
            exp, metric_val = self.run_ds_config(ds_config, exp_name)

            if metric_val:
                with open(metric_file, 'r') as f:
                    results = hjson.load(f)
                    metric_val = results[self.metric()]
                    if has_mlflow:
                        os.environ.pop('MLFLOW_RUN_ID')
                        mlflow.start_run(nested=True, run_name=exp_name)
                        for metric in results:
                            mlflow.log_metric(metric, results[metric])
                        mlflow.end_run()
                        os.environ['MLFLOW_RUN_ID'] = self.mlflow_parent_id
                self.update_records(tuning_space_name, exp, metric_val, 1)
                if metric_val > prev_best_metric_val * (1 + METRIC_PERCENT_DIFF_CONST):
                    prev_best_metric_val = metric_val
                    prev_best_mbs = mbs
                else:
                    break
            else:
                self.update_records(tuning_space_name, exp, 0, 1)
                break
        if prev_best_mbs != max_micro_batch_size:
            tuning_micro_batch_sizes[-1] = prev_best_mbs
        return tuning_micro_batch_sizes

    def get_min_max_micro_batch_size(self, stage, min_micro_batch_size, calculated_max_micro_batch_size):
        # get min and max micro batch size with gradient accumulation steps = 1
        if min_micro_batch_size > calculated_max_micro_batch_size:
            return -1, -1

        used_micro_batch_sizes = []
        tuning_space_name = TUNING_MICRO_BATCH_SIZE_PREFIX + str(stage)

        ds_config = get_first_config(self.user_config)
        ds_config[ZERO_OPTIMIZATION] = {ZERO_OPTIMIZATION_STAGE: stage}
        gas = self.get_gas_from_user_config()
        ds_config[GRADIENT_ACCUMULATION_STEPS] = gas

        # search for the min micro batch size
        if min_micro_batch_size < 1:
            if TRAIN_MICRO_BATCH_SIZE_PER_GPU in self.user_config and isinstance(
                    self.user_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU], int):
                # user specifies train_micro_batch_size_per_gpu as an int
                mbs = int(self.user_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU])
            else:
                # user does not specify train_micro_batch_size_per_gpu or sets it to "auto" when using Hugging Face
                val = self.get_val_from_user_args(TRAIN_MICRO_BATCH_SIZE_PER_GPU)
                if val:
                    mbs = int(val)
                else:
                    mbs = 1
            assert mbs > 0, "The micro batch size per GPU must be greater than 0."
            ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = mbs
            ds_config[GRADIENT_ACCUMULATION_STEPS] = gas
            ds_config[TRAIN_BATCH_SIZE] = mbs * gas * \
                self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
            exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(mbs)
            exp, metric_val = self.run_ds_config(ds_config, exp_name)
            if metric_val:
                self.update_records(tuning_space_name, exp, metric_val, 1)
                used_micro_batch_sizes.append(mbs)
                min_micro_batch_size = mbs
            else:
                self.update_records(tuning_space_name, exp, 0, 1)
                logger.info(f"User-specified micro batch size per GPU {mbs} does not run")
                if self.min_train_micro_batch_size_per_gpu() == mbs:
                    return -1, -1
                mbs = self.min_train_micro_batch_size_per_gpu()
                ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = mbs
                ds_config[GRADIENT_ACCUMULATION_STEPS] = gas
                ds_config[TRAIN_BATCH_SIZE] = mbs * gas * \
                    self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
                exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(mbs)
                exp, metric_val = self.run_ds_config(ds_config, exp_name)
                if not metric_val:
                    self.update_records(tuning_space_name, exp, 0, 1)
                    logger.info(f"min_train_micro_batch_size_per_gpu {mbs} is not runnable.")
                    return -1, -1
                self.update_records(tuning_space_name, exp, metric_val, 1)
                min_micro_batch_size = mbs
                used_micro_batch_sizes.append(mbs)
        else:
            ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = min_micro_batch_size
            ds_config[GRADIENT_ACCUMULATION_STEPS] = gas
            ds_config[TRAIN_BATCH_SIZE] = min_micro_batch_size * gas * \
                self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
            exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(min_micro_batch_size)
            exp, metric_val = self.run_ds_config(ds_config, exp_name)
            if metric_val:
                self.update_records(tuning_space_name, exp, metric_val, 1)
                used_micro_batch_sizes.append(min_micro_batch_size)
            else:
                self.update_records(tuning_space_name, exp, 0, 1)
                return -1, -1

        # search for the max micro batch size
        max_micro_batch_size = min(calculated_max_micro_batch_size, self.max_train_micro_batch_size_per_gpu())
        for mbs in [math.ceil(1.05 * max_micro_batch_size), max_micro_batch_size, int(0.95 * max_micro_batch_size)]:
            if mbs > self.max_train_micro_batch_size_per_gpu():
                continue
            if mbs in used_micro_batch_sizes:
                return min_micro_batch_size, mbs
            ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = mbs
            ds_config[TRAIN_BATCH_SIZE] = mbs * gas * \
                self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
            exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(mbs)
            exp, metric_val = self.run_ds_config(ds_config, exp_name)

            if metric_val:
                logger.info(f"mbs = {mbs} is found as max mbs")
                self.update_records(tuning_space_name, exp, metric_val, 1)
                used_micro_batch_sizes.append(mbs)
                return min_micro_batch_size, mbs
            else:
                self.update_records(tuning_space_name, exp, 0, 1)

        space_records = self.records[tuning_space_name] if tuning_space_name in self.records else []
        if space_records:
            prev_idx = min(range(len(space_records)),
                           key=lambda i: abs(space_records[i][0][DS_CONFIG][TRAIN_MICRO_BATCH_SIZE_PER_GPU] -
                                             min_micro_batch_size))
            prev_metric_val = space_records[prev_idx][1]
        else:
            prev_metric_val = None

        low = min_micro_batch_size
        high = max_micro_batch_size
        # binary search until low is the smallest micro batch size that OOMs.
        while low <= high:
            mid = int((low + high) // 2)
            logger.debug(f"trying mbs = {mid}, low = {low}, high = {high}")
            if mid not in used_micro_batch_sizes:
                ds_config[TRAIN_MICRO_BATCH_SIZE_PER_GPU] = mid
                ds_config[TRAIN_BATCH_SIZE] = mid * gas * \
                    self.exp_num_gpus * self.exp_num_nodes // self.mp_size()
                exp_name = tuning_space_name + "_gas" + str(gas) + "_tmbspg" + str(mid)
                exp, metric_val = self.run_ds_config(ds_config, exp_name)
                if metric_val:
                    low = mid + 1
                    self.update_records(tuning_space_name, exp, metric_val, 1)
                    used_micro_batch_sizes.append(mid)
                    if prev_metric_val and (
                        (metric_val - prev_metric_val) / prev_metric_val) < METRIC_PERCENT_DIFF_CONST:
                        logger.info(f"performance plateaus at mbs = {low}")
                        break
                    prev_metric_val = metric_val
                else:
                    self.update_records(tuning_space_name, exp, 0, 1)
                    high = mid - 1
            else:
                low = mid + 1
        max_micro_batch_size = low - 1

        logger.info(f"min_micro_batch_size = {min_micro_batch_size}, max_micro_batch_size = {max_micro_batch_size}.")

        return min_micro_batch_size, max_micro_batch_size

    def get_gas_from_user_config(self):
        gas = 1
        if GRADIENT_ACCUMULATION_STEPS in self.user_config:
            gas_in_config = self.user_config[GRADIENT_ACCUMULATION_STEPS]
            if isinstance(gas_in_config, int):
                gas = gas_in_config
            elif gas_in_config == "auto":  # GRADIENT_ACCUMULATION_STEPS: "auto"
                val = self.get_val_from_user_args(GRADIENT_ACCUMULATION_STEPS)
                if val:
                    gas = int(val)
            elif isinstance(gas_in_config, list):
                logger.info(
                    f"Specifying a list of {GRADIENT_ACCUMULATION_STEPS} to tune is not supported. 1 would be used.")
        assert gas > 0, "Gradient accumulation steps must be positive."
        return gas

    def get_val_from_user_args(self, ds_name):
        arg_mappings = self.autotuning_config.arg_mappings
        user_args = self.args.user_args
        if arg_mappings and ds_name in arg_mappings:
            arg_name = arg_mappings[ds_name]
            if arg_name in user_args:
                idx = user_args.index(arg_name)
                if user_args[idx + 1].isnumeric():
                    return (user_args[idx + 1])
        return None

    def get_tuning_micro_batch_size_list(self, min_micro_batch_size, max_micro_batch_size,
                                         num_tuning_micro_batch_sizes):
        """Get a list of micro batch sizes to tune based on min and max values, as well as the size of the list.
        Args:
            min_micro_batch_size ([int]): min micro batch size per GPU
            max_micro_batch_size ([int]): max micro batch size per GPU
            num_tuning_micro_batch_sizes (int): the number of items in the returned list

        Returns:
            [list]: a list of micro batch sizes to tune.
        """
        if min_micro_batch_size <= 0 or max_micro_batch_size <= 0:
            logger.info(
                f"min_micro_batch_size = {min_micro_batch_size}, max_micro_batch_size = {max_micro_batch_size}")
            return [], 0

        # NUM_GPUS=$(( ${NUM_WORKERS} * ${NUM_GPUS_PER_WORKER} ))
        # DP_SIZE=$(( ${NUM_GPUS} / (${PP_SIZE} * ${MP_SIZE}) ))
        # GRAD_ACC_STEPS=$(( ${TARGET_GLOBAL_BATCH_SIZE} / (${BATCH_SIZE} * ${DP_SIZE}) ))
        if self.max_train_batch_size(
        ) and self.max_train_batch_size() > 0:  # if the user specifies a max_train_batch_size
            max_train_batch_size_per_gpu = self.max_train_batch_size() * self.mp_size() // (self.exp_num_gpus *
                                                                                            self.exp_num_nodes)
        else:
            gas = self.get_gas_from_user_config()
            max_train_batch_size_per_gpu = max_micro_batch_size * gas // self.mp_size()
        logger.info(f"max_train_batch_size_per_gpu = {max_train_batch_size_per_gpu}")
        if min_micro_batch_size < max_micro_batch_size // 2:
            min_micro_batch_size = max_micro_batch_size // 2

        # constant stride
        stride = (max_micro_batch_size - min_micro_batch_size) // num_tuning_micro_batch_sizes
        if stride == 0:
            stride = 1
        ls = []
        min_gas = max_train_batch_size_per_gpu // max_micro_batch_size
        # if gas is the same as min_gas, do not add mbs to the tuning list
        for mbs in range(min_micro_batch_size, max_micro_batch_size, stride):
            if max_train_batch_size_per_gpu // mbs != min_gas:
                ls.append(mbs)
        ls.append(max_micro_batch_size)

        return ls, max_train_batch_size_per_gpu

    def run_ds_config(self, ds_config, exp_name):
        exp_config = {}
        exp_config['name'] = exp_name
        exp_config[DS_CONFIG] = ds_config
        exp_config['num_gpus'] = self.exp_num_gpus
        exp_config['num_nodes'] = self.exp_num_nodes
        exp_config['hostfile'] = self.args.hostfile
        exp_path = os.path.join(self.exps_dir, f'{exp_name}.json')

        logger.debug(f'run_ds_config exp_name = {exp_name}')

        with open(exp_path, 'w', buffering=BUFSIZE) as fd:
            json.dump(exp_config, fd)
            fd.flush()
            os.fsync(fd)
        self.rm.schedule_experiments([exp_path])
        self.rm.run()
        exp, metric_val = self.rm.parse_results(self.metric())
        self.rm.clear()
        return exp, metric_val

    def write_optimal_config(self):
        best_space_records = self.get_best_space_records()
        if GLOBAL_TUNING_SPACE not in best_space_records:
            return
        best_exp, best_metric_val, _ = best_space_records[GLOBAL_TUNING_SPACE]
        if best_exp:
            exp_dir = best_exp["result_dir"]
            cmd = None
            with open(os.path.join(exp_dir, "cmd.txt"), "r") as f:
                cmd = [str(i) for i in f.read().split()]

            ds_config = hjson.load(open(os.path.join(exp_dir, "ds_config.json"), "r"))
            ds_config.pop(AUTOTUNING)

            ds_config_path = os.path.join(self.results_dir, "ds_config_optimal.json")
            json.dump(ds_config, open(ds_config_path, "w"))

            cmd_path = os.path.join(self.results_dir, "cmd_optimal.txt")
            with open(cmd_path, "w") as fd:
                fd.write(" ".join(cmd))
                fd.write("\n")
                fd.flush()
            self.optimal_cmd = cmd
            self.optimal_ds_config = ds_config
            logger.info(
                f"Wrote the optimal DeepSpeed configuration found by autotuning to {ds_config_path}, and the corresponding DeepSpeed command to {cmd_path}"
            )

    def run_after_tuning(self):
        """ Launches the training with the optimal DeepSpeed configuration found through the autotuning process.
            "ds_config_optimal.json" describing the optimal DeepSpeed configuration as well the command used to launch training "cmd_optimal.txt" are saved to self.results_dir.
        """
        if self.optimal_cmd:
            result = subprocess.Popen(self.optimal_cmd)
            result.wait()

            logger.info(f"Done running with the optimal DeepSpeed configuration using {self.optimal_cmd}")
        else:
            logger.info(f"No optimal DeepSpeed configuration found by autotuning.")
