# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import re
import collections.abc
import os
import json
from deepspeed.runtime.constants import GRADIENT_ACCUMULATION_STEPS, TRAIN_MICRO_BATCH_SIZE_PER_GPU
import itertools
import copy

from ..utils import logger


def search_error(filename):
    if not os.path.exists(filename):
        return "stderr.log does not exist"
    with open(filename) as f:
        for line in f:
            for s in ["Error", "error", "ERROR"]:
                idx = line.find(s)
                if idx != -1:
                    return line[idx + len(s):].lstrip(": ")
    return None


def was_interruptted(filename):
    if not os.path.exists(filename):
        return "stderr.log does not exist"
    with open(filename) as f:
        for line in f:
            s = "KeyboardInterrupt"
            idx = line.find(s)
            if idx != -1:
                return True
    return False


def find_replace_str(value, replace_dict):
    if not isinstance(value, str):
        return str(value)

    matches = re.findall(r"\$[A-Za-z0-9_]+", value)
    for var in matches:
        var_key = var.replace("$", "").lower()
        if var_key == "nvme_path":
            continue
        assert var_key in replace_dict, f"unknown var key: {var_key}, in {replace_dict}"
        if isinstance(replace_dict[var_key], str):
            value = value.replace(var, replace_dict[var_key])
        else:
            assert len(matches) == 1, "unable to replace multiple non-string matches"
            value = replace_dict[var_key]
    return value


def find_replace(target, replace_dict):
    if isinstance(target, dict):
        for key, value in target.items():
            if isinstance(value, str):
                target[key] = find_replace_str(value, replace_dict)
            if isinstance(value, list):
                for i in range(len(value)):
                    value[i] = find_replace_str(value[i], replace_dict)
            if isinstance(value, dict):
                find_replace(value, replace_dict)
    elif isinstance(target, list):
        for i in range(len(target)):
            target[i] = str(find_replace_str(target[i], replace_dict))


def get_list(val):
    if not isinstance(val, list):
        return [val]
    else:
        return val


def combine_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = combine_dict(d.get(k, {}), v)
        else:
            if k not in d:
                d[k] = v
            else:
                if not isinstance(d[k], list):
                    d[k] = [d[k]]
                d[k].extend(i for i in get_list(v) if i not in d[k])
    return d


def del_if_exists(t, d):
    """Deletes a key from a dictionary if it exists.

    Args:
        t (string): target key to delete
        d (dict): dictionary to delete from
    """
    if t in d:
        del d[t]
        return
    for k, v in d.items():
        if isinstance(v, collections.abc.Mapping):
            del_if_exists(t, v)


def replace_dict(d, u, ignored_keys=[]):
    """Replaces values in dict d with values in dict u.

    Args:
        d (dict): the target dict to overwrite
        u (dict): the dict containing the values to overwrite the target dict

    Returns:
        dict d with values overwritten by the corresponding ones in dict u.
    """
    if u is not None:
        for k, v in u.items():
            if k not in ignored_keys:
                if v is None:
                    del_if_exists(k, d)
                    continue
                if isinstance(v, collections.abc.Mapping):
                    d[k] = replace_dict(d.get(k, {}), v, ignored_keys)
                else:
                    d[k] = v
    return d


def get_val_by_key(d: dict, k):
    if k in d:
        return d[k]
    for v in d.values():
        if isinstance(v, dict):
            return get_val_by_key(v, k)
    return None


def set_val_by_key(d: dict, k, vv):
    if k in d:
        d[k] = vv
    for v in d.values():
        if isinstance(v, dict):
            set_val_by_key(v, k, vv)


def fetch_hostfile(hostfile_path):
    if not os.path.isfile(hostfile_path):
        logger.warning("Unable to find hostfile, will proceed with training "
                       "with local resources only.")
        return None

    # e.g., worker-0 slots=16
    with open(hostfile_path, 'r') as fd:
        resource_pool = collections.OrderedDict()
        for line in fd.readlines():
            line = line.strip()
            if line == '':
                # skip empty lines
                continue
            try:
                hostname, slots = line.split()
                _, slot_count = slots.split("=")
                slot_count = int(slot_count)
            except ValueError as err:
                logger.error("Hostfile is not formatted correctly, unable to "
                             "proceed with training.")
                raise err
            if hostname in resource_pool:
                logger.error("Hostfile contains duplicate hosts, unable to "
                             "proceed with training.")
                raise ValueError("host {} is already defined".format(hostname))
            resource_pool[hostname] = slot_count

    return resource_pool


def validate_ds_config(config: dict):

    def is_False(config: dict, key):
        if config is None:
            return False
        return bool(config.get(key))

    config_zero = config.get("zero_optimization", {})
    if not config_zero:
        return True
    stage = config_zero.get("stage")
    offload = False
    if stage == 1:
        return True
    elif stage == 2:
        if is_False(config_zero, "cpu_offload") and is_False(config_zero, "cpu_offload_params"):
            return False
    elif stage == 3:
        offload_devices = ["cpu", "nvme"]
        if config_zero.get("offload_optimizer", {}).get("device") in offload_devices:
            offload = True
        if config_zero.get("offload_param", {}).get("device") in offload_devices:
            offload = True
    else:
        return True

    # HF requires that "ZeRO Offload can only work with DeepSpeed optimizers"
    if offload and not config.get("optimizer"):
        return False

    return True


def remove_dupe_dicts(l):
    """ Removes duplicate dictionaries from a list. Uses list comprehension and the json library to sort and stringify each dictionary and the set data type to ensure unique values. Works with nested data structures.

    Args:
        l (list): a list of (nested) data structures.

    Returns:
        A list of unique values.
    """
    list_of_strings = [json.dumps(d, sort_keys=True) for d in l]
    list_of_strings = set(list_of_strings)
    return [json.loads(s) for s in list_of_strings]


def prune_config(config, ignored_keys=[]):
    """ Prunes the input configurations

    Args:
        configs (dict): A configuration dictionary.
        ignored_keys (list, optional): the keys of the sections to delete. Defaults to [].

    Returns:
        A configuration dictionary.
    """
    if ignored_keys:
        for k in ignored_keys:

            def find_del_key(d: dict, k: str):
                if k in d:
                    del d[k]
                else:
                    for dd in d.values():
                        if isinstance(dd, dict):
                            find_del_key(dd, k)

            find_del_key(config, k)


def prune_configs(configs, ignored_keys=[]):
    """ Prunes the input list of configurations

    Args:
        configs (list): A list of configuration dictionaries.
        ignored_keys (list, optional): the keys of the sections to delete. Defaults to [].

    Returns:
        A list of valid and unique configuration dictionaries.
    """
    pruned_list = []
    for config in configs:
        prune_config(config, ignored_keys)
        pruned_list.append(config)

    return remove_dupe_dicts(pruned_list)


def get_tuning_keys(tuning_space: dict):
    """Outputs the list of tunable parameters in the tuning space dict.

    Args:
        tuning_space (dict): a configuration dictionary containing tunable parameters as lists of values.

    Returns:
        A list of strings
    """
    tuning_keys = []
    for key, val in tuning_space.items():
        if isinstance(val, dict):
            tuning_keys.extend(get_tuning_keys(val))
        if isinstance(val, list) and len(val) > 1:
            tuning_keys.append(key)
    return tuning_keys


def get_all_configs(tuning_space: dict, ignore_keys=None):
    """ Splits the tuning space dictionary to result in all combinations of values.

    Args:
        tuning_space (dict): the tuning space where tunable parameters are lists of values.
    """

    def gen_combinations(d: dict):
        keys, values = d.keys(), d.values()
        for v in values:
            if not isinstance(v, list):
                v = [v]
        values_choices = (gen_combinations(v) if isinstance(v, dict) else get_list(v) for v in values)
        for comb in itertools.product(*values_choices):
            yield dict(zip(keys, comb))

    all_configs = []
    ignored_key_vals = {}
    for ik in ignore_keys:
        ignored_key_vals[ik] = tuning_space.get(ik, {})
        del_if_exists(ik, tuning_space)
    for c in gen_combinations(tuning_space):
        replace_dict(c, ignored_key_vals)
        all_configs.append(c)
    return all_configs


def canonical_name(config: dict, tuning_keys=None, prefix="", omit_val=False):
    """ Generates a name from the acronyms of the tuning keys in the config dict. TRAIN_MICRO_BATCH_SIZE_PER_GPU is always included in the tuning keys.
    Args:
        config (dict): the config dict used to generate the name
        tuning_keys (list, optional):  the tuning keys used to generate the name. Defaults to None.
        prefix (str, optional): a string added to the beginning of the name. Defaults to None.
    """
    if TRAIN_MICRO_BATCH_SIZE_PER_GPU not in tuning_keys:
        tuning_keys.append(TRAIN_MICRO_BATCH_SIZE_PER_GPU)
    if GRADIENT_ACCUMULATION_STEPS not in tuning_keys:
        tuning_keys.append(GRADIENT_ACCUMULATION_STEPS)
    tuning_keys.sort()

    def get_offload_name(offload_config):
        cname = ""
        if offload_config is None:
            return "None_"
        for key, val in offload_config.items():
            key = "".join(map(lambda c: c[0], key.split('_')))
            if (isinstance(val, int) or isinstance(val, float)) and val > 9000:
                cname += key + '{:.1e}'.format(val) + "_"
            else:
                if isinstance(val, bool):
                    val = "T" if val else "F"
                cname += f"{key}{val}_"
        return cname

    def get_name_by_keys(config: dict, tuning_keys=None, omit_val=False):
        cname = ""
        if not tuning_keys or config is None:
            return cname
        for key, val in config.items():
            # skip the arg_mappings section when naming the exp file
            if key == "arg_mappings":
                continue
            if key == "offload_param":
                cname += "op_"
                if not omit_val:
                    cname += get_offload_name(val)
                continue
            if key == "offload_optimizer":
                cname += "oo_"
                if not omit_val:
                    cname += get_offload_name(val)
                continue
            # recursively call the func to get name for the child dicts
            if isinstance(val, dict):
                n = get_name_by_keys(val, tuning_keys, omit_val=omit_val)
                if n != "":
                    cname += n + "_"
            if tuning_keys and key not in tuning_keys:
                continue

            key_str = "".join(map(lambda c: c[0], key.split('_')))

            if not omit_val:
                if (isinstance(val, int) or isinstance(val, float)) and val > 9000:
                    cname += key_str + '{:.1e}'.format(val) + "_"
                else:
                    if isinstance(val, bool):
                        val = "T" if val else "F"
                    cname += f"{key_str}{val}_"
            else:
                cname += key_str + "_"

        return cname[:-1]

    name = get_name_by_keys(config, tuning_keys, omit_val=omit_val)

    return prefix + (name if name != "" else "exp")


def get_first_config(config: dict):
    if not config:
        return None
    cfg = copy.deepcopy(config)

    for key, val in cfg.items():
        if isinstance(val, dict):
            if key == "optimizer":  # use user defined optimizer which might have lists of values as params
                cfg[key] = val
            else:
                cfg[key] = get_first_config(val)
        if isinstance(val, list) and len(val) > 0:
            cfg[key] = val[0]
    return cfg


def write_experiments(exps: list, exps_dir: str):
    exp_paths = []
    for exp in exps:
        exp_name = exp['name']
        # write the expr config to a json file
        exp_path = os.path.join(exps_dir, f'{exp_name}.json')
        with open(exp_path, 'w') as fd:

            json.dump(exp, fd)
            exp_paths.append(exp_path)
    return exp_paths


def memory_to_string(n, postfix="", units=None, precision=2):
    if units is None:
        if n // 10**12 > 0:
            return str(round(n / 1024**4, precision)) + " T" + postfix
        if n // 10**9 > 0:
            return str(round(n / 1024**3, precision)) + " G" + postfix
        elif n // 10**6 > 0:
            return str(round(n / 1024**2, precision)) + " M" + postfix
        elif n // 10**3 > 0:
            return str(round(n / 1014, precision)) + " K" + postfix
        else:
            return str(n) + " "
    else:
        if units == "T":
            return str(round(n / 1024**4, precision)) + " " + units
        if units == "G" + postfix:
            return str(round(n / 1024**3, precision)) + " " + units
        elif units == "M" + postfix:
            return str(round(n / 1024**2, precision)) + " " + units
        elif units == "K" + postfix:
            return str(round(n / 1024, precision)) + " " + units
        else:
            return str(n) + " "


def number_to_string(n, postfix="", units=None, precision=2):
    if units is None:
        if n // 10**9 > 0:
            return str(round(n / 1000**3, precision)) + " B" + postfix
        if n // 10**6 > 0:
            return str(round(n / 1000**2, precision)) + " M" + postfix
        elif n // 10**3 > 0:
            return str(round(n / 1000**1, precision)) + " K" + postfix
        else:
            return str(n) + " "
    else:
        if units == "B" + postfix:
            return str(round(n / 1000**3, precision)) + " " + units
        elif units == "M" + postfix:
            return str(round(n / 1000**2, precision)) + " " + units
        elif units == "K" + postfix:
            return str(round(n / 1000**1, precision)) + " " + units
        else:
            return str(n) + " "
