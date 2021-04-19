"""
Copyright (c) Microsoft Corporation
Licensed under the MIT license.
"""
"""
Collection of DeepSpeed configuration utilities
"""
import json
import collections


# adapted from https://stackoverflow.com/a/50701137/9201239
class ScientificNotationEncoder(json.JSONEncoder):
    """
    This class overrides ``json.dumps`` default formatter.

    This version keeps everything as normal except formats numbers bigger than 1e3 using scientific notation.

    Just pass ``cls=ScientificNotationEncoder`` to ``json.dumps`` to activate it

    """
    def iterencode(self, o, _one_shot=False, level=0):
        indent = self.indent if self.indent is not None else 4
        prefix_close = " " * level * indent
        level += 1
        prefix = " " * level * indent
        if isinstance(o, bool):
            return "true" if o else "false"
        elif isinstance(o, float) or isinstance(o, int):
            if o > 1e3:
                return f"{o:e}"
            else:
                return f"{o}"
        elif isinstance(o, collections.Mapping):
            x = [
                f'\n{prefix}"{k}": {self.iterencode(v, level=level)}' for k,
                v in o.items()
            ]
            return "{" + ', '.join(x) + f"\n{prefix_close}" + "}"
        elif isinstance(o, collections.Sequence) and not isinstance(o, str):
            return f"[{ f', '.join(map(self.iterencode, o)) }]"
        return "\n, ".join(super().iterencode(o, _one_shot))


class DeepSpeedConfigObject(object):
    """
    For json serialization
    """
    def repr(self):
        return self.__dict__

    def __repr__(self):
        return json.dumps(
            self.__dict__,
            sort_keys=True,
            indent=4,
            cls=ScientificNotationEncoder,
        )


def get_scalar_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def get_list_param(param_dict, param_name, param_default_value):
    return param_dict.get(param_name, param_default_value)


def dict_raise_error_on_duplicate_keys(ordered_pairs):
    """Reject duplicate keys."""
    d = dict((k, v) for k, v in ordered_pairs)
    if len(d) != len(ordered_pairs):
        counter = collections.Counter([pair[0] for pair in ordered_pairs])
        keys = [key for key, value in counter.items() if value > 1]
        raise ValueError("Duplicate keys in DeepSpeed config: {}".format(keys))
    return d
