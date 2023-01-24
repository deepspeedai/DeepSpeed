# Automatic Tensor Parallelism
import re

from torch import nn


class AutoTP():
    def in_module_list(module, module_list):
        for item in module_list:
            if type(item).__name__ == type(module).__name__:
                return True
        return False

    def get_module_list(model):
        mlist = []
        for child in model.children():
            if isinstance(child, nn.ModuleList):
                for module in child.children():
                    if not mlist:
                        mlist = [module]
                    elif not AutoTP.in_module_list(module, mlist):
                        mlist = mlist + [module]
            else:
                mlist = mlist + AutoTP.get_module_list(child)
        return mlist

    def supported(model):
        unsupported = ['bloom', 'codegen', 'flaubert', 'xlm']
        model = str(model)
        key = re.search(r": (.*?)Model", model)
        if key is None:
            key = re.search(r": (.*?)Stack", model)
        if key is None:
            key = re.match(r"(.*?)Model", model)
        if key.group(1).lower() in unsupported:
            return False
        return True

    def get_layers(parent, module):
        layer_list = []
        for key, submodule in module._modules.items():
            if isinstance(submodule, nn.Linear):
                layer_list = layer_list + [parent + "." + key]
            elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm':
                layer_list = layer_list + ["ln"]
            else:
                layer_list = layer_list + AutoTP.get_layers(key, submodule)
        return layer_list

    def tp_parser(model):
        policy_list = []
        module_list = []
        layer_list = []
        gem_list = []

        assert AutoTP.supported(model), "Automatic policy not supported for model. Please provide policy."

        module_list = AutoTP.get_module_list(model)
        for module in module_list:
            for key, submodule in module._modules.items():
                if isinstance(submodule, nn.Linear):
                    layer_list = layer_list + ["." + key]
                elif isinstance(submodule, nn.LayerNorm) or key == 'LayerNorm':
                    layer_list = layer_list + ["ln"]
                else:
                    layer_list = layer_list + AutoTP.get_layers(key, submodule)
            for i, layer in enumerate(layer_list):
                if layer == 'ln':
                    if layer_list[i - 1] != 'ln':
                        gem_list = gem_list + [layer_list[i - 1]]
                elif 'out_proj' in layer:
                    gem_list = gem_list + [layer]
            if gem_list != []:
                policy_list.append(tuple([type(module), gem_list]))
                gem_list = []
        return policy_list
