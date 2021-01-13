import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

module_flop_count = []
old_functions = {}


class FlopsProfiler(object):
    """Measures the time, number of estimated flops and parameters of each module in a PyTorch model.

    The flops-profiler profiles the forward pass of a PyTorch model and prints the model graph with the measured profile attached to each module. It shows how time, flops and parameters are spent in the model and which modules or layers could be the bottleneck. It also outputs the names of the top k modules in terms of aggregated time, flops, and parameters at depth l with k and l specified by the user. The output profile is computed for each batch of input. If multiple forward passes are specified by the user to caputre (in the case where the model have different paths or for more accurate timing), the average profile of the multiple batches is taken.

    Args:
        object (torch.nn.Module): The PyTorch model to profile.
    """
    def __init__(self, model):
        self.model = model

    def start_profile(self, ignore_list=None):
        """Starts profiling.

        Extra attributes are added recursively to all the modules and the profiled torch.nn.functionals are monkey patched.

        Args:
            ignore_list (list, optional): the list of modules to ignore while profiling. Defaults to None.
        """
        self.reset_profile()
        _patch_functionals()

        def register_module_hooks(module, ignore_list):
            if ignore_list and type(module) in ignore_list:
                return

            # if computing the flops of a module directly
            if type(module) in MODULE_HOOK_MAPPING:
                module.__flops_handle__ = module.register_forward_hook(
                    MODULE_HOOK_MAPPING[type(module)])
                return

            # if computing the flops of the functionals in a module
            def pre_hook(module, input):
                module_flop_count.clear()
                if len(input) > 0:
                    # Can have multiple inputs, getting the first one
                    input = input[0]
                module.__steps__ += 1

            module.__pre_hook_handle__ = module.register_forward_pre_hook(pre_hook)

            def post_hook(module, input, output):
                module.__flops__ += sum([elem[1] for elem in module_flop_count])
                module_flop_count.clear()

            has_children = len(module._modules.items()) != 0
            if not has_children:
                module.__post_hook_handle__ = module.register_forward_hook(post_hook)

            def start_time_hook(module, input):
                module.__start_time__ = time.time()

            module.__start_time_hook_handle__ = module.register_forward_pre_hook(
                start_time_hook)

            def end_time_hook(module, input, output):
                module.__duration__ += time.time() - module.__start_time__

            module.__end_time_hook_handle__ = module.register_forward_hook(end_time_hook)

        self.model.apply(partial(register_module_hooks, ignore_list=ignore_list))

    def end_profile(self):
        """Ends profiling.

        Added attributes and handles are removed recursively on all the modules and the torch.nn.functionals are restored.
        """
        def remove_profile_attrs(module):
            if hasattr(module, "__steps__"):
                del module.__steps__
            if hasattr(module, "__flops__"):
                del module.__flops__
            if hasattr(module, "__params__"):
                del module.__params__
            if hasattr(module, "__start_time__"):
                del module.__start_time__
            if hasattr(module, "__duration__"):
                del module.__duration__
            if hasattr(module, "__pre_hook_handle__"):
                module.__pre_hook_handle__.remove()
                del module.__pre_hook_handle__
            if hasattr(module, "__post_hook_handle__"):
                module.__post_hook_handle__.remove()
                del module.__post_hook_handle__
            if hasattr(module, "__flops_handle__"):
                module.__flops_handle__.remove()
                del module.__flops_handle__
            if hasattr(module, "__start_time_hook_handle__"):
                module.__start_time_hook_handle__.remove()
                del module.__start_time_hook_handle__
            if hasattr(module, "__end_time_hook_handle__"):
                module.__end_time_hook_handle__.remove()
                del module.__end_time_hook_handle__

        self.model.apply(remove_profile_attrs)
        _reload_functionals()

    def reset_profile(self):
        """Resets the profiling.

        Adds or resets the extra attributes.
        """
        def add_or_reset_attrs(module):
            module.__flops__ = 0
            module.__params__ = sum(p.numel() for p in module.parameters()
                                    if p.requires_grad)
            module.__start_time__ = 0
            module.__duration__ = 0
            module.__steps__ = 0

        self.model.apply(add_or_reset_attrs)

    def get_total_flops(self, in_str=False):
        """Returns the total flops of the model.

        Args:
            in_str (bool, optional): whether to output the flops in string. Defaults to False.
        """
        if self.get_total_steps() == 0:
            return 0
        sum = 0
        for module in self.model.modules():
            sum += module.__flops__
        total_flops = sum / self.get_total_steps()
        return flops_to_string(total_flops) if in_str else total_flops

    def get_total_duration(self, in_str=False):
        """Returns the total duration of the model forward pass.

        Args:
            in_str (bool, optional): whether to output the duration in string. Defaults to False.
        """
        if self.get_total_steps() == 0:
            return 0
        total_duration = self.model.__duration__ / self.get_total_steps()
        return duration_to_string(total_duration) if in_str else total_duration

    def get_total_params(self, in_str=False):
        """Returns the total parameters of the model.

        Args:
            in_str (bool, optional): whether to output the parameters in string. Defaults to False.
        """
        return params_to_string(
            self.model.__params__) if in_str else self.model.__params__

    def get_total_steps(self):
        """Returns the total number of steps (or input batches) profiled.
        """
        def get_steps(module):
            if module.__steps__ == 0:
                sum = 0
                for m in module.children():
                    sum += get_steps(m)
                module.__steps__ = sum
            return module.__steps__

        total_steps = get_steps(self.model)
        if total_steps == 0:
            print("no step is profiled")
        return total_steps

    def print_model_profile(self):
        """Prints the model graph with the measured profile attached to each module.
        """
        total_flops = self.get_total_flops()
        total_duration = self.get_total_duration()
        total_params = self.get_total_params()
        total_steps = self.get_total_steps()

        def accumulate_flops(module):
            has_children = len(module._modules.items()) != 0
            if not has_children:
                return module.__flops__
            else:
                sum = 0
                for m in module.children():
                    sum += m.accumulate_flops()
            return sum

        def flops_repr(module):
            params = module.__params__
            flops = 0 if total_steps == 0 else module.accumulate_flops() / total_steps
            items = [
                params_to_string(params),
                "{:.2%} Params".format(params / total_params),
                flops_to_string(flops),
                "{:.2%} MACs".format(0.0 if total_flops == 0 else flops / total_flops),
            ]
            duration = 0 if total_steps == 0 else module.__duration__ / total_steps
            items.append(duration_to_string(duration))
            items.append("{:.2%} time".format(0.0 if total_duration == 0 else duration /
                                              total_duration))
            # flops = 2 * MACs
            items.append(("{:.2} TFLOPS".format(0.0 if duration == 0 else 2 * flops /
                                                duration / 10**12)))
            items.append(str(module.__steps__))
            items.append(module.original_extra_repr())
            return ", ".join(items)

        def add_extra_repr(module):
            module.accumulate_flops = accumulate_flops.__get__(module)
            flops_extra_repr = flops_repr.__get__(module)
            if module.extra_repr != flops_extra_repr:
                module.original_extra_repr = module.extra_repr
                module.extra_repr = flops_extra_repr
                assert module.extra_repr != module.original_extra_repr

        def del_extra_repr(module):
            if hasattr(module, "original_extra_repr"):
                module.extra_repr = module.original_extra_repr
                del module.original_extra_repr
            if hasattr(module, "accumulate_flops"):
                del module.accumulate_flops

        self.model.apply(add_extra_repr)
        print(self.model)
        self.model.apply(del_extra_repr)

    def print_model_aggregated_profile(self, module_depth=-1, top_modules=3):
        """Prints the names of the top top_modules modules in terms of aggregated time, flops, and parameters at depth module_depth.

        Args:
            module_depth (int, optional): the depth of the modules to show. Defaults to -1 (the innermost modules).
            top_modules (int, optional): the number of top modules to show. Defaults to 3.
        """
        info = {}
        total_steps = self.get_total_steps()
        if total_steps == 0:
            return
        if not hasattr(self.model, "__flops__"):
            print(
                "no __flops__ attribute in the model, call this function after start_profile and before end_profile"
            )
            return

        def walk_module(module, curr_depth, info):
            if curr_depth not in info:
                info[curr_depth] = {}
            if module.__class__.__name__ not in info[curr_depth]:
                info[curr_depth][module.__class__.__name__] = [
                    0,
                    0,
                    0,
                ]  # flops, params, time
            info[curr_depth][module.__class__.__name__][0] += module.__flops__
            info[curr_depth][module.__class__.__name__][1] += module.__params__
            info[curr_depth][module.__class__.__name__][2] += (module.__duration__)
            has_children = len(module._modules.items()) != 0
            if has_children:
                for child in module.children():
                    walk_module(child, curr_depth + 1, info)

        walk_module(self.model, 0, info)

        depth = module_depth
        if module_depth == -1:
            depth = len(info) - 1

        num_items = min(top_modules, len(info[depth]))

        sort_flops = {
            k: flops_to_string(v[0] / total_steps)
            for k,
            v in sorted(info[depth].items(),
                        key=lambda item: item[1][0],
                        reverse=True)[:num_items]
        }
        sort_params = {
            k: params_to_string(v[1])
            for k,
            v in sorted(info[depth].items(),
                        key=lambda item: item[1][1],
                        reverse=True)[:num_items]
        }
        sort_time = {
            k: duration_to_string(v[2] / total_steps)
            for k,
            v in sorted(info[depth].items(),
                        key=lambda item: item[1][2],
                        reverse=True)[:num_items]
        }
        print(f"Top {num_items} modules in flops at depth {depth} are {sort_flops}")
        print(f"Top {num_items} modules in params at depth {depth} are {sort_params}")
        print(f"Top {num_items} modules in time at depth {depth} are {sort_time}")


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    return torch.numel(input) * out_features


def _relu_flops_compute(input, inplace=False):
    return torch.numel(input)


def _pool_flops_compute(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    return torch.numel(input)


def _conv_flops_compute(input,
                        weight,
                        bias=None,
                        stride=1,
                        padding=0,
                        dilation=1,
                        groups=1):
    assert weight.shape[1] * groups == input.shape[1]

    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[-2:])
    input_dims = list(input.shape[2:])

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    output_dims = [0, 0]
    output_dims[0] = (input_dims[0] + 2 * paddings[0] -
                      (dilations[0] * (kernel_dims[0] - 1) + 1)) // strides[0] + 1
    output_dims[1] = (input_dims[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_dims[1] - 1) + 1)) // strides[1] + 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(output_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def _conv_trans_flops_compute(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    batch_size = input.shape[0]
    in_channels = input.shape[1]
    out_channels = weight.shape[0]
    kernel_dims = list(weight.shape[-2:])
    input_dims = list(input.shape[2:])

    paddings = padding if type(padding) is tuple else (padding, padding)
    strides = stride if type(stride) is tuple else (stride, stride)
    dilations = dilation if type(dilation) is tuple else (dilation, dilation)

    output_dims = [0, 0]
    output_dims[0] = (input_dims[0] + 2 * paddings[0] -
                      (dilations[0] * (kernel_dims[0] - 1) + 1)) // strides[0] + 1
    output_dims[1] = (input_dims[1] + 2 * paddings[1] -
                      (dilations[1] * (kernel_dims[1] - 1) + 1)) // strides[1] + 1

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(_prod(kernel_dims)) * in_channels * filters_per_channel
    active_elements_count = batch_size * int(_prod(input_dims))
    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if bias is not None:
        bias_flops = out_channels * batch_size * int(_prod(output_dims))

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops)


def _batch_norm_flops_compute(
    input,
    running_mean,
    running_var,
    weight=None,
    bias=None,
    training=False,
    momentum=0.1,
    eps=1e-05,
):
    # assume affine is true
    flops = 2 * torch.numel(input)
    return flops


def _upsample_flops_compute(input,
                            size=None,
                            scale_factor=None,
                            mode="nearest",
                            align_corners=None):
    if size is not None:
        return int(_prod(size))
    assert scale_factor is not None
    flops = torch.numel(input)
    if len(scale_factor) == len(input):
        flops * int(_prod(scale_factor))
    else:
        flops * scale_factor**len(input)
    return flops


def _softmax_flops_compute(input, dim=None, _stacklevel=3, dtype=None):
    return torch.numel(input)


def _embedding_flops_compute(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    return 0


def _dropout_flops_compute(input, p=0.5, training=True, inplace=False):
    return 0


def wrapFunc(func, funcFlopCompute):
    oldFunc = func
    name = func.__name__
    old_functions[func.__name__] = oldFunc

    def newFunc(*args, **kwds):
        flops = funcFlopCompute(*args, **kwds)
        module_flop_count.append((name, flops))
        return oldFunc(*args, **kwds)

    return newFunc


def _patch_functionals():
    # FC
    F.linear = wrapFunc(F.linear, _linear_flops_compute)

    # convolutions
    F.conv1d = wrapFunc(F.conv1d, _conv_flops_compute)
    F.conv2d = wrapFunc(F.conv2d, _conv_flops_compute)
    F.conv3d = wrapFunc(F.conv3d, _conv_flops_compute)

    # conv transposed
    F.conv_transpose1d = wrapFunc(F.conv_transpose1d, _conv_trans_flops_compute)
    F.conv_transpose2d = wrapFunc(F.conv_transpose2d, _conv_trans_flops_compute)
    F.conv_transpose3d = wrapFunc(F.conv_transpose3d, _conv_trans_flops_compute)

    # activations
    F.relu = wrapFunc(F.relu, _relu_flops_compute)
    F.prelu = wrapFunc(F.prelu, _relu_flops_compute)
    F.elu = wrapFunc(F.elu, _relu_flops_compute)
    F.leaky_relu = wrapFunc(F.leaky_relu, _relu_flops_compute)
    F.relu6 = wrapFunc(F.relu6, _relu_flops_compute)

    # BatchNorms
    F.batch_norm = wrapFunc(F.batch_norm, _batch_norm_flops_compute)

    # poolings
    F.avg_pool1d = wrapFunc(F.avg_pool1d, _pool_flops_compute)
    F.avg_pool2d = wrapFunc(F.avg_pool2d, _pool_flops_compute)
    F.avg_pool3d = wrapFunc(F.avg_pool3d, _pool_flops_compute)
    F.max_pool1d = wrapFunc(F.max_pool1d, _pool_flops_compute)
    F.max_pool2d = wrapFunc(F.max_pool2d, _pool_flops_compute)
    F.max_pool3d = wrapFunc(F.max_pool3d, _pool_flops_compute)
    F.adaptive_avg_pool1d = wrapFunc(F.adaptive_avg_pool1d, _pool_flops_compute)
    F.adaptive_avg_pool2d = wrapFunc(F.adaptive_avg_pool2d, _pool_flops_compute)
    F.adaptive_avg_pool3d = wrapFunc(F.adaptive_avg_pool3d, _pool_flops_compute)
    F.adaptive_max_pool1d = wrapFunc(F.adaptive_max_pool1d, _pool_flops_compute)
    F.adaptive_max_pool2d = wrapFunc(F.adaptive_max_pool2d, _pool_flops_compute)
    F.adaptive_max_pool3d = wrapFunc(F.adaptive_max_pool3d, _pool_flops_compute)

    # upsample
    F.upsample = wrapFunc(F.upsample, _upsample_flops_compute)
    F.interpolate = wrapFunc(F.interpolate, _upsample_flops_compute)

    # softmax
    F.softmax = wrapFunc(F.softmax, _softmax_flops_compute)

    # embedding
    F.embedding = wrapFunc(F.embedding, _embedding_flops_compute)


def _reload_functionals():
    # torch.nn.functional does not support importlib.reload()
    F.linear = old_functions["linear"]
    F.conv1d = old_functions["conv1d"]
    F.conv2d = old_functions["conv2d"]
    F.conv3d = old_functions["conv3d"]
    F.conv_transpose1d = old_functions["conv_transpose1d"]
    F.conv_transpose2d = old_functions["conv_transpose2d"]
    F.conv_transpose3d = old_functions["conv_transpose3d"]
    F.relu = old_functions["relu"]
    F.prelu = old_functions["prelu"]
    F.elu = old_functions["elu"]
    F.leaky_relu = old_functions["leaky_relu"]
    F.relu6 = old_functions["relu6"]
    F.batch_norm = old_functions["batch_norm"]
    F.avg_pool1d = old_functions["avg_pool1d"]
    F.avg_pool2d = old_functions["avg_pool2d"]
    F.avg_pool3d = old_functions["avg_pool3d"]
    F.max_pool1d = old_functions["max_pool1d"]
    F.max_pool2d = old_functions["max_pool2d"]
    F.max_pool3d = old_functions["max_pool3d"]
    F.adaptive_avg_pool1d = old_functions["adaptive_avg_pool1d"]
    F.adaptive_avg_pool2d = old_functions["adaptive_avg_pool2d"]
    F.adaptive_avg_pool3d = old_functions["adaptive_avg_pool3d"]
    F.adaptive_max_pool1d = old_functions["adaptive_max_pool1d"]
    F.adaptive_max_pool2d = old_functions["adaptive_max_pool2d"]
    F.adaptive_max_pool3d = old_functions["adaptive_max_pool3d"]
    F.upsample = old_functions["upsample"]
    F.interpolate = old_functions["interpolate"]
    F.softmax = old_functions["softmax"]
    F.embedding = old_functions["embedding"]


def _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size * 3
        # last two hadamard _product and add
        flops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size * 4
        # two hadamard _product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def _rnn_forward_hook(rnn_module, input, output):
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__("weight_ih_l" + str(i))
        w_hh = rnn_module.__getattr__("weight_hh_l" + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = _rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__("bias_ih_l" + str(i))
            b_hh = rnn_module.__getattr__("bias_hh_l" + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def _rnn_cell_forward_hook(rnn_cell_module, input, output):
    flops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__("weight_ih")
    w_hh = rnn_cell_module.__getattr__("weight_hh")
    input_size = inp.shape[1]
    flops = _rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__("bias_ih")
        b_hh = rnn_cell_module.__getattr__("bias_hh")
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


MODULE_HOOK_MAPPING = {
    # RNN
    nn.RNN: _rnn_forward_hook,
    nn.GRU: _rnn_forward_hook,
    nn.LSTM: _rnn_forward_hook,
    nn.RNNCell: _rnn_cell_forward_hook,
    nn.LSTMCell: _rnn_cell_forward_hook,
    nn.GRUCell: _rnn_cell_forward_hook,
}


def flops_to_string(flops, units=None, precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.0**9, precision)) + " GMACs"
        elif flops // 10**6 > 0:
            return str(round(flops / 10.0**6, precision)) + " MMACs"
        elif flops // 10**3 > 0:
            return str(round(flops / 10.0**3, precision)) + " KMACs"
        else:
            return str(flops) + " MACs"
    else:
        if units == "GMACs":
            return str(round(flops / 10.0**9, precision)) + " " + units
        elif units == "MMACs":
            return str(round(flops / 10.0**6, precision)) + " " + units
        elif units == "KMACs":
            return str(round(flops / 10.0**3, precision)) + " " + units
        else:
            return str(flops) + " MACs"


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10**6 > 0:
            return str(round(params_num / 10**6, 2)) + " M"
        elif params_num // 10**3:
            return str(round(params_num / 10**3, 2)) + " k"
        else:
            return str(params_num)
    else:
        if units == "M":
            return str(round(params_num / 10.0**6, precision)) + " " + units
        elif units == "K":
            return str(round(params_num / 10.0**3, precision)) + " " + units
        else:
            return str(params_num)


def duration_to_string(duration, units=None, precision=2):
    if units is None:
        if duration > 1:
            return str(round(duration, precision)) + " s"
        elif duration * 10**3 > 1:
            return str(round(duration * 10**3, precision)) + " ms"
        elif duration * 10**6 > 1:
            return str(round(duration * 10**6, precision)) + " us"
        else:
            return str(duration)
    else:
        if units == "us":
            return str(round(duration * 10.0**6, precision)) + " " + units
        elif units == "ms":
            return str(round(duration * 10.0**3, precision)) + " " + units
        else:
            return str(round(duration, precision)) + " s"


def get_model_profile(
    model,
    input_res,
    input_constructor=None,
    print_profile=True,
    print_aggregated_profile=True,
    module_depth=-1,
    top_modules=3,
    warm_up=5,
    num_steps=10,
    as_strings=True,
    ignore_modules=None,
):
    """Returns the total flops, parameters, and profiled steps of a model.

    Args:
        model ([torch.nn.Module]): the PyTorch model to be profiled.
        input_res (list): input shape or input to the input_constructor
        input_constructor (func, optional): input constructor. If specified, the constructor is applied to input_res and the constructor output is used as the input to the model. Defaults to None.
        print_profile (bool, optional): whether to print the model graph with the profile annotated. Defaults to True.
        print_aggregated_profile (bool, optional): whether to print the aggregated profile for top modules. Defaults to True.
        module_depth (int, optional): the depth into the nested modules. Defaults to -1 (the inner most modules).
        top_modules (int, optional): the number of top modules to print in the aggregated profile. Defaults to 3.
        warm_up (int, optional): the number of warm-up steps before measuring the time of each module. Defaults to 5.
        num_steps (int, optional): the number of steps to profile. Defaults to 10.
        as_strings (bool, optional): whether to print the output as strings. Defaults to True.
        ignore_modules ([type], optional): the list of modules to ignore during profiling. Defaults to None.
    """
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    prof = FlopsProfiler(model)
    model.eval()
    for _ in range(warm_up):
        if input_constructor:
            input = input_constructor(input_res)
            _ = model(**input)
        else:
            try:
                batch = torch.ones(()).new_empty(
                    (*input_res),
                    dtype=next(model.parameters()).dtype,
                    device=next(model.parameters()).device,
                )
            except StopIteration:
                batch = torch.ones(()).new_empty((*input_res))
            _ = model(batch)

    prof.start_profile(ignore_list=ignore_modules)

    for _ in range(num_steps):
        if input_constructor:
            input = input_constructor(input_res)
            _ = model(**input)
        else:
            try:
                batch = torch.ones(()).new_empty(
                    (*input_res),
                    dtype=next(model.parameters()).dtype,
                    device=next(model.parameters()).device,
                )
            except StopIteration:
                batch = torch.ones(()).new_empty((*input_res))
            _ = model(batch)

    flops = prof.get_total_flops()
    params = prof.get_total_params()
    steps = prof.get_total_steps()
    if print_profile:
        prof.print_model_profile()
    if print_aggregated_profile:
        prof.print_model_aggregated_profile(module_depth=module_depth,
                                            top_modules=top_modules)
    prof.end_profile()
    if as_strings:
        return flops_to_string(flops), params_to_string(params), steps

    return flops, params, steps
