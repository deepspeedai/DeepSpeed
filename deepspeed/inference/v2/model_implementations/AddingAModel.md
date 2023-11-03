# Adding Support for a New Model in DeepSpeed Inference V2

Adding supoprt for a new model in DeepSpeed Inference requires developing three related components:
- Containers: These describe the parameters contained in the model
- Model implementation: How should the model be computed.
- Policy: The map for adding parameters to your containers and creating the model implementation.

In this tutorial, we will assume that you'd like to use a relatively traditionally styled Transformer model and will be able to inherit from `DSTransformerModelBase` and can take advantage of the utilities that provides.

## Defining Your Containers

A container is the bridge between the original model's parameters and how to transform them to serve them for inference. For a model implementation, there are two primary kinds of containers: transformer containers and non-transformer containers. A transformer container consists of the parameters for a single Transformer layer in the model. So this includes your traditional parameters like the projections for the fully connected network, or query-key-value projections. The non-transformer container will contain basically everything else! However, before defining these containers, we need to understand how to define an individual parameter.

In DeepSpeed inference, the original model parameters are populated into the model and mapped as dependencies to a parameter. A `Parameter` has two primary components: its dependencies and its `finalize` method. Let's do an example. In Llama models, the native format is for the `query`, `key`, and `value` projections to be performed independently. However, we can achieve higher throughput by fusing them into a single larger projection. We can define this fusion with a parameter:

```python
from deepspeed.inference.module_implementations.parameter_base import ParameterBase

class UnfusedQKVParameter(ParameterBase):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor

    def finalize(self) -> torch.Tensor:
        fused_param = torch.cat([self.query, self.key, self.value], dim=0)
        return self.inference_model.transform_qkv_param(fused_param)
```

Let's walk through each part of this implementation. First, parameters should inherit from `ParameterBase`. This will allow it to automatically determine when its dependencies are met and set the appropriate components of a parent `LayerContainer`. The second key component is the type annotations on the class itself. Each type annotation represents a dependency of the parameter. Since the original Llama mode has separate query, key, and value dependencies, our fused parameter will declare dependencies for each. Finally, we have the `finalize` method. This method is automatically called once all dependencies on the layer are met and should return the final parameter.

In this `finalize` method, we are doing two things: the first is the act of fusing the parameters together through the concatenate method. Note that each of the dependencies can be accessed via `self.{name}`. The second is calling `self.inference_model.transform_qkv_param`. A parameter's finalize method always has access to the inference model. In this case we are using that to use a feature provided by `DSTransformerBase`. This method will automatically shard the parameter for tensor parallelism and then pass it to the linear module implementation to perform additional optimizations or shape transformations, like quantization.

Since many patterns are very common in Transformer models, `model_implementations.common_parameters` provides implementations for many of the patterns (all compatible with `DSTransformerBase`) to help accelerate development.

Once all parameters are created, we need to compose them into a layer container. In our simplified Llama model, let's assume there's only QKV and attention output projection matrices. A layer container would appear as the following:

```python
from deepspeed.inference.module_implementations.layer_container_base import LayerContainer

class ExampleContainer(LayerContainer):
    qkvw: UnfusedQKVParameter

    attn_o: AttentionOutputParameter

    PARAM_MAPPING: {
        "self_attn.q_proj.weight": "qkvw.query",
        "self_attn.k_proj.weight": "qkvw.key",
        "self_attn.v_proj.weight": "qkvw.value",
        "self_attn.o_proj.weight": "attn_o.params",
    }
```

Once again, we have a couple of key components. The first are parameter type annotations. Each annotation corresponds to a parameter that can be used in the model implementation. In the model implementation, I can simply write `container.qkvw` to access my fused and transformed QKV parameter. The second key component is the `PARAM_MAPPING` dictionary. This is our explicit mapping of the names of parameters in the source model to a parameter dependency. This mapping dictionary will be used by the policy to automatically populate dependencies.

Once you have written `LayerContainer`s for both the transformer and non-transformer parameters, it's time to work on the model implementation!

## Building a Model Implementation that Inherits from `DSTransformerBase`

By inheriting from `DSTransformerBase`, most of the implementation work for sharding and transforming parameters will be automatically handled for you. However, there are four key tasks that still need to be completed.

1. Defining the abstract properties based on your model configuration.
2. Configuring embedding and unembedding modules and the forward implementations for them.
3. Configuring the attention configuration and desired KV cache behaviors.
4. Writing the forward implementation for your layer.

## Writing a Policy

The `InferenceV2Policy` is the level of composition. This is the object that will be passed directly to the inference engine and will compose the model implementation and your containers to create an end-to-end solution. There are two main components to be implemented: the first is to create the model that you defined earlier. This is done by implementing the `instantiate_model` method of the policy. In general, this can just be implemented by calling the constructor for your model and passing the engine config, tensor-parallel communication object, and your custom model config.

The second component is to define how the parameters from the checkpoint will map to each container. From the section on `LayerContainer`s above, you may remember that the `LayerContainer` can handle the internal routing of a checkpoint parameter to its dependency. In order to find the correct `LayerContainer` though, we need a second abstraction: the `ContainerMap`.

A `ContainerMap` performs this mapping by categorizing checkpoint prefix strings to the type of container they map to. Typically, the easiest way to do this is through iterating over a model checkpoint's state dict or by iterating over the `named_parameters` of a PyTorch model. There are three types of mappings to define: the transformer mappings, the non-transformer mappings, and the what we'll call the rest. Let's work through an example:

```python
from deepspeed.inference.module_implementations.inference_policy_base import ContainerMap

def build_container_map(self) -> ContainerMap:
    map = ContainerMap()

    transformer_containers = [MyTransformerContainer(self.model) for _ in range(self.model.num_layers)]
    map.set_transformer_params("model.layers", transformer_containers)

    non_transformer_container = MyNonTransformerContainer(self.model)
```
