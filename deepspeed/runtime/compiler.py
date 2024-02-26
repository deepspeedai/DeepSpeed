# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Union, Callable, Dict, Any
import importlib
import torch
from ..pydantic_v1 import validator
from .config_utils import DeepSpeedConfigModel

COMPILE_CONFIG = "compile"


def is_compile_supported():
    return hasattr(torch, "compiler")


def disable(func):
    if is_compile_supported():
        return torch.compiler.disable(func)
    return func


def get_compile_config(param_dict):
    if COMPILE_CONFIG in param_dict:
        compile_config_dict = param_dict[COMPILE_CONFIG]
    else:
        compile_config_dict = {}
    return CompileConfig(**compile_config_dict)


def get_backend_fn(backend: Union[str, Callable]) -> Union[str, Callable]:
    if isinstance(backend, Callable):
        return backend

    elif isinstance(backend, str):
        if backend in torch._dynamo.list_backends(exclude_tags=()):
            return backend

        # Get module name from backend name
        module_name = '.'.join(backend.split('.')[:-1])
        fn_name = backend.split('.')[-1]

        try:
            module = importlib.import_module(module_name)
            backend_fn = getattr(module, fn_name)
        except ImportError:
            raise ValueError(
                f"The backend {backend} is not in the list of available backends and could not be imported.")
        return backend_fn

    raise ValueError(f"backend for torch.compile must be a string or Callable: {backend}")


class CompileConfig(DeepSpeedConfigModel):
    """
    [EXPERIMENTAL] This configuration enables users to activate `torch.compile` within DeepSpeed and customize its settings.
    Please be aware that these features and API designs are experimental and subject to change.
    """

    enabled: bool = False
    """
    Enable torch.compile when True.
    """

    backend: str = "inductor"
    """
    Passed to `backend` argument of torch.compile.
    If the given value is not in torch._dynamo.list_backends(),
    DeepSpeed attempts to import and instantiate the module with the given name.
    """

    kwargs: Dict[str, Any] = {}
    """
    Passed to `kwargs` argument of torch.compile.
    """

    @validator("enabled")
    def validate_enabled(cls, field_value, values):
        if field_value and not is_compile_supported():
            raise ValueError("torch.compile is not supported on this version of PyTorch.")
        return field_value


class CompiledModuleWrapper(torch.nn.Module):

    def __init__(self, module, compile_config: Union[CompileConfig, None] = None):
        super().__init__()

        assert is_compile_supported(), "torch.compile is not supported on this version of PyTorch."

        modules = self.__dict__.get('_modules')
        modules['wrapped'] = module
        self.__dict__['wrapped'] = module
        self._is_compiled = False
        self._backend = get_backend_fn(compile_config.backend)
        self._compile_kwargs = compile_config.kwargs
        self._compiler_fn = None

    def __getattr__(self, name):
        return getattr(self.__dict__['wrapped'], name)

    def set_backend(self, backend: Union[str, Callable]):
        """Set the backend for torch.compile.

        Args:
            backend (Union[str, Callable]): backend name or a function that takes a torch.nn.Module and returns a compiled module.
            You can directly pass a function that works as a backend.
            See also `backend` field in `CompileConfig` for more details.
        """
        self._backend = get_backend_fn(backend)

    def set_torch_compile_kwargs(self, kwargs: Dict[str, Union[str, Any]]) -> None:
        """Set kwargs for torch.compile. Kwargs that are set in DeepSpeed config will be overwritten.
        You can also pass a backend name with "backend" key to change the backend.

        Args:
            kwargs (Dict[str, Union[str, Any]]): kwargs passed to torch.compile.
        """

        if "backend" in kwargs:
            raise ValueError("backend cannot be set as compile kwargs. Use set_backend instead.")
        self._compile_kwargs.update(kwargs)

    def set_compiler_fn(self, compiler_fn: Callable) -> None:
        """Set a function to be used for compiling the module.
        This function should take a torch.nn.Module as input and return a compiled module.
        Note that other compile options are ignored when a compiler_fn is set.

        Example:
        ```python
            def my_compiler_fn(module: torch.nn.Module):
                ...
                return torch.compile(module, ...)

            engine.set_compiler_fn(my_compiler_fn)
        ```
        """
        self._compiler_fn = compiler_fn

    def forward(self, *args, **kwargs) -> Any:
        if not self.is_compiled:
            if self._compiler_fn is None:
                self.__dict__['wrapped'] = torch.compile(self.wrapped, backend=self._backend, **self._compile_kwargs)
            else:
                self.__dict__['wrapped'] = self._compiler_fn(self.wrapped)
            self._is_compiled = True

        return self.__dict__['wrapped'](*args, **kwargs)

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    @property
    def backend(self) -> Union[str, Callable]:
        return self._backend

    @property
    def torch_compile_kwargs(self) -> Dict[str, Any]:
        return self._compile_kwargs

    @property
    def compiler_fn(self) -> Union[Callable, None]:
        return self._compiler_fn
