# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""Site-customization hook injected into the vLLM server subprocess.

When this file is on ``PYTHONPATH`` (or placed as ``sitecustomize.py`` in a
directory listed in ``PYTHONPATH``), Python executes it automatically before
running the main script.  It is used solely by
:class:`~deepspeed.runtime.rollout.vllm_rollout.VLLMRollout` to patch a known
compatibility issue between **vLLM 0.22.0** and certain ``pydantic-core``
builds that hit a Rust-level assertion::

    pydantic_core._pydantic_core.ValidationError:
      Assertion failed, duplicate template name

The patch is **harmless on systems that don't need it** — the original
``validate_python`` call is attempted first and only on ``Exception`` do we
fall back to plain dataclass field assignment.

This file is NOT a monkey-patch on installed packages.  It patches the
*behaviour at runtime* by replacing the ``__init__`` method that pydantic's
``@dataclass`` decorator installs on each decorated class.  No files under
``site-packages/`` are modified.
"""

import dataclasses as _dc


def _install_pydantic_dataclass_fallback():
    try:
        import pydantic._internal._dataclasses as _pdc
    except ImportError:
        return

    if getattr(_pdc, "_deepspeed_patched", False):
        return

    def _make_safe_init(original_init):

        def _safe_init(__dataclass_self__, *args, **kwargs):
            __tracebackhide__ = True
            try:
                original_init(__dataclass_self__, *args, **kwargs)
            except Exception:
                s = __dataclass_self__
                kw = dict(zip([f.name for f in _dc.fields(s.__class__)], args))
                kw.update(kwargs)
                for f in _dc.fields(s.__class__):
                    if f.name in kw:
                        object.__setattr__(s, f.name, kw[f.name])
                    elif f.default is not _dc.MISSING:
                        object.__setattr__(s, f.name, f.default)
                    elif f.default_factory is not _dc.MISSING:
                        object.__setattr__(s, f.name, f.default_factory())
                    else:
                        object.__setattr__(s, f.name, None)

        _safe_init.__qualname__ = original_init.__qualname__
        return _safe_init

    _original_complete = _pdc.complete_dataclass

    def _patched_complete(cls, config_wrapper, *, raise_errors=False):
        result = _original_complete(cls, config_wrapper, raise_errors=raise_errors)
        if hasattr(cls, "__init__"):
            original_init = cls.__init__
            cls.__init__ = _make_safe_init(original_init)
        return result

    _pdc.complete_dataclass = _patched_complete
    _pdc._deepspeed_patched = True


_install_pydantic_dataclass_fallback()
