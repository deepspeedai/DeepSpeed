"""Compatibility functions to support wider version ranges for python and dependencies."""

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Mapping, Any, Dict

try:
    import annotationlib  # python >= 3.14
except ImportError:
    annotationlib = None

# Deal with annotations in python versions >=3.14. See:
#   - Python 3.14 release notes: https://docs.python.org/3/whatsnew/3.14.html
#     Porting annotations: https://docs.python.org/3/whatsnew/3.14.html#whatsnew314-porting-annotations
#   - PEP649: https://peps.python.org/pep-0649/
#   - PEP749: https://peps.python.org/pep-0749/
# Backwards compatible, applies best practices (use annotationlib) from python 3.14 onwards.


def get_annotations_from_namespace(namespace: Mapping[str, object]) -> Dict[str, Any]:
    if annotationlib:
        annotate_func = annotationlib.get_annotate_from_class_namespace(namespace)
        if annotate_func is not None:
            return annotationlib.call_annotate_function(annotate_func, annotationlib.Format.VALUE)
    return namespace.get("__annotations__", {})


def get_annotations(obj: Any) -> Dict[str, Any]:
    if annotationlib:
        if not isinstance(obj, type):
            obj = type(obj)
        return annotationlib.get_annotations(obj)
    try:
        return obj.__annotations__
    except AttributeError:
        return {}
