# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""A rollout rank's weights as views of one buffer that is never reallocated.

CUDA graphs record raw device pointers. As long as every weight update writes into the same storage, a decode graph
captured once stays valid across updates; a weight that is reallocated (``param.data = new``, a gathered temporary)
leaves the graph reading freed memory. Keeping all of a rank's weights in one buffer makes that invariant easy to
hold and to check (:meth:`WeightArena.address`), and gives weight-sync plans a single flat destination: a piece's
``dest_offset`` is its position in the arena.
"""

from typing import Dict, Iterator, Sequence, Tuple

import torch

# Each view starts on a 256-byte boundary, as allocator blocks do, so kernels see the alignment they expect.
_ALIGN_BYTES = 256


class WeightArena:
    """Named views of one flat buffer.

    Args:
        layout: ``{name: shape}`` in allocation order.
        dtype, device: of the buffer.
    """

    def __init__(self, layout: Dict[str, Sequence[int]], dtype: torch.dtype, device):
        align = max(1, _ALIGN_BYTES // torch.empty(0, dtype=dtype).element_size())
        self.offsets: Dict[str, int] = {}
        self.shapes: Dict[str, Tuple[int, ...]] = {}
        total = 0
        for name, shape in layout.items():
            total = -(-total // align) * align
            self.offsets[name] = total
            self.shapes[name] = tuple(shape)
            total += _numel(shape)
        # Zeros: padding (e.g. vocabulary rows past the end) must not hold garbage the forward could read.
        self.buffer = torch.zeros(total, dtype=dtype, device=device)
        self._views = {
            name: self.buffer[off:off + _numel(self.shapes[name])].view(self.shapes[name])
            for name, off in self.offsets.items()
        }

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._views[name]

    def __contains__(self, name: str) -> bool:
        return name in self._views

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return iter(self._views.items())

    @property
    def nbytes(self) -> int:
        return self.buffer.numel() * self.buffer.element_size()

    def address(self) -> Tuple[int, int]:
        """``(data_ptr, numel)`` of the buffer: equal before and after an update when nothing was reallocated."""
        return self.buffer.data_ptr(), self.buffer.numel()


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for dim in shape:
        n *= dim
    return n
