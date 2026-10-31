# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Index and box arithmetic shared by affine maps, transfer planning and storage maps.

Pure functions on shapes, strides, offsets and axis-aligned boxes. They know nothing about pieces,
ranks or tensors, so one implementation can serve every module that describes or plans shards.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    'LogicalBox', 'coordinate', 'dot', 'product', 'row_major_range_boxes', 'row_major_strides', 'strided_address'
]


def product(shape: Sequence[int]) -> int:
    count = 1
    for size in shape:
        count *= size
    return count


def row_major_strides(shape: Sequence[int]) -> Tuple[int, ...]:
    """Element stride per axis for a densely packed tensor of this shape."""
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    return tuple(strides)


def coordinate(offset: int, strides: Tuple[int, ...], shape: Tuple[int, ...]) -> Optional[Tuple[int, ...]]:
    """The grid coordinate a storage offset names, or None when it names nothing."""
    coordinates = []
    rest = offset
    for stride, size in zip(strides, shape):
        position, remainder = divmod(rest, stride)
        if position >= size:
            return None
        coordinates.append(position)
        rest = remainder
    return tuple(coordinates) if rest == 0 else None


def dot(index, strides):
    return sum(position * stride for position, stride in zip(index, strides))


def strided_address(shape, offset, strides, index):
    address = offset
    for size, stride in reversed(list(zip(shape, strides))):
        index, position = divmod(index, size)
        address += position * stride
    return address


def row_major_range_boxes(shape, low, high, dense_rows=0) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Split the row-major index range ``[low, high)`` of ``shape`` into ``(origin, extent)`` boxes.

    At most two boxes per axis: a partial run at either end and whole runs between. The first
    ``dense_rows`` axes are never merged, so a box spans more than one position of such an axis only
    where the storage is contiguous across it.
    """
    if low >= high:
        return []
    if not shape:
        return [((), ())]
    if len(shape) == 1:
        return [((low, ), (high - low, ))]
    inner = product(shape[1:])
    first, low_rest = divmod(low, inner)
    last, high_rest = divmod(high, inner)

    def within(row, start, stop):
        return [((row, ) + origin, (1, ) + extent)
                for origin, extent in row_major_range_boxes(shape[1:], start, stop, dense_rows - 1)]

    if first == last:
        return within(first, low_rest, high_rest)
    boxes = []
    if low_rest:
        boxes.extend(within(first, low_rest, inner))
        first += 1
    if dense_rows > 0:
        for row in range(first, last):
            boxes.extend(within(row, 0, inner))
    elif last > first:
        boxes.append(((first, ) + (0, ) * (len(shape) - 1), (last - first, ) + tuple(shape[1:])))
    if high_rest:
        boxes.extend(within(last, 0, high_rest))
    return boxes


@dataclass(frozen=True)
class LogicalBox:
    """An axis-aligned block of the logical parameter: its first coordinate and its extents."""
    origin: Tuple[int, ...]
    extent: Tuple[int, ...]

    @property
    def volume(self) -> int:
        return product(self.extent)

    @property
    def is_empty(self) -> bool:
        return any(size == 0 for size in self.extent)

    def upper(self) -> Tuple[int, ...]:
        return tuple(low + size for low, size in zip(self.origin, self.extent))

    def intersection(self, other: 'LogicalBox') -> 'LogicalBox':
        low = tuple(max(a, b) for a, b in zip(self.origin, other.origin))
        high = tuple(min(a, b) for a, b in zip(self.upper(), other.upper()))
        return LogicalBox(low, tuple(max(0, h - l) for h, l in zip(high, low)))

    def contains(self, other: 'LogicalBox') -> bool:
        return all(other_low >= low and other_high <= high
                   for low, high, other_low, other_high in zip(self.origin, self.upper(), other.origin, other.upper()))

    def __repr__(self) -> str:
        return f'Box({self.origin}+{self.extent})'


def boxes_disjoint(boxes: Sequence[LogicalBox]) -> bool:
    """Whether no two boxes share an element.

    Sweeps along the axis whose box origins are most varied, so a box is compared only with the boxes
    still open where it starts. For the grid and run layouts the builders emit that is a handful of
    boxes, not all of them.
    """
    boxes = [box for box in boxes if not box.is_empty]
    if len(boxes) < 2:
        return True
    if not boxes[0].origin:
        return False
    axis = max(range(len(boxes[0].origin)), key=lambda a: len({box.origin[a] for box in boxes}))
    open_boxes: List[LogicalBox] = []
    for box in sorted(boxes, key=lambda b: b.origin[axis]):
        start = box.origin[axis]
        open_boxes = [other for other in open_boxes if other.origin[axis] + other.extent[axis] > start]
        if any(not box.intersection(other).is_empty for other in open_boxes):
            return False
        open_boxes.append(box)
    return True
