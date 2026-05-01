"""Numeric tile codec helpers for Crafter pattern learning.

The pattern-learning stack uses integer token grids rather than string grids.
This module provides the small conversion layer that keeps the rest of the
code clean while still interoperating with :mod:`crafter.known_world`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from crafter import constants
from crafter.known_world import KnownWorld, SimpleWorld

_WORLDGEN_MATERIALS = tuple(
  material for material in constants.materials if material not in {'table', 'furnace'}
)


@dataclass(frozen=True)
class CrafterTileCodec:
  """Bidirectional mapping between Crafter materials and integer tokens.

  Parameters
  ----------
  materials : tuple[str, ...]
      Ordered material vocabulary used for prediction classes.
  unknown_id : int
      Token used for hidden cells in partial maps.
  boundary_id : int
      Token used when a cross-shaped pattern touches the map boundary.
  """

  materials: tuple[str, ...]
  unknown_id: int
  boundary_id: int

  @classmethod
  def for_worldgen_materials(cls) -> 'CrafterTileCodec':
    """Build the default codec for Crafter initial world-generation materials.

    Returns
    -------
    CrafterTileCodec
        Codec whose classes match the initial-world material vocabulary.
    """
    materials = _WORLDGEN_MATERIALS
    return cls(
      materials=materials,
      unknown_id=len(materials),
      boundary_id=len(materials) + 1,
    )

  @property
  def num_classes(self) -> int:
    """Return the number of predicted material classes."""
    return len(self.materials)

  def encode_material(self, material: str) -> int:
    """Encode one material string into its prediction-class integer id.

    Parameters
    ----------
    material : str
        Material name from the Crafter initial-world vocabulary.

    Returns
    -------
    int
        Integer class id in ``[0, num_classes)``.
    """
    try:
      return self.materials.index(material)
    except ValueError as exc:
      raise ValueError(f'Unknown codec material: {material!r}') from exc

  def decode_material(self, material_id: int) -> str:
    """Decode one prediction-class integer id into a material string."""
    if not 0 <= int(material_id) < self.num_classes:
      raise ValueError(
        f'Expected material_id in [0, {self.num_classes}), got {material_id}'
      )
    return self.materials[int(material_id)]

  def encode_material_name_grid(
    self,
    material_name_grid: np.ndarray,
    device: torch.device | str | None = None,
  ) -> torch.Tensor:
    """Encode a material-name grid into a dense integer tensor.

    Parameters
    ----------
    material_name_grid : np.ndarray
        Two-dimensional array whose entries are Crafter material strings.
    device : torch.device | str, optional
        Destination torch device.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(H, W)``.
    """
    if material_name_grid.ndim != 2:  # noqa: PLR2004
      raise ValueError(
        f'Expected a 2D material-name grid, got {material_name_grid.ndim}D'
      )
    encoded = np.empty(material_name_grid.shape, dtype=np.int64)
    for index, material in np.ndenumerate(material_name_grid):
      encoded[index] = self.encode_material(str(material))
    return torch.as_tensor(encoded, dtype=torch.long, device=device)

  def encode_simple_world(
    self,
    world: SimpleWorld,
    device: torch.device | str | None = None,
  ) -> torch.Tensor:
    """Encode a :class:`SimpleWorld` snapshot into class ids.

    Parameters
    ----------
    world : SimpleWorld
        Source world snapshot.
    device : torch.device | str, optional
        Destination torch device.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(H, W)``.
    """
    return self.encode_material_name_grid(world.material_name_grid(), device=device)

  def encode_known_world_initial_partial(
    self,
    known_world: KnownWorld,
    device: torch.device | str | None = None,
  ) -> torch.Tensor:
    """Encode a known world into a partial initial-world token grid.

    Parameters
    ----------
    known_world : KnownWorld
        Known-world state whose immutable initial-world view is used.
    device : torch.device | str, optional
        Destination torch device.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(H, W)`` where hidden cells are ``unknown_id``.
    """
    full_ids = self.encode_simple_world(known_world.initial_world, device='cpu')
    partial_ids = torch.full_like(full_ids, fill_value=self.unknown_id)
    perceived_mask = torch.as_tensor(known_world.perceived_mask, dtype=torch.bool)
    partial_ids[perceived_mask] = full_ids[perceived_mask]
    for x, y in zip(*np.nonzero(known_world.imputed_mask), strict=True):
      partial_ids[int(x), int(y)] = self.encode_material(
        known_world.imputed_material_at((int(x), int(y)))
      )
    if device is not None:
      partial_ids = partial_ids.to(device)
    return partial_ids

  def one_hot(self, material_ids: torch.Tensor) -> torch.Tensor:
    """Return class one-hot encodings for a material-id tensor.

    Parameters
    ----------
    material_ids : torch.Tensor
        Integer class-id tensor whose entries are in ``[0, num_classes)``.

    Returns
    -------
    torch.Tensor
        Float tensor with shape ``material_ids.shape + (num_classes,)``.
    """
    if material_ids.dtype not in {torch.int32, torch.int64, torch.long}:
      raise ValueError(
        f'Expected integer material ids, got dtype={material_ids.dtype!s}'
      )
    one_hot = torch.nn.functional.one_hot(
      material_ids.to(torch.long), num_classes=self.num_classes
    )
    return one_hot.to(dtype=torch.float32)
