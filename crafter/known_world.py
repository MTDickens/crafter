"""Known-world state and task metadata for the Crafter planner.

This module provides three layers used by the task-and-motion planner:

``SimpleWorld``
    A lightweight, mostly static world snapshot that keeps just enough state
    for planning over terrain and placeable materials.
``Task``
    A small task catalog plus logic that expands a high-level achievement-like
    task into material requirements, layout requirements, and micro-actions.
``KnownWorld``
    The agent's evolving belief/state during planning. Unknown cells are
    revealed from the initial world snapshot, while already-revealed cells are
    allowed to change over time as the planner simulates actions.
"""

from __future__ import annotations

import collections
from copy import copy
from dataclasses import dataclass
from typing import Final

import numpy as np

from crafter import constants
from crafter.engine import RuntimeRules


def _inside(lhs, mid, rhs):
  return (lhs[0] <= mid[0] < rhs[0]) and (lhs[1] <= mid[1] < rhs[1])


def _positions_within(area):
  for x in range(area[0]):
    for y in range(area[1]):
      yield (x, y)


@dataclass(frozen=True)
class LayoutRequirement:
  """A layout-shaped prerequisite for a task.

  Parameters
  ----------
  kind : str
      Requirement family. In the current implementation this is always
      ``"layout"``.
  name : str
      Concrete layout identifier, e.g. ``"craft_cluster_4x4"``.
  """

  kind: str
  name: str


class SimpleWorld:
  """A lightweight planner world with only terrain/material state.

  The class mirrors the subset of :class:`crafter.engine.World` that the
  planner needs. It intentionally ignores dynamic game mechanics such as mobs,
  hunger, thirst, and daylight. The world can still *evolve* from the
  planner's point of view, because material tiles can change after collect,
  place, and make actions.

  Parameters
  ----------
  area : tuple[int, int]
      World size in ``(width, height)`` order.
  materials : sequence[str]
      Material vocabulary used to build the internal integer encoding.
  runtime_rules : RuntimeRules, optional
      Runtime flags carried over from the environment. They are mostly inert in
      the planner, but the state is kept for compatibility with the original
      world abstraction.

  Notes
  -----
  The original TODO text is preserved below on purpose.

  TODO preserved from the original file:
  - Implement this "SimpleWorld" as the docstring says.
  """

  _CHUNK_SIZE: Final[tuple[int, int]] = (12, 12)

  def __init__(self, area, materials, runtime_rules=None):
    self.area = tuple(area)
    self._chunk_size = self._CHUNK_SIZE
    self._mat_names = {i: x for i, x in enumerate([None] + list(materials))}
    self._mat_ids = {x: i for i, x in enumerate([None] + list(materials))}
    self.runtime_rules = runtime_rules or RuntimeRules()
    self._chunks = collections.defaultdict(set)
    self._objects = [None]
    self._mat_map = np.zeros(self.area, np.uint8)
    self._obj_map = np.zeros(self.area, np.uint32)

  @classmethod
  def from_world(cls, world):
    """Build a :class:`SimpleWorld` from an environment world snapshot.

    Parameters
    ----------
    world : crafter.engine.World
        Source world. Only material-state information is copied.

    Returns
    -------
    SimpleWorld
        Planner-facing world snapshot.
    """
    simple_world = cls(world.area, constants.materials, world.runtime_rules)
    simple_world._mat_names = world._mat_names.copy()
    simple_world._mat_ids = world._mat_ids.copy()
    simple_world._mat_map = world._mat_map.copy()
    simple_world._obj_map = np.zeros(world.area, np.uint32)
    return simple_world

  def __copy__(self):
    """Return a planner-safe deep copy of the material-state world.

    Returns
    -------
    SimpleWorld
        Independent copy with duplicated arrays and maps.

    Notes
    -----
    The original TODO text is preserved below on purpose.

    TODO preserved from the original file:
    - TODO
    """
    new_world = type(self)(self.area, list(constants.materials), self.runtime_rules)
    new_world._mat_names = self._mat_names.copy()
    new_world._mat_ids = self._mat_ids.copy()
    new_world._mat_map = self._mat_map.copy()
    new_world._obj_map = self._obj_map.copy()
    new_world._objects = [None]
    new_world._chunks = collections.defaultdict(set)
    return new_world

  @property
  def objects(self):
    """Return currently tracked objects.

    Returns
    -------
    list
        Copy-like list view of non-null objects. Objects are not used by the
        current planner, but the accessor is retained for API compatibility.
    """
    return [obj for obj in self._objects if obj]

  def add(self, obj):
    assert hasattr(obj, 'pos')
    obj.pos = np.array(obj.pos)
    assert self._obj_map[tuple(obj.pos)] == 0
    index = len(self._objects)
    self._objects.append(obj)
    self._obj_map[tuple(obj.pos)] = index
    self._chunks[self.chunk_key(obj.pos)].add(obj)

  def remove(self, obj):
    if obj.removed:
      return
    self._objects[self._obj_map[tuple(obj.pos)]] = None
    self._obj_map[tuple(obj.pos)] = 0
    self._chunks[self.chunk_key(obj.pos)].remove(obj)
    obj.removed = True

  def move(self, obj, pos):
    if obj.removed:
      return
    pos = np.array(pos)
    assert self._obj_map[tuple(pos)] == 0
    index = self._obj_map[tuple(obj.pos)]
    self._obj_map[tuple(pos)] = index
    self._obj_map[tuple(obj.pos)] = 0
    old_chunk = self.chunk_key(obj.pos)
    new_chunk = self.chunk_key(pos)
    if old_chunk != new_chunk:
      self._chunks[old_chunk].remove(obj)
      self._chunks[new_chunk].add(obj)
    obj.pos = pos

  def __setitem__(self, pos, material):
    if material not in self._mat_ids:
      id_ = len(self._mat_ids)
      self._mat_ids[material] = id_
      self._mat_names[id_] = material
    self._mat_map[tuple(pos)] = self._mat_ids[material]

  def __getitem__(self, pos):
    if not _inside((0, 0), pos, self.area):
      return None, None
    material = self._mat_names[self._mat_map[tuple(pos)]]
    obj = self._objects[self._obj_map[tuple(pos)]]
    return material, obj

  def nearby(self, pos, distance):
    (x, y), d = pos, distance
    xmin = max(0, x - d)
    xmax = min(self.area[0], x + d + 1)
    ymin = max(0, y - d)
    ymax = min(self.area[1], y + d + 1)
    ids = set(self._mat_map[xmin:xmax, ymin:ymax].flatten().tolist())
    materials = tuple(self._mat_names[idx] for idx in ids)
    indices = self._obj_map[xmin:xmax, ymin:ymax].flatten().tolist()
    objs = {self._objects[i] for i in indices if i > 0}
    return materials, objs

  def mask(self, xmin, xmax, ymin, ymax, material):
    region = self._mat_map[xmin:xmax, ymin:ymax]
    return region == self._mat_ids[material]

  def count(self, material):
    return int((self._mat_map == self._mat_ids[material]).sum())

  def material_name_grid(self) -> np.ndarray:
    """Return the current material-name grid.

    Returns
    -------
    np.ndarray
        Array of shape ``(width, height)`` whose entries are material-name
        strings.
    """
    grid = np.empty(self.area, dtype=object)
    for pos in _positions_within(self.area):
      grid[pos] = self[pos][0]
    return grid

  def chunk_key(self, pos):
    (x, y), (csx, csy) = pos, self._chunk_size
    xmin, ymin = (x // csx) * csx, (y // csy) * csy
    xmax = min(xmin + csx, self.area[0])
    ymax = min(ymin + csy, self.area[1])
    return (xmin, xmax, ymin, ymax)


class Task:
  """High-level planner task definition.

  A task corresponds to one item in ``planner.task_order`` and expands into:

  - required inventory materials,
  - required spatial layouts,
  - ordered micro-actions that the planner will simulate.

  Parameters
  ----------
  name : str
      Task identifier, e.g. ``"collect_wood"``.
  material_required : dict[str, int]
      Total inventory requirements that must hold before the terminal
      micro-action can be executed.
  micro_action_tasks : list[dict]
      Ordered micro-actions emitted by this task.
  """

  _TASK_DEFINITIONS: Final[dict[str, dict]] = {
      'collect_wood': dict(kind='collect', target='wood', count=1),
      'place_table': dict(kind='place_cluster', name='table'),
      'make_wood_pickaxe': dict(kind='make', name='wood_pickaxe'),
      'make_wood_sword': dict(kind='make', name='wood_sword'),
      'collect_stone': dict(kind='collect', target='stone', count=1),
      'place_stone': dict(kind='place_single', name='stone'),
      'make_stone_pickaxe': dict(kind='make', name='stone_pickaxe'),
      'make_stone_sword': dict(kind='make', name='stone_sword'),
      'place_furnace': dict(kind='place_cluster', name='furnace'),
      'collect_coal': dict(kind='collect', target='coal', count=1),
      'collect_iron': dict(kind='collect', target='iron', count=1),
      'make_iron_pickaxe': dict(kind='make', name='iron_pickaxe'),
      'make_iron_sword': dict(kind='make', name='iron_sword'),
      'collect_diamond': dict(kind='collect', target='diamond', count=1),
  }

  def __init__(self, name: str, material_required: dict[str, int], micro_action_tasks: list):
    self.name = name
    self._materials_required: dict[str, int] = material_required
    self._micro_action_tasks: list = micro_action_tasks

  @classmethod
  def from_name(cls, name: str) -> 'Task':
    """Construct a task from the built-in task catalog.

    Parameters
    ----------
    name : str
        Name from ``planner.task_order``.

    Returns
    -------
    Task
        Normalized task object.
    """
    assert name in cls._TASK_DEFINITIONS, f'Unknown task name: {name}'
    definition = cls._TASK_DEFINITIONS[name]
    kind = definition['kind']
    if kind == 'collect':
      target = definition['target']
      count = definition['count']
      return cls(
          name=name,
          material_required={target: count},
          micro_action_tasks=[
              {'kind': 'collect', 'item': target, 'count': count},
          ],
      )
    if kind == 'place_cluster':
      place_name = definition['name']
      return cls(
          name=name,
          material_required=constants.place[place_name]['uses'].copy(),
          micro_action_tasks=[
              {'kind': 'ensure_layout', 'layout': 'craft_cluster_4x4'},
              {'kind': 'place_named', 'name': place_name, 'target': place_name},
          ],
      )
    if kind == 'place_single':
      place_name = definition['name']
      return cls(
          name=name,
          material_required=constants.place[place_name]['uses'].copy(),
          micro_action_tasks=[
              {'kind': 'place_1x1', 'name': place_name},
          ],
      )
    if kind == 'make':
      make_name = definition['name']
      return cls(
          name=name,
          material_required=constants.make[make_name]['uses'].copy(),
          micro_action_tasks=[
              {'kind': 'make', 'name': make_name},
          ],
      )
    raise AssertionError(f'Unsupported task definition: {definition}')

  def get_micro_tasks(self, world, inventory):
    """Expand a task against the current known world state.

    Parameters
    ----------
    world : KnownWorld
        Current known world used to count visible pre-materials and layouts.
    inventory : dict[str, int]
        Current inventory snapshot.

    Returns
    -------
    dict
        Dictionary with keys ``missing_pre_materials``,
        ``required_layouts``, and ``micro_action_tasks``.

    Notes
    -----
    The long TODO/spec text from the original file is intentionally preserved
    below because it still documents the intended semantics and future
    refinements.

    NOTE: for now, you don't need to worry about "REACHABILITY", since by our extension algorithm, we will never find a block that is non-reachable.

    e.g. You might get something like (take the second task [which Minecrafters call "achievement"] "place table" as an example)
    - We need, in total, 2 pieces of wood.
    - Since there's 1 piece of wood in our inventory, and there's 0 pieces of wood in REACHABLE and KNOWN world, we need 1 extra piece of wood.
    - Which means, ([xxx] stands for local places, will only be meaningful in this current task; {xxx} stands for global places, which will be meaningful globally)
      - the collect-micro-perception task is
        - find a tree that hasn't been found, mark it as [tree 0]
      - the place-micro-perception task is
          - find a nearest currently REACHABLE walkable 4x4 place, mark the specific locations as {crafting stand}, {furnace} and {crafting table} respectively.

          [[passable, passable, passable, passable],
           [passable, furnace, crafting stand, passable],
           [passable, passable, crafting table, passable],
           [passable, passable, passable, passable]]

    I will give an another example: for `place_stone` task, it will be like this,
    - We need, in total, 1 piece of stone.
    - Since there's 3 pieces of stone in our inventory, we need no more stone.
    - Which means,
      - there will be no collect-micro-perception tasks
      - the place-micro-perception task is
          - find a nearest currently REACHABLE walkable 1x1 place, mark the specific locations as {place stone}.

          [[place stone]]

      - the micro-action tasks are
        - navigate near and face {place stone}
        - action `place_stone`

    Note:
    - The place micro material task(s) and the micro actions tasks will be stored in hydra yaml.
    - The materials (in total) needed will be stored in hydra yaml.
    - The collect micro task(s) will be generated dynamically based on the current world and inventory state, and the materials (in total) needed.

    ----

    UPDATE (TODO): Maybe we should change the formulation. We will consider all known "pre-material" (i.e. tree -> woord; coal/iron/diamond ore -> coal/iron/diamond; that's why I call them "pre-")
    and inventory stuff as "a whole" thing (we can maintain "pre-material" dict/list/whatever just like the "inventory" one).

    And there won't be any specific "collect-micro-perception task". All we need to do, is to reveal grids until "pre-material" + "inventory" >= required.
    """
    missing_pre_materials = {}
    for item, required in self._materials_required.items():
      total_known = inventory.get(item, 0) + world.available_pre_material_count(item)
      if total_known < required:
        missing_pre_materials[item] = required - total_known
    required_layouts = []
    for micro_action in self._micro_action_tasks:
      if micro_action['kind'] == 'ensure_layout':
        required_layouts.append(LayoutRequirement('layout', micro_action['layout']))
      elif micro_action['kind'] in ('place_named', 'make'):
        required_layouts.append(LayoutRequirement('layout', 'craft_cluster_4x4'))
    return {
        'missing_pre_materials': missing_pre_materials,
        'required_layouts': required_layouts,
        'micro_action_tasks': list(self._micro_action_tasks),
    }


class KnownWorld:
  """Planner-visible world state that evolves during simulated execution.

  ``KnownWorld`` stores two worlds:

  ``_initial_world``
      Immutable reveal source. Unknown cells are revealed from this snapshot.
  ``_world``
      Evolving known world. Once a cell is known, subsequent simulated actions
      update this world rather than the immutable source.

  Parameters
  ----------
  initial_world : SimpleWorld
      Full world snapshot used as the reveal source.
  player_pos : array-like of int, shape (2,)
      Initial player position in grid coordinates.
  inventory : dict[str, int]
      Initial inventory used for passability checks and simulated actions.
  facing : tuple[int, int], default=(0, 1)
      Initial facing direction.

  Notes
  -----
  Reachability vocabulary used by this class:

  ``walkable``
      A tile that can be stood on without any prior ``do`` action.
  ``passable``
      A tile that can become standable immediately under the current
      inventory/tool state, possibly by collecting it first.
  ``on-reachable``
      Reachable through a path of currently known passable tiles from the
      current player position.
  """

  _CRAFT_CLUSTER_SHAPE: Final[tuple[int, int]] = (4, 4)
  _CRAFT_CLUSTER_OFFSETS: Final[dict[str, tuple[int, int]]] = {
      'furnace': (1, 1),
      'crafting_stand': (1, 2),
      'table': (2, 2),
  }
  _DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
      (-1, 0), (1, 0), (0, -1), (0, 1),
  )
  _ITEM_TO_SOURCES: Final[dict[str, tuple[str, ...]]] = {
      item: tuple(
          material
          for material, info in constants.collect.items()
          if info['receive'].get(item, 0) > 0
      )
      for item in constants.items
  }

  def __init__(
      self,
      initial_world: SimpleWorld,
      player_pos,
      inventory: dict[str, int],
      facing=(0, 1),
  ):
    self._initial_world: Final[SimpleWorld] = copy(initial_world)
    self._world: SimpleWorld = copy(self._initial_world)
    self._mask: np.ndarray = np.zeros(self._world.area, dtype=bool)
    self._perceived_mask: np.ndarray = np.zeros(self._world.area, dtype=bool)
    self._imputed_mask: np.ndarray = np.zeros(self._world.area, dtype=bool)
    self._imputed_materials: dict[tuple[int, int], str] = {}
    self.player_pos = np.array(player_pos, dtype=int)
    self.facing = tuple(facing)
    self.inventory = dict(inventory)
    self._craft_cluster_anchor: tuple[int, int] | None = None
    self._mask[tuple(self.player_pos)] = True
    self._perceived_mask[tuple(self.player_pos)] = True

  def copy(self):
    """Return a planner-state copy of the known world.

    Returns
    -------
    KnownWorld
        Independent copy of the evolving known-world state.
    """
    new_world = type(self)(
        initial_world=self._initial_world,
        player_pos=self.player_pos.copy(),
        inventory=self.inventory.copy(),
        facing=self.facing,
    )
    new_world._world = copy(self._world)
    new_world._mask = self._mask.copy()
    new_world._perceived_mask = self._perceived_mask.copy()
    new_world._imputed_mask = self._imputed_mask.copy()
    new_world._imputed_materials = dict(self._imputed_materials)
    new_world._craft_cluster_anchor = self._craft_cluster_anchor
    return new_world

  def is_known(self, pos) -> bool:
    """Check whether a cell has been revealed.

    Parameters
    ----------
    pos : array-like of int, shape (2,)
        Target cell.

    Returns
    -------
    bool
        ``True`` if the cell is currently known.
    """
    return bool(self._mask[tuple(pos)])

  @property
  def revealed_cell_count(self) -> int:
    """Return the number of cells perceived from the reveal source.

    Returns
    -------
    int
        Count of cells whose material came from an actual reveal, excluding
        hard-inferred cells.
    """
    return int(self._perceived_mask.sum())

  @property
  def known_mask(self) -> np.ndarray:
    """Return a copy of the current reveal mask.

    Returns
    -------
    np.ndarray
        Boolean mask whose ``True`` entries are exactly the currently revealed
        cells.
    """
    return self._mask.copy()

  @property
  def perceived_mask(self) -> np.ndarray:
    """Return cells whose material came from the immutable reveal source."""
    return self._perceived_mask.copy()

  @property
  def imputed_mask(self) -> np.ndarray:
    """Return cells whose material was filled by hard pattern inference."""
    return self._imputed_mask.copy()

  @property
  def initial_world(self) -> SimpleWorld:
    """Return the immutable reveal-source world snapshot.

    Returns
    -------
    SimpleWorld
        Initial world used as the source of future reveals.
    """
    return self._initial_world

  def material_at(self, pos) -> str:
    """Return the current material at a revealed cell."""
    assert self.is_known(pos), f'Attempted to access unknown material at {tuple(pos)}'
    return self._world[tuple(pos)][0]

  def initial_material_at(self, pos) -> str:
    """Return the immutable reveal-source material at a cell."""
    return self._initial_world[tuple(pos)][0]

  def imputed_material_at(self, pos) -> str:
    """Return the original material predicted for a hard-inferred cell."""
    return self._imputed_materials[tuple(pos)]

  def reveal_cell(self, pos):
    """Reveal one unknown frontier cell.

    Parameters
    ----------
    pos : array-like of int, shape (2,)
        Cell to reveal. The cell must be unknown, in bounds, adjacent to the
        current on-reachable region, and currently passable under planner
        semantics.
    """
    pos = tuple(pos)
    assert _inside((0, 0), pos, self._world.area), f'Out-of-bounds reveal: {pos}'
    assert not self.is_known(pos), f'Grid is already known: {pos}'
    frontier = set(self.frontier_unknowns())
    assert pos in frontier, f'Reveal target is not a legal frontier cell: {pos}'
    self._mask[pos] = True
    self._perceived_mask[pos] = True

  def impute_cell(self, pos, material: str):
    """Fill one unknown cell with a material predicted by hard inference."""
    pos = tuple(pos)
    assert _inside((0, 0), pos, self._world.area), f'Out-of-bounds imputation: {pos}'
    assert not self.is_known(pos), f'Grid is already known: {pos}'
    self._world[pos] = material
    self._mask[pos] = True
    self._imputed_mask[pos] = True
    self._imputed_materials[pos] = material

  def neighbors(self, pos):
    """Yield in-bounds 4-neighborhood cells around ``pos``."""
    x, y = tuple(pos)
    for dx, dy in self._DIRECTIONS:
      neighbor = (x + dx, y + dy)
      if _inside((0, 0), neighbor, self._world.area):
        yield neighbor

  def is_walkable_material(self, material: str | None) -> bool:
    """Return whether a material is directly standable."""
    return material in constants.walkable

  def _collect_info(self, material: str):
    return constants.collect.get(material)

  def _is_collectible_passable(self, material: str) -> bool:
    info = self._collect_info(material)
    if not info:
      return False
    if not self.is_walkable_material(info['leaves']):
      return False
    return all(self.inventory.get(item, 0) >= amount for item, amount in info['require'].items())

  def is_passable_material(self, material: str | None) -> bool:
    """Return whether a material is currently passable under the inventory."""
    if material is None:
      return False
    return self.is_walkable_material(material) or self._is_collectible_passable(material)

  def reachable_mask(self) -> np.ndarray:
    """Compute the current on-reachable mask over known cells.

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as the world, marking the connected
        component of currently known passable tiles that contains the player.
    """
    mask = np.zeros(self._world.area, dtype=bool)
    start = tuple(self.player_pos)
    assert self.is_known(start), f'Player position must always be known: {start}'
    assert self.is_passable_material(self.material_at(start)), (
        f'Player must stand on a passable known grid: {start}, {self.material_at(start)}')
    queue = collections.deque([start])
    mask[start] = True
    while queue:
      current = queue.popleft()
      for neighbor in self.neighbors(current):
        if mask[neighbor]:
          continue
        if not self.is_known(neighbor):
          continue
        if not self.is_passable_material(self.material_at(neighbor)):
          continue
        mask[neighbor] = True
        queue.append(neighbor)
    return mask

  def frontier_unknowns(self):
    """Return legal unknown cells that may be revealed next.

    Returns
    -------
    list[tuple[int, int]]
        Unknown cells adjacent to the current reachable region whose reveal
        source material is currently passable.
    """
    reachable = self.reachable_mask()
    frontier = []
    for pos in _positions_within(self._world.area):
      if self.is_known(pos):
        continue
      if not any(reachable[neighbor] for neighbor in self.neighbors(pos)):
        continue
      if not self.is_passable_material(self.initial_material_at(pos)):
        continue
      frontier.append(pos)
    return frontier

  def available_pre_material_positions(self, item: str):
    """Return known pre-material cells that can currently produce ``item``."""
    positions = []
    sources = self._ITEM_TO_SOURCES[item]
    reachable = self.reachable_mask()
    for pos in _positions_within(self._world.area):
      if not self.is_known(pos):
        continue
      material = self.material_at(pos)
      if material not in sources:
        continue
      if not self.is_passable_material(material):
        continue
      if not any(reachable[neighbor] for neighbor in self.neighbors(pos)):
        continue
      positions.append(pos)
    return positions

  def available_pre_material_count(self, item: str) -> int:
    """Count known reachable pre-material cells for ``item``."""
    return len(self.available_pre_material_positions(item))

  def _shortest_path_distances(self):
    reachable = self.reachable_mask()
    distances = {tuple(self.player_pos): 0}
    queue = collections.deque([tuple(self.player_pos)])
    while queue:
      current = queue.popleft()
      for neighbor in self.neighbors(current):
        if not reachable[neighbor]:
          continue
        if neighbor in distances:
          continue
        distances[neighbor] = distances[current] + 1
        queue.append(neighbor)
    return distances

  def _nearest(self, positions):
    positions = list(positions)
    assert positions, 'Expected a non-empty candidate set'
    distances = self._shortest_path_distances()
    scored = []
    for pos in positions:
      if pos not in distances:
        continue
      scored.append((distances[pos], pos))
    assert scored, 'No candidate position is reachable'
    scored.sort()
    return scored[0][1]

  def find_collect_target(self, item: str):
    """Choose the nearest known collect target for ``item``."""
    targets = self.available_pre_material_positions(item)
    assert targets, f'No known target found for item: {item}'
    distances = self._shortest_path_distances()
    best = []
    for target in targets:
      adjacent = [neighbor for neighbor in self.neighbors(target) if neighbor in distances]
      assert adjacent, f'Collect target must have a reachable adjacent grid: {target}'
      best.append((min(distances[neighbor] for neighbor in adjacent), target))
    best.sort()
    return best[0][1]

  def find_adjacent_reachable(self, pos):
    """Return the nearest reachable neighbor of ``pos``."""
    distances = self._shortest_path_distances()
    candidates = [neighbor for neighbor in self.neighbors(pos) if neighbor in distances]
    assert candidates, f'No reachable adjacent position for {pos}'
    return min(candidates, key=lambda x: distances[x])

  def find_placeable_1x1(self, place_name: str):
    """Return the nearest known 1x1 placement target for ``place_name``."""
    where = set(constants.place[place_name]['where'])
    candidates = []
    for pos in _positions_within(self._world.area):
      if not self.is_known(pos):
        continue
      material = self.material_at(pos)
      if material not in where:
        continue
      if not self.is_walkable_material(material):
        continue
      if self._is_reserved_craft_cluster_cell(pos):
        continue
      try:
        self.find_adjacent_reachable(pos)
      except AssertionError:
        continue
      candidates.append(pos)
    assert candidates, f'No known placeable 1x1 position found for {place_name}'
    return self._nearest(candidates)

  def _craft_anchor_cells(self, anchor):
    ax, ay = anchor
    for dx in range(self._CRAFT_CLUSTER_SHAPE[0]):
      for dy in range(self._CRAFT_CLUSTER_SHAPE[1]):
        yield (ax + dx, ay + dy)

  def _is_reserved_craft_cluster_cell(self, pos) -> bool:
    if self._craft_cluster_anchor is None:
      return False
    return tuple(pos) in set(self._craft_anchor_cells(self._craft_cluster_anchor))

  def _craft_anchor_compatible(self, anchor, fully_known: bool):
    for pos in self._craft_anchor_cells(anchor):
      if not _inside((0, 0), pos, self._world.area):
        return False
      if fully_known and not self.is_known(pos):
        return False
      if not self.is_known(pos):
        continue
      material = self.material_at(pos)
      local = (pos[0] - anchor[0], pos[1] - anchor[1])
      if local == self._CRAFT_CLUSTER_OFFSETS['table']:
        if material not in ('table', 'grass', 'sand', 'path'):
          return False
      elif local == self._CRAFT_CLUSTER_OFFSETS['furnace']:
        if material not in ('furnace', 'grass', 'sand', 'path'):
          return False
      else:
        if not self.is_walkable_material(material):
          return False
    return True

  def craft_cluster_candidates(self, fully_known: bool):
    """Enumerate feasible 4x4 craft-cluster anchors."""
    if self._craft_cluster_anchor is not None:
      anchor = self._craft_cluster_anchor
      assert self._craft_anchor_compatible(anchor, fully_known), (
          f'Reserved craft cluster is no longer compatible: {anchor}')
      return [anchor]
    candidates = []
    max_x = self._world.area[0] - self._CRAFT_CLUSTER_SHAPE[0] + 1
    max_y = self._world.area[1] - self._CRAFT_CLUSTER_SHAPE[1] + 1
    for ax in range(max_x):
      for ay in range(max_y):
        anchor = (ax, ay)
        if self._craft_anchor_compatible(anchor, fully_known):
          candidates.append(anchor)
    return candidates

  def craft_cluster_frontier_scores(self):
    """Score frontier cells by how many feasible 4x4 anchors they preserve."""
    anchors = self.craft_cluster_candidates(fully_known=False)
    frontier = set(self.frontier_unknowns())
    scores = collections.Counter()
    for anchor in anchors:
      for pos in self._craft_anchor_cells(anchor):
        if pos in frontier:
          scores[pos] += 1
    return scores

  def find_or_reserve_craft_cluster(self):
    """Choose and memoize the craft-cluster anchor.

    Returns
    -------
    tuple[int, int]
        Anchor of the selected 4x4 craft cluster.
    """
    if self._craft_cluster_anchor is not None:
      return self._craft_cluster_anchor
    candidates = self.craft_cluster_candidates(fully_known=True)
    assert candidates, 'No fully known craft cluster candidate exists'
    scoring_positions = []
    for anchor in candidates:
      stand_pos = (
          anchor[0] + self._CRAFT_CLUSTER_OFFSETS['crafting_stand'][0],
          anchor[1] + self._CRAFT_CLUSTER_OFFSETS['crafting_stand'][1],
      )
      scoring_positions.append((self._nearest([stand_pos]), anchor))
    scored = []
    for stand_pos, anchor in scoring_positions:
      dist = abs(stand_pos[0] - self.player_pos[0]) + abs(stand_pos[1] - self.player_pos[1])
      scored.append((dist, anchor))
    scored.sort()
    self._craft_cluster_anchor = scored[0][1]
    return self._craft_cluster_anchor

  def named_position(self, name: str):
    """Return the absolute grid position of a named craft-cluster slot."""
    assert self._craft_cluster_anchor is not None, 'Craft cluster anchor is not reserved'
    assert name in self._CRAFT_CLUSTER_OFFSETS, f'Unknown craft cluster name: {name}'
    offset = self._CRAFT_CLUSTER_OFFSETS[name]
    return np.array((
        self._craft_cluster_anchor[0] + offset[0],
        self._craft_cluster_anchor[1] + offset[1],
    ))

  def apply_move_to(self, pos, facing):
    """Apply a successful move to the known-world player state."""
    pos = tuple(pos)
    facing = tuple(facing)
    assert self.is_known(pos), f'Cannot move to unknown position: {pos}'
    assert self.is_walkable_material(self.material_at(pos)), (
        f'Player can only stand on walkable tiles, got {self.material_at(pos)} at {pos}')
    self.player_pos = np.array(pos, dtype=int)
    self.facing = facing

  def apply_collect(self, pos):
    """Apply a successful ``do`` collect action at ``pos``."""
    pos = tuple(pos)
    material = self.material_at(pos)
    info = self._collect_info(material)
    assert info is not None, f'Material is not collectible: {material}'
    assert all(self.inventory.get(item, 0) >= amount for item, amount in info['require'].items()), (
        f'Collect requirements not satisfied for {material}: {info["require"]}')
    self._world[pos] = info['leaves']
    for item, amount in info['receive'].items():
      self.inventory[item] = self.inventory.get(item, 0) + amount

  def apply_place(self, name: str, pos):
    """Apply a successful ``place_*`` action at ``pos``."""
    pos = tuple(pos)
    material = self.material_at(pos)
    info = constants.place[name]
    if name not in ('table', 'furnace'):
      assert not self._is_reserved_craft_cluster_cell(pos), (
          f'Cannot place {name} inside the reserved craft cluster: {pos}')
    assert material in info['where'], f'Cannot place {name} on {material} at {pos}'
    assert all(self.inventory.get(item, 0) >= amount for item, amount in info['uses'].items()), (
        f'Not enough inventory to place {name}: need {info["uses"]}')
    assert info['type'] == 'material', f'Unsupported place type in planner: {info["type"]}'
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self._world[pos] = name

  def apply_make(self, name: str):
    """Apply a successful ``make_*`` action at the current player position."""
    info = constants.make[name]
    nearby, _ = self._world.nearby(tuple(self.player_pos), 1)
    assert all(util in nearby for util in info['nearby']), (
        f'Not near enough utilities to make {name}: need {info["nearby"]}, got {nearby}')
    assert all(self.inventory.get(item, 0) >= amount for item, amount in info['uses'].items()), (
        f'Not enough inventory to make {name}: need {info["uses"]}')
    for item, amount in info['uses'].items():
      self.inventory[item] -= amount
    self.inventory[name] = self.inventory.get(name, 0) + info['gives']

  def current_make_spot(self):
    """Return the designated craft-cluster standing position for making tools."""
    return self.named_position('crafting_stand')

  def count_known_material(self, material: str) -> int:
    """Count revealed cells whose current material matches ``material``."""
    return sum(
        1
        for pos in _positions_within(self._world.area)
        if self.is_known(pos) and self.material_at(pos) == material
    )

  def is_named_position_filled(self, name: str) -> bool:
    """Return whether a named craft-cluster slot already contains its material."""
    pos = tuple(self.named_position(name))
    return self.is_known(pos) and self.material_at(pos) == name

  def layout_reveal_scores(self, layout_name: str):
    """Return reveal scores for a layout-style reveal requirement."""
    if layout_name == 'craft_cluster_4x4':
      return self.craft_cluster_frontier_scores()
    raise AssertionError(f'Unknown layout reveal request: {layout_name}')

  def placeable_frontier_candidates_1x1(self, place_name: str):
    """Return frontier cells that could serve a future 1x1 placement."""
    frontier = self.frontier_unknowns()
    where = set(constants.place[place_name]['where'])
    return [
        pos for pos in frontier
        if (
            self.initial_material_at(pos) in where
            and self.is_walkable_material(self.initial_material_at(pos))
            and not self._is_reserved_craft_cluster_cell(pos)
        )
    ]

  # TODO preserved from the original file:
  # The known world should be initialized with the start known and only the start known. All others are unknown.
  # It should work well with TaskAndMotionPlanner. You should think of which function this known_world should provide.
  # Make sure this is concisely implemented.
