"""Store the known world state for the agent."""

from typing import Final

import numpy as np
from copy import copy


class SimpleWorld:
  """A simple world where there's no mobs, hunger, thirst, energy, day-light cycle and etc.
  The world is almost static and only contains static blocks.
  """

  # TODO: Implement this "SimpleWorld" as the docstring.
  
  def __copy__(self):
    # TODO

  def __init__(self, area, materials, runtime_rules=None):
    self.area = area
    self._mat_names = {i: x for i, x in enumerate([None] + materials)}
    self._mat_ids = {x: i for i, x in enumerate([None] + materials)}
    self.runtime_rules = runtime_rules or RuntimeRules()

  @property
  def objects(self):
    # Return a new list so the objects cannot change while being iterated over.
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
    self._mat_map[tuple(pos)] = self._mat_ids[material]

  def __getitem__(self, pos):
    if not _inside((0, 0), pos, self.area):
      return None, None
    material = self._mat_names[self._mat_map[tuple(pos)]]
    obj = self._objects[self._obj_map[tuple(pos)]]
    return material, obj

  def nearby(self, pos, distance):
    (x, y), d = pos, distance
    ids = set(self._mat_map[x - d : x + d + 1, y - d : y + d + 1].flatten().tolist())
    materials = tuple(self._mat_names[x] for x in ids)
    indices = self._obj_map[x - d : x + d + 1, y - d : y + d + 1].flatten().tolist()
    objs = {self._objects[i] for i in indices if i > 0}
    return materials, objs

  def mask(self, xmin, xmax, ymin, ymax, material):
    region = self._mat_map[xmin:xmax, ymin:ymax]
    return region == self._mat_ids[material]

  def count(self, material):
    return (self._mat_map == self._mat_ids[material]).sum()

  def chunk_key(self, pos):
    (x, y), (csx, csy) = pos, self._chunk_size
    xmin, ymin = (x // csx) * csx, (y // csy) * csy
    xmax = min(xmin + csx, self.area[0])
    ymax = min(ymin + csy, self.area[1])
    return (xmin, xmax, ymin, ymax)

class Task:
  def __init__(self, material_required: dict[str, int], micro_action_tasks: list):
    self._materials_required: dict[str, int] = material_required # Materials and how many do we need them each
    self._micro_action_tasks: list = micro_action_tasks # e.g. "navigate to and face tree 0; `do` the tree"
    
  def get_micro_tasks(self, world, inventory):
    """Get collect/place-perception micro tasks and action micro tasks.
    
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
  
  
    
    

class KnownWorld:
  """Store the known world state for the agent."""

  def __init__(self, initial_world: SimpleWorld):
    self._initial_world: Final[SimpleWorld] = copy(initial_world)
    self._world: SimpleWorld = copy(self._initial_world)
    self._mask: np.ndarray = np.zeros(self._world.area, dtype=bool)
    
    
