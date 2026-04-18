"""Task-and-motion planning for the simplified Crafter setup.

The planner in this module works over :class:`crafter.known_world.KnownWorld`
rather than the live environment. It repeatedly:

1. reveals new cells when required materials or layouts are still unknown,
2. simulates collect/place/make actions on the known-world state,
3. returns the resulting low-level Crafter action names.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np
import torch

from crafter import constants
from crafter.known_world import KnownWorld, Task
from crafter.skills.crafter_patterns import CrafterSkillLibrary
from crafter.utils.crafter_codec import CrafterTileCodec


@dataclass
class TaskBoundaryTrace:
  """Planner trace entry for one task boundary.

  Parameters
  ----------
  task_name : str
      Name of the high-level task.
  pre_action_index : int
      Number of environment actions planned before the task's execution phase.
      This is the action index immediately after ``_reveal_until_ready``.
  post_action_index : int
      Number of environment actions planned after the task finishes.
  pre_action_mask : np.ndarray
      Known-world mask after the reveal phase of this task.
  post_action_mask : np.ndarray
      Known-world mask after the full task finishes.
  """

  task_name: str
  pre_action_index: int
  post_action_index: int
  pre_action_mask: np.ndarray
  post_action_mask: np.ndarray


@dataclass
class PlanTrace:
  """Structured planner output used by ``run_gui``.

  Parameters
  ----------
  actions : list[str]
      Planned Crafter action names.
  result_metrics : dict[str, int]
      Planner-side metrics such as reveal counts.
  initial_known_mask : np.ndarray
      Reveal mask before planning the first task.
  task_boundaries : list[TaskBoundaryTrace]
      Per-task trace entries used to align environment execution with planner
      masks.
  """

  actions: list[str]
  result_metrics: dict[str, int]
  initial_known_mask: np.ndarray
  task_boundaries: list[TaskBoundaryTrace]


class TaskAndMotionPlanner:
  """High-level task planner plus low-level grid motion planner.

  Parameters
  ----------
  task_list : sequence[Task | str], optional
      Ordered task list. String entries are resolved through
      :meth:`crafter.known_world.Task.from_name`.
  random : numpy.random.RandomState, optional
      Random generator used for tie-breaking in reveal decisions.
  """

  _DIR_TO_ACTION = {
      (-1, 0): 'move_left',
      (1, 0): 'move_right',
      (0, -1): 'move_up',
      (0, 1): 'move_down',
  }
  _CLOCKWISE = {
      (-1, 0): (0, -1),
      (0, -1): (1, 0),
      (1, 0): (0, 1),
      (0, 1): (-1, 0),
  }
  _COUNTERCLOCKWISE = {value: key for key, value in _CLOCKWISE.items()}

  def __init__(
      self,
      task_list=None,
      random=None,
      inference_library: CrafterSkillLibrary | None = None,
      skill_learning_cfg=None,
  ):
    """Construct the planner.

    Parameters
    ----------
    task_list : sequence[Task | str], optional
        Ordered task sequence to execute.
    random : numpy.random.RandomState, optional
        Random generator used for tie-breaking in reveal selection.

    Notes
    -----
    The original TODO text is preserved below on purpose.

    TODO preserved from the original file:
    - Add what we need. For example, it could be `task_list: list[Task]`, etc
    """
    task_list = task_list or []
    self.task_list = [
        task if isinstance(task, Task) else Task.from_name(task)
        for task in task_list
    ]
    self.random = random or np.random.RandomState()
    self._known_world: KnownWorld | None = None
    self.inference_library = inference_library
    self.skill_learning_cfg = skill_learning_cfg
    self.codec = CrafterTileCodec.for_worldgen_materials()

  def plan(self, known_world: KnownWorld) -> list[str]:
    """Plan and simulate the whole task sequence.

    Parameters
    ----------
    known_world : KnownWorld
        Evolving planner state. The planner mutates this object in-place while
        simulating reveals and successful actions.

    Returns
    -------
    list[str]
        Crafter action names such as ``"move_left"``, ``"do"``, and
        ``"place_table"``.

    Notes
    -----
    The original TODO text is preserved below on purpose.

    TODO preserved from the original file:
    - You have to finish tasks in the task_list one by one.
    - For micro-perception tasks, you need to reveal the unknown grids in the known_world, until you find the target object or a place that satisfies the requirements.
    - For micro-action tasks, it's simpler: you just need to finish them one by one.
    - UPDATE: see `known_world.py` - micro-perception tasks will not be considered as "hard" tasks that must be finished in order.
      Instead, we will just need to reveal grids until "pre-material" + "inventory" >= required.
    """
    return self.plan_with_trace(known_world).actions

  def plan_with_trace(self, known_world: KnownWorld) -> PlanTrace:
    """Plan the whole task sequence and return a structured trace.

    Parameters
    ----------
    known_world : KnownWorld
        Evolving planner state. The planner mutates this object in-place while
        simulating reveals and successful actions.

    Returns
    -------
    PlanTrace
        Planned action sequence plus per-task reveal-mask snapshots.
    """
    self._known_world = known_world
    actions: list[str] = []
    task_boundaries: list[TaskBoundaryTrace] = []
    initial_known_mask = self._known_world.known_mask
    for task in self.task_list:
      micro_tasks = task.get_micro_tasks(self._known_world, self._known_world.inventory)
      actions.extend(self._reveal_until_ready(task, micro_tasks))
      pre_action_index = len(actions)
      pre_action_mask = self._known_world.known_mask
      actions.extend(self._ensure_material_inventory(task))
      for micro_action in micro_tasks['micro_action_tasks']:
        actions.extend(self._execute_micro_action(micro_action))
      task_boundaries.append(TaskBoundaryTrace(
          task_name=task.name,
          pre_action_index=pre_action_index,
          post_action_index=len(actions),
          pre_action_mask=pre_action_mask,
          post_action_mask=self._known_world.known_mask,
      ))
    return PlanTrace(
        actions=actions,
        result_metrics={'reveal_count': self._known_world.revealed_cell_count},
        initial_known_mask=initial_known_mask,
        task_boundaries=task_boundaries,
    )

  def reveal_next_cell_for_requirement(self, reveal_requirement):
    """Choose and reveal the next frontier cell for a requirement.

    This method is intentionally named after what it actually does: it does
    *not* reveal an arbitrary caller-specified cell. Instead, it selects the
    next cell according to the planner's reveal policy for a material/layout
    requirement and then reveals that cell in the bound known world.

    Parameters
    ----------
    reveal_requirement : dict
        Requirement descriptor. Supported forms currently include collect-style
        requirements, layout-style requirements, and 1x1 placement-style
        requirements.

    Returns
    -------
    tuple[int, int]
        The revealed cell.

    Notes
    -----
    The original TODO text is preserved below on purpose.

    TODO preserved from the original file:
    - You need to reveal the unknown grids in the known_world, until you find the target object or a place that satisfies the requirements.
    - Where to reveal? We adopt a Bayesian approach:
    - We need to reveal a place where things will mostly satisfy the micro perception task's requirements.
      - btw, the revealed place need to be reachable from current location (assuming unknown grids are impassable).
        This method will have an argument about this. For now, it must be asserted to be true (i.e. must be reachable).
        - Although, for now, since passable grids on the border of the known area will never become impassable,
          if you always reveal an adjacent grid of a certain passable grid on the border,
          you can prove that: you will always be able to reach any grid of the known area (quiz: how to prove it? "Reachable" means you can get near it, not on it).
          In this sense, you don't have to ACTUALLY check reachability. You just need to always reveal an adjacent grid of a passable grid on the border.
    - We will consider obtaining a tensor showing the probability distribution of the attribute of each grid.
      - For now, we adopt the vanilla way: known grids have 1.0 prob of its known attribute, where unknown grids will have a uniform distribution of all attributes,
        though it's subject to change (we will add a pattern-based inference engine later on, and the vanilla will serve as a baseline)
        - If you are doing 1x1 reveal, just reveal a most likely adjacent grid (for now, since it's all uniform, you will just have to randomly select one).
          - For example, if you are still short of coal and iron, you will find an adjacent grid that is most probable to be either coal or iron;
            if there are more than one grids that are equally most likely, you will randomly choose one among them.
        - If you are doing n by n reveal (e.g. 4x4), you are still trying to find a set of most likely grids. e.g. For "place table" task, you need to find a 4x4 set of grids that are jointly
          most likely to all be walkable (i.e. grass or path). But unlike 1x1, you will reveal only one grid and the probs will change. So TODO: you might have to carefully consider this. Like,
          how to formulate this problem in this way (if this way is possible) and select the next one grid to see? And can you mix 4x4 with 1x1 (you can mix 1x1 together; like "find coal" and "find wood"
          can be mixed into "find coal or wood". But can it be the case for 4x4?), if not, is it the right way to order these two as well,
          e.g. first 1x1 (if any), then 2x2 (if any), then 3x3 (if any), then 4x4 (if any)
    """
    known_world = self._require_known_world()
    self._maybe_apply_hard_inference(reveal_requirement)
    kind = reveal_requirement['kind']
    if kind == 'collect':
      candidates = known_world.frontier_unknowns()
      assert candidates, f'No legal frontier cells exist for collect reveal: {reveal_requirement}'
      target = self._pick_collect_reveal_target(candidates, reveal_requirement)
      known_world.reveal_cell(target)
      return target
    if kind == 'layout':
      scores = known_world.layout_reveal_scores(reveal_requirement['name'])
      assert scores, f'No legal frontier cells exist for layout reveal: {reveal_requirement}'
      target = self._pick_layout_reveal_target(scores)
      known_world.reveal_cell(target)
      return target
    if kind == 'place_1x1':
      candidates = known_world.placeable_frontier_candidates_1x1(reveal_requirement['name'])
      assert candidates, f'No legal frontier cells exist for 1x1 placement reveal: {reveal_requirement}'
      target = self._pick_place_1x1_reveal_target(candidates, reveal_requirement['name'])
      known_world.reveal_cell(target)
      return target
    raise AssertionError(f'Unknown reveal requirement: {reveal_requirement}')

  def plan_path_to_any(self, goal_positions):
    """Plan a shortest path to any one of several goal positions.

    Parameters
    ----------
    goal_positions : iterable[array-like]
        Candidate destination cells. They are interpreted over the current
        known passable graph.

    Returns
    -------
    list[tuple[int, int]]
        Shortest path as a sequence of positions, including the start cell and
        the reached goal cell.

    Notes
    -----
    The original TODO text is preserved below on purpose.

    TODO preserved from the original file:
    - you can change the name of this function if needed. This is to plan the actual motion of micro-action tasks,
      it shouldn't be too hard.
    """
    known_world = self._require_known_world()
    goal_positions = {tuple(pos) for pos in goal_positions}
    reachable = known_world.reachable_mask()
    start = tuple(known_world.player_pos)
    assert start in goal_positions or reachable[start], f'Invalid motion-planning start state: {start}'
    parents = {start: None}
    queue = collections.deque([start])
    found = None
    while queue:
      current = queue.popleft()
      if current in goal_positions:
        found = current
        break
      for neighbor in known_world.neighbors(current):
        if not reachable[neighbor]:
          continue
        if neighbor in parents:
          continue
        parents[neighbor] = current
        queue.append(neighbor)
    assert found is not None, f'No motion path found to any of: {sorted(goal_positions)}'
    path = [found]
    while parents[path[-1]] is not None:
      path.append(parents[path[-1]])
    path.reverse()
    return path

  def _reveal_until_ready(self, task: Task, micro_tasks: dict):
    known_world = self._require_known_world()
    actions = []
    while True:
      missing_pre_materials = {}
      for item, required in task._materials_required.items():
        total_known = known_world.inventory.get(item, 0) + known_world.available_pre_material_count(item)
        if total_known < required:
          missing_pre_materials[item] = required - total_known

      layout_missing = False
      for layout_requirement in micro_tasks['required_layouts']:
        if layout_requirement.name == 'craft_cluster_4x4':
          if not known_world.craft_cluster_candidates(fully_known=True):
            layout_missing = True
            break

      if not missing_pre_materials and not layout_missing:
        return actions

      if missing_pre_materials:
        self.reveal_next_cell_for_requirement({'kind': 'collect', 'items': tuple(sorted(missing_pre_materials))})
      else:
        self.reveal_next_cell_for_requirement({'kind': 'layout', 'name': 'craft_cluster_4x4'})

  def _ensure_material_inventory(self, task: Task):
    actions = []
    for item, required in task._materials_required.items():
      actions.extend(self._collect_until(item, required))
    return actions

  def _collect_until(self, item: str, required: int):
    known_world = self._require_known_world()
    actions = []
    while known_world.inventory.get(item, 0) < required:
      positions = known_world.available_pre_material_positions(item)
      
      while not positions:
        raise AssertionError(f'Since _ensure_material_inventory is only called after _reveal_until_ready, there should be some known pre-material positions for {item}, but got none')
        self.reveal_next_cell_for_requirement({'kind': 'collect', 'items': (item,)})
        positions = known_world.available_pre_material_positions(item)

      target = known_world.find_collect_target(item)
      actions.extend(self._navigate_adjacent_and_face(target))
      actions.append('do')
      known_world.apply_collect(target)
    return actions

  def _execute_micro_action(self, micro_action: dict):
    known_world = self._require_known_world()
    kind = micro_action['kind']
    if kind == 'collect':
      return self._collect_until(micro_action['item'], micro_action['count'])
    if kind == 'ensure_layout':
      while not known_world.craft_cluster_candidates(fully_known=True):
        self.reveal_next_cell_for_requirement({'kind': 'layout', 'name': micro_action['layout']})
      known_world.find_or_reserve_craft_cluster()
      return []
    if kind == 'place_named':
      known_world.find_or_reserve_craft_cluster()
      target = tuple(known_world.named_position(micro_action['target']))
      actions = self._navigate_adjacent_and_face(target)
      actions.append(f'place_{micro_action["name"]}')
      known_world.apply_place(micro_action['name'], target)
      return actions
    if kind == 'place_1x1':
      place_name = micro_action['name']
      while True:
        try:
          target = known_world.find_placeable_1x1(place_name)
          break
        except AssertionError:
          self.reveal_next_cell_for_requirement({'kind': 'place_1x1', 'name': place_name})
      actions = self._navigate_adjacent_and_face(target)
      actions.append(f'place_{place_name}')
      known_world.apply_place(place_name, target)
      return actions
    if kind == 'make':
      known_world.find_or_reserve_craft_cluster()
      make_spot = tuple(known_world.current_make_spot())
      actions = self._navigate_to_exact(make_spot)
      actions.append(f'make_{micro_action["name"]}')
      known_world.apply_make(micro_action['name'])
      return actions
    raise AssertionError(f'Unknown micro action: {micro_action}')

  def _navigate_to_exact(self, goal_pos):
    """Move the player to one exact goal position."""
    path = self.plan_path_to_any([goal_pos])
    return self._execute_position_path(path)

  def _navigate_adjacent_and_face(self, target_pos) -> list[str]:
    """Move adjacent to a target cell and rotate to face it."""
    known_world = self._require_known_world()
    adjacent = [neighbor for neighbor in known_world.neighbors(target_pos)]
    path = self.plan_path_to_any(adjacent)
    actions: list[str] = self._execute_position_path(path)
    direction = self._delta(tuple(known_world.player_pos), tuple(target_pos))
    actions.extend(self._rotate_to(direction))
    return actions

  def _execute_position_path(self, path) -> list[str]:
    """Convert a position path into Crafter actions and simulate them."""
    known_world = self._require_known_world()
    actions: list[str] = []
    for current, next_pos in zip(path, path[1:]):
      direction = self._delta(current, next_pos)
      material = known_world.material_at(next_pos)
      if not known_world.is_walkable_material(material):
        assert known_world.is_passable_material(material), (
            f'Path stepped through a non-passable material: {material} at {next_pos}')
        actions.extend(self._rotate_to(direction))
        actions.append('do')
        known_world.apply_collect(next_pos)
      move_action: str = self._DIR_TO_ACTION[direction]
      actions.append(move_action)
      known_world.apply_move_to(next_pos, direction)
    return actions

  def _rotate_to(self, target_direction):
    """Rotate the player to ``target_direction`` using discrete turn actions."""
    known_world = self._require_known_world()
    target_direction = tuple(target_direction)
    assert target_direction in self._DIR_TO_ACTION, f'Invalid direction: {target_direction}'
    actions: list[str] = []
    while tuple(known_world.facing) != target_direction:
      current = tuple(known_world.facing)
      clockwise = self._CLOCKWISE[current]
      counterclockwise = self._COUNTERCLOCKWISE[current]
      if clockwise == target_direction:
        actions.append('rotate_clockwise')
        known_world.facing = clockwise
      elif counterclockwise == target_direction:
        actions.append('rotate_counterclockwise')
        known_world.facing = counterclockwise
      else:
        actions.append('rotate_clockwise')
        known_world.facing = clockwise
    return actions

  def _pick_uniform_best(self, candidates):
    """Sample uniformly among equally good reveal candidates."""
    assert candidates, 'Expected at least one reveal candidate'
    index = self.random.randint(0, len(candidates))
    return tuple(candidates[index])

  def _known_world_partial_ids(self) -> torch.Tensor:
    known_world = self._require_known_world()
    return self.codec.encode_known_world_initial_partial(known_world, device='cpu')

  def _candidate_mask(self, candidates) -> torch.Tensor:
    known_world = self._require_known_world()
    mask = torch.zeros(known_world.known_mask.shape, dtype=torch.bool)
    for pos in candidates:
      mask[pos] = True
    return mask

  def _maybe_apply_hard_inference(self, reveal_requirement):
    del reveal_requirement
    if self.inference_library is None or self.skill_learning_cfg is None:
      return
    if not bool(self.skill_learning_cfg.enable_hard_inference):
      return
    threshold = float(self.skill_learning_cfg.infer_prob_threshold)
    if threshold >= 1.0:
      return
    partial_ids = self._known_world_partial_ids()
    candidate_mask = self._candidate_mask(self._require_known_world().frontier_unknowns())
    updated, inferred_positions = self.inference_library.maybe_hard_infer(
        partial_map_ids=partial_ids,
        eps=float(self.skill_learning_cfg.eps),
        threshold=threshold,
        candidate_mask=candidate_mask,
    )
    del updated
    for pos in inferred_positions:
      if not self._require_known_world().is_known(pos):
        self._require_known_world().reveal_cell(pos)

  def _pick_collect_reveal_target(self, candidates, reveal_requirement):
    if self.inference_library is None or self.skill_learning_cfg is None:
      return self._pick_uniform_best(candidates)
    known_world = self._require_known_world()
    target_materials = []
    for item in reveal_requirement['items']:
      target_materials.extend(known_world._ITEM_TO_SOURCES[item])
    partial_ids = self._known_world_partial_ids()
    target_class_ids = [
        self.codec.encode_material(material)
        for material in sorted(set(target_materials))
    ]
    scores = self.inference_library.score_candidate_cells(
        partial_map_ids=partial_ids,
        candidate_mask=self._candidate_mask(candidates),
        class_ids=target_class_ids,
        eps=float(self.skill_learning_cfg.eps),
    )
    return self._pick_best_scored_cell(candidates, scores)

  def _pick_place_1x1_reveal_target(self, candidates, place_name: str):
    if self.inference_library is None or self.skill_learning_cfg is None:
      return self._pick_uniform_best(candidates)
    target_class_ids = [
        self.codec.encode_material(material)
        for material in set(constants.place[place_name]['where'])
        if material in self.codec.materials
    ]
    scores = self.inference_library.score_candidate_cells(
        partial_map_ids=self._known_world_partial_ids(),
        candidate_mask=self._candidate_mask(candidates),
        class_ids=target_class_ids,
        eps=float(self.skill_learning_cfg.eps),
    )
    return self._pick_best_scored_cell(candidates, scores)

  def _pick_layout_reveal_target(self, vanilla_scores):
    candidates = list(vanilla_scores)
    if self.inference_library is None or self.skill_learning_cfg is None:
      best_score = max(vanilla_scores.values())
      return self._pick_uniform_best([pos for pos, score in vanilla_scores.items() if score == best_score])
    partial_ids = self._known_world_partial_ids()
    walkable_ids = [
        self.codec.encode_material(material)
        for material in ('grass', 'path', 'sand')
    ]
    walkable_scores = self.inference_library.score_candidate_cells(
        partial_map_ids=partial_ids,
        candidate_mask=self._candidate_mask(candidates),
        class_ids=walkable_ids,
        eps=float(self.skill_learning_cfg.eps),
    )
    combined = {
        pos: float(walkable_scores[pos].item()) + float(vanilla_scores[pos])
        for pos in candidates
    }
    best_score = max(combined.values())
    return self._pick_uniform_best([pos for pos, score in combined.items() if score == best_score])

  def _pick_best_scored_cell(self, candidates, scores: torch.Tensor):
    best_score = max(float(scores[pos].item()) for pos in candidates)
    best = [pos for pos in candidates if float(scores[pos].item()) == best_score]
    return self._pick_uniform_best(best)

  def _delta(self, lhs, rhs):
    return (rhs[0] - lhs[0], rhs[1] - lhs[1])

  def _require_known_world(self):
    """Return the currently bound known world.

    Returns
    -------
    KnownWorld
        Known-world instance passed to :meth:`plan`.
    """
    assert self._known_world is not None, 'Planner is not bound to a KnownWorld'
    return self._known_world
