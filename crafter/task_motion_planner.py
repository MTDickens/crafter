"""Task-level action synthesis on top of a partially observed Crafter world.

This module turns a configured sequence of high-level tasks such as
``collect_wood``, ``place_table``, and ``make_wood_pickaxe`` into concrete
single-step environment actions. The implementation combines three layers:

- ``KnownWorld`` stores the planner's partial belief over terrain and objects.
- ``LazyMotionPlanner`` searches for short paths while treating unknown tiles
  as traversable-but-risky until they are revealed.
- ``TaskMotionPlanner`` decomposes task semantics into movement, rotation, and
  interaction actions using Crafter's crafting and placement rules.

The planner is intentionally optimistic: it uses the currently observed world
to build a plan, reveals additional frontier tiles only when needed, and
re-plans if an assumed traversable route turns out to be blocked.
"""

from __future__ import annotations

import collections
import dataclasses
import heapq
import itertools
from typing import Iterable

import numpy as np

from . import constants

Position = tuple[int, int]
Facing = tuple[int, int]

DIRECTIONS: tuple[Facing, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
MOVE_ACTION_BY_FACING: dict[Facing, str] = {
    (-1, 0): "move_left",
    (1, 0): "move_right",
    (0, -1): "move_up",
    (0, 1): "move_down",
}
ACTION_TO_FACING: dict[str, Facing] = {
    action: facing for facing, action in MOVE_ACTION_BY_FACING.items()
}
CLOCKWISE_ORDER: tuple[Facing, ...] = ((-1, 0), (0, -1), (1, 0), (0, 1))
ROTATE_CLOCKWISE: dict[Facing, Facing] = {
    lhs: rhs for lhs, rhs in zip(CLOCKWISE_ORDER, CLOCKWISE_ORDER[1:] + CLOCKWISE_ORDER[:1])
}
ROTATE_COUNTERCLOCKWISE: dict[Facing, Facing] = {
    lhs: rhs for lhs, rhs in zip(CLOCKWISE_ORDER, CLOCKWISE_ORDER[-1:] + CLOCKWISE_ORDER[:-1])
}


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Normalized representation of one configured task.

    The planner accepts task names from configuration and derives a coarse task
    kind plus the concrete target item or structure that the task refers to.
    The original ``name`` is preserved because Crafter action and achievement
    identifiers already use the same string form.
    """

    name: str
    kind: str
    target: str

    @property
    def achievement_name(self) -> str:
        """Return the Crafter achievement key associated with this task."""
        return self.name

    @property
    def action_name(self) -> str:
        """Return the Crafter action name associated with this task."""
        return self.name


@dataclasses.dataclass(frozen=True)
class WorkspacePlan:
    """Reserved layout for the table/furnace crafting workspace.

    Attributes:
        stand: Tile where the player should stand while crafting or placing
            utilities in the reserved workspace.
        table_pos: Tile reserved for table placement.
        furnace_pos: Tile reserved for furnace placement.
    """

    stand: Position
    table_pos: Position
    furnace_pos: Position


class KnownWorld:
    """Planner-side cache of observed terrain and objects.

    ``KnownWorld`` tracks only what the planner has explicitly observed from
    the environment. Unknown tiles are absent from the internal dictionaries,
    which allows callers to distinguish between "known blocked" and "not yet
    revealed". The class also counts how many additional peeks were spent on
    revealing previously unseen tiles.
    """

    def __init__(self, area: Iterable[int], allow_lava: bool):
        """Initialize an empty partial map for a Crafter world.

        Args:
            area: World shape as ``(width, height)`` or any iterable of two
                integers.
            allow_lava: Whether lava should be considered standable by the
                planner.
        """
        self.area = tuple(int(x) for x in area)
        self.allow_lava = allow_lava
        self.peek_count = 0
        self._materials: dict[Position, str] = {}
        self._objects: dict[Position, str | None] = {}

    def in_bounds(self, pos: Position) -> bool:
        """Return whether ``pos`` is inside the world area."""
        return 0 <= pos[0] < self.area[0] and 0 <= pos[1] < self.area[1]

    def sync(self, world, pos: Position, charge_cost: bool) -> None:
        """Refresh one tile from the environment into the known-world cache.

        Args:
            world: Crafter world object supporting ``world[pos]`` lookups.
            pos: Tile to reveal or refresh.
            charge_cost: Whether revealing a previously unknown tile should
                increment ``peek_count``.

        Raises:
            RuntimeError: If ``pos`` is outside the world or the world returns
                an invalid material entry.
        """
        if not self.in_bounds(pos):
            raise RuntimeError(f"Position out of bounds: {pos}")
        material, obj = world[pos]
        if material is None:
            raise RuntimeError(f"World returned no material at {pos}")
        if pos not in self._materials and charge_cost:
            self.peek_count += 1
        self._materials[pos] = material
        self._objects[pos] = _object_name(obj)

    def reveal_all(self, world, charge_cost: bool) -> None:
        """Reveal the entire map by calling :meth:`sync` on every tile."""
        for x in range(self.area[0]):
            for y in range(self.area[1]):
                self.sync(world, (x, y), charge_cost=charge_cost)

    def is_known(self, pos: Position) -> bool:
        """Return whether this tile has been observed at least once."""
        return pos in self._materials

    def material(self, pos: Position) -> str:
        """Return the observed material at a known tile."""
        return self._materials[pos]

    def object_name(self, pos: Position) -> str | None:
        """Return the observed non-player object name at a known tile."""
        return self._objects[pos]

    def positions_with_material(self, material: str) -> list[Position]:
        """Return all known positions currently observed as ``material``."""
        return [pos for pos, observed in self._materials.items() if observed == material]

    def neighbors(self, pos: Position) -> tuple[Position, ...]:
        """Return in-bounds 4-neighbors of ``pos``."""
        neighbors = []
        for dx, dy in DIRECTIONS:
            candidate = (pos[0] + dx, pos[1] + dy)
            if self.in_bounds(candidate):
                neighbors.append(candidate)
        return tuple(neighbors)

    def can_stand_known(self, pos: Position) -> bool:
        """Return whether a known tile is currently standable.

        A tile is standable only when it has no blocking object and its
        material is walkable under the planner's current lava policy.
        Unknown tiles are not permitted here; callers must reveal them first or
        use :meth:`optimistic_can_stand`.
        """
        material = self.material(pos)
        obj_name = self.object_name(pos)
        if obj_name is not None:
            return False
        if material in constants.walkable:
            return True
        if self.allow_lava and material == "lava":
            return True
        return False

    def optimistic_can_stand(self, pos: Position) -> bool:
        """Return a permissive standability check for planning under uncertainty.

        Unknown in-bounds tiles are treated as traversable so the planner can
        form tentative routes through unexplored space. Known tiles are checked
        with the stricter :meth:`can_stand_known` logic.
        """
        if not self.in_bounds(pos):
            return False
        if not self.is_known(pos):
            return True
        return self.can_stand_known(pos)

    def known_distance_map(self, start: Position) -> dict[Position, int]:
        """Compute BFS distances inside the revealed traversable region.

        Only already known and standable tiles are expanded. This makes the
        result useful for ranking candidate stands, frontier tiles, and
        workspace locations without speculating through unknown space.

        Args:
            start: Revealed starting tile for the search.

        Returns:
            Mapping from reachable known positions to their shortest distance in
            number of moves.

        Raises:
            RuntimeError: If ``start`` is unknown or currently not standable.
        """
        if not self.is_known(start):
            raise RuntimeError(f"Start position is not known: {start}")
        if not self.can_stand_known(start):
            raise RuntimeError(f"Start position is not standable: {start}")
        queue: collections.deque[Position] = collections.deque([start])
        distances = {start: 0}
        while queue:
            pos = queue.popleft()
            for neighbor in self.neighbors(pos):
                if neighbor in distances:
                    continue
                if not self.is_known(neighbor):
                    continue
                if not self.can_stand_known(neighbor):
                    continue
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)
        return distances


class LazyMotionPlanner:
    """Path planner that is optimistic about unknown tiles.

    The planner searches in the current ``KnownWorld`` while assigning an extra
    penalty to stepping into unrevealed tiles. Candidate paths are therefore
    biased toward confirmed safe terrain, but the search can still route
    through unknown regions when no revealed alternative is available.
    """

    def __init__(self, known_world: KnownWorld, unknown_penalty: float):
        """Create a planner backed by a mutable :class:`KnownWorld`."""
        self.known_world = known_world
        self.unknown_penalty = float(unknown_penalty)

    def estimate_path(self, start: Position, goals: Iterable[Position]) -> list[Position] | None:
        """Return the current best optimistic path without revealing new tiles."""
        return self._optimistic_shortest_path(start, goals)

    def plan_validated_path(
        self, world, start: Position, goals: Iterable[Position]
    ) -> list[Position] | None:
        """Plan a path and validate every unknown step against the real world.

        The method repeatedly computes an optimistic shortest path, reveals
        unknown tiles along that path as needed, and restarts planning if one
        of those tiles turns out to be blocked. The returned path therefore
        contains only known standable tiles by the time the method succeeds.

        Args:
            world: Crafter world object used to reveal unknown tiles.
            start: Current player position.
            goals: Acceptable goal positions.

        Returns:
            A validated path from ``start`` to one goal, inclusive, or ``None``
            if no route can be found even after revealing the necessary tiles.
        """
        goal_set = tuple(sorted(set(goals)))
        if not goal_set:
            return None
        while True:
            path = self._optimistic_shortest_path(start, goal_set)
            if path is None:
                return None
            blocked = False
            for pos in path[1:]:
                if not self.known_world.is_known(pos):
                    self.known_world.sync(world, pos, charge_cost=True)
                if not self.known_world.can_stand_known(pos):
                    blocked = True
                    break
            if not blocked:
                return path

    def _optimistic_shortest_path(
        self, start: Position, goals: Iterable[Position]
    ) -> list[Position] | None:
        """Run Dijkstra search with unknown-tile penalties.

        Unknown tiles are traversable at cost ``1 + unknown_penalty``.
        Revealed standable tiles cost ``1``. Revealed blocked tiles are
        excluded entirely.
        """
        goal_set = frozenset(goals)
        if not goal_set:
            return None
        if start in goal_set:
            return [start]
        frontier: list[tuple[float, int, Position]] = [(0.0, 0, start)]
        best_cost = {start: 0.0}
        parents = {start: start}
        tie_breaker = 1
        while frontier:
            cost, _, pos = heapq.heappop(frontier)
            if cost != best_cost[pos]:
                continue
            if pos in goal_set:
                return _reconstruct_path(parents, pos)
            for neighbor in self.known_world.neighbors(pos):
                step_cost = self._step_cost(neighbor)
                if step_cost is None:
                    continue
                new_cost = cost + step_cost
                if neighbor in best_cost and new_cost >= best_cost[neighbor]:
                    continue
                best_cost[neighbor] = new_cost
                parents[neighbor] = pos
                heapq.heappush(frontier, (new_cost, tie_breaker, neighbor))
                tie_breaker += 1
        return None

    def _step_cost(self, pos: Position) -> float | None:
        """Return movement cost for stepping onto ``pos`` under optimistic planning."""
        if not self.known_world.is_known(pos):
            return 1.0 + self.unknown_penalty
        if not self.known_world.can_stand_known(pos):
            return None
        return 1.0


class TaskMotionPlanner:
    """Convert a task sequence into single-step Crafter actions.

    The planner maintains a partial world model, tracks progress through the
    configured task order, and lazily synthesizes action queues for the current
    task. It assumes the environment evolves according to the planner's chosen
    actions and verifies that assumption on every step via expected position
    and facing checks.
    """

    def __init__(self, env, config):
        """Initialize planner state from the environment and config.

        Args:
            env: Crafter environment instance whose ``_world`` and ``_player``
                fields are used for planning.
            config: Planner configuration object with task order, knowledge
                mode, reveal budget, motion penalty, and debug flags.

        Raises:
            ValueError: If the requested initial knowledge mode is unsupported.
        """
        self.tasks = tuple(parse_task_name(name) for name in config.task_order)
        self.debug = bool(config.debug)
        self.frontier_reveal_budget = int(config.frontier_reveal_budget)
        self.known_world = KnownWorld(env._world.area, allow_lava=bool(config.allow_lava))
        self.motion_planner = LazyMotionPlanner(
            self.known_world, unknown_penalty=float(config.unknown_motion_penalty)
        )
        self.workspace: WorkspacePlan | None = None
        self.task_index = 0
        self._queued_actions: list[str] = []
        self._expected_pos: Position | None = None
        self._expected_facing: Facing | None = None
        self._has_reported_success = False

        if config.initial_knowledge == "global":
            self.known_world.reveal_all(env._world, charge_cost=False)
        elif config.initial_knowledge == "local":
            self._refresh_local_state(env)
        else:
            raise ValueError(
                f"Unsupported planner.initial_knowledge: {config.initial_knowledge}"
            )
        self._refresh_local_state(env)
        print(
            f"[planner] loaded {len(self.tasks)} tasks, initial known tiles = {len(self.known_world._materials)}"
        )

    def next_action(self, env) -> str:
        """Return the next environment action needed to advance the task list.

        The method refreshes local observations, checks that the environment is
        consistent with the planner's previously predicted state, advances past
        any tasks that are already complete, and synthesizes a new action queue
        for the current task when necessary.

        Args:
            env: Crafter environment providing current player/world state.

        Returns:
            A single Crafter action name. Once all tasks are complete the
            planner returns ``"noop"`` indefinitely.

        Raises:
            RuntimeError: If internal state prediction fails or if no valid
                action sequence can be synthesized for the current task.
        """
        self._refresh_local_state(env)
        self._validate_expected_state(env)
        self._advance_completed_tasks(env)

        if self.task_index >= len(self.tasks):
            self._report_finished_once()
            action = "noop"
            self._set_expected_state(env, action)
            return action

        if not self._queued_actions:
            task = self.tasks[self.task_index]
            if self.debug:
                print(f"[planner] synthesizing plan for {task.name}")
            self._queued_actions = self._synthesize_actions(env, task)
            if not self._queued_actions:
                raise RuntimeError(f"Planner failed to synthesize actions for {task.name}")

        action = self._queued_actions.pop(0)
        self._set_expected_state(env, action)
        return action

    def is_finished(self, env) -> bool:
        """Return whether the planner considers all configured tasks complete.

        This method updates planner-internal knowledge and task progress from
        the current environment observation before checking completion, so the
        result reflects the planner's latest internal state rather than a stale
        cached task index.
        """
        self._refresh_local_state(env)
        self._validate_expected_state(env)
        self._advance_completed_tasks(env)
        if self.task_index >= len(self.tasks):
            self._report_finished_once()
            return True
        return False

    def _refresh_local_state(self, env) -> None:
        """Reveal the player tile and its immediate neighbors for free."""
        player_pos = _to_position(env._player.pos)
        self.known_world.sync(env._world, player_pos, charge_cost=False)
        for neighbor in self.known_world.neighbors(player_pos):
            self.known_world.sync(env._world, neighbor, charge_cost=False)

    def _validate_expected_state(self, env) -> None:
        """Assert that the environment matches the planner's last prediction.

        The planner caches the expected post-action position and facing after
        every emitted action. A mismatch indicates that the environment changed
        in an unexpected way or that the planner made an invalid assumption.
        """
        if self._expected_pos is None:
            return
        current_pos = _to_position(env._player.pos)
        current_facing = _to_facing(env._player.facing)
        if current_pos != self._expected_pos:
            raise RuntimeError(
                f"Planner state mismatch: expected pos {self._expected_pos}, got {current_pos}"
            )
        if current_facing != self._expected_facing:
            raise RuntimeError(
                f"Planner state mismatch: expected facing {self._expected_facing}, got {current_facing}"
            )

    def _set_expected_state(self, env, action: str) -> None:
        """Record the position/facing the planner expects after ``action``."""
        pos = _to_position(env._player.pos)
        facing = _to_facing(env._player.facing)
        if action in ACTION_TO_FACING:
            facing = ACTION_TO_FACING[action]
            pos = (pos[0] + facing[0], pos[1] + facing[1])
        elif action == "rotate_clockwise":
            facing = ROTATE_CLOCKWISE[facing]
        elif action == "rotate_counterclockwise":
            facing = ROTATE_COUNTERCLOCKWISE[facing]
        self._expected_pos = pos
        self._expected_facing = facing

    def _advance_completed_tasks(self, env) -> None:
        """Skip over tasks that are already satisfied in the live environment."""
        while self.task_index < len(self.tasks):
            task = self.tasks[self.task_index]
            if not self._task_is_complete(env, self.task_index):
                break
            print(
                f"[planner] completed {task.name} at step {env._step}, inventory = {env._player.inventory}"
            )
            self.task_index += 1
            self._queued_actions = []

    def _report_finished_once(self) -> None:
        """Emit the one-time completion log message for the entire task list."""
        if self._has_reported_success:
            return
        self._has_reported_success = True
        print(
            f"[planner] all tasks finished, total additional peeks = {self.known_world.peek_count}"
        )

    def _task_is_complete(self, env, task_index: int) -> bool:
        """Return whether the task at ``task_index`` is already complete.

        Collect tasks require both an achievement signal and enough remaining
        inventory to satisfy future downstream uses of the collected item.
        Other task kinds follow the achievement signal alone.
        """
        task = self.tasks[task_index]
        achievement_count = env._player.achievements[task.achievement_name]
        if task.kind == "collect":
            required_inventory = self._required_inventory_floor_after_collect(task_index, task.target)
            return achievement_count > 0 and env._player.inventory[task.target] >= required_inventory
        return achievement_count > 0

    def _required_inventory_floor_after_collect(self, task_index: int, item_name: str) -> int:
        """Return the minimum inventory that should remain after a collect task.

        This prevents the planner from marking a collect task complete if the
        newly collected item will immediately be consumed by future place/make
        tasks and therefore still needs to be gathered again.
        """
        required = 0
        for future_task in self.tasks[task_index + 1 :]:
            if future_task.kind == "place":
                uses = constants.place[future_task.target]["uses"]
                if item_name in uses:
                    required += uses[item_name]
            elif future_task.kind == "make":
                uses = constants.make[future_task.target]["uses"]
                if item_name in uses:
                    required += uses[item_name]
        return max(1, required)

    def _synthesize_actions(self, env, task: TaskSpec) -> list[str]:
        """Dispatch task planning to the task-kind-specific helper."""
        if task.kind == "collect":
            return self._plan_collect_task(env, task)
        if task.kind == "place":
            return self._plan_place_task(env, task)
        if task.kind == "make":
            return self._plan_make_task(env, task)
        raise RuntimeError(f"Unsupported task kind: {task.kind}")

    def _plan_collect_task(self, env, task: TaskSpec) -> list[str]:
        """Plan how to collect the item targeted by ``task``.

        The planner first tries direct collection from already known target
        material tiles, then spends a limited reveal budget on promising
        frontier tiles, and finally falls back to digging auxiliary path-like
        materials that may expose more map.
        """
        target_material = collect_material_for_item(task.target)
        if not requirements_met(constants.collect[target_material]["require"], env._player.inventory):
            raise RuntimeError(
                f"Missing required tools for collecting {target_material}: {constants.collect[target_material]['require']}"
            )

        direct_plan = self._plan_collect_from_material(env, target_material)
        if direct_plan is not None:
            return direct_plan

        self._reveal_frontier_toward(env, target_material)
        direct_plan = self._plan_collect_from_material(env, target_material)
        if direct_plan is not None:
            return direct_plan

        auxiliary_tile = self._choose_auxiliary_excavation_tile(env, task, target_material)
        if auxiliary_tile is not None:
            return self._plan_collect_from_tile(env, auxiliary_tile)

        raise RuntimeError(
            f"Could not find a way to collect {task.target} for task {task.name}"
        )

    def _plan_collect_from_material(self, env, material: str) -> list[str] | None:
        """Plan collection from any known tile with the requested material."""
        target_tiles = self.known_world.positions_with_material(material)
        if not target_tiles:
            return None
        return self._plan_collect_from_tiles(env, target_tiles)

    def _plan_collect_from_tiles(
        self, env, target_tiles: Iterable[Position]
    ) -> list[str] | None:
        """Choose a reachable stand next to one of ``target_tiles`` and interact.

        Multiple target tiles may share the same stand tile. In that case the
        validated path is planned to the stand, and a deterministic target is
        chosen among the tiles adjacent to that stand.
        """
        goal_to_targets: dict[Position, list[Position]] = collections.defaultdict(list)
        for target in sorted(set(target_tiles)):
            for stand in self.known_world.neighbors(target):
                goal_to_targets[stand].append(target)
        if not goal_to_targets:
            return None
        start = _to_position(env._player.pos)
        path = self.motion_planner.plan_validated_path(env._world, start, goal_to_targets)
        if path is None:
            return None
        stand = path[-1]
        target = tuple(sorted(goal_to_targets[stand]))[0]
        return self._assemble_interaction_plan(
            path=path,
            current_facing=_to_facing(env._player.facing),
            stand=stand,
            target=target,
            action_name="do",
        )

    def _plan_collect_from_tile(self, env, target: Position) -> list[str]:
        """Plan a collection interaction against one specific target tile."""
        start = _to_position(env._player.pos)
        path = self.motion_planner.plan_validated_path(
            env._world, start, self.known_world.neighbors(target)
        )
        if path is None:
            raise RuntimeError(f"No path to collect from tile {target}")
        return self._assemble_interaction_plan(
            path=path,
            current_facing=_to_facing(env._player.facing),
            stand=path[-1],
            target=target,
            action_name="do",
        )

    def _reveal_frontier_toward(self, env, target_material: str) -> None:
        """Spend reveal budget on frontier tiles that may uncover a target.

        Revealing stops early if there is no frontier left or if one of the
        newly revealed tiles already contains the requested material.
        """
        for _ in range(self.frontier_reveal_budget):
            frontier = self._choose_frontier_to_reveal(env, target_material)
            if frontier is None:
                return
            self.known_world.sync(env._world, frontier, charge_cost=True)
            if self.known_world.material(frontier) == target_material:
                return

    def _choose_frontier_to_reveal(
        self, env, target_material: str
    ) -> Position | None:
        """Rank unknown frontier tiles and return the best one to reveal.

        Frontier tiles are unknown tiles adjacent to some reachable known stand.
        When a target material has already been seen elsewhere, the heuristic
        prefers frontiers with smaller Manhattan distance to those targets;
        otherwise it prefers frontiers that are cheap to reach from the current
        revealed region.
        """
        current_pos = _to_position(env._player.pos)
        distance_map = self.known_world.known_distance_map(current_pos)
        known_targets = self.known_world.positions_with_material(target_material)
        best_score = None
        best_frontier = None
        frontier_best_distance: dict[Position, int] = {}
        for stand, distance in distance_map.items():
            for frontier in self.known_world.neighbors(stand):
                if self.known_world.is_known(frontier):
                    continue
                if frontier not in frontier_best_distance or distance < frontier_best_distance[frontier]:
                    frontier_best_distance[frontier] = distance
        for frontier, distance in frontier_best_distance.items():
            if known_targets:
                hint = min(manhattan(frontier, target) for target in known_targets)
                score = (hint, distance, frontier)
            else:
                score = (distance, frontier)
            if best_score is None or score < best_score:
                best_score = score
                best_frontier = frontier
        return best_frontier

    def _choose_auxiliary_excavation_tile(
        self, env, task: TaskSpec, target_material: str
    ) -> Position | None:
        """Pick a secondary tile to mine when the direct target is unavailable.

        The fallback prefers already allowed path-leaving materials from the
        completed/current collect tasks. If the target material is already
        known, the heuristic prefers auxiliary tiles close to it; otherwise it
        prefers tiles that border more unrevealed space.
        """
        current_pos = _to_position(env._player.pos)
        distance_map = self.known_world.known_distance_map(current_pos)
        known_targets = self.known_world.positions_with_material(target_material)
        allowed_auxiliary_materials = self._allowed_auxiliary_materials(task)
        best_score = None
        best_target = None
        for material in sorted(allowed_auxiliary_materials):
            if not requirements_met(constants.collect[material]["require"], env._player.inventory):
                continue
            for target in self.known_world.positions_with_material(material):
                stand_options = [
                    stand for stand in self.known_world.neighbors(target) if stand in distance_map
                ]
                if not stand_options:
                    continue
                if not known_targets:
                    frontier_bonus = self._unknown_neighbor_count(target)
                    if frontier_bonus == 0:
                        continue
                    hint = -frontier_bonus
                else:
                    hint = min(manhattan(target, known_target) for known_target in known_targets)
                best_stand = min(stand_options, key=lambda stand: (distance_map[stand], stand))
                score = (
                    0 if material == target_material else 1,
                    hint,
                    distance_map[best_stand],
                    target,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_target = target
        return best_target

    def _allowed_auxiliary_materials(self, task: TaskSpec) -> set[str]:
        """Return materials that are safe to excavate as exploratory fallback."""
        current_index = self.tasks.index(task)
        allowed = set()
        for completed_or_current in self.tasks[: current_index + 1]:
            if completed_or_current.kind != "collect":
                continue
            material = collect_material_for_item(completed_or_current.target)
            if constants.collect[material]["leaves"] == "path":
                allowed.add(material)
        return allowed

    def _unknown_neighbor_count(self, pos: Position) -> int:
        """Count unrevealed 4-neighbors of ``pos``."""
        return sum(
            1 for neighbor in self.known_world.neighbors(pos) if not self.known_world.is_known(neighbor)
        )

    def _plan_place_task(self, env, task: TaskSpec) -> list[str]:
        """Plan how to place a structure or block for ``task``."""
        if task.target == "table":
            workspace = self._ensure_workspace(env)
            return self._plan_place_at(env, workspace.stand, workspace.table_pos, task.action_name)
        if task.target == "furnace":
            workspace = self._ensure_workspace(env)
            return self._plan_place_at(env, workspace.stand, workspace.furnace_pos, task.action_name)
        if task.target == "stone":
            stand, target = self._choose_stone_placement(env)
            return self._plan_place_at(env, stand, target, task.action_name)
        raise RuntimeError(f"Unsupported place task: {task.name}")

    def _choose_stone_placement(self, env) -> tuple[Position, Position]:
        """Choose a low-risk location for placing stone.

        The heuristic avoids reserved workspace tiles, prefers nearby stand
        positions, and strongly prefers filling hazardous terrain such as water
        or lava over consuming generic buildable terrain.
        """
        current_pos = _to_position(env._player.pos)
        distance_map = self.known_world.known_distance_map(current_pos)
        reserved = set()
        if self.workspace is not None:
            reserved.add(self.workspace.stand)
            reserved.add(self.workspace.table_pos)
            reserved.add(self.workspace.furnace_pos)
        best_score = None
        best_choice = None
        for target_material in constants.place["stone"]["where"]:
            for target in self.known_world.positions_with_material(target_material):
                if target in reserved:
                    continue
                if self.known_world.object_name(target) is not None:
                    continue
                stand_options = [
                    stand for stand in self.known_world.neighbors(target) if stand in distance_map
                ]
                if not stand_options:
                    continue
                best_stand = min(stand_options, key=lambda stand: (distance_map[stand], stand))
                degree = self._known_standable_neighbor_count(target, reserved | {best_stand})
                terrain_bias = 0 if target_material in ("water", "lava") else 5
                score = (
                    distance_map[best_stand] + terrain_bias,
                    degree,
                    target,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_choice = (best_stand, target)
        if best_choice is None:
            raise RuntimeError("Could not find a safe tile for place_stone")
        return best_choice

    def _plan_place_at(
        self, env, stand: Position, target: Position, action_name: str
    ) -> list[str]:
        """Plan movement and facing adjustment for one placement action."""
        start = _to_position(env._player.pos)
        path = self.motion_planner.plan_validated_path(env._world, start, [stand])
        if path is None:
            raise RuntimeError(f"No path to placement stand {stand} for {action_name}")
        return self._assemble_interaction_plan(
            path=path,
            current_facing=_to_facing(env._player.facing),
            stand=stand,
            target=target,
            action_name=action_name,
        )

    def _plan_make_task(self, env, task: TaskSpec) -> list[str]:
        """Plan a crafting action at the reserved workspace.

        The method verifies recipe inputs, confirms that any required nearby
        utilities have already been placed at the reserved workspace slots, and
        then moves to the workspace stand before issuing the craft action.
        """
        make_info = constants.make[task.target]
        if not requirements_met(make_info["uses"], env._player.inventory):
            raise RuntimeError(
                f"Missing resources for {task.name}: need {make_info['uses']}, have {env._player.inventory}"
            )
        workspace = self._ensure_workspace(env)
        required_utilities = tuple(make_info["nearby"])
        missing = []
        if "table" in required_utilities and self.known_world.material(workspace.table_pos) != "table":
            missing.append("table")
        if "furnace" in required_utilities and self.known_world.material(workspace.furnace_pos) != "furnace":
            missing.append("furnace")
        if missing:
            raise RuntimeError(f"Workspace is missing required utilities for {task.name}: {missing}")
        start = _to_position(env._player.pos)
        path = self.motion_planner.plan_validated_path(env._world, start, [workspace.stand])
        if path is None:
            raise RuntimeError(f"No path to workspace stand {workspace.stand} for {task.name}")
        actions = path_to_actions(path)
        actions.append(task.action_name)
        return actions

    def _ensure_workspace(self, env) -> WorkspacePlan:
        """Reserve a reusable table/furnace layout if one is not chosen yet.

        A valid workspace provides one stand tile, two adjacent build slots for
        table and furnace, and at least one remaining escape/access neighbor so
        the player is not boxed in after placing both utilities.
        """
        if self.workspace is not None:
            return self.workspace
        current_pos = _to_position(env._player.pos)
        distance_map = self.known_world.known_distance_map(current_pos)
        best_score = None
        best_workspace = None
        for stand, distance in distance_map.items():
            candidate_slots = []
            for neighbor in self.known_world.neighbors(stand):
                if not self.known_world.is_known(neighbor):
                    continue
                if self.known_world.object_name(neighbor) is not None:
                    continue
                if self.known_world.material(neighbor) in constants.place["table"]["where"]:
                    candidate_slots.append(neighbor)
            if len(candidate_slots) < 2:
                continue
            for table_pos, furnace_pos in itertools.combinations(sorted(candidate_slots), 2):
                remaining_access = 0
                for neighbor in self.known_world.neighbors(stand):
                    if neighbor in (table_pos, furnace_pos):
                        continue
                    if self.known_world.is_known(neighbor) and self.known_world.can_stand_known(neighbor):
                        remaining_access += 1
                    elif not self.known_world.is_known(neighbor):
                        remaining_access += 1
                if remaining_access < 1:
                    continue
                slot_penalty = self._known_standable_neighbor_count(table_pos, {stand})
                slot_penalty += self._known_standable_neighbor_count(furnace_pos, {stand})
                score = (distance, slot_penalty, -remaining_access, stand, table_pos, furnace_pos)
                if best_score is None or score < best_score:
                    best_score = score
                    best_workspace = WorkspacePlan(
                        stand=stand,
                        table_pos=table_pos,
                        furnace_pos=furnace_pos,
                    )
        if best_workspace is None:
            raise RuntimeError("Could not find a workspace for table and furnace placement")
        self.workspace = best_workspace
        print(f"[planner] reserved workspace at {self.workspace}")
        return self.workspace

    def _known_standable_neighbor_count(
        self, pos: Position, excluded: set[Position]
    ) -> int:
        """Count known standable neighbors of ``pos`` excluding reserved tiles."""
        count = 0
        for neighbor in self.known_world.neighbors(pos):
            if neighbor in excluded:
                continue
            if self.known_world.is_known(neighbor) and self.known_world.can_stand_known(neighbor):
                count += 1
        return count

    def _assemble_interaction_plan(
        self,
        path: list[Position],
        current_facing: Facing,
        stand: Position,
        target: Position,
        action_name: str,
    ) -> list[str]:
        """Convert movement plus final facing adjustment into action strings."""
        actions = path_to_actions(path)
        facing_after_path = current_facing if len(path) == 1 else direction_between(path[-2], path[-1])
        desired_facing = direction_between(stand, target)
        actions.extend(rotation_actions(facing_after_path, desired_facing))
        actions.append(action_name)
        return actions


def parse_task_name(task_name: str) -> TaskSpec:
    """Parse one configured task name into a normalized :class:`TaskSpec`.

    Args:
        task_name: Task identifier such as ``collect_wood`` or
            ``make_stone_pickaxe``.

    Returns:
        Parsed task specification containing the original name, task kind, and
        target suffix.

    Raises:
        ValueError: If ``task_name`` does not use a supported prefix.
    """
    if task_name.startswith("collect_"):
        return TaskSpec(name=task_name, kind="collect", target=task_name[len("collect_") :])
    if task_name.startswith("place_"):
        return TaskSpec(name=task_name, kind="place", target=task_name[len("place_") :])
    if task_name.startswith("make_"):
        return TaskSpec(name=task_name, kind="make", target=task_name[len("make_") :])
    raise ValueError(f"Unsupported task name: {task_name}")


def collect_material_for_item(item_name: str) -> str:
    """Return the unique collectable material that yields ``item_name``.

    The planner relies on ``constants.collect`` declaring an unambiguous source
    material for each collectible item.

    Args:
        item_name: Inventory item produced by a collect action.

    Returns:
        Material name that should be interacted with to obtain the item.

    Raises:
        RuntimeError: If zero or multiple materials claim to produce the item.
    """
    matches = []
    for material, info in constants.collect.items():
        receive = info["receive"]
        if item_name in receive:
            matches.append(material)
    if len(matches) != 1:
        raise RuntimeError(f"Expected a unique collect material for {item_name}, got {matches}")
    return matches[0]


def requirements_met(requirements: dict[str, int], inventory: dict[str, int]) -> bool:
    """Return whether ``inventory`` satisfies all required item counts."""
    for item_name, amount in requirements.items():
        if inventory[item_name] < amount:
            return False
    return True


def path_to_actions(path: list[Position]) -> list[str]:
    """Translate a path into per-step movement action names.

    Args:
        path: Consecutive 4-neighbor positions including the start tile.

    Returns:
        Movement actions needed to walk the path. A length-1 path produces an
        empty action list.

    Raises:
        RuntimeError: If two consecutive positions are not 4-neighbors.
    """
    actions = []
    for lhs, rhs in zip(path, path[1:]):
        actions.append(MOVE_ACTION_BY_FACING[direction_between(lhs, rhs)])
    return actions


def rotation_actions(current_facing: Facing, desired_facing: Facing) -> list[str]:
    """Return the shorter rotation sequence from one facing to another.

    Ties are broken in favor of clockwise rotation.
    """
    if current_facing == desired_facing:
        return []
    current_index = CLOCKWISE_ORDER.index(current_facing)
    desired_index = CLOCKWISE_ORDER.index(desired_facing)
    clockwise_steps = (desired_index - current_index) % 4
    counterclockwise_steps = (current_index - desired_index) % 4
    if clockwise_steps <= counterclockwise_steps:
        return ["rotate_clockwise"] * clockwise_steps
    return ["rotate_counterclockwise"] * counterclockwise_steps


def direction_between(lhs: Position, rhs: Position) -> Facing:
    """Return the unit facing vector from ``lhs`` to neighboring ``rhs``.

    Raises:
        RuntimeError: If the positions are not 4-neighbors.
    """
    offset = (rhs[0] - lhs[0], rhs[1] - lhs[1])
    if offset not in MOVE_ACTION_BY_FACING:
        raise RuntimeError(f"Positions are not 4-neighbors: {lhs} -> {rhs}")
    return offset


def manhattan(lhs: Position, rhs: Position) -> int:
    """Return Manhattan distance between two positions."""
    return abs(lhs[0] - rhs[0]) + abs(lhs[1] - rhs[1])


def _reconstruct_path(parents: dict[Position, Position], goal: Position) -> list[Position]:
    """Reconstruct a root-to-goal path from a predecessor map."""
    path = [goal]
    cursor = goal
    while parents[cursor] != cursor:
        cursor = parents[cursor]
        path.append(cursor)
    path.reverse()
    return path


def _to_position(pos) -> Position:
    """Coerce a 2D coordinate-like value to the ``Position`` alias."""
    return (int(pos[0]), int(pos[1]))


def _to_facing(facing) -> Facing:
    """Coerce a facing-like value to the ``Facing`` alias."""
    return (int(facing[0]), int(facing[1]))


def _object_name(obj) -> str | None:
    """Return a normalized object name, treating the player as empty space."""
    if obj is None:
        return None
    name = type(obj).__name__
    if name == "Player":
        return None
    return name.lower()


def render_known_world_image(
    planner: TaskMotionPlanner,
    env,
    textures,
    tile_size: int = 16,
    background_color: tuple[int, int, int] = (127, 127, 127),
) -> np.ndarray:
    """Render the planner's currently known world as a full-map RGB image.

    Unknown tiles are left as a flat background color. Known materials are
    rendered using existing Crafter textures, known non-player objects are
    alpha-blended on top, and the current player texture is drawn last.

    Args:
        planner: Planner whose ``KnownWorld`` should be visualized.
        env: Crafter environment providing the current player state.
        textures: Crafter texture atlas, typically ``env._textures``.
        tile_size: Pixel size for each world tile in the exported image.
        background_color: RGB fill color for unrevealed tiles.

    Returns:
        ``(height, width, 3)`` uint8 image suitable for saving as a PNG.
    """
    known_world = planner.known_world
    unit = np.array([tile_size, tile_size], dtype=int)
    canvas = np.zeros(
        (known_world.area[0] * tile_size, known_world.area[1] * tile_size, 3),
        dtype=np.uint8,
    )
    canvas[:] = np.array(background_color, dtype=np.uint8)

    def draw_texture(texture_name: str, pos: Position, *, alpha: bool = False) -> None:
        texture = textures.get(texture_name, unit)
        top_left = np.array(pos) * unit
        x, y = int(top_left[0]), int(top_left[1])
        width, height = texture.shape[:2]
        if texture.shape[-1] == 4 and alpha:
            alpha_channel = texture[..., 3:].astype(np.float32) / 255.0
            rgb = texture[..., :3].astype(np.float32) / 255.0
            current = canvas[x : x + width, y : y + height].astype(np.float32) / 255.0
            blended = alpha_channel * rgb + (1.0 - alpha_channel) * current
            canvas[x : x + width, y : y + height] = (255.0 * blended).astype(np.uint8)
            return
        canvas[x : x + width, y : y + height] = texture[..., :3]

    for pos in sorted(known_world._materials):
        draw_texture(known_world.material(pos), pos)
        obj_name = known_world.object_name(pos)
        if obj_name is not None:
            draw_texture(obj_name, pos, alpha=True)

    draw_texture(env._player.texture, _to_position(env._player.pos), alpha=True)
    return canvas.transpose((1, 0, 2))
