"""Run the Crafter environment with a series of actions provided by the user."""

import datetime
import pathlib

import hydra
import numpy as np
import pygame
from omegaconf import DictConfig
from PIL import Image

import crafter
from crafter.task_motion_planner import TaskMotionPlanner, render_known_world_image


class KnownWorldFrameExporter:
    """Save planner known-world renders as a PNG frame sequence."""

    def __init__(
        self,
        directory: pathlib.Path,
        textures,
        mode: str,
    ):
        self._directory = directory
        self._directory.mkdir(exist_ok=True, parents=True)
        self._textures = textures
        self._frame_index = 0
        self._mode = mode

    def save_step_frame(self, planner: TaskMotionPlanner, env) -> None:
        """Save a zero-padded per-step frame."""
        image = render_known_world_image(planner, env, self._textures)
        filename = self._directory / f"frame_{self._frame_index:06d}.png"
        Image.fromarray(image).save(filename)
        self._frame_index += 1

    def save_pending_task_frame(self, planner: TaskMotionPlanner, env) -> None:
        """Save one frame when a new task starts executing in task mode."""
        if self._mode != "task":
            return
        event = planner.consume_known_world_task_render()
        if event is None:
            return
        task_name, frame_index = event
        image = render_known_world_image(planner, env, self._textures)
        filename = self._directory / f"frame_{frame_index:06d}-{task_name}.png"
        Image.fromarray(image).save(filename)


def _make_known_world_episode_dir(root: pathlib.Path, run_id: str, episode_index: int) -> pathlib.Path:
    """Return the output directory for one GUI-run planner episode."""
    return root / run_id / "worker_0000" / f"episode_{episode_index:06d}"


def _make_known_world_run_id() -> str:
    """Return a timestamp-based run identifier for known-world exports."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def _print_actions(keymap):
    print("Actions:")
    for key, action in keymap.items():
        print(f"  {pygame.key.name(key)}: {action}")


def get_actions_from_user(keymap: dict[int, str], config: DictConfig) -> list[str]:
    """Get action prefixes list from user input (either stdin or file) and parse it into actual actions."""
    action_parts_by_name = {action: action.split("_") for action in keymap.values()}

    def matching_actions(
        prefix_parts: list[str], skip_first_component: bool = False
    ) -> list[str]:
        """Return actions whose components match the given per-component prefixes."""
        matches = []
        for action, action_parts in action_parts_by_name.items():
            candidate_parts = action_parts[1:] if skip_first_component else action_parts
            if len(prefix_parts) != len(candidate_parts):
                continue
            if all(
                part.startswith(prefix)
                for prefix, part in zip(prefix_parts, candidate_parts)
            ):
                matches.append(action)
        return matches

    def parse_action_prefix(prefix: str) -> str:
        """Resolve a user-provided action token to one concrete action name.

        Matching is intentionally tiered to keep shorthand convenient without
        introducing surprising collisions:

        1. Exact action-name match, e.g. ``do`` -> ``do``.
        2. Per-component prefix match with the same number of components,
            e.g. ``mo_ri`` -> ``move_right``.
        3. Per-component prefix match after dropping the first action component,
            which allows movement shorthand such as ``up`` -> ``move_up`` and
            ``down`` -> ``move_down``.

        A token is accepted only if exactly one action matches at the highest
        applicable tier. Otherwise a ``ValueError`` is raised for either
        ambiguity or no match.
        """
        prefix = prefix.strip()
        if not prefix:
            raise ValueError("Action prefix cannot be empty.")

        if prefix in action_parts_by_name:
            return prefix

        prefix_parts = prefix.split("_")
        matches = matching_actions(prefix_parts)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Prefix '{prefix}' is ambiguous and matches multiple actions: {matches}"
            )

        shorthand_matches = matching_actions(prefix_parts, skip_first_component=True)
        if len(shorthand_matches) == 1:
            return shorthand_matches[0]
        if len(shorthand_matches) > 1:
            raise ValueError(
                f"Prefix '{prefix}' is ambiguous and matches multiple actions: {shorthand_matches}"
            )
        raise ValueError(f"Prefix '{prefix}' did not match any action.")

    actions_input_source = config.actions_input_source
    if actions_input_source == "stdin":
        print(
            "Enter actions (separated by comma, can be prefix of one of the components of an action, e.g., 'move_left' to 'le')."
            "Press Ctrl+D (Unix) or Ctrl+Z (Windows) to end input."
        )
        action_prefixes = input("Actions: ")
        action_tokens = [token.strip() for token in action_prefixes.split(",")]
        actions = [parse_action_prefix(token) for token in action_tokens if token]
    elif actions_input_source == "file":
        if not config.actions_input_file_path:
            raise ValueError(
                "actions_input_file_path must be provided when actions_input_source is 'file'"
            )
        with open(config.actions_input_file_path, "r") as f:
            actions = [
                a
                for a in [
                    parse_action_prefix(line.strip()) for line in f if line.strip()
                ]
                if a is not None
            ]
    else:
        raise ValueError(
            f"Invalid actions_input_source: {actions_input_source}, should be 'stdin' or 'file'"
        )
    return actions


@hydra.main(version_base=None, config_path="conf", config_name="run_gui_from_actions")
def main(config: DictConfig):
    keymap = {
        pygame.K_a: "move_left",
        pygame.K_d: "move_right",
        pygame.K_w: "move_up",
        pygame.K_s: "move_down",
        pygame.K_e: "rotate_clockwise",
        pygame.K_q: "rotate_counterclockwise",
        pygame.K_SPACE: "do",
        pygame.K_TAB: "sleep",
        pygame.K_r: "place_stone",
        pygame.K_t: "place_table",
        pygame.K_f: "place_furnace",
        pygame.K_p: "place_plant",
        pygame.K_1: "make_wood_pickaxe",
        pygame.K_2: "make_stone_pickaxe",
        pygame.K_3: "make_iron_pickaxe",
        pygame.K_4: "make_wood_sword",
        pygame.K_5: "make_stone_sword",
        pygame.K_6: "make_iron_sword",
    }
    _print_actions(keymap)

    crafter.constants.items["health"]["max"] = config.health
    crafter.constants.items["health"]["initial"] = config.health
    crafter.constants.collect["grass"]["probability"] = (
        config.sapling_from_grass_probability
    )
    known_world_frames_dir = config.planner.known_world_frames_dir
    known_world_frames_mode = str(config.planner.known_world_frames_mode)
    if known_world_frames_dir and config.actions_input_source != "planner":
        raise ValueError(
            "planner.known_world_frames_dir can only be used when "
            "actions_input_source is 'planner'"
        )
    if known_world_frames_mode not in {"step", "task"}:
        raise ValueError(
            "planner.known_world_frames_mode must be either 'step' or 'task'"
        )
    planner_exit_on_completion = bool(config.planner.exit_on_completion)

    size = list(config.size)
    size[0] = size[0] or config.window[0]
    size[1] = size[1] or config.window[1]

    record = (
        pathlib.Path(hydra.utils.to_absolute_path(config.record))
        if config.record
        else None
    )
    env = crafter.Env(
        area=config.area,
        view=config.view,
        size=size,
        length=config.length,
        seed=config.seed,
        spawn_objects=config.runtime.spawn_objects,
        spawn_random_objects=config.runtime.spawn_random_objects,
        move_objects=config.runtime.move_objects,
        move_random_objects=config.runtime.move_random_objects,
        hunger_decreases=config.runtime.hunger_decreases,
        thirst_decreases=config.runtime.thirst_decreases,
        energy_decreases=config.runtime.energy_decreases,
        daylight_cycle=config.runtime.daylight_cycle,
    )
    env_recorded = crafter.Recorder(env, record)
    env_recorded.reset()
    achievements = set()
    duration = 0
    return_ = 0
    was_done = False
    print("Diamonds exist:", env_recorded._world.count("diamond"))

    # Main loop
    pygame.init()
    screen = pygame.display.set_mode(config.window)
    clock = pygame.time.Clock()
    running = True

    actions: list[str] = []
    action_idx = 0
    actions_loaded = False
    planner: TaskMotionPlanner | None = None
    known_world_exporter: KnownWorldFrameExporter | None = None
    known_world_episode_index = 0
    known_world_run_id = _make_known_world_run_id() if known_world_frames_dir else None
    known_world_root = (
        pathlib.Path(hydra.utils.to_absolute_path(known_world_frames_dir))
        if known_world_frames_dir
        else None
    )

    while running:
        if (
            actions_loaded
            and planner is not None
            and planner_exit_on_completion
            and planner.is_finished(env_recorded)
        ):
            print("[planner] exiting run because all configured tasks are complete")
            break

        # Rendering.
        image = env_recorded.render(size)
        if size != config.window:
            image = Image.fromarray(image)
            image = image.resize(config.window, resample=Image.NEAREST)
            image = np.array(image)
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        screen.blit(surface, (0, 0))
        pygame.display.flip()
        clock.tick(config.fps)

        # Keyboard input.
        action = None
        pygame.event.pump()

        # Get actions from user at the beginning of the episode after rendering the initial state
        # so that user can see the initial state before providing actions.
        if not actions_loaded:
            if config.actions_input_source == "planner":
                planner = TaskMotionPlanner(env_recorded, config.planner)
                actions = []
                if planner_exit_on_completion and planner.is_finished(env_recorded):
                    print("[planner] exiting run because all configured tasks are complete")
                    break
                if known_world_root is not None:
                    known_world_exporter = KnownWorldFrameExporter(
                        _make_known_world_episode_dir(
                            known_world_root,
                            known_world_run_id,
                            known_world_episode_index,
                        ),
                        env_recorded._textures,
                        known_world_frames_mode,
                    )
                    known_world_episode_index += 1
                    if known_world_frames_mode == "step":
                        known_world_exporter.save_step_frame(planner, env_recorded)
            else:
                actions = get_actions_from_user(keymap, config)
            actions_loaded = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif (
                action_idx >= len(actions)
                and event.type == pygame.KEYDOWN
                and event.key in keymap
            ):
                action = keymap[event.key]
        if action_idx < len(actions):
            action = actions[action_idx]
            action_idx += 1
        elif planner is not None:
            action = planner.next_action(env_recorded)
            if known_world_exporter is not None:
                known_world_exporter.save_pending_task_frame(planner, env_recorded)
        elif action is None:
            pressed = pygame.key.get_pressed()
            for key, action in keymap.items():
                if pressed[key]:
                    break
            else:
                if config.wait and not env_recorded._player.sleeping:
                    continue
                else:
                    action = "noop"

        # Environment step.
        _, reward, done, _ = env_recorded.step(env_recorded.action_names.index(action))
        if (
            planner is not None
            and known_world_exporter is not None
            and known_world_frames_mode == "step"
        ):
            planner._refresh_local_state(env_recorded)
            known_world_exporter.save_step_frame(planner, env_recorded)
        duration += 1

        # Achievements.
        unlocked = {
            name
            for name, count in env_recorded._player.achievements.items()
            if count > 0 and name not in achievements
        }
        for name in unlocked:
            achievements |= unlocked
            total = len(env_recorded._player.achievements.keys())
            print(f"Achievement ({len(achievements)}/{total}): {name}")
        if env_recorded._step > 0 and env_recorded._step % 100 == 0:
            print(f"Time step: {env_recorded._step}")
        if reward:
            print(f"Reward: {reward}")
            return_ += reward

        # Episode end.
        if done and not was_done:
            was_done = True
            print("Episode done!")
            print("Duration:", duration)
            print("Return:", return_)
            if config.death == "quit":
                running = False
            if config.death == "reset":
                print("\nStarting a new episode.")
                env_recorded.reset()
                achievements = set()
                was_done = False
                duration = 0
                return_ = 0
                actions = []
                action_idx = 0
                actions_loaded = False
                planner = None
                known_world_exporter = None
            if config.death == "continue":
                pass

    pygame.quit()


if __name__ == "__main__":
    main()
