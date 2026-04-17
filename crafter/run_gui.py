import pathlib

import hydra
import numpy as np
import pygame
from omegaconf import DictConfig
from PIL import Image

import crafter
from crafter.known_world import KnownWorld, SimpleWorld
from crafter.task_and_motion_planner import TaskAndMotionPlanner


def _print_actions(keymap):
    print("Actions:")
    for key, action in keymap.items():
        print(f"  {pygame.key.name(key)}: {action}")


def _planner_tasks_finished(config: DictConfig, env_recorded) -> bool:
    return all(
        env_recorded._player.achievements.get(task_name, 0) > 0
        for task_name in config.planner.task_order
    )


def _log_planner_results(config: DictConfig, known_world: KnownWorld):
    for result_name in config.planner.result_logging:
        assert (
            result_name == "reveal_count"
        ), f"Unsupported planner result logging entry: {result_name}"
        print(f"Planner result ({result_name}): {known_world.revealed_cell_count}")


def _build_planner_actions(config: DictConfig, env_recorded) -> list[str]:
    planner_name = config.planner.name
    assert planner_name == "vanilla", f"Unsupported planner implementation: {planner_name}"
    known_world = KnownWorld(
        initial_world=SimpleWorld.from_world(env_recorded._world),
        player_pos=env_recorded._player.pos,
        inventory=env_recorded._player.inventory.copy(),
        facing=tuple(env_recorded._player.facing),
    )
    planner = TaskAndMotionPlanner(config.planner.task_order)
    actions: list[str] = planner.plan(known_world)
    print(f"Planner ({planner_name}) produced {len(actions)} actions.")
    _log_planner_results(config, known_world)
    return actions


@hydra.main(version_base=None, config_path="conf", config_name="run_gui")
def main(config: DictConfig):
    keymap = {
        pygame.K_a: "move_left",
        pygame.K_d: "move_right",
        pygame.K_w: "move_up",
        pygame.K_s: "move_down",
        pygame.K_q: "rotate_counterclockwise",
        pygame.K_e: "rotate_clockwise",
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
    planner_actions: list[str] = []
    planner_action_idx = 0
    print("Diamonds exist:", env_recorded._world.count("diamond"))
    if config.planner.enabled:
        planner_actions: list[str] = _build_planner_actions(config, env_recorded)

    pygame.init()
    screen = pygame.display.set_mode(config.window)
    clock = pygame.time.Clock()
    running = True
    while running:
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap:
                action = keymap[event.key]
        if config.planner.enabled and planner_action_idx < len(planner_actions):
            action = planner_actions[planner_action_idx]
            planner_action_idx += 1
        elif (
            config.planner.enabled
            and config.planner.exit_on_finish
            and planner_action_idx >= len(planner_actions)
        ):
            assert _planner_tasks_finished(
                config, env_recorded
            ), "Planner exhausted its actions before finishing all configured tasks."
            print("Planner finished all configured tasks.")
            running = False
            continue
        if action is None:
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
        if config.planner.enabled and config.planner.exit_on_finish and _planner_tasks_finished(
            config, env_recorded
        ):
            print("Planner finished all configured tasks.")
            running = False
            continue

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
                planner_actions = []
                planner_action_idx = 0
                if config.planner.enabled:
                    planner_actions = _build_planner_actions(config, env_recorded)
            if config.death == "continue":
                pass

    pygame.quit()


if __name__ == "__main__":
    main()
