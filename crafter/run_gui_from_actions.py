"""Run the Crafter environment with a series of actions provided by the user."""

import pathlib

import hydra
import numpy as np
import pygame
from omegaconf import DictConfig
from PIL import Image

import crafter


def _print_actions(keymap):
    print("Actions:")
    for key, action in keymap.items():
        print(f"  {pygame.key.name(key)}: {action}")


@hydra.main(version_base=None, config_path="conf", config_name="run_gui_from_actions")
def main(config: DictConfig):
    keymap = {
        pygame.K_a: "move_left",
        pygame.K_d: "move_right",
        pygame.K_w: "move_up",
        pygame.K_s: "move_down",
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
    action_components_list = [
        action.split("_") for action in keymap.values()
    ]  # Split actions into components for future prefix matching
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
    print("Diamonds exist:", env_recorded._world.count("diamond"))

    # Get action prefixes list from user input (either stdin or file) and parse it into actual actions
    def parse_action_prefix(prefix):
        """Parse action prefix and return matching action."""
        prefix = prefix.strip()
        matching_actions = [
            action
            for action in keymap.values()
            if all(
                comp.startswith(p)
                for p, comp in zip(prefix.split("_"), action.split("_"))
            )
            and len(prefix.split("_")) == len(action.split("_"))
        ]
        if len(matching_actions) == 1:
            return matching_actions[0]
        else:
            # print(f"Warning: prefix '{prefix}' matches {len(matching_actions)} actions, skipping")
            # return None
            raise ValueError(
                f"Prefix '{prefix}' is ambiguous and matches multiple actions: {matching_actions}"
            )

    actions_input_source = config.actions_input_source
    if actions_input_source == "stdin":
        print(
            "Enter actions (separated by comma, can be prefix of one of the components of an action, e.g., 'move_left' to 'le')."
            "Press Ctrl+D (Unix) or Ctrl+Z (Windows) to end input."
        )
        action_prefixes = input("Actions: ")
        actions = [
            a
            for a in [parse_action_prefix(p) for p in action_prefixes.split(",")]
            if a is not None
        ]
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

    # Main loop
    pygame.init()
    screen = pygame.display.set_mode(config.window)
    clock = pygame.time.Clock()
    running = True
    action_idx = 0
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
            if action_idx < len(actions):
                # Prioritize scripted actions from user input over keyboard input
                action = actions[action_idx]
                action_idx += 1
            elif event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in keymap:
                action = keymap[event.key]
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
            if config.death == "continue":
                pass

    pygame.quit()


if __name__ == "__main__":
    main()
