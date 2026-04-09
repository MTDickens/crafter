import pathlib

import hydra
import numpy as np
try:
  import pygame
except ImportError:
  print('Please install the pygame package to use the GUI.')
  raise
from omegaconf import DictConfig
from PIL import Image

import crafter


def _print_actions(keymap):
  print('Actions:')
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')


@hydra.main(version_base=None, config_path='conf', config_name='run_gui')
def main(config: DictConfig):
  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_SPACE: 'do',
      pygame.K_TAB: 'sleep',

      pygame.K_r: 'place_stone',
      pygame.K_t: 'place_table',
      pygame.K_f: 'place_furnace',
      pygame.K_p: 'place_plant',

      pygame.K_1: 'make_wood_pickaxe',
      pygame.K_2: 'make_stone_pickaxe',
      pygame.K_3: 'make_iron_pickaxe',
      pygame.K_4: 'make_wood_sword',
      pygame.K_5: 'make_stone_sword',
      pygame.K_6: 'make_iron_sword',
  }
  _print_actions(keymap)

  crafter.constants.items['health']['max'] = config.health
  crafter.constants.items['health']['initial'] = config.health

  size = list(config.size)
  size[0] = size[0] or config.window[0]
  size[1] = size[1] or config.window[1]

  record = (
      pathlib.Path(hydra.utils.to_absolute_path(config.record))
      if config.record else None)
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
      daylight_cycle=config.runtime.daylight_cycle)
  env = crafter.Recorder(env, record)
  env.reset()
  achievements = set()
  duration = 0
  return_ = 0
  was_done = False
  print('Diamonds exist:', env._world.count('diamond'))

  pygame.init()
  screen = pygame.display.set_mode(config.window)
  clock = pygame.time.Clock()
  running = True
  while running:

    # Rendering.
    image = env.render(size)
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
    if action is None:
      pressed = pygame.key.get_pressed()
      for key, action in keymap.items():
        if pressed[key]:
          break
      else:
        if config.wait and not env._player.sleeping:
          continue
        else:
          action = 'noop'

    # Environment step.
    _, reward, done, _ = env.step(env.action_names.index(action))
    duration += 1

    # Achievements.
    unlocked = {
        name for name, count in env._player.achievements.items()
        if count > 0 and name not in achievements}
    for name in unlocked:
      achievements |= unlocked
      total = len(env._player.achievements.keys())
      print(f'Achievement ({len(achievements)}/{total}): {name}')
    if env._step > 0 and env._step % 100 == 0:
      print(f'Time step: {env._step}')
    if reward:
      print(f'Reward: {reward}')
      return_ += reward

    # Episode end.
    if done and not was_done:
      was_done = True
      print('Episode done!')
      print('Duration:', duration)
      print('Return:', return_)
      if config.death == 'quit':
        running = False
      if config.death == 'reset':
        print('\nStarting a new episode.')
        env.reset()
        achievements = set()
        was_done = False
        duration = 0
        return_ = 0
      if config.death == 'continue':
        pass

  pygame.quit()


if __name__ == '__main__':
  main()
