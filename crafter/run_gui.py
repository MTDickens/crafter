import concurrent.futures
import json
import pathlib
from datetime import datetime

import hydra
import numpy as np
import pygame
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import crafter
from crafter.known_world import KnownWorld, SimpleWorld
from crafter.task_and_motion_planner import TaskAndMotionPlanner


def _print_actions(keymap):
  print('Actions:')
  for key, action in keymap.items():
    print(f'  {pygame.key.name(key)}: {action}')


def _apply_runtime_constants(config: DictConfig):
  crafter.constants.items['health']['max'] = config.health
  crafter.constants.items['health']['initial'] = config.health
  crafter.constants.collect['grass']['probability'] = (
      config.sapling_from_grass_probability
  )


def _planner_tasks_finished(task_names, achievements) -> bool:
  return all(achievements.get(task_name, 0) > 0 for task_name in task_names)


def _finished_task_count(task_names, achievements) -> int:
  return sum(int(achievements.get(task_name, 0) > 0) for task_name in task_names)


def _selected_planner_metrics(config: DictConfig, planner_trace) -> dict[str, int]:
  metrics = {}
  for result_name in config.planner.result_logging:
    assert (
        result_name == 'reveal_count'
    ), f'Unsupported planner result logging entry: {result_name}'
    metrics[result_name] = planner_trace.result_metrics[result_name]
  return metrics


def _timestamp_string() -> str:
  return datetime.now().strftime('%Y%m%dT%H%M%S')


def _episode_seed(config: DictConfig, episode_index: int):
  if config.seed is None:
    return None
  return int(config.seed) + int(episode_index)


def _resolved_path(path: str | None) -> pathlib.Path | None:
  if path is None:
    return None
  return pathlib.Path(path).expanduser().resolve()


def _env_size(config: DictConfig):
  size = list(config.size)
  size[0] = size[0] or config.window[0]
  size[1] = size[1] or config.window[1]
  return size


def _resource_counts(env_recorded) -> dict[str, int]:
  return {
      resource_name: int(env_recorded._world.count(resource_name))
      for resource_name in config_resource_names()
  }


def config_resource_names():
  return ('tree', 'stone', 'coal', 'iron', 'diamond')


def _resource_requirements_satisfied(config: DictConfig, env_recorded) -> bool:
  for resource_name in config_resource_names():
    required = int(config.map_generation.minimum_resources[resource_name])
    actual = int(env_recorded._world.count(resource_name))
    if actual < required:
      return False
  return True


def _reset_until_resource_requirements_satisfied(config: DictConfig, env_recorded):
  max_attempts = int(config.map_generation.max_reset_attempts)
  last_counts = None
  for attempt in range(1, max_attempts + 1):
    env_recorded.reset()
    last_counts = _resource_counts(env_recorded)
    if _resource_requirements_satisfied(config, env_recorded):
      if attempt > 1:
        print(f'Map generation succeeded on attempt {attempt}/{max_attempts}.')
      return
  raise AssertionError(
      'Failed to generate a map satisfying minimum resource counts after '
      f'{max_attempts} attempts. Last counts: {last_counts}, '
      f'required: {dict(config.map_generation.minimum_resources)}'
  )


def _planner_output_root(config: DictConfig) -> pathlib.Path | None:
  if not config.planner.enabled:
    return None
  root = _resolved_path(config.planner.output_dir)
  root = root / _timestamp_string()
  root.mkdir(parents=True, exist_ok=False)
  return root


def _episode_output_dir(output_root: pathlib.Path | None, episode_index: int):
  if output_root is None:
    return None
  episode_dir = output_root / f'episode-{episode_index:05d}'
  episode_dir.mkdir(parents=True, exist_ok=False)
  return episode_dir


def _episode_record_dir(config: DictConfig, episode_index: int):
  if not config.record:
    return None
  record_root = _resolved_path(config.record)
  if config.episodes == 1 and config.workers == 1:
    return record_root
  return record_root / f'episode-{episode_index:05d}'


def _validate_config(config: DictConfig):
  if config.episodes > 1:
    assert (
        config.death != 'continue'
    ), 'episodes > 1 is unsupported when death == "continue".'
  if not config.headless:
    assert config.workers == 1, 'workers > 1 requires headless mode.'
  if config.headless:
    assert config.planner.enabled, 'headless mode requires planner.enabled = true.'
  if config.planner.enabled and config.planner.render_known_world == 'step':
    raise NotImplementedError('planner.render_known_world=step is not implemented.')
  if config.planner.enabled and config.planner.plan_only:
    assert config.headless, 'planner.plan_only requires headless mode.'
    assert config.record is None, 'planner.plan_only is incompatible with record.'
    assert (
        config.planner.render_known_world == 'none'
    ), 'planner.plan_only is incompatible with planner.render_known_world.'
  assert config.map_generation.max_reset_attempts >= 1, (
      'map_generation.max_reset_attempts must be at least 1.'
  )
  for resource_name in config_resource_names():
    assert config.map_generation.minimum_resources[resource_name] >= 0, (
        f'map_generation.minimum_resources.{resource_name} must be non-negative.'
    )


def _make_env(config: DictConfig, episode_index: int):
  size = _env_size(config)
  env = crafter.Env(
      area=config.area,
      view=config.view,
      size=size,
      length=config.length,
      seed=_episode_seed(config, episode_index),
      spawn_objects=config.runtime.spawn_objects,
      spawn_random_objects=config.runtime.spawn_random_objects,
      move_objects=config.runtime.move_objects,
      move_random_objects=config.runtime.move_random_objects,
      hunger_decreases=config.runtime.hunger_decreases,
      thirst_decreases=config.runtime.thirst_decreases,
      energy_decreases=config.runtime.energy_decreases,
      daylight_cycle=config.runtime.daylight_cycle,
  )
  record = _episode_record_dir(config, episode_index)
  return crafter.Recorder(env, record)


def _build_planner_trace(config: DictConfig, env_recorded):
  planner_name = config.planner.name
  assert planner_name == 'vanilla', f'Unsupported planner implementation: {planner_name}'
  known_world = KnownWorld(
      initial_world=SimpleWorld.from_world(env_recorded._world),
      player_pos=env_recorded._player.pos,
      inventory=env_recorded._player.inventory.copy(),
      facing=tuple(env_recorded._player.facing),
  )
  planner = TaskAndMotionPlanner(config.planner.task_order)
  planner_trace = planner.plan_with_trace(known_world)
  print(f'Planner ({planner_name}) produced {len(planner_trace.actions)} actions.')
  for name, value in _selected_planner_metrics(config, planner_trace).items():
    print(f'Planner result ({name}): {value}')
  return planner_trace


def _snapshot_specs(planner_trace):
  specs = [(0, '000-initial', planner_trace.initial_known_mask)]
  next_index = 1
  for boundary in planner_trace.task_boundaries:
    specs.append((
        boundary.pre_action_index,
        f'{next_index:03d}-{boundary.task_name}-pre',
        boundary.pre_action_mask,
    ))
    next_index += 1
    specs.append((
        boundary.post_action_index,
        f'{next_index:03d}-{boundary.task_name}-post',
        boundary.post_action_mask,
    ))
    next_index += 1
  return specs


def _blit_texture(canvas: np.ndarray, pos, texture: np.ndarray):
  x, y = pos
  width, height = texture.shape[:2]
  if texture.shape[-1] == 4:
    alpha = texture[..., 3:].astype(np.float32) / 255
    foreground = texture[..., :3].astype(np.float32) / 255
    background = canvas[x:x + width, y:y + height].astype(np.float32) / 255
    blended = alpha * foreground + (1 - alpha) * background
    canvas[x:x + width, y:y + height] = (255 * blended).astype(np.uint8)
    return
  canvas[x:x + width, y:y + height] = texture[..., :3]


def _render_known_world_image(env_recorded, known_mask: np.ndarray, tile_size: int = 16):
  world = env_recorded._world
  textures = env_recorded._textures
  unit = np.array((tile_size, tile_size))
  canvas = np.zeros((world.area[0] * tile_size, world.area[1] * tile_size, 3), np.uint8) + 127
  for x in range(world.area[0]):
    for y in range(world.area[1]):
      if not known_mask[x, y]:
        continue
      texture = textures.get(world[(x, y)][0], unit)
      _blit_texture(canvas, (x * tile_size, y * tile_size), texture)
  player_pos = tuple(env_recorded._player.pos)
  if known_mask[player_pos]:
    texture = textures.get(env_recorded._player.texture, unit)
    _blit_texture(canvas, (player_pos[0] * tile_size, player_pos[1] * tile_size), texture)
  return canvas.transpose((1, 0, 2))


def _save_known_world_snapshot(episode_dir: pathlib.Path, snapshot_name: str, known_mask: np.ndarray, env_recorded):
  mask_dir = episode_dir / 'known_world_masks'
  image_dir = episode_dir / 'known_world_images'
  mask_dir.mkdir(parents=True, exist_ok=True)
  image_dir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(mask_dir / f'{snapshot_name}.npz', known_mask=known_mask)
  image = _render_known_world_image(env_recorded, known_mask)
  Image.fromarray(image).save(image_dir / f'{snapshot_name}.png')


def _flush_task_snapshots(pending_snapshots: dict[int, list], action_count: int, episode_dir: pathlib.Path, env_recorded):
  for snapshot_name, known_mask in pending_snapshots.pop(action_count, []):
    _save_known_world_snapshot(episode_dir, snapshot_name, known_mask, env_recorded)


def _planner_result_payload(config: DictConfig, planner_trace, episode_index: int, executed_summary: dict | None):
  payload = {
      'episode_index': episode_index,
      'seed': _episode_seed(config, episode_index),
      'planner_name': config.planner.name,
      'plan_only': bool(config.planner.plan_only),
      'planned_action_count': len(planner_trace.actions),
      'metrics': _selected_planner_metrics(config, planner_trace),
  }
  if executed_summary is not None:
    payload.update(executed_summary)
  return payload


def _write_planner_results(episode_dir: pathlib.Path | None, payload: dict):
  if episode_dir is None:
    return
  with (episode_dir / 'planner_results.json').open('w') as file:
    json.dump(payload, file, indent=2, sort_keys=True)


def _print_step_updates(env_recorded, achievements: set[str], duration: int, return_: float):
  unlocked = {
      name
      for name, count in env_recorded._player.achievements.items()
      if count > 0 and name not in achievements
  }
  for name in unlocked:
    achievements |= unlocked
    total = len(env_recorded._player.achievements.keys())
    print(f'Achievement ({len(achievements)}/{total}): {name}')
  if env_recorded._step > 0 and env_recorded._step % 100 == 0:
    print(f'Time step: {env_recorded._step}')
  reward = env_recorded._player.health - env_recorded._last_health
  if reward:
    print(f'Health delta since last env step state: {reward}')
  del duration, return_


def _episode_execution_summary(config: DictConfig, env_recorded, duration: int, return_: float):
  task_names = list(config.planner.task_order) if config.planner.enabled else []
  summary = {
      'duration_steps': duration,
      'return': return_,
  }
  if config.planner.enabled:
    finished_task_count = _finished_task_count(task_names, env_recorded._player.achievements)
    summary.update({
        'finished_task_count': finished_task_count,
        'task_count': len(task_names),
        'finished_all_tasks': finished_task_count == len(task_names),
    })
  return summary


def _run_headless_episode(config: DictConfig, env_recorded, planner_trace, episode_dir: pathlib.Path | None):
  planner_actions = list(planner_trace.actions)
  pending_snapshots = {}
  if config.planner.render_known_world == 'task':
    assert episode_dir is not None, 'Task-level rendering requires an episode output directory.'
    for action_index, snapshot_name, known_mask in _snapshot_specs(planner_trace):
      pending_snapshots.setdefault(action_index, []).append((snapshot_name, known_mask))
    _flush_task_snapshots(pending_snapshots, 0, episode_dir, env_recorded)
  if config.planner.plan_only:
    return {
        'duration_steps': None,
        'return': None,
        'finished_task_count': None,
        'task_count': None,
        'finished_all_tasks': None,
    }

  duration = 0
  return_ = 0.0
  planner_action_idx = 0
  while True:
    if planner_action_idx < len(planner_actions):
      action = planner_actions[planner_action_idx]
      planner_action_idx += 1
    elif config.planner.exit_on_finish:
      assert _planner_tasks_finished(
          config.planner.task_order, env_recorded._player.achievements
      ), 'Planner exhausted its actions before finishing all configured tasks.'
      print('Planner finished all configured tasks.')
      break
    else:
      action = 'noop'

    _, reward, done, _ = env_recorded.step(env_recorded.action_names.index(action))
    duration += 1
    return_ += reward
    _flush_task_snapshots(pending_snapshots, planner_action_idx, episode_dir, env_recorded)
    if config.planner.exit_on_finish and _planner_tasks_finished(
        config.planner.task_order, env_recorded._player.achievements
    ):
      print('Planner finished all configured tasks.')
      break
    if done:
      print('Episode done!')
      print('Duration:', duration)
      print('Return:', return_)
      break

  return _episode_execution_summary(config, env_recorded, duration, return_)


def _run_gui_episode(
    config: DictConfig,
    env_recorded,
    planner_trace,
    episode_dir: pathlib.Path | None,
    screen,
    clock,
    keymap,
):
  planner_actions = list(planner_trace.actions) if planner_trace is not None else []
  planner_action_idx = 0
  duration = 0
  return_ = 0.0
  achievements: set[str] = set()
  was_done = False
  pending_snapshots = {}
  if planner_trace is not None and config.planner.render_known_world == 'task':
    assert episode_dir is not None, 'Task-level rendering requires an episode output directory.'
    for action_index, snapshot_name, known_mask in _snapshot_specs(planner_trace):
      pending_snapshots.setdefault(action_index, []).append((snapshot_name, known_mask))
    _flush_task_snapshots(pending_snapshots, 0, episode_dir, env_recorded)

  running = True
  user_stopped = False
  while running:
    image = env_recorded.render(_env_size(config))
    if tuple(image.shape[:2][::-1]) != tuple(config.window):
      image = np.array(
          Image.fromarray(image).resize(config.window, resample=Image.NEAREST)
      )
    surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
    screen.blit(surface, (0, 0))
    pygame.display.flip()
    clock.tick(config.fps)

    action = None
    pygame.event.pump()
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
        user_stopped = True
      elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
        running = False
        user_stopped = True
      elif event.type == pygame.KEYDOWN and event.key in keymap:
        action = keymap[event.key]

    if planner_trace is not None and planner_action_idx < len(planner_actions):
      action = planner_actions[planner_action_idx]
      planner_action_idx += 1
    elif (
        planner_trace is not None
        and config.planner.exit_on_finish
        and planner_action_idx >= len(planner_actions)
    ):
      assert _planner_tasks_finished(
          config.planner.task_order, env_recorded._player.achievements
      ), 'Planner exhausted its actions before finishing all configured tasks.'
      print('Planner finished all configured tasks.')
      break

    if action is None:
      pressed = pygame.key.get_pressed()
      for key, mapped_action in keymap.items():
        if pressed[key]:
          action = mapped_action
          break
      if action is None:
        if config.wait and not env_recorded._player.sleeping:
          continue
        action = 'noop'

    _, reward, done, _ = env_recorded.step(env_recorded.action_names.index(action))
    duration += 1
    return_ += reward
    _flush_task_snapshots(pending_snapshots, planner_action_idx, episode_dir, env_recorded)

    unlocked = {
        name
        for name, count in env_recorded._player.achievements.items()
        if count > 0 and name not in achievements
    }
    for name in unlocked:
      achievements |= unlocked
      total = len(env_recorded._player.achievements.keys())
      print(f'Achievement ({len(achievements)}/{total}): {name}')
    if env_recorded._step > 0 and env_recorded._step % 100 == 0:
      print(f'Time step: {env_recorded._step}')
    if reward:
      print(f'Reward: {reward}')
    if planner_trace is not None and config.planner.exit_on_finish and _planner_tasks_finished(
        config.planner.task_order, env_recorded._player.achievements
    ):
      print('Planner finished all configured tasks.')
      break

    if done and not was_done:
      was_done = True
      print('Episode done!')
      print('Duration:', duration)
      print('Return:', return_)
      if config.death != 'continue':
        break

  return _episode_execution_summary(config, env_recorded, duration, return_), user_stopped


def _run_single_episode(config: DictConfig, episode_index: int, episode_dir: pathlib.Path | None, screen=None, clock=None):
  _apply_runtime_constants(config)
  env_recorded = _make_env(config, episode_index)
  _reset_until_resource_requirements_satisfied(config, env_recorded)
  print('Resource counts:', _resource_counts(env_recorded))
  planner_trace = _build_planner_trace(config, env_recorded) if config.planner.enabled else None

  if config.headless:
    assert planner_trace is not None, 'Headless execution requires a planner trace.'
    execution_summary = _run_headless_episode(config, env_recorded, planner_trace, episode_dir)
    user_stopped = False
  else:
    execution_summary, user_stopped = _run_gui_episode(
        config=config,
        env_recorded=env_recorded,
        planner_trace=planner_trace,
        episode_dir=episode_dir,
        screen=screen,
        clock=clock,
        keymap={
            pygame.K_a: 'move_left',
            pygame.K_d: 'move_right',
            pygame.K_w: 'move_up',
            pygame.K_s: 'move_down',
            pygame.K_q: 'rotate_counterclockwise',
            pygame.K_e: 'rotate_clockwise',
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
        },
    )

  if planner_trace is not None:
    _write_planner_results(
        episode_dir,
        _planner_result_payload(config, planner_trace, episode_index, execution_summary),
    )
  return user_stopped


def _run_single_episode_worker(config_dict: dict, episode_index: int, output_root: str | None):
  config = OmegaConf.create(config_dict)
  episode_dir = _episode_output_dir(pathlib.Path(output_root) if output_root else None, episode_index)
  _run_single_episode(config, episode_index, episode_dir)


@hydra.main(version_base=None, config_path='conf', config_name='run_gui')
def main(config: DictConfig):
  _validate_config(config)
  keymap = {
      pygame.K_a: 'move_left',
      pygame.K_d: 'move_right',
      pygame.K_w: 'move_up',
      pygame.K_s: 'move_down',
      pygame.K_q: 'rotate_counterclockwise',
      pygame.K_e: 'rotate_clockwise',
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

  output_root = _planner_output_root(config)
  if config.headless:
    config_dict = OmegaConf.to_container(config, resolve=True)
    if config.workers == 1:
      for episode_index in range(config.episodes):
        episode_dir = _episode_output_dir(output_root, episode_index)
        _run_single_episode(config, episode_index, episode_dir)
      return
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.workers) as executor:
      futures = [
          executor.submit(
              _run_single_episode_worker,
              config_dict,
              episode_index,
              str(output_root) if output_root else None,
          )
          for episode_index in range(config.episodes)
      ]
      for future in concurrent.futures.as_completed(futures):
        future.result()
    return

  pygame.init()
  screen = pygame.display.set_mode(config.window)
  clock = pygame.time.Clock()
  try:
    for episode_index in range(config.episodes):
      episode_dir = _episode_output_dir(output_root, episode_index)
      user_stopped = _run_single_episode(config, episode_index, episode_dir, screen=screen, clock=clock)
      if user_stopped:
        break
  finally:
    pygame.quit()


if __name__ == '__main__':
  main()
