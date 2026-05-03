import concurrent.futures
import json
import pathlib
from datetime import datetime

import hydra
import numpy as np
import pygame
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import crafter
from crafter.known_world import KnownWorld, SimpleWorld
from crafter.skills.crafter_patterns import CrafterSkillLibrary
from crafter.training.crafter_pattern_learning import CrafterSkillLearningManager
from crafter.task_and_motion_planner import TaskAndMotionPlanner
from crafter.utils.crafter_codec import CrafterTileCodec


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
  assert config.map_generation.noise_scale > 0, (
      'map_generation.noise_scale must be positive.'
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
      worldgen_noise_scale=config.map_generation.noise_scale,
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
  assert planner_name in {'vanilla', 'pattern_learning'}, (
      f'Unsupported planner implementation: {planner_name}')
  return _build_planner_trace_with_manager(config, env_recorded, skill_learning_manager=None)


def _build_planner_trace_with_manager(
    config: DictConfig,
    env_recorded,
    skill_learning_manager: CrafterSkillLearningManager | None,
):
  """Build one planner trace, optionally with online pattern inference."""
  planner_name = config.planner.name
  known_world = KnownWorld(
      initial_world=SimpleWorld.from_world(env_recorded._world),
      player_pos=env_recorded._player.pos,
      inventory=env_recorded._player.inventory.copy(),
      facing=tuple(env_recorded._player.facing),
  )
  inference_library = None
  skill_learning_cfg = None
  if planner_name == 'pattern_learning':
    assert skill_learning_manager is not None, 'pattern_learning requires a skill-learning manager.'
    if skill_learning_manager.should_use_inference():
      inference_library = skill_learning_manager.library
    skill_learning_cfg = config.planner.skill_learning
  planner = TaskAndMotionPlanner(
      config.planner.task_order,
      random=np.random.RandomState(env_recorded._world.random.randint(0, 2 ** 31 - 1)),
      inference_library=inference_library,
      skill_learning_cfg=skill_learning_cfg,
  )
  planner_trace = planner.plan_with_trace(known_world)
  print(f'Planner ({planner_name}) produced {len(planner_trace.actions)} actions.')
  for name, value in _selected_planner_metrics(config, planner_trace).items():
    print(f'Planner result ({name}): {value}')
  return planner_trace, known_world


def _rebuild_pattern_learning_manager(
    config: DictConfig,
    library_state: dict | None,
) -> CrafterSkillLearningManager:
  """Rebuild a worker-local pattern-learning manager from a library snapshot."""
  manager = CrafterSkillLearningManager(config.planner.skill_learning)
  if library_state is None:
    return manager
  manager.library = CrafterSkillLibrary(
      codec=manager.codec,
      patterns=list(library_state['patterns']),
      raw_weights=library_state['raw_weights'],
      device=torch.device(str(config.planner.skill_learning.device)),
  )
  return manager


def _pattern_learning_group_endpoints(config: DictConfig) -> list[int]:
  """Return all episode indices that end a state-sharing worker group."""
  if not (config.planner.enabled and config.planner.name == 'pattern_learning'):
    return list(range(int(config.episodes)))
  manager = CrafterSkillLearningManager(config.planner.skill_learning)
  endpoints = set()
  total_episodes = int(config.episodes)
  for episode_index in range(total_episodes - 1):
    if manager.should_trigger_proposal(episode_index, total_episodes):
      endpoints.add(episode_index)
    if manager.should_trigger_reweight(episode_index, total_episodes):
      endpoints.add(episode_index)
  endpoints.add(total_episodes - 1)
  return sorted(endpoints)


def _pattern_learning_episode_groups(config: DictConfig) -> list[list[int]]:
  """Partition episodes into concurrency-safe groups."""
  endpoints = _pattern_learning_group_endpoints(config)
  groups = []
  start = 0
  for endpoint in endpoints:
    groups.append(list(range(start, endpoint + 1)))
    start = endpoint + 1
  return groups


def _make_pattern_learning_replay_payload(
    known_world: KnownWorld,
    start_pos: tuple[int, int],
    episode_index: int,
    seed: int | None,
    codec,
) -> dict:
  """Serialize one replay entry for parent-process aggregation."""
  return {
      'full_map_ids': codec.encode_simple_world(known_world.initial_world, device='cpu').numpy(),
      'final_known_mask': known_world.perceived_mask,
      'start_pos': tuple(start_pos),
      'episode_index': int(episode_index),
      'seed': seed,
  }


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


def _save_final_full_map(episode_dir: pathlib.Path, env_recorded):
  image_dir = episode_dir / 'known_world_images'
  image_dir.mkdir(parents=True, exist_ok=True)
  known_mask = np.ones(env_recorded._world.area, dtype=bool)
  image = _render_known_world_image(env_recorded, known_mask)
  Image.fromarray(image).save(image_dir / 'final-full-map.png')


def _map_ids_to_text(map_ids: np.ndarray, codec: CrafterTileCodec) -> str:
  rows = []
  for y in range(map_ids.shape[1]):
    tokens = []
    for x in range(map_ids.shape[0]):
      value = int(map_ids[x, y])
      if value == codec.unknown_id:
        tokens.append('unknown')
      else:
        tokens.append(codec.decode_material(value))
    rows.append(' '.join(tokens))
  return '\n'.join(rows) + '\n'


def _save_planner_map_artifacts(
    episode_dir: pathlib.Path | None,
    known_world: KnownWorld | None,
    codec: CrafterTileCodec,
) -> dict | None:
  if episode_dir is None or known_world is None:
    return None
  map_dir = episode_dir / 'maps'
  map_dir.mkdir(parents=True, exist_ok=True)
  ground_truth_ids = (
      codec.encode_simple_world(known_world.initial_world, device='cpu')
      .detach()
      .cpu()
      .numpy()
  )
  perceived_mask = known_world.perceived_mask
  imputed_mask = known_world.imputed_mask
  total_mask = known_world.known_mask
  perceived_ids = np.full_like(ground_truth_ids, fill_value=codec.unknown_id)
  imputed_ids = np.full_like(ground_truth_ids, fill_value=codec.unknown_id)
  total_ids = np.full_like(ground_truth_ids, fill_value=codec.unknown_id)
  perceived_ids[perceived_mask] = ground_truth_ids[perceived_mask]
  total_ids[perceived_mask] = ground_truth_ids[perceived_mask]
  for x, y in zip(*np.nonzero(imputed_mask), strict=True):
    pos = (int(x), int(y))
    material_id = codec.encode_material(known_world.imputed_material_at(pos))
    imputed_ids[pos] = material_id
    total_ids[pos] = material_id
  np.savez_compressed(
      map_dir / 'maps.npz',
      ground_truth_ids=ground_truth_ids,
      perceived_ids=perceived_ids,
      perceived_mask=perceived_mask,
      imputed_ids=imputed_ids,
      imputed_mask=imputed_mask,
      total_ids=total_ids,
      total_mask=total_mask,
      materials=np.array(codec.materials),
      unknown_id=np.array(codec.unknown_id, dtype=np.int64),
  )
  text_maps = {
      'ground_truth': ground_truth_ids,
      'perceived': perceived_ids,
      'imputed': imputed_ids,
      'total': total_ids,
  }
  for name, map_ids in text_maps.items():
    (map_dir / f'{name}.txt').write_text(_map_ids_to_text(map_ids, codec))
  return {
      'npz': 'maps/maps.npz',
      'ground_truth_text': 'maps/ground_truth.txt',
      'perceived_text': 'maps/perceived.txt',
      'imputed_text': 'maps/imputed.txt',
      'total_text': 'maps/total.txt',
  }


def _flush_task_snapshots(pending_snapshots: dict[int, list], action_count: int, episode_dir: pathlib.Path, env_recorded):
  for snapshot_name, known_mask in pending_snapshots.pop(action_count, []):
    _save_known_world_snapshot(episode_dir, snapshot_name, known_mask, env_recorded)


def _pattern_key(codec, pattern) -> str:
  center = codec.decode_material(pattern.center_id)
  if not pattern.use_gating:
    return f'ungated:center={center}'
  top = codec.decode_material(pattern.top_id)
  bottom = codec.decode_material(pattern.bottom_id)
  left = codec.decode_material(pattern.left_id)
  right = codec.decode_material(pattern.right_id)
  return (
      f'gated:center={center},top={top},bottom={bottom},left={left},right={right}'
  )


def _pattern_learning_library_payload(
    skill_learning_manager: CrafterSkillLearningManager | None,
) -> dict | None:
  if skill_learning_manager is None:
    return None
  library = skill_learning_manager.library
  codec = skill_learning_manager.codec
  positive_weights = library.positive_weights().detach().cpu().tolist()
  raw_weights = library.raw_weights.detach().cpu().tolist()
  positive_weight_sum = float(sum(float(value) for value in positive_weights))
  patterns = []
  for index, pattern in enumerate(library.patterns):
    center = codec.decode_material(pattern.center_id)
    if pattern.use_gating:
      top = codec.decode_material(pattern.top_id)
      bottom = codec.decode_material(pattern.bottom_id)
      left = codec.decode_material(pattern.left_id)
      right = codec.decode_material(pattern.right_id)
      pattern_kind = 'gated_cross'
    else:
      top = bottom = left = right = None
      pattern_kind = 'ungated_center_only'
    patterns.append({
        'pattern_index': index,
        'pattern_key': _pattern_key(codec, pattern),
        'kind': pattern_kind,
        'use_gating': bool(pattern.use_gating),
        'center': center,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right,
        'raw_weight': float(raw_weights[index]),
        'positive_weight': float(positive_weights[index]),
        'normalized_weight': float(positive_weights[index]) / positive_weight_sum,
    })
  return {
      'num_patterns': len(patterns),
      'patterns': patterns,
  }


def _planner_result_payload(
    config: DictConfig,
    planner_trace,
    episode_index: int,
    executed_summary: dict | None,
    skill_learning_manager: CrafterSkillLearningManager | None = None,
    map_paths: dict | None = None,
):
  payload = {
      'episode_index': episode_index,
      'seed': _episode_seed(config, episode_index),
      'planner_name': config.planner.name,
      'plan_only': bool(config.planner.plan_only),
      'planned_action_count': len(planner_trace.actions),
      'metrics': _selected_planner_metrics(config, planner_trace),
  }
  pattern_learning_library = _pattern_learning_library_payload(skill_learning_manager)
  if pattern_learning_library is not None:
    payload['pattern_learning_library'] = pattern_learning_library
  if map_paths is not None:
    payload['map_paths'] = map_paths
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


def _run_single_episode(
    config: DictConfig,
    episode_index: int,
    episode_dir: pathlib.Path | None,
    screen=None,
    clock=None,
    skill_learning_manager: CrafterSkillLearningManager | None = None,
    return_pattern_learning_replay_payload: bool = False,
):
  _apply_runtime_constants(config)
  env_recorded = _make_env(config, episode_index)
  _reset_until_resource_requirements_satisfied(config, env_recorded)
  print('Resource counts:', _resource_counts(env_recorded))
  start_pos = tuple(int(x) for x in env_recorded._player.pos)
  planner_trace = None
  known_world = None
  local_skill_learning_manager = skill_learning_manager
  if config.planner.enabled:
    if config.planner.name == 'pattern_learning' and local_skill_learning_manager is None:
      local_skill_learning_manager = CrafterSkillLearningManager(config.planner.skill_learning)
    planner_trace, known_world = _build_planner_trace_with_manager(
        config,
        env_recorded,
        skill_learning_manager=local_skill_learning_manager,
    )

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
    if config.planner.render_final_full_map:
      assert episode_dir is not None, 'Final full-map rendering requires an episode output directory.'
      _save_final_full_map(episode_dir, env_recorded)
    codec = (
        local_skill_learning_manager.codec
        if local_skill_learning_manager is not None
        else CrafterTileCodec.for_worldgen_materials()
    )
    map_paths = _save_planner_map_artifacts(episode_dir, known_world, codec)
    _write_planner_results(
        episode_dir,
        _planner_result_payload(
            config,
            planner_trace,
            episode_index,
            execution_summary,
            skill_learning_manager=local_skill_learning_manager,
            map_paths=map_paths,
        ),
    )
  if config.planner.enabled and config.planner.name == 'pattern_learning':
    assert known_world is not None
    if local_skill_learning_manager is None:
      local_skill_learning_manager = CrafterSkillLearningManager(config.planner.skill_learning)
    replay_payload = _make_pattern_learning_replay_payload(
        known_world=known_world,
        start_pos=start_pos,
        episode_index=episode_index,
        seed=_episode_seed(config, episode_index),
        codec=local_skill_learning_manager.codec,
    )
    if return_pattern_learning_replay_payload:
      return user_stopped, replay_payload
    assert skill_learning_manager is not None
    skill_learning_manager.add_replay_entry(
        full_map_ids=torch.as_tensor(replay_payload['full_map_ids'], dtype=torch.long),
        final_known_mask=torch.as_tensor(replay_payload['final_known_mask'], dtype=torch.bool),
        start_pos=tuple(replay_payload['start_pos']),
        episode_index=int(replay_payload['episode_index']),
        seed=replay_payload['seed'],
    )
    trained = skill_learning_manager.maybe_train(
        episode_index=episode_index,
        total_episodes=int(config.episodes),
    )
    if trained:
      print(f'Updated pattern-learning library after episode {episode_index}.')
  return user_stopped, None


def _run_single_episode_worker(
    config_dict: dict,
    episode_index: int,
    output_root: str | None,
    pattern_learning_library_state: dict | None = None,
):
  config = OmegaConf.create(config_dict)
  episode_dir = _episode_output_dir(pathlib.Path(output_root) if output_root else None, episode_index)
  skill_learning_manager = None
  if config.planner.enabled and config.planner.name == 'pattern_learning':
    skill_learning_manager = _rebuild_pattern_learning_manager(
        config,
        pattern_learning_library_state,
    )
  _, replay_payload = _run_single_episode(
      config,
      episode_index,
      episode_dir,
      skill_learning_manager=skill_learning_manager,
      return_pattern_learning_replay_payload=(
          config.planner.enabled and config.planner.name == 'pattern_learning'
      ),
  )
  return replay_payload


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
  skill_learning_manager = None
  if config.planner.enabled and config.planner.name == 'pattern_learning':
    skill_learning_manager = CrafterSkillLearningManager(config.planner.skill_learning)
    skill_learning_manager.proposal_output_root = output_root
  if config.headless:
    config_dict = OmegaConf.to_container(config, resolve=True)
    if config.workers == 1:
      for episode_index in range(config.episodes):
        episode_dir = _episode_output_dir(output_root, episode_index)
        _run_single_episode(
            config,
            episode_index,
            episode_dir,
            skill_learning_manager=skill_learning_manager,
        )
      return
    if config.planner.name == 'pattern_learning':
      with concurrent.futures.ProcessPoolExecutor(max_workers=config.workers) as executor:
        for episode_group in _pattern_learning_episode_groups(config):
          library_state = skill_learning_manager.export_library_state()
          futures = [
              executor.submit(
                  _run_single_episode_worker,
                  config_dict,
                  episode_index,
                  str(output_root) if output_root else None,
                  library_state,
              )
              for episode_index in episode_group
          ]
          replay_payloads = [future.result() for future in futures]
          replay_payloads = [payload for payload in replay_payloads if payload is not None]
          replay_payloads.sort(key=lambda payload: payload['episode_index'])
          for payload in replay_payloads:
            skill_learning_manager.add_replay_entry(
                full_map_ids=torch.as_tensor(payload['full_map_ids'], dtype=torch.long),
                final_known_mask=torch.as_tensor(payload['final_known_mask'], dtype=torch.bool),
                start_pos=tuple(payload['start_pos']),
                episode_index=int(payload['episode_index']),
                seed=payload['seed'],
            )
          last_episode_index = episode_group[-1]
          trained = skill_learning_manager.maybe_train(
              episode_index=last_episode_index,
              total_episodes=int(config.episodes),
          )
          if trained:
            print(f'Updated pattern-learning library after episode {last_episode_index}.')
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
      user_stopped, _ = _run_single_episode(
          config,
          episode_index,
          episode_dir,
          screen=screen,
          clock=clock,
          skill_learning_manager=skill_learning_manager,
      )
      if user_stopped:
        break
  finally:
    pygame.quit()


if __name__ == '__main__':
  main()
