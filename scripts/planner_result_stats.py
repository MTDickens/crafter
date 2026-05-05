"""Summarize and plot Crafter planner-result directories.

This script reads one or more timestamped directories under `planner_results/`
and supports two operations:

1. Compute simple aggregate statistics for selected metrics. Currently only the
   mean is implemented.
2. Plot one selected metric across episodes for multiple timestamps, either on
   the same axes or on separate axes with shared limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont


_PLACEHOLDER_MATERIALS = (
  'water',
  'grass',
  'stone',
  'path',
  'sand',
  'tree',
  'lava',
  'coal',
  'iron',
  'diamond',
)


def _episode_dirs(timestamp_dir: Path) -> list[Path]:
  episode_dirs = sorted(
    path
    for path in timestamp_dir.iterdir()
    if path.is_dir() and path.name.startswith('episode-')
  )
  assert episode_dirs, f'No episode directories found under {timestamp_dir}'
  return episode_dirs


def _max_episode_index(config: DictConfig) -> int | None:
  value = config.get('max_episode_index', None)
  if value is None:
    return None
  value = int(value)
  assert value >= 0, f'max_episode_index must be non-negative, got {value}'
  return value


def _load_episode_payloads(
  timestamp_dir: Path,
  max_episode_index: int | None = None,
) -> list[dict]:
  payloads = []
  for episode_dir in _episode_dirs(timestamp_dir):
    result_path = episode_dir / 'planner_results.json'
    assert result_path.exists(), f'Missing planner_results.json: {result_path}'
    with result_path.open() as file:
      payload = json.load(file)
    if (
      max_episode_index is not None
      and int(payload['episode_index']) > max_episode_index
    ):
      continue
    payloads.append(payload)
  assert payloads, (
    f'No episode payloads found under {timestamp_dir} '
    f'with max_episode_index={max_episode_index}'
  )
  return payloads


def _get_metric(payload: dict, dotted_key: str):
  value = payload
  for key in dotted_key.split('.'):
    assert isinstance(value, dict), (
      f'Expected dict while reading {dotted_key}, got {type(value)}'
    )
    value = value[key]
  assert isinstance(value, int | float), (
    f'Metric must be numeric: {dotted_key} -> {value!r}'
  )
  return float(value)


def _episode_indices(payloads: list[dict]) -> list[int]:
  indices = [int(payload['episode_index']) for payload in payloads]
  assert indices == sorted(indices), f'Episode indices are not sorted: {indices}'
  return indices


def _compute_mean(payloads: list[dict], metric_name: str) -> float:
  values = [_get_metric(payload, metric_name) for payload in payloads]
  return float(np.mean(values))


def _sample_stdev_or_none(values: np.ndarray) -> float | None:
  if values.size < 2:
    return None
  return float(np.std(values, ddof=1))


def _add_all_timestamp_stats(config: DictConfig, stats: dict) -> None:
  all_timestamp_cfg = config.stats.get('all_timestamps', {})
  if not bool(all_timestamp_cfg.get('enabled', False)):
    return
  include_mean = bool(all_timestamp_cfg.get('mean', True))
  include_stdev = bool(all_timestamp_cfg.get('stdev', True))
  timestamp_keys = [str(timestamp) for timestamp in config.timestamps]
  aggregate = {}
  for metric_name in config.metrics:
    metric_key = str(metric_name)
    values = np.asarray(
      [float(stats[timestamp_key][metric_key]) for timestamp_key in timestamp_keys],
      dtype=float,
    )
    metric_stats = {}
    if include_mean:
      metric_stats['mean'] = float(np.mean(values))
    if include_stdev:
      metric_stats['stdev'] = _sample_stdev_or_none(values)
    aggregate[metric_key] = metric_stats
  stats[str(all_timestamp_cfg.get('key', 'ALL_TIMESTAMPS'))] = aggregate


def _timestamp_dir(input_root: Path, timestamp: str) -> Path:
  timestamp_dir = input_root / timestamp
  assert timestamp_dir.exists(), f'Timestamp directory does not exist: {timestamp_dir}'
  return timestamp_dir


def _collect_series(config: DictConfig) -> dict[str, tuple[list[int], list[float]]]:
  input_root = Path(config.input_root).resolve()
  max_episode_index = _max_episode_index(config)
  series = {}
  for timestamp in config.timestamps:
    payloads = _load_episode_payloads(
      _timestamp_dir(input_root, str(timestamp)),
      max_episode_index=max_episode_index,
    )
    indices = _episode_indices(payloads)
    values = [_get_metric(payload, str(config.plot.metric)) for payload in payloads]
    series[str(timestamp)] = (indices, values)
  return series


def _moving_average_series(
  indices: list[int],
  values: list[float],
  window: int,
) -> tuple[list[int], list[float]]:
  assert window >= 1, f'plot.moving_average must be at least 1, got {window}'
  assert len(indices) == len(values), (
    f'Expected matching index/value lengths, got {len(indices)} and {len(values)}'
  )
  if window == 1:
    return indices, values
  assert len(values) >= window, (
    f'plot.moving_average={window} requires at least {window} episodes, '
    f'but only found {len(values)}.'
  )
  averaged_indices = indices[window - 1 :]
  averaged_values = [
    float(np.mean(values[index - window + 1 : index + 1]))
    for index in range(window - 1, len(values))
  ]
  return averaged_indices, averaged_values


def _apply_plot_moving_average(
  config: DictConfig,
  series: dict[str, tuple[list[int], list[float]]],
) -> dict[str, tuple[list[int], list[float]]]:
  window = int(config.plot.moving_average)
  return {
    label: _moving_average_series(indices, values, window)
    for label, (indices, values) in series.items()
  }


def _pattern_entries(payload: dict) -> list[dict]:
  assert 'pattern_learning_library' in payload, (
    'Pattern-library output is missing from planner_results.json. '
    'Re-run the experiment after enabling the updated logging.'
  )
  library_payload = payload['pattern_learning_library']
  assert isinstance(library_payload, dict), (
    f'Expected dict pattern_learning_library payload, got {type(library_payload)}'
  )
  patterns = library_payload['patterns']
  assert isinstance(patterns, list), f'Expected pattern list, got {type(patterns)}'
  return patterns


def _collect_pattern_histories(
  config: DictConfig,
) -> dict[str, dict[str, object]]:
  input_root = Path(config.input_root).resolve()
  histories = {}
  weight_field = str(config.pattern_library.weight_field)
  assert weight_field in {'raw_weight', 'positive_weight'}, (
    f'Unsupported pattern weight field: {weight_field}'
  )
  for timestamp in config.timestamps:
    timestamp_key = str(timestamp)
    payloads = _load_episode_payloads(
      _timestamp_dir(input_root, timestamp_key),
      max_episode_index=_max_episode_index(config),
    )
    indices = _episode_indices(payloads)
    pattern_metadata: dict[str, dict] = {}
    pattern_weights: dict[str, np.ndarray] = {}
    for payload_index, payload in enumerate(payloads):
      for pattern in _pattern_entries(payload):
        pattern_key = str(pattern['pattern_key'])
        if pattern_key not in pattern_metadata:
          pattern_metadata[pattern_key] = {
            key: value
            for key, value in pattern.items()
            if key not in {'raw_weight', 'positive_weight', 'pattern_index'}
          }
          pattern_weights[pattern_key] = np.full(len(payloads), np.nan, dtype=float)
        pattern_weights[pattern_key][payload_index] = float(pattern[weight_field])
    histories[timestamp_key] = {
      'episode_indices': indices,
      'pattern_metadata': pattern_metadata,
      'pattern_weights': pattern_weights,
    }
  return histories


def _pattern_spec_from_mapping(mapping, kind: str) -> dict[str, object]:
  center = str(mapping.center)
  top = getattr(mapping, 'top', None)
  bottom = getattr(mapping, 'bottom', None)
  left = getattr(mapping, 'left', None)
  right = getattr(mapping, 'right', None)
  if top is None or bottom is None or left is None or right is None:
    return {
      'kind': kind,
      'center': center,
      'top': 'unknown',
      'bottom': 'unknown',
      'left': 'unknown',
      'right': 'unknown',
      'pattern_key': f'ungated:center={center}',
      'label_lines': [
        kind,
        f'center={center}',
        'top=unknown',
        'bottom=unknown',
        'left=unknown',
        'right=unknown',
      ],
    }
  top = str(top)
  bottom = str(bottom)
  left = str(left)
  right = str(right)
  return {
    'kind': kind,
    'center': center,
    'top': top,
    'bottom': bottom,
    'left': left,
    'right': right,
    'pattern_key': (
      f'gated:center={center},top={top},bottom={bottom},left={left},right={right}'
    ),
    'label_lines': [
      f'center={center}',
      f'top={top}',
      f'bottom={bottom}',
      f'left={left}',
      f'right={right}',
    ],
  }


def _selected_pattern_specs(config: DictConfig) -> list[dict[str, object]]:
  if not bool(config.pattern_library.plot.use_selected_patterns):
    return []
  specs = [
    _pattern_spec_from_mapping(pattern, kind='selected')
    for pattern in config.pattern_library.plot.selected_patterns
  ]
  if bool(config.pattern_library.plot.add_center_only_placeholders):
    specs = [
      _pattern_spec_from_mapping(
        type('M', (), {'center': material})(), kind='placeholder'
      )
      for material in _PLACEHOLDER_MATERIALS
    ] + specs
  return specs


def _filter_pattern_histories(
  config: DictConfig,
  histories: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
  selected = _selected_pattern_specs(config)
  if not selected:
    return histories
  selected_by_key = {pattern['pattern_key']: pattern for pattern in selected}
  filtered = {}
  for timestamp, payload in histories.items():
    kept_weights = {}
    kept_metadata = {}
    for pattern_key, weights in payload['pattern_weights'].items():
      if pattern_key not in selected_by_key:
        continue
      if not np.isfinite(weights).any():
        continue
      kept_weights[pattern_key] = weights
      kept_metadata[pattern_key] = {
        **payload['pattern_metadata'][pattern_key],
        'label_lines': selected_by_key[pattern_key]['label_lines'],
      }
    filtered[timestamp] = {
      'episode_indices': payload['episode_indices'],
      'pattern_metadata': kept_metadata,
      'pattern_weights': kept_weights,
    }
  return filtered


def _pattern_colors(count: int) -> list[tuple[float, float, float, float]]:
  cmap_name = 'tab20' if count <= 20 else 'hsv'
  cmap = plt.get_cmap(cmap_name, count)
  return [cmap(index) for index in range(count)]


def _rgba255(color: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
  return tuple(int(round(channel * 255.0)) for channel in color)


def _load_font(font_size: int) -> ImageFont.ImageFont:
  for font_name in (
    'DejaVuSans.ttf',
    '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
  ):
    try:
      return ImageFont.truetype(font_name, font_size)
    except OSError:
      continue
  return ImageFont.load_default()


def _asset_image(asset_dir: Path, name: str, tile_size: int) -> Image.Image:
  path = asset_dir / f'{name}.png'
  assert path.exists(), f'Missing asset image: {path}'
  image = Image.open(path).convert('RGBA')
  return image.resize((tile_size, tile_size), resample=Image.NEAREST)


def _render_pattern_card(
  asset_dir: Path,
  pattern: dict[str, object],
  tile_size: int,
  card_padding: int,
  label_height: int,
  font_size: int,
  outline_rgba: tuple[int, int, int, int],
) -> Image.Image:
  card_size = tile_size * 3 + card_padding * 2
  card = Image.new('RGBA', (card_size, card_size + label_height), (255, 255, 255, 255))
  draw = ImageDraw.Draw(card)
  font = _load_font(font_size)
  origin = card_padding
  positions = {
    'top': (origin + tile_size, origin),
    'left': (origin, origin + tile_size),
    'center': (origin + tile_size, origin + tile_size),
    'right': (origin + tile_size * 2, origin + tile_size),
    'bottom': (origin + tile_size, origin + tile_size * 2),
  }
  for key, pos in positions.items():
    tile = _asset_image(asset_dir, str(pattern[key]), tile_size)
    card.alpha_composite(tile, pos)
  draw.rectangle((0, 0, card_size - 1, card_size - 1), outline=outline_rgba, width=2)
  label_y = card_size + 2
  for line in pattern['label_lines']:
    draw.text((card_padding, label_y), str(line), fill=(0, 0, 0, 255), font=font)
    bbox = draw.textbbox((card_padding, label_y), str(line), font=font)
    label_y += (bbox[3] - bbox[1]) + 2
  return card


def _render_pattern_cards_strip(
  config: DictConfig,
  pattern_order: list[str],
  metadata_by_key: dict[str, dict],
  colors: list[tuple[float, float, float, float]],
) -> Image.Image:
  asset_dir = Path(config.pattern_library.plot.asset_dir).resolve()
  resolution_scale = int(config.pattern_library.plot.resolution_scale)
  tile_size = int(config.pattern_library.plot.tile_size) * resolution_scale
  card_padding = int(config.pattern_library.plot.card_padding) * resolution_scale
  label_height = int(config.pattern_library.plot.label_height) * resolution_scale
  font_size = int(config.pattern_library.plot.font_size) * resolution_scale
  background_rgba = tuple(config.pattern_library.plot.background_rgba)
  cards = [
    _render_pattern_card(
      asset_dir=asset_dir,
      pattern=metadata_by_key[pattern_key],
      tile_size=tile_size,
      card_padding=card_padding,
      label_height=label_height,
      font_size=font_size,
      outline_rgba=_rgba255(color),
    )
    for pattern_key, color in zip(pattern_order, colors, strict=True)
  ]
  card_width, card_height = cards[0].size
  strip = Image.new('RGBA', (len(cards) * card_width, card_height), background_rgba)
  for index, card in enumerate(cards):
    strip.alpha_composite(card, (index * card_width, 0))
  return strip.convert('RGB')


def _pattern_histories_jsonable(histories: dict[str, dict[str, object]]) -> dict:
  jsonable = {}
  for timestamp, payload in histories.items():
    indices = payload['episode_indices']
    pattern_metadata = payload['pattern_metadata']
    pattern_weights = payload['pattern_weights']
    timestamp_payload = {
      'episode_indices': list(indices),
      'patterns': {},
    }
    for pattern_key, weights in pattern_weights.items():
      weight_by_episode = {}
      for episode_index, weight in zip(indices, weights.tolist(), strict=True):
        weight_by_episode[str(episode_index)] = (
          None if np.isnan(weight) else float(weight)
        )
      timestamp_payload['patterns'][pattern_key] = {
        **pattern_metadata[pattern_key],
        'weights_by_episode': weight_by_episode,
      }
    jsonable[timestamp] = timestamp_payload
  return jsonable


def _print_and_optionally_save_stats(config: DictConfig):
  if not bool(config.stats.enabled):
    return
  reduction = str(config.stats.reduction)
  assert reduction == 'mean', f'Unsupported stats reduction: {reduction}'
  input_root = Path(config.input_root).resolve()
  max_episode_index = _max_episode_index(config)
  stats = {}
  for timestamp in config.timestamps:
    timestamp_key = str(timestamp)
    payloads = _load_episode_payloads(
      _timestamp_dir(input_root, timestamp_key),
      max_episode_index=max_episode_index,
    )
    stats[timestamp_key] = {
      metric_name: _compute_mean(payloads, str(metric_name))
      for metric_name in config.metrics
    }
  _add_all_timestamp_stats(config, stats)
  print(json.dumps(stats, indent=2, sort_keys=True))
  if config.stats.output_path is not None:
    output_path = Path(config.stats.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as file:
      json.dump(stats, file, indent=2, sort_keys=True)


def _plot_series(config: DictConfig):
  if not bool(config.plot.enabled):
    return
  series = _apply_plot_moving_average(config, _collect_series(config))
  layout = str(config.plot.layout)
  assert layout in {'same_axes', 'separate_axes'}, f'Unsupported plot layout: {layout}'
  output_path = Path(config.plot.output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  global_xmin = min(min(xs) for xs, _ in series.values())
  global_xmax = max(max(xs) for xs, _ in series.values())
  global_ymin = min(min(ys) for _, ys in series.values())
  global_ymax = max(max(ys) for _, ys in series.values())

  if layout == 'same_axes':
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, (xs, ys) in series.items():
      ax.plot(xs, ys, marker='o', label=label)
    ax.set_xlim(global_xmin, global_xmax)
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_xlabel('Episode Index')
    ax.set_ylabel(str(config.plot.metric))
    ax.set_title(str(config.plot.title))
    ax.legend()
    ax.grid(True, alpha=0.3)
  else:
    fig, axes = plt.subplots(
      nrows=len(series),
      ncols=1,
      figsize=(8, 3 * len(series)),
      sharex=True,
      sharey=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (label, (xs, ys)) in zip(axes, series.items(), strict=True):
      ax.plot(xs, ys, marker='o', label=label)
      ax.set_xlim(global_xmin, global_xmax)
      ax.set_ylim(global_ymin, global_ymax)
      ax.set_ylabel(str(config.plot.metric))
      ax.set_title(label)
      ax.grid(True, alpha=0.3)
      ax.legend()
    axes[-1].set_xlabel('Episode Index')
    fig.suptitle(str(config.plot.title))

  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  print(f'Saved plot to {output_path}')


def _print_and_optionally_save_pattern_library(config: DictConfig):
  if not bool(config.pattern_library.enabled):
    return
  histories = _collect_pattern_histories(config)
  payload = _pattern_histories_jsonable(histories)
  print(json.dumps(payload, indent=2, sort_keys=True))
  if config.pattern_library.output_path is None:
    return
  output_path = Path(config.pattern_library.output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open('w') as file:
    json.dump(payload, file, indent=2, sort_keys=True)


def _plot_pattern_library(config: DictConfig):
  if not bool(config.pattern_library.plot.enabled):
    return
  histories = _filter_pattern_histories(config, _collect_pattern_histories(config))
  output_path = Path(config.pattern_library.plot.output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)

  global_xmin = min(min(payload['episode_indices']) for payload in histories.values())
  global_xmax = max(max(payload['episode_indices']) for payload in histories.values())
  all_weights = []
  for payload in histories.values():
    for weights in payload['pattern_weights'].values():
      finite = weights[np.isfinite(weights)]
      if finite.size > 0:
        all_weights.append(finite)
  assert all_weights, 'No pattern weights were found to plot.'
  concatenated_weights = np.concatenate(all_weights)
  global_ymin = float(np.min(concatenated_weights))
  global_ymax = float(np.max(concatenated_weights))
  if global_ymin == global_ymax:
    global_ymin *= 0.9
    global_ymax *= 1.1
  yscale = str(config.pattern_library.plot.yscale)
  assert yscale in {'linear', 'log'}, f'Unsupported yscale: {yscale}'
  if yscale == 'log':
    assert global_ymin > 0.0, (
      'Log-scale pattern plots require strictly positive weights.'
    )

  draw_patterns = bool(config.pattern_library.plot.draw_patterns)
  if draw_patterns:
    fig, axes = plt.subplots(
      nrows=len(histories),
      ncols=2,
      figsize=(14, 4 * len(histories)),
      gridspec_kw={'width_ratios': [1.7, 3.0]},
      sharey=False,
    )
    axes = np.atleast_2d(axes)
  else:
    fig, axes = plt.subplots(
      nrows=len(histories),
      ncols=1,
      figsize=(10, 4 * len(histories)),
      sharex=True,
      sharey=True,
    )
    axes = np.atleast_1d(axes)

  mark_first_appearance = bool(config.pattern_library.plot.mark_first_appearance)
  axes = np.atleast_1d(axes)
  for axis_entry, (timestamp, payload) in zip(axes, histories.items(), strict=True):
    if draw_patterns:
      card_ax, ax = axis_entry
    else:
      ax = axis_entry
    xs = payload['episode_indices']
    pattern_order = list(payload['pattern_weights'])
    colors = _pattern_colors(len(pattern_order))
    if draw_patterns:
      if pattern_order:
        card_image = _render_pattern_cards_strip(
          config,
          pattern_order,
          payload['pattern_metadata'],
          colors,
        )
        card_ax.imshow(card_image)
      else:
        card_ax.text(
          0.5, 0.5, 'No selected patterns appeared.', ha='center', va='center'
        )
      card_ax.axis('off')
      card_ax.set_title(f'{timestamp} patterns')
    for pattern_key, color in zip(pattern_order, colors, strict=True):
      weights = payload['pattern_weights'][pattern_key]
      ax.plot(
        xs,
        weights,
        color=color,
        marker='o',
        linewidth=1.5,
        markersize=3,
        label=pattern_key,
      )
      if mark_first_appearance:
        finite_indices = np.flatnonzero(np.isfinite(weights))
        if finite_indices.size > 0:
          first_index = int(finite_indices[0])
          ax.scatter(
            [xs[first_index]],
            [weights[first_index]],
            color=color,
            s=45,
            zorder=4,
          )
    ax.set_xlim(global_xmin, global_xmax)
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_yscale(yscale)
    ax.set_ylabel(str(config.pattern_library.weight_field))
    ax.set_title(timestamp)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))
  if draw_patterns:
    axes[-1, 1].set_xlabel('Episode Index')
  else:
    axes[-1].set_xlabel('Episode Index')
  fig.suptitle(str(config.pattern_library.plot.title))
  fig.tight_layout()
  fig.savefig(output_path, dpi=300, bbox_inches='tight')
  print(f'Saved pattern plot to {output_path}')


@hydra.main(
  version_base=None,
  config_path='conf',
  config_name='planner_result_stats',
)
def main(config: DictConfig):
  """Run metric aggregation and plotting from a Hydra config."""
  assert config.timestamps, 'Expected at least one timestamp.'
  _print_and_optionally_save_stats(config)
  _plot_series(config)
  _print_and_optionally_save_pattern_library(config)
  _plot_pattern_library(config)


if __name__ == '__main__':
  main()
