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


def _episode_dirs(timestamp_dir: Path) -> list[Path]:
  episode_dirs = sorted(
    path
    for path in timestamp_dir.iterdir()
    if path.is_dir() and path.name.startswith('episode-')
  )
  assert episode_dirs, f'No episode directories found under {timestamp_dir}'
  return episode_dirs


def _load_episode_payloads(timestamp_dir: Path) -> list[dict]:
  payloads = []
  for episode_dir in _episode_dirs(timestamp_dir):
    result_path = episode_dir / 'planner_results.json'
    assert result_path.exists(), f'Missing planner_results.json: {result_path}'
    with result_path.open() as file:
      payload = json.load(file)
    payloads.append(payload)
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


def _timestamp_dir(input_root: Path, timestamp: str) -> Path:
  timestamp_dir = input_root / timestamp
  assert timestamp_dir.exists(), f'Timestamp directory does not exist: {timestamp_dir}'
  return timestamp_dir


def _collect_series(config: DictConfig) -> dict[str, tuple[list[int], list[float]]]:
  input_root = Path(config.input_root).resolve()
  series = {}
  for timestamp in config.timestamps:
    payloads = _load_episode_payloads(_timestamp_dir(input_root, str(timestamp)))
    indices = _episode_indices(payloads)
    values = [_get_metric(payload, str(config.plot.metric)) for payload in payloads]
    series[str(timestamp)] = (indices, values)
  return series


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
    payloads = _load_episode_payloads(_timestamp_dir(input_root, timestamp_key))
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
        weight_by_episode[str(episode_index)] = None if np.isnan(weight) else float(weight)
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
  stats = {}
  for timestamp in config.timestamps:
    timestamp_key = str(timestamp)
    payloads = _load_episode_payloads(_timestamp_dir(input_root, timestamp_key))
    stats[timestamp_key] = {
      metric_name: _compute_mean(payloads, str(metric_name))
      for metric_name in config.metrics
    }
  print(json.dumps(stats, indent=2, sort_keys=True))
  if config.stats.output_path is not None:
    output_path = Path(config.stats.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as file:
      json.dump(stats, file, indent=2, sort_keys=True)


def _plot_series(config: DictConfig):
  if not bool(config.plot.enabled):
    return
  series = _collect_series(config)
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
  histories = _collect_pattern_histories(config)
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
    assert global_ymin > 0.0, 'Log-scale pattern plots require strictly positive weights.'

  fig, axes = plt.subplots(
    nrows=len(histories),
    ncols=1,
    figsize=(10, 4 * len(histories)),
    sharex=True,
    sharey=True,
  )
  axes = np.atleast_1d(axes)
  for ax, (timestamp, payload) in zip(axes, histories.items(), strict=True):
    xs = payload['episode_indices']
    for pattern_key, weights in payload['pattern_weights'].items():
      ax.plot(xs, weights, marker='o', linewidth=1, markersize=2, label=pattern_key)
    ax.set_xlim(global_xmin, global_xmax)
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_yscale(yscale)
    ax.set_ylabel(str(config.pattern_library.weight_field))
    ax.set_title(timestamp)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))
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
