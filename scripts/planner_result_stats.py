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


if __name__ == '__main__':
  main()
