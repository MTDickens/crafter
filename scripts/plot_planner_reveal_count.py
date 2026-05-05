"""Plot per-episode reveal counts across Crafter planner-result runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import numpy as np
from omegaconf import DictConfig, OmegaConf

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


MAX_TARGETS = 4
TARGET_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']


@dataclass(frozen=True, slots=True)
class TargetSeries:
  """Center/std reveal-count series for one plotted target."""

  name: str
  timestamps: list[str]
  episode_indices: np.ndarray
  center: np.ndarray
  std: np.ndarray


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
  """Gaussian smoothing settings for plotted curves."""

  enabled: bool = False
  sigma: float = 1.0
  truncate: float = 3.0


def _target_fields(target: Mapping[str, Any]) -> tuple[str, str]:
  """Return `(dir, name)` from a YAML target entry."""
  output_dir = str(target.get('dir', ''))
  name = str(target.get('name', ''))
  if not output_dir:
    raise ValueError('Each target must define a non-empty dir')
  if not name:
    raise ValueError('Each target must define a non-empty name')
  return output_dir, name


def validate_targets(targets: list[Mapping[str, Any]]) -> None:
  """Validate the configured target count."""
  if not 1 <= len(targets) <= MAX_TARGETS:
    raise ValueError(f'Expected 1 to {MAX_TARGETS} targets, got {len(targets)}')


def discover_timestamps(*, repo_root: Path, output_dir: str) -> list[str]:
  """Return timestamp subdirectory names under one planner output directory."""
  root = repo_root / Path(output_dir)
  if not root.exists():
    raise ValueError(f'Planner output directory does not exist: {root}')
  timestamps = sorted(path.name for path in root.iterdir() if path.is_dir())
  if not timestamps:
    raise ValueError(f'No timestamp dirs discovered under {root}')
  return timestamps


def _episode_dirs(timestamp_dir: Path) -> list[Path]:
  episode_dirs = sorted(
    path
    for path in timestamp_dir.iterdir()
    if path.is_dir() and path.name.startswith('episode-')
  )
  if not episode_dirs:
    raise ValueError(f'No episode directories found under {timestamp_dir}')
  return episode_dirs


def _load_episode_payloads(timestamp_dir: Path) -> list[dict[str, Any]]:
  payloads = []
  for episode_dir in _episode_dirs(timestamp_dir):
    result_path = episode_dir / 'planner_results.json'
    if not result_path.exists():
      raise ValueError(f'Missing planner_results.json: {result_path}')
    with result_path.open() as file:
      payload = json.load(file)
    if not isinstance(payload, dict):
      raise ValueError(f'Expected object in {result_path}, got {type(payload)}')
    payloads.append(payload)
  return payloads


def _get_metric(payload: Mapping[str, Any], dotted_key: str) -> float:
  value: Any = payload
  for key in dotted_key.split('.'):
    if not isinstance(value, Mapping):
      raise ValueError(
        f'Expected mapping while reading {dotted_key}, got {type(value)}'
      )
    value = value[key]
  if not isinstance(value, int | float):
    raise ValueError(f'Metric must be numeric: {dotted_key} -> {value!r}')
  return float(value)


def _aggregate_center(values: np.ndarray, aggregation: str) -> np.ndarray:
  if aggregation == 'median':
    return np.median(values, axis=0)
  if aggregation == 'mean':
    return values.mean(axis=0)
  raise ValueError(f'Unsupported aggregation: {aggregation!r}. Expected median or mean')


def _run_reveal_counts(
  *,
  timestamp_dir: Path,
  expected_episode_count: int,
  metric: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Return `(episode_indices, metric_values)` ordered by episode index."""
  payloads = _load_episode_payloads(timestamp_dir)
  if len(payloads) != expected_episode_count:
    raise ValueError(
      f'{timestamp_dir} has {len(payloads)} episodes; '
      f'expected exactly {expected_episode_count}'
    )

  payloads_by_index = {int(payload['episode_index']): payload for payload in payloads}
  if len(payloads_by_index) != len(payloads):
    raise ValueError(f'{timestamp_dir} has duplicate episode_index values')

  actual_indices = sorted(payloads_by_index)
  expected_indices = list(
    range(actual_indices[0], actual_indices[0] + expected_episode_count)
  )
  if actual_indices != expected_indices:
    raise ValueError(
      f'{timestamp_dir} has episode indices {actual_indices}; '
      f'expected contiguous {expected_indices[0]}..{expected_indices[-1]}'
    )

  return (
    np.asarray(expected_indices, dtype=int),
    np.asarray(
      [_get_metric(payloads_by_index[index], metric) for index in expected_indices],
      dtype=float,
    ),
  )


def load_target_series(
  *,
  repo_root: Path,
  output_dir: str,
  name: str,
  expected_episode_count: int,
  metric: str,
  aggregation: str,
) -> TargetSeries:
  """Load one target directory and aggregate reveal counts across timestamp runs."""
  timestamps = discover_timestamps(repo_root=repo_root, output_dir=output_dir)

  episode_indices: np.ndarray | None = None
  per_run_counts: list[np.ndarray] = []
  for timestamp in timestamps:
    indices, counts = _run_reveal_counts(
      timestamp_dir=repo_root / Path(output_dir) / timestamp,
      expected_episode_count=expected_episode_count,
      metric=metric,
    )
    if episode_indices is None:
      episode_indices = indices
    elif not np.array_equal(episode_indices, indices):
      raise ValueError(
        f'{output_dir}/{timestamp} episode indices differ from previous runs'
      )
    per_run_counts.append(counts)

  values = np.stack(per_run_counts, axis=0)
  std = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1])
  if episode_indices is None:
    raise ValueError(f'No runs discovered for target {name}')
  return TargetSeries(
    name=name,
    timestamps=timestamps,
    episode_indices=episode_indices,
    center=_aggregate_center(values, aggregation),
    std=std,
  )


def load_all_target_series(
  *,
  repo_root: Path,
  targets: list[Mapping[str, Any]],
  expected_episode_count: int,
  metric: str,
  aggregation: str,
) -> list[TargetSeries]:
  """Load all configured target series."""
  validate_targets(targets)
  series: list[TargetSeries] = []
  for target in targets:
    output_dir, name = _target_fields(target)
    series.append(
      load_target_series(
        repo_root=repo_root,
        output_dir=output_dir,
        name=name,
        expected_episode_count=expected_episode_count,
        metric=metric,
        aggregation=aggregation,
      )
    )
  return series


def gaussian_smooth(values: np.ndarray, *, sigma: float, truncate: float) -> np.ndarray:
  """Smooth a 1D series with a Gaussian kernel using NumPy only."""
  if sigma <= 0:
    raise ValueError(f'sigma must be positive, got {sigma}')
  if truncate <= 0:
    raise ValueError(f'truncate must be positive, got {truncate}')

  radius = max(1, int(truncate * sigma + 0.5))
  offsets = np.arange(-radius, radius + 1, dtype=float)
  kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
  kernel = kernel / kernel.sum()
  padded = np.pad(np.asarray(values, dtype=float), pad_width=radius, mode='edge')
  return np.convolve(padded, kernel, mode='valid')


def maybe_smooth(values: np.ndarray, smoothing: SmoothingConfig) -> np.ndarray:
  """Apply optional Gaussian smoothing."""
  if not smoothing.enabled:
    return values
  return gaussian_smooth(values, sigma=smoothing.sigma, truncate=smoothing.truncate)


def plot_reveal_count(
  *,
  series: list[TargetSeries],
  title: str,
  show_std: bool,
  zero_based_episode_index: bool,
  smoothing: SmoothingConfig,
  out_path: Path | None,
) -> None:
  """Plot and optionally save reveal-count curves."""
  fig, ax = plt.subplots(figsize=(12.8, 6.4))
  for index, target_series in enumerate(series):
    color = TARGET_COLORS[index]
    episode_indices = target_series.episode_indices
    if not zero_based_episode_index and episode_indices[0] == 0:
      episode_indices = episode_indices + 1
    center = maybe_smooth(target_series.center, smoothing)
    std = maybe_smooth(target_series.std, smoothing)
    ax.plot(
      episode_indices,
      center,
      marker='o',
      markersize=4,
      linewidth=1.8,
      color=color,
      label=target_series.name,
    )
    if show_std:
      ax.fill_between(
        episode_indices,
        center - std,
        center + std,
        color=color,
        alpha=0.18,
        linewidth=0,
      )

  ax.set_title(title)
  ax.set_xlabel('Episode Index')
  ax.set_ylabel('reveal count')
  ax.grid(True, alpha=0.3)
  ax.legend(loc='upper right')
  fig.tight_layout()

  if out_path is not None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f'Wrote {out_path}')
  else:
    plt.show()
  plt.close(fig)


def _smoothing_from_cfg(cfg: DictConfig) -> SmoothingConfig:
  smoothing_cfg = cfg.get('gaussian_smoothing', {})
  return SmoothingConfig(
    enabled=bool(smoothing_cfg.get('enabled', False)),
    sigma=float(smoothing_cfg.get('sigma', 1.0)),
    truncate=float(smoothing_cfg.get('truncate', 3.0)),
  )


@hydra.main(
  version_base=None, config_path='conf', config_name='plot_planner_reveal_count'
)
def _main(cfg: DictConfig) -> None:
  repo_root = Path(__file__).resolve().parent.parent
  raw_targets = OmegaConf.to_container(cfg.targets, resolve=True)
  if not isinstance(raw_targets, list):
    raise ValueError('targets must be a YAML list')
  targets = [target for target in raw_targets if isinstance(target, Mapping)]
  if len(targets) != len(raw_targets):
    raise ValueError('Each target must be a mapping with dir and name')

  expected_episode_count = int(cfg.episode)
  metric = str(cfg.get('metric', 'metrics.reveal_count'))
  aggregation = str(cfg.get('aggregation', 'median'))
  series = load_all_target_series(
    repo_root=repo_root,
    targets=targets,
    expected_episode_count=expected_episode_count,
    metric=metric,
    aggregation=aggregation,
  )

  out_path_value = cfg.get('out_path', None)
  out_path = None
  if out_path_value is not None:
    out_path = Path(str(out_path_value))
    if not out_path.is_absolute():
      out_path = repo_root / out_path

  plot_reveal_count(
    series=series,
    title=str(cfg.get('title', 'Crafter')),
    show_std=bool(cfg.get('show_std', False)),
    zero_based_episode_index=bool(cfg.get('zero_based_episode_index', True)),
    smoothing=_smoothing_from_cfg(cfg),
    out_path=out_path,
  )


if __name__ == '__main__':
  _main()
