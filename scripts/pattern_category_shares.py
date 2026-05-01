"""Plot pattern-library weight shares by semantic category.

The selected patterns in `scripts/conf/planner_result_stats.yaml` are treated
as critical patterns. Ungated center-only patterns are treated as placeholders.
All remaining patterns are treated as non-critical learned patterns.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf


_CATEGORY_ORDER = (
  'critical',
  'placeholder',
  'non_critical_learned',
)
_CATEGORY_COLORS = {
  'critical': '#2ca02c',
  'placeholder': '#9e9e9e',
  'non_critical_learned': '#d62728',
}
_CATEGORY_LABELS = {
  'critical': 'Critical patterns',
  'placeholder': 'Placeholder patterns',
  'non_critical_learned': 'Non-critical learned patterns',
}


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
      payloads.append(json.load(file))
  return payloads


def _timestamp_dir(input_root: Path, timestamp: str) -> Path:
  timestamp_dir = input_root / timestamp
  assert timestamp_dir.exists(), f'Timestamp directory does not exist: {timestamp_dir}'
  return timestamp_dir


def _pattern_key_from_mapping(mapping) -> str:
  center = str(mapping.center)
  top = getattr(mapping, 'top', None)
  bottom = getattr(mapping, 'bottom', None)
  left = getattr(mapping, 'left', None)
  right = getattr(mapping, 'right', None)
  if top is None or bottom is None or left is None or right is None:
    return f'ungated:center={center}'
  return (
    f'gated:center={center},top={top},bottom={bottom},left={left},right={right}'
  )


def _critical_pattern_keys(config: DictConfig) -> set[str]:
  selected_patterns = OmegaConf.select(
    config,
    'pattern_library.plot.selected_patterns',
    default=[],
  )
  return {_pattern_key_from_mapping(pattern) for pattern in selected_patterns}


def _pattern_entries(payload: dict) -> list[dict]:
  assert 'pattern_learning_library' in payload, (
    'Pattern-library output is missing from planner_results.json.'
  )
  library_payload = payload['pattern_learning_library']
  assert isinstance(library_payload, dict), (
    f'Expected dict pattern_learning_library payload, got {type(library_payload)}'
  )
  patterns = library_payload['patterns']
  assert isinstance(patterns, list), f'Expected pattern list, got {type(patterns)}'
  return patterns


def _pattern_weight(pattern: dict) -> float:
  if 'normalized_weight' in pattern:
    return float(pattern['normalized_weight'])
  if 'positive_weight' in pattern:
    return float(pattern['positive_weight'])
  raise AssertionError(f'Pattern has no plottable weight: {pattern}')


def _pattern_category(pattern: dict, critical_keys: set[str]) -> str:
  if str(pattern['pattern_key']) in critical_keys:
    return 'critical'
  if pattern.get('kind') == 'ungated_center_only':
    return 'placeholder'
  return 'non_critical_learned'


def _episode_category_shares(
    payload: dict,
    critical_keys: set[str],
) -> dict[str, float]:
  category_weights = {category: 0.0 for category in _CATEGORY_ORDER}
  for pattern in _pattern_entries(payload):
    category = _pattern_category(pattern, critical_keys)
    category_weights[category] += _pattern_weight(pattern)

  total = sum(category_weights.values())
  assert total > 0.0, (
    f'No positive pattern weight found for episode {payload.get("episode_index")}'
  )
  return {
    category: category_weights[category] / total
    for category in _CATEGORY_ORDER
  }


def _collect_category_histories(config: DictConfig) -> dict[str, dict[str, object]]:
  input_root = Path(config.input_root).resolve()
  critical_keys = _critical_pattern_keys(config)
  histories = {}
  for timestamp in config.timestamps:
    timestamp_key = str(timestamp)
    payloads = _load_episode_payloads(_timestamp_dir(input_root, timestamp_key))
    episode_indices = [int(payload['episode_index']) for payload in payloads]
    assert episode_indices == sorted(episode_indices), (
      f'Episode indices are not sorted: {episode_indices}'
    )

    shares = {category: [] for category in _CATEGORY_ORDER}
    for payload in payloads:
      episode_shares = _episode_category_shares(payload, critical_keys)
      for category in _CATEGORY_ORDER:
        shares[category].append(episode_shares[category])

    histories[timestamp_key] = {
      'episode_indices': episode_indices,
      'shares': {
        category: np.array(values, dtype=float)
        for category, values in shares.items()
      },
    }
  return histories


def _output_path(config: DictConfig) -> Path:
  configured_path = OmegaConf.select(
    config,
    'category_shares.output_path',
    default='plots/pattern_category_shares.png',
  )
  return Path(str(configured_path)).resolve()


def _plot_category_histories(config: DictConfig):
  histories = _collect_category_histories(config)
  output_path = _output_path(config)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(
    nrows=len(histories),
    ncols=1,
    figsize=(10, 3.8 * len(histories)),
    sharex=False,
    sharey=True,
  )
  axes = np.atleast_1d(axes)

  for ax, (timestamp, payload) in zip(axes, histories.items(), strict=True):
    xs = payload['episode_indices']
    ys = [payload['shares'][category] for category in _CATEGORY_ORDER]
    colors = [_CATEGORY_COLORS[category] for category in _CATEGORY_ORDER]
    labels = [_CATEGORY_LABELS[category] for category in _CATEGORY_ORDER]

    ax.stackplot(xs, ys, colors=colors, labels=labels, alpha=0.35)
    for category, values in zip(_CATEGORY_ORDER, ys, strict=True):
      ax.plot(
        xs,
        values,
        color=_CATEGORY_COLORS[category],
        linewidth=1.8,
      )
    stacked_total = np.sum(np.vstack(ys), axis=0)
    ax.plot(xs, stacked_total, color='black', linewidth=1.0, linestyle='--')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Normalized total weight share')
    ax.set_title(timestamp)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))

  axes[-1].set_xlabel('Episode Index')
  title = OmegaConf.select(
    config,
    'category_shares.title',
    default='Pattern Category Weight Shares',
  )
  fig.suptitle(str(title))
  fig.tight_layout()
  fig.savefig(output_path, dpi=300, bbox_inches='tight')
  print(f'Saved pattern category share plot to {output_path}')


@hydra.main(
  version_base=None,
  config_path='conf',
  config_name='planner_result_stats',
)
def main(config: DictConfig):
  _plot_category_histories(config)


if __name__ == '__main__':
  main()
