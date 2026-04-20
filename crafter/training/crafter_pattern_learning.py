"""Online Crafter pattern learning and replay management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from omegaconf import DictConfig

from crafter.skills.crafter_patterns import CrafterSkillLibrary, CrossShapedPattern
from crafter.skills.crafter_proposal import (
  CrafterProposalPromptExample,
  build_crafter_proposal_prompt_examples,
  propose_crafter_patterns_from_partial_maps,
)
from crafter.skills.crafter_weight_optimization import (
  CrafterTransitionExample,
  PositiveWeightCrafterSolver,
)
from crafter.utils.crafter_codec import CrafterTileCodec


@dataclass(slots=True)
class CrafterReplayEntry:
  """One Crafter episode stored in the replay buffer.

  Parameters
  ----------
  full_map_ids : torch.Tensor
      Integer full initial-world material grid.
  final_known_mask : torch.Tensor
      Boolean known mask at the end of planning or execution.
  start_pos : tuple[int, int]
      Episode start position.
  episode_index : int
      Episode index for debugging.
  seed : int or None
      Episode seed for debugging.
  """

  full_map_ids: torch.Tensor
  final_known_mask: torch.Tensor
  start_pos: tuple[int, int]
  episode_index: int
  seed: int | None


class CrafterSkillLearningReplayBuffer:
  """FIFO replay buffer for Crafter online pattern learning."""

  def __init__(self, capacity: int):
    self.capacity = int(capacity)
    self._buffer: deque[CrafterReplayEntry] = deque(maxlen=self.capacity)

  def add(self, item: CrafterReplayEntry) -> None:
    """Append one replay entry."""
    self._buffer.append(item)

  def get_all(self) -> list[CrafterReplayEntry]:
    """Return all replay entries currently stored."""
    return list(self._buffer)

  def __len__(self) -> int:
    return len(self._buffer)


def _random_connected_order(
  known_mask: np.ndarray,
  start_pos: tuple[int, int],
  rng: np.random.Generator,
) -> list[tuple[int, int]]:
  """Return a random connected reveal order over known cells."""
  assert known_mask[start_pos], 'The start position must be known.'
  order = [tuple(start_pos)]
  visited = {tuple(start_pos)}
  frontier = [tuple(start_pos)]
  directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  while frontier:
    current_index = int(rng.integers(0, len(frontier)))
    current = frontier[current_index]
    neighbors = []
    for dx, dy in directions:
      nxt = (current[0] + dx, current[1] + dy)
      if not (0 <= nxt[0] < known_mask.shape[0] and 0 <= nxt[1] < known_mask.shape[1]):
        continue
      if not known_mask[nxt] or nxt in visited:
        continue
      neighbors.append(nxt)
    if neighbors:
      nxt = neighbors[int(rng.integers(0, len(neighbors)))]
      visited.add(nxt)
      frontier.append(nxt)
      order.append(nxt)
      continue
    frontier.pop(current_index)
  assert len(visited) == int(known_mask.sum()), (
    'Known mask must be connected from the start.'
  )
  return order


def build_connected_reverse_trajectory_examples(
  replay_entry: CrafterReplayEntry,
  pure_samples_per_map: int,
  seed: int,
  unknown_id: int,
) -> list[CrafterTransitionExample]:
  """Build single-target reveal-trajectory examples from one replay entry.

  Parameters
  ----------
  replay_entry : CrafterReplayEntry
      Replay-buffer entry that provides the final known mask.
  pure_samples_per_map : int
      Number of random reveal trajectories to sample.
  seed : int
      Sampling seed.
  unknown_id : int
      Token used for hidden cells.

  Returns
  -------
  list[CrafterTransitionExample]
      Training examples whose target list always has length 1.
  """
  rng = np.random.default_rng(seed)
  examples: list[CrafterTransitionExample] = []
  known_mask_np = replay_entry.final_known_mask.detach().cpu().numpy().astype(bool)
  full_map_ids = replay_entry.full_map_ids.to(dtype=torch.long)
  for sample_index in range(int(pure_samples_per_map)):
    order = _random_connected_order(
      known_mask=known_mask_np,
      start_pos=replay_entry.start_pos,
      rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
    )
    prefix_mask = np.zeros_like(known_mask_np, dtype=bool)
    prefix_mask[replay_entry.start_pos] = True
    for pos in order[1:]:
      partial = torch.full_like(full_map_ids, fill_value=-1)
      partial[:] = unknown_id
      partial[prefix_mask] = full_map_ids[prefix_mask]
      examples.append(
        CrafterTransitionExample(
          partial_map_ids=partial,
          target_positions=(tuple(pos),),
          target_class_ids=(int(full_map_ids[pos]),),
        )
      )
      prefix_mask[pos] = True
    del sample_index
  return examples


def build_iid_known_drop_examples(
  replay_entry: CrafterReplayEntry,
  pure_samples_per_map: int,
  pure_drop_prob: float,
  seed: int,
  unknown_id: int,
) -> list[CrafterTransitionExample]:
  """Build IID-known-drop training examples from one replay entry.

  Each sampled example independently drops every currently known grid with
  probability ``pure_drop_prob``. The remaining known grids form the input
  partial map; every dropped known grid is supervised jointly.

  Parameters
  ----------
  replay_entry : CrafterReplayEntry
      Replay-buffer entry that provides the final known mask.
  pure_samples_per_map : int
      Number of valid IID-drop examples to produce.
  pure_drop_prob : float
      Bernoulli drop probability for each known cell.
  seed : int
      Sampling seed.
  unknown_id : int
      Token used for hidden cells.

  Returns
  -------
  list[CrafterTransitionExample]
      Joint-supervision training examples.
  """
  assert 0.0 <= pure_drop_prob <= 1.0, (
      f'pure_drop_prob must lie in [0, 1], got {pure_drop_prob}'
  )
  rng = np.random.default_rng(seed)
  examples: list[CrafterTransitionExample] = []
  known_mask_np = replay_entry.final_known_mask.detach().cpu().numpy().astype(bool)
  full_map_ids = replay_entry.full_map_ids.to(dtype=torch.long)
  max_attempts = max(int(pure_samples_per_map) * 100, 100)
  attempts = 0
  while len(examples) < int(pure_samples_per_map) and attempts < max_attempts:
    attempts += 1
    drop_mask = known_mask_np & (rng.random(known_mask_np.shape) < pure_drop_prob)
    if not drop_mask.any():
      continue
    observed_mask = known_mask_np & (~drop_mask)
    partial = torch.full_like(full_map_ids, fill_value=unknown_id)
    partial[observed_mask] = full_map_ids[observed_mask]
    target_positions = tuple(
        (int(x), int(y))
        for x, y in zip(*np.nonzero(drop_mask), strict=True)
    )
    target_class_ids = tuple(int(full_map_ids[pos]) for pos in target_positions)
    examples.append(CrafterTransitionExample(
        partial_map_ids=partial,
        target_positions=target_positions,
        target_class_ids=target_class_ids,
    ))
  assert len(examples) == int(pure_samples_per_map), (
      'Failed to construct enough iid_known_drop examples. '
      f'Generated {len(examples)} / {pure_samples_per_map} valid examples '
      f'after {attempts} attempts with pure_drop_prob={pure_drop_prob}.'
  )
  return examples


class CrafterSkillLearningManager:
  """Own the online Crafter replay buffer, pattern library, and training loop.

  Parameters
  ----------
  cfg : DictConfig
      ``planner.skill_learning`` config subtree.
  codec : CrafterTileCodec, optional
      Integer token codec.
  """

  def __init__(self, cfg: DictConfig, codec: CrafterTileCodec | None = None):
    self.cfg = cfg
    self.codec = codec or CrafterTileCodec.for_worldgen_materials()
    self.replay_buffer = CrafterSkillLearningReplayBuffer(
      capacity=int(cfg.replay_buffer.capacity)
    )
    self._rng = np.random.default_rng(int(cfg.sampling_method.seed))
    self.library = CrafterSkillLibrary(
      codec=self.codec,
      device=torch.device(str(cfg.device)),
    )
    self._proposal_example_cache: dict[tuple, list[CrafterProposalPromptExample]] = {}
    self._initialize_patterns()

  def _initialize_patterns(self):
    if bool(self.cfg.initial_patterns.add_center_only_placeholders):
      for class_id in range(self.codec.num_classes):
        self.library.add_pattern(CrossShapedPattern.make_ungated_center_only(class_id))
    for preset in self.cfg.initial_patterns.presets:
      self.library.add_pattern(
        CrossShapedPattern(
          center_id=self.codec.encode_material(str(preset.center)),
          top_id=self.codec.encode_material(str(preset.top)),
          bottom_id=self.codec.encode_material(str(preset.bottom)),
          left_id=self.codec.encode_material(str(preset.left)),
          right_id=self.codec.encode_material(str(preset.right)),
        )
      )

  def should_use_inference(self) -> bool:
    """Return whether reveal-time pattern inference is currently enabled."""
    return bool(self.cfg.enable_inference) and len(self.library) > 0

  def should_trigger_proposal(self, episode_index: int, total_episodes: int) -> bool:
    """Return whether this episode boundary should add proposed patterns."""
    if not bool(self.cfg.skill_proposal.enabled):
      return False
    trigger_interval = int(self.cfg.skill_proposal.trigger_interval)
    if trigger_interval <= 0:
      return False
    if (episode_index + 1) % trigger_interval != 0:
      return False
    return episode_index + 1 < total_episodes

  def should_trigger_reweight(self, episode_index: int, total_episodes: int) -> bool:
    """Return whether this episode boundary should reweight the library."""
    if not bool(self.cfg.enable_reweighting):
      return False
    trigger_interval = int(self.cfg.train_trigger_interval)
    if trigger_interval <= 0:
      return False
    if (episode_index + 1) % trigger_interval != 0:
      return False
    return episode_index + 1 < total_episodes

  def add_replay_entry(
    self,
    full_map_ids: torch.Tensor,
    final_known_mask: torch.Tensor,
    start_pos: tuple[int, int],
    episode_index: int,
    seed: int | None,
  ) -> None:
    """Append one finished episode to the replay buffer."""
    self.replay_buffer.add(
      CrafterReplayEntry(
        full_map_ids=full_map_ids.detach().cpu().clone(),
        final_known_mask=final_known_mask.detach().cpu().clone(),
        start_pos=tuple(start_pos),
        episode_index=int(episode_index),
        seed=seed,
      )
    )

  def maybe_train(self, episode_index: int, total_episodes: int) -> bool:
    """Trigger online training if the configured interval is reached."""
    did_update = False
    if self.should_trigger_proposal(episode_index, total_episodes):
      self._maybe_add_proposals(self.replay_buffer.get_all())
      did_update = True
    if not self.should_trigger_reweight(episode_index, total_episodes):
      return did_update
    replay_entries = self.replay_buffer.get_all()
    if not replay_entries:
      return did_update
    examples = self._build_examples(replay_entries)
    selected_examples: list[CrafterTransitionExample] = (
      self._select_examples_for_training(examples)
    )
    solver = PositiveWeightCrafterSolver(
      max_iters=int(self.cfg.lbfgs.max_iters),
      lr=float(self.cfg.lbfgs.lr),
    )
    solver.solve(self.library, selected_examples, eps=float(self.cfg.eps))
    return True

  def export_library_state(self) -> dict:
    """Return a process-safe snapshot of the current pattern library."""
    return {
      'patterns': list(self.library.patterns),
      'raw_weights': self.library.raw_weights.detach().cpu().clone(),
    }

  def _build_examples(
    self,
    replay_entries: Iterable[CrafterReplayEntry],
  ) -> list[CrafterTransitionExample]:
    examples: list[CrafterTransitionExample] = []
    mode = str(self.cfg.sampling_method.mode)
    for replay_entry in replay_entries:
      entry_seed = int(self._rng.integers(0, 2**31 - 1))
      if mode == 'connected_reverse_trajectory':
        examples.extend(
          build_connected_reverse_trajectory_examples(
            replay_entry=replay_entry,
            pure_samples_per_map=int(self.cfg.sampling_method.pure_samples_per_map),
            seed=entry_seed,
            unknown_id=self.codec.unknown_id,
          )
        )
        continue
      if mode == 'iid_known_drop':
        examples.extend(
          build_iid_known_drop_examples(
            replay_entry=replay_entry,
            pure_samples_per_map=int(self.cfg.sampling_method.pure_samples_per_map),
            pure_drop_prob=float(self.cfg.sampling_method.pure_drop_prob),
            seed=entry_seed,
            unknown_id=self.codec.unknown_id,
          )
        )
        continue
      raise AssertionError(
        f'Unsupported sampling_method.mode: {mode}. '
        'Expected one of: connected_reverse_trajectory, iid_known_drop.'
      )
    return examples

  def _select_examples_for_training(
    self,
    examples: list[CrafterTransitionExample],
  ) -> list[CrafterTransitionExample]:
    """Select the training subset implied by ``max_examples_learned``.

    Parameters
    ----------
    examples : list[CrafterTransitionExample]
        Full list of training examples expanded from every replay-buffer entry.

    Returns
    -------
    list[CrafterTransitionExample]
        Selected example subset. ``replay_buffer.capacity`` controls how many
        episodes are stored; ``max_examples_learned`` only controls how many
        expanded transition examples are used for one training run.
    """
    max_examples_learned = int(self.cfg.max_examples_learned)
    if max_examples_learned <= 0 or len(examples) <= max_examples_learned:
      return examples

    order = str(self.cfg.max_examples_learned_order)
    if order == 'first':
      return examples[:max_examples_learned]
    if order == 'last':
      return examples[-max_examples_learned:]
    if order == 'random':
      indices = self._rng.choice(len(examples), size=max_examples_learned, replace=False)
      return [examples[int(index)] for index in sorted(indices.tolist())]
    raise AssertionError(
      f'Unsupported max_examples_learned_order: {order}. '
      'Expected one of: first, last, random.'
    )

  def _maybe_add_proposals(self, replay_entries: list[CrafterReplayEntry]) -> int:
    if not bool(self.cfg.skill_proposal.enabled):
      return 0
    self._prune_proposal_example_cache(replay_entries)
    prompt_examples: list[CrafterProposalPromptExample] = []
    for replay_entry in replay_entries:
      prompt_examples.extend(self._proposal_prompt_examples(replay_entry))
    selected_prompt_examples = self._select_prompt_examples(prompt_examples)
    proposed = propose_crafter_patterns_from_partial_maps(
      prompt_examples=selected_prompt_examples,
      codec=self.codec,
      skill_proposal_cfg=self.cfg.skill_proposal,
    )
    added = 0
    for pattern in proposed[: int(self.cfg.skill_proposal.patterns_per_trigger)]:
      added += int(self.library.add_pattern(pattern))
    return added

  def _proposal_partial_map(self, replay_entry: CrafterReplayEntry) -> torch.Tensor:
    partial = torch.full_like(
      replay_entry.full_map_ids, fill_value=self.codec.unknown_id
    )
    partial[replay_entry.final_known_mask] = replay_entry.full_map_ids[
      replay_entry.final_known_mask
    ]
    return partial

  def _proposal_cache_key(self, replay_entry: CrafterReplayEntry) -> tuple:
    cfg = self.cfg.skill_proposal
    return (
      int(replay_entry.episode_index),
      bool(cfg.use_bounding_box),
      bool(cfg.enable_tiling),
      tuple(int(value) for value in cfg.tile_size),
      tuple(int(value) for value in cfg.tile_overlap),
    )

  def _proposal_prompt_examples(
    self,
    replay_entry: CrafterReplayEntry,
  ) -> list[CrafterProposalPromptExample]:
    cache_key = self._proposal_cache_key(replay_entry)
    if cache_key not in self._proposal_example_cache:
      self._proposal_example_cache[cache_key] = build_crafter_proposal_prompt_examples(
        partial_map_ids=self._proposal_partial_map(replay_entry),
        source_episode_index=int(replay_entry.episode_index),
        codec=self.codec,
        skill_proposal_cfg=self.cfg.skill_proposal,
      )
    return self._proposal_example_cache[cache_key]

  def _prune_proposal_example_cache(
    self,
    replay_entries: list[CrafterReplayEntry],
  ) -> None:
    active_episode_indices = {int(entry.episode_index) for entry in replay_entries}
    stale_keys = [
      key for key in self._proposal_example_cache
      if int(key[0]) not in active_episode_indices
    ]
    for key in stale_keys:
      del self._proposal_example_cache[key]

  def _select_prompt_examples(
    self,
    prompt_examples: list[CrafterProposalPromptExample],
  ) -> list[CrafterProposalPromptExample]:
    max_examples = int(self.cfg.skill_proposal.max_examples_in_prompt)
    if max_examples <= 0 or len(prompt_examples) <= max_examples:
      return prompt_examples
    order = str(self.cfg.skill_proposal.max_examples_in_prompt_order)
    if order == 'first':
      return prompt_examples[:max_examples]
    if order == 'last':
      return prompt_examples[-max_examples:]
    if order == 'random':
      indices = self._rng.choice(len(prompt_examples), size=max_examples, replace=False)
      return [prompt_examples[int(index)] for index in sorted(indices.tolist())]
    raise AssertionError(
      f'Unsupported max_examples_in_prompt_order: {order}. '
      'Expected one of: first, last, random.'
    )
