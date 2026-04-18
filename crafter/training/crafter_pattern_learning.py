"""Online Crafter pattern learning and replay management."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from omegaconf import DictConfig

from crafter.skills.crafter_patterns import CrafterSkillLibrary, CrossShapedPattern
from crafter.skills.crafter_proposal import propose_crafter_patterns_from_partial_maps
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
  assert len(visited) == int(known_mask.sum()), 'Known mask must be connected from the start.'
  return order


def build_connected_reverse_trajectory_examples(
    replay_entry: CrafterReplayEntry,
    pure_samples_per_map: int,
    seed: int,
    unknown_id: int,
) -> list[CrafterTransitionExample]:
  """Build next-reveal transition examples from one replay entry."""
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
      examples.append(CrafterTransitionExample(
          partial_map_ids=partial,
          target_pos=tuple(pos),
          target_class_id=int(full_map_ids[pos]),
      ))
      prefix_mask[pos] = True
    del sample_index
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
        capacity=int(cfg.replay_buffer.capacity))
    self._rng = np.random.default_rng(int(cfg.sampling_method.seed))
    self.library = CrafterSkillLibrary(
        codec=self.codec,
        device=torch.device(str(cfg.device)),
    )
    self._initialize_patterns()

  def _initialize_patterns(self):
    if bool(self.cfg.initial_patterns.add_center_only_placeholders):
      for class_id in range(self.codec.num_classes):
        self.library.add_pattern(CrossShapedPattern.make_ungated_center_only(class_id))
    for preset in self.cfg.initial_patterns.presets:
      self.library.add_pattern(CrossShapedPattern(
          center_id=self.codec.encode_material(str(preset.center)),
          top_id=self.codec.encode_material(str(preset.top)),
          bottom_id=self.codec.encode_material(str(preset.bottom)),
          left_id=self.codec.encode_material(str(preset.left)),
          right_id=self.codec.encode_material(str(preset.right)),
      ))

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
    self.replay_buffer.add(CrafterReplayEntry(
        full_map_ids=full_map_ids.detach().cpu().clone(),
        final_known_mask=final_known_mask.detach().cpu().clone(),
        start_pos=tuple(start_pos),
        episode_index=int(episode_index),
        seed=seed,
    ))

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
    max_example_learned = int(self.cfg.max_example_learned)
    if max_example_learned > 0:
      examples = examples[:max_example_learned]
    solver = PositiveWeightCrafterSolver(
        max_iters=int(self.cfg.lbfgs.max_iters),
        lr=float(self.cfg.lbfgs.lr),
    )
    solver.solve(self.library, examples, eps=float(self.cfg.eps))
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
    for replay_entry in replay_entries:
      entry_seed = int(self._rng.integers(0, 2**31 - 1))
      examples.extend(build_connected_reverse_trajectory_examples(
          replay_entry=replay_entry,
          pure_samples_per_map=int(self.cfg.sampling_method.pure_samples_per_map),
          seed=entry_seed,
          unknown_id=self.codec.unknown_id,
      ))
    return examples

  def _maybe_add_proposals(self, replay_entries: list[CrafterReplayEntry]) -> int:
    if not bool(self.cfg.skill_proposal.enabled):
      return 0
    examples = replay_entries[: int(self.cfg.skill_proposal.max_examples_in_prompt)]
    partial_maps = []
    for replay_entry in examples:
      partial = torch.full_like(replay_entry.full_map_ids, fill_value=self.codec.unknown_id)
      partial[replay_entry.final_known_mask] = replay_entry.full_map_ids[replay_entry.final_known_mask]
      partial_maps.append(partial)
    proposed = propose_crafter_patterns_from_partial_maps(
        partial_map_ids_examples=partial_maps,
        codec=self.codec,
        skill_proposal_cfg=self.cfg.skill_proposal,
    )
    added = 0
    for pattern in proposed[: int(self.cfg.skill_proposal.patterns_per_trigger)]:
      added += int(self.library.add_pattern(pattern))
    return added
