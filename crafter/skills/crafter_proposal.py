"""Crafter pattern proposal helpers."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from omegaconf import DictConfig

from crafter.skills.crafter_patterns import CrossShapedPattern
from crafter.utils.crafter_codec import CrafterTileCodec
from crafter.utils.llm_call import get_completion_text


@dataclass(frozen=True)
class CrafterProposalPromptExample:
  """One preprocessed prompt example for pattern proposal.

  Parameters
  ----------
  partial_map_ids : torch.Tensor
      Integer token grid shown in the prompt.
  source_episode_index : int
      Replay-buffer episode index from which the example came.
  original_shape : tuple[int, int]
      Shape of the original partial map before preprocessing.
  bbox_bounds : tuple[int, int, int, int] or None
      Optional ``(xmin, xmax, ymin, ymax)`` bbox in original coordinates.
  tile_bounds : tuple[int, int, int, int] or None
      Optional ``(xmin, xmax, ymin, ymax)`` tile bounds in original
      coordinates.
  """

  partial_map_ids: torch.Tensor
  source_episode_index: int
  original_shape: tuple[int, int]
  bbox_bounds: tuple[int, int, int, int] | None = None
  tile_bounds: tuple[int, int, int, int] | None = None


def partial_map_ids_to_prompt_block(
  partial_map_ids,
  codec: CrafterTileCodec,
) -> str:
  """Format one numeric partial map as prompt text."""
  rows = []
  for y in range(partial_map_ids.shape[1]):
    tokens = []
    for x in range(partial_map_ids.shape[0]):
      value = int(partial_map_ids[x, y])
      if value == codec.unknown_id:
        tokens.append('UNKNOWN')
      else:
        tokens.append(codec.decode_material(value).upper())
    rows.append(' '.join(tokens))
  return '\n'.join(rows)


def crop_partial_map_to_known_bbox(
  partial_map_ids: torch.Tensor,
  unknown_id: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int] | None]:
  """Crop a partial map to the minimum bbox containing all known cells."""
  known = partial_map_ids != unknown_id
  if not bool(torch.any(known)):
    return partial_map_ids, None
  xs, ys = torch.nonzero(known, as_tuple=True)
  xmin = int(xs.min())
  xmax = int(xs.max()) + 1
  ymin = int(ys.min())
  ymax = int(ys.max()) + 1
  return partial_map_ids[xmin:xmax, ymin:ymax], (xmin, xmax, ymin, ymax)


def _axis_tile_starts(length: int, tile_length: int, overlap_length: int) -> list[int]:
  """Return evenly spread tile starts covering one axis."""
  if tile_length < 1 or overlap_length < 0 or tile_length <= overlap_length:
    raise AssertionError(
      'Invalid tile settings: expected tile_length >= 1, '
      'overlap_length >= 0, and tile_length > overlap_length, got '
      f'tile_length={tile_length}, overlap_length={overlap_length}.'
    )
  if length <= tile_length:
    return [0]
  stride = tile_length - overlap_length
  tile_count = math.ceil((length - tile_length) / stride) + 1
  max_start = length - tile_length
  if tile_count == 1:
    return [0]
  starts = [
      int(round(index * max_start / (tile_count - 1)))
      for index in range(tile_count)
  ]
  starts[0] = 0
  starts[-1] = max_start
  return starts


def build_tiled_prompt_examples(
  partial_map_ids: torch.Tensor,
  source_episode_index: int,
  original_shape: tuple[int, int],
  tile_size: tuple[int, int],
  tile_overlap: tuple[int, int],
  bbox_bounds: tuple[int, int, int, int] | None = None,
) -> list[CrafterProposalPromptExample]:
  """Split one partial map into overlapping tiles."""
  tile_width, tile_height = tile_size
  overlap_width, overlap_height = tile_overlap
  x_starts = _axis_tile_starts(partial_map_ids.shape[0], tile_width, overlap_width)
  y_starts = _axis_tile_starts(partial_map_ids.shape[1], tile_height, overlap_height)
  x_offset = 0 if bbox_bounds is None else bbox_bounds[0]
  y_offset = 0 if bbox_bounds is None else bbox_bounds[2]
  return [
      CrafterProposalPromptExample(
          partial_map_ids=partial_map_ids[x:x + tile_width, y:y + tile_height],
          source_episode_index=source_episode_index,
          original_shape=original_shape,
          bbox_bounds=bbox_bounds,
          tile_bounds=(x + x_offset, x + x_offset + tile_width, y + y_offset, y + y_offset + tile_height),
      )
      for x in x_starts
      for y in y_starts
  ]


def build_crafter_proposal_prompt_examples(
  partial_map_ids: torch.Tensor,
  source_episode_index: int,
  codec: CrafterTileCodec,
  skill_proposal_cfg: DictConfig,
) -> list[CrafterProposalPromptExample]:
  """Preprocess one replay partial map into prompt examples."""
  original_shape = tuple(int(value) for value in partial_map_ids.shape)
  working_map = partial_map_ids
  bbox_bounds = None
  if bool(skill_proposal_cfg.use_bounding_box):
    working_map, bbox_bounds = crop_partial_map_to_known_bbox(
      partial_map_ids=working_map,
      unknown_id=codec.unknown_id,
    )
  if not bool(skill_proposal_cfg.enable_tiling):
    return [CrafterProposalPromptExample(
      partial_map_ids=working_map,
      source_episode_index=source_episode_index,
      original_shape=original_shape,
      bbox_bounds=bbox_bounds,
    )]
  tile_width, tile_height = [int(value) for value in skill_proposal_cfg.tile_size]
  overlap_width, overlap_height = [int(value) for value in skill_proposal_cfg.tile_overlap]
  return build_tiled_prompt_examples(
    partial_map_ids=working_map,
    source_episode_index=source_episode_index,
    original_shape=original_shape,
    tile_size=(tile_width, tile_height),
    tile_overlap=(overlap_width, overlap_height),
    bbox_bounds=bbox_bounds,
  )


def build_crafter_skill_proposal_prompt(
  prompt_examples: list[CrafterProposalPromptExample],
  codec: CrafterTileCodec,
  patterns_per_trigger: int,
  include_tile_coordinates: bool = False,
) -> str:
  """Build a single-shot prompt for proposing Crafter cross patterns."""
  rendered_examples = []
  for index, example in enumerate(prompt_examples, start=1):
    metadata = [f'episode={example.source_episode_index}']
    if example.bbox_bounds is not None:
      xmin, xmax, ymin, ymax = example.bbox_bounds
      metadata.append(f'bbox: x=[{xmin},{xmax}), y=[{ymin},{ymax})')
    if include_tile_coordinates and example.tile_bounds is not None:
      xmin, xmax, ymin, ymax = example.tile_bounds
      metadata.append(f'tile: x=[{xmin},{xmax}), y=[{ymin},{ymax})')
    rendered_examples.append(
      f'Example {index}\n'
      + '\n'.join(metadata)
      + '\n'
      + partial_map_ids_to_prompt_block(example.partial_map_ids, codec)
    )
  return (
    "You are given partially revealed Crafter material maps.\n"
    "Crafter is essentially 2D Minecraft; so also use that world knowledge to infer.\n"
    'Each pattern should use top/bottom/left/right to predict the center.\n'
    f'Propose at most {patterns_per_trigger} reusable cross-shaped patterns.\n'
    'Return strict JSON only with shape '
    '{"patterns": [{"center": "...", "top": "...", "bottom": "...", '
    '"left": "...", "right": "..."}]}.\n'
    'Use only world-generation material names. Do not use UNKNOWN.\n\n'
    + '\n\n'.join(rendered_examples)
  )


def get_crafter_skill_proposal_text(
  skill_proposal_cfg: DictConfig,
  prompt: str,
) -> str:
  """Return one proposal response from dummy text or the configured LLM."""
  if skill_proposal_cfg.get('dummy_response_path', None):
    return Path(skill_proposal_cfg.dummy_response_path).read_text()
  llm_cfg = cast(DictConfig, skill_proposal_cfg.llm_cfg)
  return get_completion_text([prompt], llm_cfg)


def parse_crafter_skill_proposal_response(
  response_text: str,
  codec: CrafterTileCodec,
) -> list[CrossShapedPattern]:
  """Parse proposed Crafter cross patterns from strict JSON."""
  raw_patterns = json.loads(response_text)['patterns']
  parsed: list[CrossShapedPattern] = []
  for raw_pattern in raw_patterns:
    try:
      parsed.append(
        CrossShapedPattern(
          center_id=codec.encode_material(raw_pattern['center']),
          top_id=codec.encode_material(raw_pattern['top']),
          bottom_id=codec.encode_material(raw_pattern['bottom']),
          left_id=codec.encode_material(raw_pattern['left']),
          right_id=codec.encode_material(raw_pattern['right']),
        )
      )
    except Exception as exc:  # pragma: no cover - parser failures are input-dependent.
      warnings.warn(f'Skipping invalid Crafter proposal pattern {raw_pattern!r}: {exc}')
  return parsed


def propose_crafter_patterns_from_partial_maps(
  prompt_examples: list[CrafterProposalPromptExample],
  codec: CrafterTileCodec,
  skill_proposal_cfg: DictConfig,
) -> list[CrossShapedPattern]:
  """Propose new Crafter cross patterns from prepared prompt examples."""
  prompt = build_crafter_skill_proposal_prompt(
    prompt_examples=prompt_examples,
    codec=codec,
    patterns_per_trigger=int(skill_proposal_cfg.patterns_per_trigger),
    include_tile_coordinates=bool(skill_proposal_cfg.include_tile_coordinates),
  )
  response_text = get_crafter_skill_proposal_text(skill_proposal_cfg, prompt)
  patterns = parse_crafter_skill_proposal_response(response_text, codec)
  if len(patterns) > int(skill_proposal_cfg.patterns_per_trigger):
    raise AssertionError(
      'LLM proposed too many patterns: '
      f'{len(patterns)} > {int(skill_proposal_cfg.patterns_per_trigger)}'
    )
  return patterns
