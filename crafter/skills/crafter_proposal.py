"""Crafter pattern proposal helpers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import cast

from omegaconf import DictConfig

from crafter.skills.crafter_patterns import CrossShapedPattern
from crafter.utils.crafter_codec import CrafterTileCodec
from crafter.utils.llm_call import get_completion_text


def partial_map_ids_to_prompt_block(
  partial_map_ids,
  codec: CrafterTileCodec,
) -> str:
  """Format one numeric partial map as prompt text.

  Parameters
  ----------
  partial_map_ids : torch.Tensor or np.ndarray
      Integer token grid.
  codec : CrafterTileCodec
      Codec used to decode token ids.

  Returns
  -------
  str
      Human-readable prompt block.
  """
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


def build_crafter_skill_proposal_prompt(
  partial_map_ids_examples: list,
  codec: CrafterTileCodec,
  patterns_per_trigger: int,
) -> str:
  """Build a single-shot prompt for proposing Crafter cross patterns."""
  rendered_examples = [
    f'Example {index}\n{partial_map_ids_to_prompt_block(partial_map_ids, codec)}'
    for index, partial_map_ids in enumerate(partial_map_ids_examples, start=1)
  ]
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
  partial_map_ids_examples: list,
  codec: CrafterTileCodec,
  skill_proposal_cfg: DictConfig,
) -> list[CrossShapedPattern]:
  """Propose new Crafter cross patterns from replay examples."""
  prompt = build_crafter_skill_proposal_prompt(
    partial_map_ids_examples=partial_map_ids_examples,
    codec=codec,
    patterns_per_trigger=int(skill_proposal_cfg.patterns_per_trigger),
  )
  response_text = get_crafter_skill_proposal_text(skill_proposal_cfg, prompt)
  patterns = parse_crafter_skill_proposal_response(response_text, codec)
  if len(patterns) > int(skill_proposal_cfg.patterns_per_trigger):
    raise AssertionError(
      'LLM proposed too many patterns: '
      f'{len(patterns)} > {int(skill_proposal_cfg.patterns_per_trigger)}'
    )
  return patterns
