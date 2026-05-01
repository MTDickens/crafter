"""Crafter cross-shaped patterns and numeric skill inference.

This module implements the Crafter-specific pattern semantics used by the
online learner:

- the four neighbors gate a pattern,
- the center class is predicted,
- patterns operate on integer token tensors rather than symbolic strings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from crafter.utils.crafter_codec import CrafterTileCodec


@dataclass(frozen=True)
class CrossShapedPattern:
  """Conditional cross-shaped Crafter pattern.

  Parameters
  ----------
  center_id : int
      Predicted center class.
  top_id : int
      Required top-neighbor class.
  bottom_id : int
      Required bottom-neighbor class.
  left_id : int
      Required left-neighbor class.
  right_id : int
      Required right-neighbor class.
  use_gating : bool, default=True
      Whether the pattern should check the four neighbors. Ungated placeholder
      patterns must be created through :meth:`make_ungated_center_only`.
  """

  center_id: int
  top_id: int
  bottom_id: int
  left_id: int
  right_id: int
  use_gating: bool = True

  def __post_init__(self):
    if self.use_gating:
      assert min(
          self.center_id,
          self.top_id,
          self.bottom_id,
          self.left_id,
          self.right_id,
      ) >= 0, 'Regular patterns must use normal non-negative token ids.'
    else:
      assert self.top_id == self.bottom_id == self.left_id == self.right_id == -1, (
          'Ungated placeholder patterns must use the dedicated factory method.')

  @classmethod
  def make_ungated_center_only(cls, center_id: int) -> 'CrossShapedPattern':
    """Create an ungated center-only placeholder pattern.

    Parameters
    ----------
    center_id : int
        Predicted center class.

    Returns
    -------
    CrossShapedPattern
        Placeholder pattern that always gates successfully.
    """
    return cls(
        center_id=center_id,
        top_id=-1,
        bottom_id=-1,
        left_id=-1,
        right_id=-1,
        use_gating=False,
    )

  def key(self) -> tuple[int, int, int, int, int, bool]:
    """Return a hashable exact-deduplication key."""
    return (
        self.center_id,
        self.top_id,
        self.bottom_id,
        self.left_id,
        self.right_id,
        self.use_gating,
    )


class GatedCrafterCrossSkill:
  """One conditional Crafter cross-pattern skill.

  Parameters
  ----------
  pattern : CrossShapedPattern
      Pattern specification.
  codec : CrafterTileCodec
      Integer codec used by the learner.
  """

  def __init__(self, pattern: CrossShapedPattern, codec: CrafterTileCodec):
    self.pattern = pattern
    self.codec = codec

  def output_distribution(
      self,
      partial_map_ids: torch.Tensor,
      eps: float,
  ) -> torch.Tensor:
    """Return the pattern's smoothed center distribution.

    Parameters
    ----------
    partial_map_ids : torch.Tensor
        Partial map of shape ``(H, W)`` or ``(W, H)`` with integer tokens.
        The tensor is only used for shape/device propagation.
    eps : float
        Total smoothing mass placed on all non-center classes.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(C, H, W)`` representing the pattern's categorical
        output at every center cell.
    """
    num_classes = self.codec.num_classes
    assert 0.0 <= eps < 1.0, f'eps must lie in [0, 1), got {eps}'
    off_value = eps / max(num_classes - 1, 1)
    logits = torch.full(
        (num_classes, *partial_map_ids.shape),
        off_value,
        dtype=torch.float32,
        device=partial_map_ids.device,
    )
    logits[self.pattern.center_id, :, :] = 1.0 - eps
    return logits

  def gating_mask(self, partial_map_ids: torch.Tensor) -> torch.Tensor:
    """Return the pattern gate mask for every candidate center cell.

    Parameters
    ----------
    partial_map_ids : torch.Tensor
        Partial map integer grid of shape ``(H, W)``.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(H, W)``. ``True`` means the pattern gates at
        the corresponding center cell.
    """
    if not self.pattern.use_gating:
      return torch.ones_like(partial_map_ids, dtype=torch.bool)

    padded = F.pad(
        partial_map_ids,
        (1, 1, 1, 1),
        mode='constant',
        value=self.codec.boundary_id,
    )
    top = padded[:-2, 1:-1]
    bottom = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    neighbors = (
        (top, self.pattern.top_id),
        (bottom, self.pattern.bottom_id),
        (left, self.pattern.left_id),
        (right, self.pattern.right_id),
    )
    gating = torch.ones_like(partial_map_ids, dtype=torch.bool)
    for observed, expected in neighbors:
      known = observed != self.codec.unknown_id
      gating &= (~known) | (observed == expected)
    return gating


class CrafterSkillLibrary:
  """Weighted library of Crafter cross-pattern skills.

  Parameters
  ----------
  codec : CrafterTileCodec
      Integer token codec.
  patterns : list[CrossShapedPattern], optional
      Initial pattern list.
  raw_weights : torch.Tensor, optional
      Initial raw positive-weight parameters. They are converted to actual
      weights through ``softplus``.
  device : torch.device or str, optional
      Torch device used for inference and optimization.
  """

  def __init__(
      self,
      codec: CrafterTileCodec,
      patterns: list[CrossShapedPattern] | None = None,
      raw_weights: torch.Tensor | None = None,
      device: torch.device | str | None = None,
  ):
    self.codec = codec
    self.device = torch.device(device) if device is not None else torch.device('cpu')
    self.patterns: list[CrossShapedPattern] = list(patterns or [])
    if raw_weights is None:
      raw_weights = torch.zeros(len(self.patterns), dtype=torch.float32)
    assert raw_weights.ndim == 1, 'Expected 1D raw_weights tensor.'
    assert raw_weights.numel() == len(self.patterns), (
        'raw_weights must match the number of patterns.')
    self.raw_weights = raw_weights.to(device=self.device, dtype=torch.float32)

  def __len__(self) -> int:
    """Return the number of patterns currently stored."""
    return len(self.patterns)

  def exact_pattern_keys(self) -> set[tuple[int, int, int, int, int, bool]]:
    """Return exact deduplication keys for all active patterns."""
    return {pattern.key() for pattern in self.patterns}

  def add_pattern(self, pattern: CrossShapedPattern, raw_weight: float = 0.0) -> bool:
    """Add one exact-novel pattern.

    Parameters
    ----------
    pattern : CrossShapedPattern
        Candidate pattern.
    raw_weight : float, default=0.0
        Initial raw positive-weight parameter.

    Returns
    -------
    bool
        ``True`` if the pattern was appended, ``False`` if it already existed.
    """
    if pattern.key() in self.exact_pattern_keys():
      return False
    self.patterns.append(pattern)
    extra = torch.tensor([raw_weight], dtype=torch.float32, device=self.device)
    self.raw_weights = torch.cat([self.raw_weights, extra], dim=0)
    return True

  def positive_weights(self) -> torch.Tensor:
    """Return strictly positive pattern weights.

    Returns
    -------
    torch.Tensor
        Positive tensor of shape ``(num_patterns,)``.
    """
    return F.softplus(self.raw_weights)

  def _known_one_hot(self, partial_map_ids: torch.Tensor) -> torch.Tensor:
    known = partial_map_ids != self.codec.unknown_id
    clipped = partial_map_ids.clamp(min=0, max=self.codec.num_classes - 1)
    one_hot = F.one_hot(clipped, num_classes=self.codec.num_classes)
    one_hot = one_hot.permute(2, 0, 1).to(dtype=torch.float32)
    return one_hot * known.unsqueeze(0)

  def infer_probs(self, partial_map_ids: torch.Tensor, eps: float) -> torch.Tensor:
    """Infer a categorical material posterior for every cell.

    Parameters
    ----------
    partial_map_ids : torch.Tensor
        Integer partial map of shape ``(H, W)``.
    eps : float
        Skill-level smoothing mass.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(C, H, W)`` whose first axis enumerates material
        classes.
    """
    partial_map_ids = partial_map_ids.to(device=self.device, dtype=torch.long)
    num_classes = self.codec.num_classes
    known = partial_map_ids != self.codec.unknown_id
    inference_dtype = torch.float64
    uniform = torch.full(
        (num_classes, *partial_map_ids.shape),
        1.0 / num_classes,
        dtype=inference_dtype,
        device=self.device,
    )
    if not self.patterns:
      inferred = uniform
    else:
      numerators = torch.zeros_like(uniform)
      denominator = torch.zeros(
          partial_map_ids.shape, dtype=inference_dtype, device=self.device)
      weights = self.positive_weights().to(dtype=inference_dtype)
      if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
        raise AssertionError('Pattern weights contain nan/inf.')
      for index, pattern in enumerate(self.patterns):
        skill = GatedCrafterCrossSkill(pattern, self.codec)
        gate = skill.gating_mask(partial_map_ids).to(dtype=inference_dtype)
        output = skill.output_distribution(
            partial_map_ids, eps=eps).to(dtype=inference_dtype)
        numerators += weights[index] * gate.unsqueeze(0) * output
        denominator += weights[index] * gate
      inferred = uniform.clone()
      active = denominator > 0
      if torch.any(torch.isnan(denominator)) or torch.any(torch.isinf(denominator)):
        raise AssertionError('Pattern-weight denominator contains nan/inf.')
      inferred[:, active] = numerators[:, active] / denominator[active].unsqueeze(0)
    known_one_hot = self._known_one_hot(partial_map_ids).to(dtype=inference_dtype)
    inferred[:, known] = known_one_hot[:, known]
    column_sums = inferred.sum(dim=0, keepdim=True)
    if torch.any(torch.isnan(column_sums)) or torch.any(torch.isinf(column_sums)):
      raise AssertionError('Inferred probability sums contain nan/inf.')
    assert torch.all(column_sums > 0), 'Inferred probability sums must be positive.'
    inferred = inferred / column_sums
    column_sums = inferred.sum(dim=0)
    assert torch.allclose(
        column_sums,
        torch.ones_like(column_sums),
        atol=1e-5,
    ), 'Inferred probabilities must sum to 1 at every cell.'
    return inferred.to(dtype=torch.float32)

  def maybe_hard_infer(
      self,
      partial_map_ids: torch.Tensor,
      eps: float,
      threshold: float,
      candidate_mask: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Hard-fill high-confidence unknown cells.

    Parameters
    ----------
    partial_map_ids : torch.Tensor
        Integer partial map of shape ``(H, W)``.
    eps : float
        Skill-level smoothing mass.
    threshold : float
        Confidence threshold. Cells with maximum posterior probability at least
        this value are hard-filled.
    candidate_mask : torch.Tensor, optional
        Optional boolean mask restricting which cells may be filled.

    Returns
    -------
    tuple[torch.Tensor, list[tuple[int, int]]]
        Updated partial map and the list of newly inferred cells.
    """
    inferred = self.infer_probs(partial_map_ids, eps=eps)
    max_probs, argmax = inferred.max(dim=0)
    updated = partial_map_ids.clone()
    unknown = partial_map_ids == self.codec.unknown_id
    fillable = unknown & (max_probs >= threshold)
    if candidate_mask is not None:
      fillable &= candidate_mask.to(device=partial_map_ids.device, dtype=torch.bool)
    positions = list(zip(*torch.nonzero(fillable, as_tuple=True), strict=True))
    for x, y in positions:
      updated[x, y] = argmax[x, y]
    return updated, [(int(x), int(y)) for x, y in positions]

  def score_candidate_cells(
      self,
      partial_map_ids: torch.Tensor,
      candidate_mask: torch.Tensor,
      class_ids: list[int],
      eps: float,
  ) -> torch.Tensor:
    """Score candidate cells by posterior mass on target classes.

    Parameters
    ----------
    partial_map_ids : torch.Tensor
        Integer partial map of shape ``(H, W)``.
    candidate_mask : torch.Tensor
        Boolean candidate mask of shape ``(H, W)``.
    class_ids : list[int]
        Material-class ids whose posterior mass should be summed.
    eps : float
        Skill-level smoothing mass.

    Returns
    -------
    torch.Tensor
        Score tensor of shape ``(H, W)``.
    """
    inferred = self.infer_probs(partial_map_ids, eps=eps)
    scores = inferred[class_ids].sum(dim=0)
    return scores * candidate_mask.to(device=scores.device, dtype=scores.dtype)
