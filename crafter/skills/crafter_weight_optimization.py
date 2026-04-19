"""Positive-weight optimization for Crafter pattern libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from crafter.skills.crafter_patterns import CrafterSkillLibrary


@dataclass(slots=True)
class CrafterTransitionExample:
  """One partial-map training example with one or more supervised targets.

  Parameters
  ----------
  partial_map_ids : torch.Tensor
      Integer partial map shown to the learner.
  target_positions : tuple[tuple[int, int], ...]
      Positions supervised by this example.
  target_class_ids : tuple[int, ...]
      True classes of the supervised positions. This tuple must align with
      ``target_positions`` one-to-one.
  """

  partial_map_ids: torch.Tensor
  target_positions: tuple[tuple[int, int], ...]
  target_class_ids: tuple[int, ...]


@dataclass(slots=True)
class PositiveWeightOptimizationResult:
  """Result of positive-weight fitting."""

  raw_weights: torch.Tensor
  loss_history: list[float]


def single_example_loss(
    library: CrafterSkillLibrary,
    example: CrafterTransitionExample,
    eps: float,
) -> torch.Tensor:
  """Return the smoothed mean cross-entropy loss for one training example.

  Parameters
  ----------
  library : CrafterSkillLibrary
      Pattern library being optimized.
  example : CrafterTransitionExample
      Training example.
  eps : float
      Smoothing mass.

  Returns
  -------
  torch.Tensor
      Scalar loss tensor equal to the mean loss over all supervised target
      positions in ``example``.
  """
  if len(example.target_positions) == 0:
    raise ValueError('Expected at least one target position per training example.')
  if len(example.target_positions) != len(example.target_class_ids):
    raise ValueError(
      'target_positions and target_class_ids must have the same length.'
    )
  inferred = library.infer_probs(example.partial_map_ids, eps=eps)
  losses = []
  for target_pos, target_class_id in zip(
      example.target_positions, example.target_class_ids, strict=True):
    target = torch.full(
        (library.codec.num_classes,),
        eps / max(library.codec.num_classes - 1, 1),
        dtype=torch.float32,
        device=library.device,
    )
    target[target_class_id] = 1.0 - eps
    predicted = inferred[:, target_pos[0], target_pos[1]]
    log_predicted = torch.log(predicted)
    losses.append(-(target * log_predicted).sum())
  return torch.stack(losses).mean()


def mean_example_loss(
    library: CrafterSkillLibrary,
    examples: Sequence[CrafterTransitionExample],
    eps: float,
) -> torch.Tensor:
  """Return the mean loss over training examples."""
  if not examples:
    raise ValueError('Expected at least one training example.')
  losses = [single_example_loss(library, example, eps=eps) for example in examples]
  return torch.stack(losses).mean()


class PositiveWeightCrafterSolver:
  """Fit positive pattern weights with L-BFGS.

  Parameters
  ----------
  max_iters : int, default=150
      Maximum L-BFGS iterations.
  lr : float, default=1.0
      L-BFGS learning rate.
  """

  def __init__(self, max_iters: int = 150, lr: float = 1.0):
    self.max_iters = int(max_iters)
    self.lr = float(lr)

  def solve(
      self,
      library: CrafterSkillLibrary,
      examples: Sequence[CrafterTransitionExample],
      eps: float,
  ) -> PositiveWeightOptimizationResult:
    """Optimize the library's raw weights in-place.

    Parameters
    ----------
    library : CrafterSkillLibrary
        Pattern library whose raw weights are updated.
    examples : sequence[CrafterTransitionExample]
        Training set.
    eps : float
        Skill-level smoothing mass.

    Returns
    -------
    PositiveWeightOptimizationResult
        Optimized raw weights and the recorded closure losses.
    """
    if not examples or len(library) == 0:
      return PositiveWeightOptimizationResult(
          raw_weights=library.raw_weights.detach().clone(),
          loss_history=[],
      )

    parameter = torch.nn.Parameter(library.raw_weights.detach().clone())
    optimizer = torch.optim.LBFGS([parameter], lr=self.lr, max_iter=self.max_iters)
    loss_history: list[float] = []

    def closure():
      optimizer.zero_grad()
      library.raw_weights = parameter
      loss = mean_example_loss(library, examples, eps=eps)
      if torch.isnan(loss) or torch.isinf(loss):
        raise AssertionError('Weight optimization produced nan/inf loss.')
      loss.backward()
      loss_history.append(float(loss.detach().cpu().item()))
      return loss

    optimizer.step(closure)
    library.raw_weights = parameter.detach().clone()
    return PositiveWeightOptimizationResult(
        raw_weights=library.raw_weights.detach().clone(),
        loss_history=loss_history,
    )
