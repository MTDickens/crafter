"""Positive-weight optimization for Crafter pattern libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from crafter.skills.crafter_patterns import CrafterSkillLibrary


@dataclass(slots=True)
class CrafterTransitionExample:
  """One single-cell next-reveal training example.

  Parameters
  ----------
  partial_map_ids : torch.Tensor
      Integer partial map before the reveal.
  target_pos : tuple[int, int]
      Position of the newly revealed center cell.
  target_class_id : int
      True class of that newly revealed center cell.
  """

  partial_map_ids: torch.Tensor
  target_pos: tuple[int, int]
  target_class_id: int


@dataclass(slots=True)
class PositiveWeightOptimizationResult:
  """Result of positive-weight fitting."""

  raw_weights: torch.Tensor
  loss_history: list[float]


def single_transition_loss(
    library: CrafterSkillLibrary,
    example: CrafterTransitionExample,
    eps: float,
) -> torch.Tensor:
  """Return the smoothed single-cell cross-entropy loss.

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
      Scalar loss tensor.
  """
  inferred = library.infer_probs(example.partial_map_ids, eps=eps)
  target = torch.full(
      (library.codec.num_classes,),
      eps / max(library.codec.num_classes - 1, 1),
      dtype=torch.float32,
      device=library.device,
  )
  target[example.target_class_id] = 1.0 - eps
  predicted = inferred[:, example.target_pos[0], example.target_pos[1]]
  log_predicted = torch.log(predicted)
  return -(target * log_predicted).sum()


def mean_transition_loss(
    library: CrafterSkillLibrary,
    examples: Sequence[CrafterTransitionExample],
    eps: float,
) -> torch.Tensor:
  """Return the mean loss over transition examples."""
  if not examples:
    raise ValueError('Expected at least one transition example.')
  losses = [single_transition_loss(library, example, eps=eps) for example in examples]
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
      loss = mean_transition_loss(library, examples, eps=eps)
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
