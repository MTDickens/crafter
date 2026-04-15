"""Cross-shaped pattern learning and inference for Crafter maps.

The implementation is intentionally white-box:

- One pattern template, :class:`CrossShapedPattern`, captures a center tile plus
  its 4-neighborhood.
- Training stores one non-negative weight for each instantiated exact pattern.
- Inference on partially observed maps marginalizes over the unobserved slots by
  summing weights of all exact patterns that agree with the observed subset.

That marginalization is the key simplification: even with only one exact
cross-shaped template family, the same library can express pairwise evidence,
three-neighbor evidence, and full 5-tile evidence without adding more template
families or hidden neural state.
"""

from __future__ import annotations

from abc import ABC
import dataclasses
import json
import math
import pathlib
from typing import Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from crafter import constants

Position = tuple[int, int]
DirectionName = str

DIRECTION_NAMES: tuple[DirectionName, ...] = ("top", "bottom", "left", "right")
DIRECTION_OFFSETS: dict[DirectionName, Position] = {
    "top": (0, -1),
    "bottom": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}
_UNKNOWN_MATERIAL_INDEX = -1


class Pattern(ABC):
    """Marker base class for hand-designed pattern templates."""


@dataclasses.dataclass(frozen=True)
class CrossShapedPattern(Pattern):
    """Exact center-plus-4-neighbor pattern."""

    center: str
    top: str
    bottom: str
    left: str
    right: str


@dataclasses.dataclass(frozen=True)
class MaterialPosterior:
    """Posterior over candidate center materials at one map position."""

    position: Position
    probabilities: dict[str, float]
    best_material: str
    confidence: float


@dataclasses.dataclass(frozen=True)
class EvaluationMetrics:
    """Compact held-out prediction metrics for one trained library."""

    total_positions: int
    top1_accuracy: float
    average_true_probability: float
    average_entropy: float
    average_negative_log_likelihood: float


class PatternLibrary:
    """Weighted library of exact cross-shaped terrain patterns.

    The library stores one non-negative weight per exact
    :class:`CrossShapedPattern` instance plus a center-material prior. Inference
    on a partially observed neighborhood uses exact matching on the observed
    slots and marginalizes the rest by summing compatible pattern weights.
    Internally, pattern weights and contexts are encoded as torch tensors so
    training and batched inference can run on CPU, CUDA, or MPS.
    """

    def __init__(
        self,
        materials: Iterable[str],
        prior_counts: Mapping[str, float],
        pattern_weights: Mapping[CrossShapedPattern, float],
        *,
        prior_scale: float,
        metadata: Mapping[str, object] | None = None,
        device: str | torch.device = "cpu",
    ):
        self.materials = tuple(materials)
        if not self.materials:
            raise RuntimeError("Pattern library must contain at least one material")
        self.material_to_index = {material: index for index, material in enumerate(self.materials)}
        self.prior_counts = {
            material: float(prior_counts[material]) for material in self.materials
        }
        self.pattern_weights = {
            pattern: float(weight) for pattern, weight in pattern_weights.items()
        }
        self.prior_scale = float(prior_scale)
        self.metadata = {} if metadata is None else dict(metadata)
        self.device = _resolve_torch_device(device)
        self.total_prior_count = float(sum(self.prior_counts.values()))
        if self.total_prior_count <= 0.0:
            raise RuntimeError("Pattern library has zero total prior count")
        self._patterns_by_center = {material: [] for material in self.materials}
        for pattern, weight in self.pattern_weights.items():
            if pattern.center not in self._patterns_by_center:
                raise RuntimeError(f"Pattern center is outside material vocabulary: {pattern}")
            if weight <= 0.0:
                raise RuntimeError(f"Pattern weight must be positive: {pattern} -> {weight}")
            self._patterns_by_center[pattern.center].append((pattern, weight))
        self._posterior_cache: dict[tuple[tuple[str, str], ...], dict[str, float]] = {}
        self._rebuild_tensors()

    @classmethod
    def empty(
        cls,
        materials: Iterable[str],
        *,
        prior_scale: float,
        metadata: Mapping[str, object] | None = None,
        device: str | torch.device = "cpu",
    ) -> "PatternLibrary":
        material_list = tuple(materials)
        prior_counts = {material: 1e-6 for material in material_list}
        return cls(
            material_list,
            prior_counts,
            {},
            prior_scale=prior_scale,
            metadata=metadata,
            device=device,
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "PatternLibrary":
        payload = json.loads(pathlib.Path(path).read_text())
        if payload["version"] != 1:
            raise RuntimeError(f"Unsupported pattern library version: {payload['version']}")
        materials = tuple(payload["materials"])
        prior_counts_payload = payload["prior_counts"]
        prior_counts = {
            material: float(prior_counts_payload[material]) for material in materials
        }
        pattern_weights: dict[CrossShapedPattern, float] = {}
        for record in payload["patterns"]:
            pattern = CrossShapedPattern(
                center=record["center"],
                top=record["top"],
                bottom=record["bottom"],
                left=record["left"],
                right=record["right"],
            )
            pattern_weights[pattern] = float(record["weight"])
        metadata = payload["metadata"]
        return cls(
            materials,
            prior_counts,
            pattern_weights,
            prior_scale=float(payload["prior_scale"]),
            metadata=metadata,
            device=device,
        )

    def save(self, path: str | pathlib.Path) -> None:
        records = []
        for pattern, weight in sorted(
            self.pattern_weights.items(),
            key=lambda item: (
                item[0].center,
                item[0].top,
                item[0].bottom,
                item[0].left,
                item[0].right,
            ),
        ):
            records.append(
                {
                    "center": pattern.center,
                    "top": pattern.top,
                    "bottom": pattern.bottom,
                    "left": pattern.left,
                    "right": pattern.right,
                    "weight": weight,
                }
            )
        payload = {
            "version": 1,
            "materials": list(self.materials),
            "prior_counts": dict(self.prior_counts),
            "prior_scale": self.prior_scale,
            "patterns": records,
            "metadata": dict(self.metadata),
        }
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def num_patterns(self) -> int:
        return len(self.pattern_weights)

    def add_pattern_weight(self, pattern: CrossShapedPattern, weight: float) -> None:
        if weight <= 0.0:
            raise RuntimeError(f"Pattern weight must be positive, got {weight}")
        self._apply_pattern_count_updates({pattern: float(weight)})

    def prior_distribution(self) -> dict[str, float]:
        raw = {material: self.prior_counts[material] for material in self.materials}
        return _normalize_distribution(raw)

    def posterior_from_context(self, observed_context: Mapping[str, str]) -> dict[str, float]:
        """Return ``P(center | observed subset of 4-neighbors)``.

        ``observed_context`` uses direction keys from :data:`DIRECTION_NAMES`.
        Unobserved directions are omitted and are marginalized by summing the
        weights of all exact patterns that match the observed subset.
        """
        for direction in observed_context:
            if direction not in DIRECTION_OFFSETS:
                raise RuntimeError(f"Unsupported direction in observed context: {direction}")
        context_key = tuple(sorted(observed_context.items()))
        if context_key in self._posterior_cache:
            return dict(self._posterior_cache[context_key])

        encoded_context = torch.full(
            (1, len(DIRECTION_NAMES)),
            _UNKNOWN_MATERIAL_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        observed_mask = torch.zeros((1, len(DIRECTION_NAMES)), dtype=torch.bool, device=self.device)
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            material = observed_context.get(direction)
            if material is None:
                continue
            encoded_context[0, direction_index] = self.material_to_index[material]
            observed_mask[0, direction_index] = True
        posterior_tensor = self._posterior_from_encoded_contexts(encoded_context, observed_mask)[0]
        posterior = self._posterior_tensor_to_dict(posterior_tensor)
        self._posterior_cache[context_key] = posterior
        return dict(posterior)

    def infer_unknown_materials(
        self,
        *,
        area: tuple[int, int],
        known_materials: Mapping[Position, str],
        candidate_positions: Iterable[Position] | None = None,
        commit_threshold: float | None = None,
        commit_max_passes: int = 1,
    ) -> dict[Position, MaterialPosterior]:
        """Infer material posteriors for unknown tiles.

        Args:
            area: Full map shape as ``(width, height)``.
            known_materials: Mapping from observed tile positions to material.
            candidate_positions: Unknown tiles whose posteriors should be
                returned. When omitted, every unknown tile is inferred.
            commit_threshold: Optional confidence threshold for recursively
                committing high-confidence inferred tiles into the context used
                by later passes. This does not mutate the real ``KnownWorld``;
                it only affects pattern-based inference.
            commit_max_passes: Maximum number of inference/commit passes.
        """
        if commit_threshold is not None and not (0.0 < commit_threshold <= 1.0):
            raise RuntimeError(f"commit_threshold must lie in (0, 1], got {commit_threshold}")
        if commit_max_passes < 1:
            raise RuntimeError(f"commit_max_passes must be >= 1, got {commit_max_passes}")

        current_materials = dict(known_materials)
        if candidate_positions is None:
            requested_positions = [
                (x, y)
                for x in range(area[0])
                for y in range(area[1])
                if (x, y) not in current_materials
            ]
        else:
            requested_positions = []
            for pos in candidate_positions:
                if pos in current_materials:
                    continue
                requested_positions.append(pos)

        if commit_threshold is None:
            return self._infer_one_pass(area, current_materials, requested_positions)

        all_unknown_positions = [
            (x, y)
            for x in range(area[0])
            for y in range(area[1])
            if (x, y) not in current_materials
        ]
        latest_predictions: dict[Position, MaterialPosterior] = {}
        for _ in range(commit_max_passes):
            latest_predictions = self._infer_one_pass(area, current_materials, all_unknown_positions)
            new_commits = []
            for pos in all_unknown_positions:
                posterior = latest_predictions[pos]
                if posterior.confidence < commit_threshold:
                    continue
                if pos in current_materials:
                    continue
                new_commits.append((pos, posterior.best_material))
            if not new_commits:
                break
            for pos, material in new_commits:
                current_materials[pos] = material
            all_unknown_positions = [
                pos for pos in all_unknown_positions if pos not in current_materials
            ]
            if not all_unknown_positions:
                break
        output: dict[Position, MaterialPosterior] = {}
        for pos in requested_positions:
            output[pos] = latest_predictions[pos]
        return output

    def _infer_one_pass(
        self,
        area: tuple[int, int],
        current_materials: Mapping[Position, str],
        positions: Iterable[Position],
    ) -> dict[Position, MaterialPosterior]:
        positions = list(positions)
        if not positions:
            return {}

        known_grid = torch.full(
            area,
            _UNKNOWN_MATERIAL_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        for pos, material in current_materials.items():
            known_grid[pos] = self.material_to_index[material]

        position_tensor = torch.tensor(positions, dtype=torch.long, device=self.device)
        encoded_context = torch.full(
            (len(positions), len(DIRECTION_NAMES)),
            _UNKNOWN_MATERIAL_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        observed_mask = torch.zeros(
            (len(positions), len(DIRECTION_NAMES)),
            dtype=torch.bool,
            device=self.device,
        )
        for direction_index, direction in enumerate(DIRECTION_NAMES):
            offset = DIRECTION_OFFSETS[direction]
            neighbor_x = position_tensor[:, 0] + offset[0]
            neighbor_y = position_tensor[:, 1] + offset[1]
            in_bounds = (
                (neighbor_x >= 0)
                & (neighbor_x < area[0])
                & (neighbor_y >= 0)
                & (neighbor_y < area[1])
            )
            if not bool(in_bounds.any()):
                continue
            candidate_rows = torch.nonzero(in_bounds, as_tuple=False).squeeze(1)
            neighbor_values = known_grid[neighbor_x[candidate_rows], neighbor_y[candidate_rows]]
            is_observed = neighbor_values >= 0
            if not bool(is_observed.any()):
                continue
            observed_rows = candidate_rows[is_observed]
            encoded_context[observed_rows, direction_index] = neighbor_values[is_observed]
            observed_mask[observed_rows, direction_index] = True

        posterior_tensor = self._posterior_from_encoded_contexts(encoded_context, observed_mask)
        predictions: dict[Position, MaterialPosterior] = {}
        for row_index, pos in enumerate(positions):
            posterior = self._posterior_tensor_to_dict(posterior_tensor[row_index])
            best_material = max(
                self.materials,
                key=lambda material: (posterior[material], material),
            )
            predictions[pos] = MaterialPosterior(
                position=pos,
                probabilities=posterior,
                best_material=best_material,
                confidence=posterior[best_material],
            )
        return predictions

    def _posterior_from_encoded_contexts(
        self,
        encoded_context: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_context.ndim != 2 or encoded_context.shape[1] != len(DIRECTION_NAMES):
            raise RuntimeError(
                f"encoded_context must have shape (N, {len(DIRECTION_NAMES)}), got {tuple(encoded_context.shape)}"
            )
        if observed_mask.shape != encoded_context.shape:
            raise RuntimeError(
                f"observed_mask shape {tuple(observed_mask.shape)} must match encoded_context shape "
                f"{tuple(encoded_context.shape)}"
            )
        encoded_context = encoded_context.to(device=self.device, dtype=torch.long)
        observed_mask = observed_mask.to(device=self.device, dtype=torch.bool)
        raw_scores = self.prior_scale * self._prior_counts_tensor.unsqueeze(0).expand(
            encoded_context.shape[0], -1
        )
        if self._pattern_weights_tensor.numel() > 0:
            matches = self._pattern_neighbor_tensor.unsqueeze(0) == encoded_context.unsqueeze(1)
            compatible = torch.logical_or(~observed_mask.unsqueeze(1), matches)
            matched_weights = compatible.all(dim=2).to(self._float_dtype)
            matched_weights = matched_weights * self._pattern_weights_tensor.unsqueeze(0)
            raw_scores = raw_scores + matched_weights @ self._pattern_center_one_hot
        return raw_scores / raw_scores.sum(dim=1, keepdim=True)

    def fit_material_map(self, material_map: np.ndarray, *, position_mask: np.ndarray | None = None) -> None:
        """Add exact cross-pattern counts from one fully observed material map.

        Args:
            material_map: ``(width, height)`` array of material-token strings.
            position_mask: Optional boolean mask selecting which center
                positions should contribute training weight.
        """
        _validate_material_map(material_map)
        width, height = material_map.shape
        if position_mask is not None and position_mask.shape != material_map.shape:
            raise RuntimeError(
                f"position_mask shape {position_mask.shape} must match material_map shape {material_map.shape}"
            )
        encoded_map = torch.as_tensor(
            _encode_material_map(material_map, self.material_to_index),
            dtype=torch.long,
            device=self.device,
        )
        center = encoded_map[1:-1, 1:-1]
        pattern_tensor = torch.stack(
            (
                center,
                encoded_map[1:-1, :-2],
                encoded_map[1:-1, 2:],
                encoded_map[:-2, 1:-1],
                encoded_map[2:, 1:-1],
            ),
            dim=-1,
        ).reshape(-1, 5)
        if position_mask is not None:
            mask_tensor = torch.as_tensor(
                position_mask[1:-1, 1:-1].reshape(-1),
                dtype=torch.bool,
                device=self.device,
            )
            pattern_tensor = pattern_tensor[mask_tensor]
        if pattern_tensor.numel() == 0:
            return
        self._apply_pattern_tensor_counts(pattern_tensor)

    def evaluate_on_material_maps(
        self,
        material_maps: Iterable[np.ndarray],
        *,
        evaluation_policy: str,
        masks_per_map: int,
        observed_fraction_min: float,
        observed_fraction_max: float,
        rng: np.random.Generator,
        start_position: Position | None,
    ) -> EvaluationMetrics:
        total_positions = 0
        total_correct = 0
        total_true_probability = 0.0
        total_entropy = 0.0
        total_nll = 0.0
        for material_map in material_maps:
            _validate_material_map(material_map)
            width, height = material_map.shape
            if evaluation_policy == "all_tiles":
                known_materials = {}
                predictions = self.infer_unknown_materials(
                    area=(width, height),
                    known_materials=known_materials,
                )
                evaluation_positions = [
                    (x, y)
                    for x in range(width)
                    for y in range(height)
                ]
                for pos in evaluation_positions:
                    posterior = predictions[pos].probabilities
                    true_material = str(material_map[pos])
                    total_positions += 1
                    if predictions[pos].best_material == true_material:
                        total_correct += 1
                    total_true_probability += posterior[true_material]
                    total_entropy += _distribution_entropy(posterior)
                    total_nll += -math.log(max(posterior[true_material], 1e-12))
                continue

            if evaluation_policy != "connected_frontier":
                raise RuntimeError(f"Unsupported evaluation_policy: {evaluation_policy}")

            for _ in range(masks_per_map):
                observed_mask = sample_connected_observation_mask(
                    (width, height),
                    observed_fraction_min=observed_fraction_min,
                    observed_fraction_max=observed_fraction_max,
                    rng=rng,
                    start_position=start_position,
                )
                known_materials = material_map_to_position_dict(material_map, observed_mask)
                frontier_positions = connected_frontier_positions(observed_mask)
                evaluation_positions = [
                    pos for pos in frontier_positions if _is_interior((width, height), pos)
                ]
                predictions = self.infer_unknown_materials(
                    area=(width, height),
                    known_materials=known_materials,
                    candidate_positions=evaluation_positions,
                )
                for pos in evaluation_positions:
                    posterior = predictions[pos].probabilities
                    true_material = str(material_map[pos])
                    total_positions += 1
                    if predictions[pos].best_material == true_material:
                        total_correct += 1
                    total_true_probability += posterior[true_material]
                    total_entropy += _distribution_entropy(posterior)
                    total_nll += -math.log(max(posterior[true_material], 1e-12))
        if total_positions == 0:
            raise RuntimeError("No positions were evaluated")
        return EvaluationMetrics(
            total_positions=total_positions,
            top1_accuracy=total_correct / total_positions,
            average_true_probability=total_true_probability / total_positions,
            average_entropy=total_entropy / total_positions,
            average_negative_log_likelihood=total_nll / total_positions,
        )

    def _apply_pattern_tensor_counts(self, pattern_tensor: torch.Tensor) -> None:
        pattern_tensor = pattern_tensor.to(device=self.device, dtype=torch.long)
        center_counts = torch.bincount(
            pattern_tensor[:, 0],
            minlength=len(self.materials),
        ).to(dtype=self._float_dtype)
        try:
            unique_patterns, counts = torch.unique(
                pattern_tensor, dim=0, return_counts=True
            )
        except (NotImplementedError, RuntimeError):
            unique_patterns, counts = torch.unique(
                pattern_tensor.to("cpu"), dim=0, return_counts=True
            )
        pattern_count_updates: dict[CrossShapedPattern, float] = {}
        for pattern_row, count in zip(unique_patterns.tolist(), counts.tolist(), strict=True):
            pattern = CrossShapedPattern(
                center=self.materials[int(pattern_row[0])],
                top=self.materials[int(pattern_row[1])],
                bottom=self.materials[int(pattern_row[2])],
                left=self.materials[int(pattern_row[3])],
                right=self.materials[int(pattern_row[4])],
            )
            pattern_count_updates[pattern] = float(count)
        self._apply_pattern_count_updates(
            pattern_count_updates,
            center_count_updates=center_counts.detach().cpu().tolist(),
        )

    def _apply_pattern_count_updates(
        self,
        pattern_count_updates: Mapping[CrossShapedPattern, float],
        *,
        center_count_updates: list[float] | None = None,
    ) -> None:
        if center_count_updates is None:
            center_count_updates = [0.0] * len(self.materials)
            for pattern, weight in pattern_count_updates.items():
                center_count_updates[self.material_to_index[pattern.center]] += float(weight)
        if len(center_count_updates) != len(self.materials):
            raise RuntimeError(
                f"center_count_updates must have length {len(self.materials)}, "
                f"got {len(center_count_updates)}"
            )
        total_added = 0.0
        for material, update in zip(self.materials, center_count_updates, strict=True):
            update = float(update)
            if update < 0.0:
                raise RuntimeError(f"Center count updates must be non-negative, got {update}")
            self.prior_counts[material] += update
            total_added += update
        for pattern, weight in pattern_count_updates.items():
            weight = float(weight)
            if weight <= 0.0:
                raise RuntimeError(f"Pattern weight must be positive, got {weight}")
            if pattern.center not in self.prior_counts:
                raise RuntimeError(f"Pattern center outside vocabulary: {pattern.center}")
            if pattern in self.pattern_weights:
                self.pattern_weights[pattern] += weight
            else:
                self.pattern_weights[pattern] = weight
        self.total_prior_count += total_added
        self._rebuild_pattern_groups()
        self._rebuild_tensors()

    def _rebuild_pattern_groups(self) -> None:
        self._patterns_by_center = {material: [] for material in self.materials}
        for pattern, weight in self.pattern_weights.items():
            self._patterns_by_center[pattern.center].append((pattern, weight))
        self._posterior_cache = {}

    def _rebuild_tensors(self) -> None:
        self._float_dtype = torch.float32
        self._prior_counts_tensor = torch.tensor(
            [self.prior_counts[material] for material in self.materials],
            dtype=self._float_dtype,
            device=self.device,
        )
        if self.pattern_weights:
            sorted_patterns = sorted(
                self.pattern_weights.items(),
                key=lambda item: (
                    item[0].center,
                    item[0].top,
                    item[0].bottom,
                    item[0].left,
                    item[0].right,
                ),
            )
            self._pattern_center_tensor = torch.tensor(
                [self.material_to_index[pattern.center] for pattern, _ in sorted_patterns],
                dtype=torch.long,
                device=self.device,
            )
            self._pattern_neighbor_tensor = torch.tensor(
                [
                    [
                        self.material_to_index[pattern.top],
                        self.material_to_index[pattern.bottom],
                        self.material_to_index[pattern.left],
                        self.material_to_index[pattern.right],
                    ]
                    for pattern, _ in sorted_patterns
                ],
                dtype=torch.long,
                device=self.device,
            )
            self._pattern_weights_tensor = torch.tensor(
                [weight for _, weight in sorted_patterns],
                dtype=self._float_dtype,
                device=self.device,
            )
            self._pattern_center_one_hot = F.one_hot(
                self._pattern_center_tensor, num_classes=len(self.materials)
            ).to(dtype=self._float_dtype)
        else:
            self._pattern_center_tensor = torch.empty(0, dtype=torch.long, device=self.device)
            self._pattern_neighbor_tensor = torch.empty(
                (0, len(DIRECTION_NAMES)), dtype=torch.long, device=self.device
            )
            self._pattern_weights_tensor = torch.empty(0, dtype=self._float_dtype, device=self.device)
            self._pattern_center_one_hot = torch.empty(
                (0, len(self.materials)), dtype=self._float_dtype, device=self.device
            )

    def _posterior_tensor_to_dict(self, posterior_tensor: torch.Tensor) -> dict[str, float]:
        values = posterior_tensor.detach().to("cpu").tolist()
        return {
            material: float(values[index])
            for index, material in enumerate(self.materials)
        }


def extract_cross_shaped_pattern(material_map: np.ndarray, pos: Position) -> CrossShapedPattern:
    """Extract the exact cross pattern centered at ``pos`` from ``material_map``."""
    _validate_material_map(material_map)
    if not _is_interior(material_map.shape, pos):
        raise RuntimeError(f"Cross pattern requires an interior position, got {pos}")
    x, y = pos
    return CrossShapedPattern(
        center=str(material_map[x, y]),
        top=str(material_map[x, y - 1]),
        bottom=str(material_map[x, y + 1]),
        left=str(material_map[x - 1, y]),
        right=str(material_map[x + 1, y]),
    )


def material_map_to_position_dict(
    material_map: np.ndarray,
    observed_mask: np.ndarray,
) -> dict[Position, str]:
    """Convert an observed boolean mask into a sparse known-material mapping."""
    _validate_material_map(material_map)
    if material_map.shape != observed_mask.shape:
        raise RuntimeError(
            f"observed_mask shape {observed_mask.shape} must match material_map shape {material_map.shape}"
        )
    known_materials: dict[Position, str] = {}
    width, height = material_map.shape
    for x in range(width):
        for y in range(height):
            if not bool(observed_mask[x, y]):
                continue
            known_materials[(x, y)] = str(material_map[x, y])
    return known_materials


def connected_frontier_positions(observed_mask: np.ndarray) -> list[Position]:
    """Return unknown tiles adjacent to the connected observed region."""
    width, height = observed_mask.shape
    frontier = []
    for x in range(width):
        for y in range(height):
            if bool(observed_mask[x, y]):
                continue
            pos = (x, y)
            for neighbor in neighbors((width, height), pos):
                if bool(observed_mask[neighbor]):
                    frontier.append(pos)
                    break
    frontier.sort()
    return frontier


def sample_connected_observation_mask(
    area: tuple[int, int],
    *,
    observed_fraction_min: float,
    observed_fraction_max: float,
    rng: np.random.Generator,
    start_position: Position | None,
) -> np.ndarray:
    """Sample a blob-like connected observed region.

    The mask is grown outward from ``start_position`` or from the map center.
    This approximates the planner's observed region more closely than i.i.d.
    random masking, which is why it is used as the default policy for training
    and evaluation.
    """
    if not (0.0 < observed_fraction_min <= observed_fraction_max <= 1.0):
        raise RuntimeError(
            "observed fractions must satisfy 0 < min <= max <= 1"
        )
    width, height = area
    if start_position is None:
        start = (width // 2, height // 2)
    else:
        start = start_position
    if not _in_bounds(area, start):
        raise RuntimeError(f"start_position out of bounds: {start}")
    total_tiles = width * height
    observed_fraction = float(rng.uniform(observed_fraction_min, observed_fraction_max))
    target_tiles = max(1, int(round(observed_fraction * total_tiles)))

    observed = np.zeros(area, dtype=bool)
    frontier_list: list[Position] = []
    frontier_set: set[Position] = set()

    def add_frontier(pos: Position) -> None:
        if not _in_bounds(area, pos):
            return
        if bool(observed[pos]):
            return
        if pos in frontier_set:
            return
        frontier_set.add(pos)
        frontier_list.append(pos)

    observed[start] = True
    for neighbor in neighbors(area, start):
        add_frontier(neighbor)

    observed_count = 1
    while observed_count < target_tiles and frontier_list:
        index = int(rng.integers(0, len(frontier_list)))
        pos = frontier_list[index]
        frontier_list[index] = frontier_list[-1]
        frontier_list.pop()
        frontier_set.remove(pos)
        if bool(observed[pos]):
            continue
        observed[pos] = True
        observed_count += 1
        for neighbor in neighbors(area, pos):
            add_frontier(neighbor)
    return observed


def fit_pattern_library_from_material_maps(
    material_maps: Iterable[np.ndarray],
    *,
    materials: Iterable[str] | None = None,
    prior_scale: float,
    training_policy: str,
    masks_per_map: int,
    observed_fraction_min: float,
    observed_fraction_max: float,
    rng: np.random.Generator,
    start_position: Position | None,
    metadata: Mapping[str, object] | None = None,
    device: str | torch.device = "cpu",
) -> PatternLibrary:
    """Fit a :class:`PatternLibrary` from fully observed historical maps."""
    if materials is None:
        vocabulary = tuple(constants.materials)
    else:
        vocabulary = tuple(materials)
    library = PatternLibrary.empty(
        vocabulary,
        prior_scale=prior_scale,
        metadata=metadata,
        device=device,
    )
    for material_map in material_maps:
        _validate_material_map(material_map)
        width, height = material_map.shape
        if training_policy == "all_tiles":
            library.fit_material_map(material_map)
            continue
        if training_policy != "connected_frontier":
            raise RuntimeError(f"Unsupported training_policy: {training_policy}")
        for _ in range(masks_per_map):
            observed_mask = sample_connected_observation_mask(
                (width, height),
                observed_fraction_min=observed_fraction_min,
                observed_fraction_max=observed_fraction_max,
                rng=rng,
                start_position=start_position,
            )
            position_mask = np.zeros((width, height), dtype=bool)
            for pos in connected_frontier_positions(observed_mask):
                if not _is_interior((width, height), pos):
                    continue
                position_mask[pos] = True
            library.fit_material_map(material_map, position_mask=position_mask)
    return library


def material_map_from_world(world) -> np.ndarray:
    """Extract the full material map from a Crafter ``world`` instance."""
    area = tuple(int(x) for x in world.area)
    material_map = np.empty(area, dtype=object)
    for x in range(area[0]):
        for y in range(area[1]):
            material, _ = world[(x, y)]
            material_map[(x, y)] = material
    return material_map


def neighbors(area: tuple[int, int], pos: Position) -> tuple[Position, ...]:
    result = []
    for offset in DIRECTION_OFFSETS.values():
        candidate = (pos[0] + offset[0], pos[1] + offset[1])
        if _in_bounds(area, candidate):
            result.append(candidate)
    return tuple(result)


def _normalize_distribution(raw_scores: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(raw_scores.values()))
    if total <= 0.0:
        raise RuntimeError(f"Cannot normalize non-positive scores: {raw_scores}")
    return {key: float(value) / total for key, value in raw_scores.items()}


def _distribution_entropy(distribution: Mapping[str, float]) -> float:
    entropy = 0.0
    for probability in distribution.values():
        if probability <= 0.0:
            continue
        entropy -= probability * math.log(probability)
    return entropy


def _resolve_torch_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    resolved = torch.device(str(device))
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        print("[pattern-library] requested device 'mps' is unavailable; falling back to 'cpu'")
        return torch.device("cpu")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        print("[pattern-library] requested device 'cuda' is unavailable; falling back to 'cpu'")
        return torch.device("cpu")
    return resolved


def _encode_material_map(
    material_map: np.ndarray,
    material_to_index: Mapping[str, int],
) -> np.ndarray:
    try:
        encode_fn = np.vectorize(lambda item: material_to_index[str(item)], otypes=[np.int64])
        return encode_fn(material_map)
    except KeyError as exc:
        raise RuntimeError(f"Material map contains unknown material: {exc}") from exc


def _in_bounds(area: tuple[int, int], pos: Position) -> bool:
    return 0 <= pos[0] < area[0] and 0 <= pos[1] < area[1]


def _is_interior(area: tuple[int, int], pos: Position) -> bool:
    return 1 <= pos[0] < area[0] - 1 and 1 <= pos[1] < area[1] - 1


def _validate_material_map(material_map: np.ndarray) -> None:
    if material_map.ndim != 2:
        raise RuntimeError(f"material_map must be 2D, got shape {material_map.shape}")
    if material_map.shape[0] < 3 or material_map.shape[1] < 3:
        raise RuntimeError(
            f"material_map must be at least 3x3 for cross patterns, got {material_map.shape}"
        )
