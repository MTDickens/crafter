"""Helpers for constrained Crafter map generation via repeated reset."""

from __future__ import annotations

from typing import Callable, Mapping

from crafter import constants


MaterialCounts = dict[str, int]


def normalize_material_minimums(
    material_minimums: Mapping[str, int] | None,
) -> MaterialCounts:
    """Validate and normalize configured per-material lower bounds."""
    if material_minimums is None:
        return {}
    normalized: MaterialCounts = {}
    for material, minimum in material_minimums.items():
        if material not in constants.materials:
            raise ValueError(
                f"Unsupported material in map_generation.material_minimums: {material}"
            )
        minimum = int(minimum)
        if minimum < 0:
            raise ValueError(
                f"material minimum must be non-negative for {material}, got {minimum}"
            )
        if minimum == 0:
            continue
        normalized[str(material)] = minimum
    return normalized


def world_material_counts(world, materials: Mapping[str, int] | list[str] | tuple[str, ...]) -> MaterialCounts:
    """Return current counts for the requested world materials."""
    if isinstance(materials, Mapping):
        material_names = materials.keys()
    else:
        material_names = materials
    return {material: int(world.count(material)) for material in material_names}


def reset_env_with_material_minimums(
    env,
    *,
    material_minimums: Mapping[str, int] | None,
    max_attempts: int,
    log_fn: Callable[[str], None] | None = None,
):
    """Reset an environment until the generated map satisfies material bounds."""
    normalized_minimums = normalize_material_minimums(material_minimums)
    max_attempts = int(max_attempts)
    if max_attempts < 1:
        raise ValueError(f"map_generation.max_attempts must be at least 1, got {max_attempts}")

    if not normalized_minimums:
        obs = env.reset()
        return obs, {}, 1

    last_counts: MaterialCounts | None = None
    for attempt in range(1, max_attempts + 1):
        obs = env.reset()
        counts = world_material_counts(env._world, normalized_minimums)
        if all(counts[material] >= minimum for material, minimum in normalized_minimums.items()):
            if log_fn is not None and attempt > 1:
                log_fn(
                    "[map-gen] accepted map after "
                    f"{attempt} attempts with counts={counts}"
                )
            return obs, counts, attempt
        last_counts = counts
        if log_fn is not None and attempt < max_attempts:
            log_fn(
                "[map-gen] rejected map attempt "
                f"{attempt}/{max_attempts}: counts={counts}, required={dict(normalized_minimums)}"
            )

    raise RuntimeError(
        "Failed to generate a map satisfying material minimums after "
        f"{max_attempts} attempts. Required={dict(normalized_minimums)}, "
        f"last_counts={last_counts}"
    )
