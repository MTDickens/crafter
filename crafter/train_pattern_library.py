"""Train a white-box cross-pattern library on historical Crafter maps."""

from __future__ import annotations

import pathlib

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

import crafter
from crafter.pattern_library import (
    fit_pattern_library_from_material_maps,
    material_map_from_world,
)


def _make_env(config: DictConfig, seed: int):
    return crafter.Env(
        area=tuple(config.area),
        view=tuple(config.view),
        size=tuple(config.size),
        length=1,
        seed=seed,
        spawn_objects=bool(config.runtime.spawn_objects),
        spawn_random_objects=bool(config.runtime.spawn_random_objects),
        move_objects=bool(config.runtime.move_objects),
        move_random_objects=bool(config.runtime.move_random_objects),
        hunger_decreases=bool(config.runtime.hunger_decreases),
        thirst_decreases=bool(config.runtime.thirst_decreases),
        energy_decreases=bool(config.runtime.energy_decreases),
        daylight_cycle=bool(config.runtime.daylight_cycle),
    )


def _generate_material_maps(config: DictConfig, *, base_seed: int, count: int) -> list[np.ndarray]:
    material_maps = []
    for index in range(count):
        seed = base_seed + index
        env = _make_env(config, seed)
        env.reset()
        material_maps.append(material_map_from_world(env._world))
    return material_maps


@hydra.main(version_base=None, config_path="conf", config_name="train_pattern_library")
def main(config: DictConfig) -> None:
    output_path = pathlib.Path(str(config.output_path))
    requested_device = str(config.get("device", "cpu"))
    train_maps = _generate_material_maps(
        config,
        base_seed=int(config.train_base_seed),
        count=int(config.train_maps),
    )
    training_rng = np.random.default_rng(int(config.training_mask_seed))
    library = fit_pattern_library_from_material_maps(
        train_maps,
        prior_scale=float(config.prior_scale),
        training_policy=str(config.training_policy),
        masks_per_map=int(config.masks_per_map),
        observed_fraction_min=float(config.observed_fraction_min),
        observed_fraction_max=float(config.observed_fraction_max),
        rng=training_rng,
        start_position=None,
        device=requested_device,
        metadata={
            "train_maps": int(config.train_maps),
            "train_base_seed": int(config.train_base_seed),
            "training_policy": str(config.training_policy),
            "masks_per_map": int(config.masks_per_map),
            "observed_fraction_min": float(config.observed_fraction_min),
            "observed_fraction_max": float(config.observed_fraction_max),
            "prior_scale": float(config.prior_scale),
            "device": requested_device,
        },
    )
    library.save(output_path)

    print(f"[pattern-train] requested device = {requested_device}")
    print(f"[pattern-train] resolved device = {library.device}")
    print(f"[pattern-train] torch version = {torch.__version__}")
    print(f"[pattern-train] saved library to {output_path}")
    print(f"[pattern-train] number of exact cross patterns = {library.num_patterns()}")
    print(f"[pattern-train] prior distribution = {library.prior_distribution()}")

    eval_maps = int(config.eval_maps)
    if eval_maps <= 0:
        return
    heldout_maps = _generate_material_maps(
        config,
        base_seed=int(config.eval_base_seed),
        count=eval_maps,
    )
    evaluation_rng = np.random.default_rng(int(config.evaluation_mask_seed))
    metrics = library.evaluate_on_material_maps(
        heldout_maps,
        evaluation_policy=str(config.evaluation_policy),
        masks_per_map=int(config.evaluation_masks_per_map),
        observed_fraction_min=float(config.observed_fraction_min),
        observed_fraction_max=float(config.observed_fraction_max),
        rng=evaluation_rng,
        start_position=None,
    )
    print(
        "[pattern-eval] "
        f"positions={metrics.total_positions} "
        f"top1={metrics.top1_accuracy:.4f} "
        f"true_prob={metrics.average_true_probability:.4f} "
        f"entropy={metrics.average_entropy:.4f} "
        f"nll={metrics.average_negative_log_likelihood:.4f}"
    )


if __name__ == "__main__":
    main()
