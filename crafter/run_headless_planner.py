"""Run the Crafter task planner without a GUI, optionally in parallel."""

from __future__ import annotations

import concurrent.futures
import datetime
import multiprocessing
import pathlib
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import crafter
from crafter.task_motion_planner import TaskMotionPlanner, render_known_world_image


class KnownWorldFrameExporter:
    """Save planner known-world renders for one episode."""

    def __init__(
        self,
        directory: pathlib.Path,
        textures,
        mode: str,
    ):
        self._directory = directory
        self._directory.mkdir(exist_ok=True, parents=True)
        self._textures = textures
        self._frame_index = 0
        self._mode = mode

    def save_step_frame(self, planner: TaskMotionPlanner, env) -> None:
        """Save one per-step known-world frame."""
        image = render_known_world_image(planner, env, self._textures)
        filename = self._directory / f"frame_{self._frame_index:06d}.png"
        Image.fromarray(image).save(filename)
        self._frame_index += 1

    def save_pending_task_frame(self, planner: TaskMotionPlanner, env) -> None:
        """Save one frame when the planner starts a new task."""
        if self._mode != "task":
            return
        event = planner.consume_known_world_task_render()
        if event is None:
            return
        task_name, frame_index = event
        image = render_known_world_image(planner, env, self._textures)
        filename = self._directory / f"frame_{frame_index:06d}-{task_name}.png"
        Image.fromarray(image).save(filename)


def _log(worker_index: int, episode_index: int, message: str) -> None:
    prefix = f"[worker {worker_index:04d} episode {episode_index:06d}]"
    print(f"{prefix} {message}")


def _episode_seed(base_seed: int | None, episode_index: int) -> int:
    base = 0 if base_seed is None else int(base_seed)
    return (base + episode_index) % (2**31 - 1)


def _run_id_from_config(config: dict[str, Any]) -> str:
    run_id = config.get("run_id")
    if run_id:
        return str(run_id)
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")


def _validate_config(config: dict[str, Any]) -> None:
    workers = int(config["workers"])
    episodes = int(config["episodes"])
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    mode = str(config["planner"]["known_world_frames_mode"])
    if mode not in {"step", "task"}:
        raise ValueError("planner.known_world_frames_mode must be either 'step' or 'task'")


def _apply_global_config(config: dict[str, Any]) -> None:
    crafter.constants.items["health"]["max"] = int(config["health"])
    crafter.constants.items["health"]["initial"] = int(config["health"])
    crafter.constants.collect["grass"]["probability"] = float(
        config["sapling_from_grass_probability"]
    )


def _make_env(config: dict[str, Any], seed: int, record_dir: pathlib.Path | None):
    env = crafter.Env(
        area=tuple(config["area"]),
        view=tuple(config["view"]),
        size=tuple(config["size"]),
        length=config["length"],
        seed=seed,
        spawn_objects=config["runtime"]["spawn_objects"],
        spawn_random_objects=config["runtime"]["spawn_random_objects"],
        move_objects=config["runtime"]["move_objects"],
        move_random_objects=config["runtime"]["move_random_objects"],
        hunger_decreases=config["runtime"]["hunger_decreases"],
        thirst_decreases=config["runtime"]["thirst_decreases"],
        energy_decreases=config["runtime"]["energy_decreases"],
        daylight_cycle=config["runtime"]["daylight_cycle"],
    )
    return crafter.Recorder(env, record_dir) if record_dir is not None else env


def _make_episode_dir(root: str | None, run_id: str, worker_index: int, episode_index: int) -> pathlib.Path | None:
    if root is None:
        return None
    return (
        pathlib.Path(root).expanduser()
        / run_id
        / f"worker_{worker_index:04d}"
        / f"episode_{episode_index:06d}"
    )


def run_one_episode(
    config: dict[str, Any],
    run_id: str,
    worker_index: int,
    episode_index: int,
) -> dict[str, Any]:
    """Run one planner-controlled episode and return a compact result summary."""
    _apply_global_config(config)

    planner_config = config["planner"]
    seed = _episode_seed(config.get("base_seed"), episode_index)
    record_dir = _make_episode_dir(
        config.get("headless_record"), run_id, worker_index, episode_index
    )
    known_world_dir = _make_episode_dir(
        planner_config.get("known_world_frames_dir"), run_id, worker_index, episode_index
    )

    env_recorded = _make_env(config, seed, record_dir)
    env_recorded.reset()
    _log(worker_index, episode_index, f"Diamonds exist: {env_recorded._world.count('diamond')}")

    planner = TaskMotionPlanner(env_recorded, OmegaConf.create(planner_config))
    known_world_exporter = None
    if known_world_dir is not None:
        known_world_exporter = KnownWorldFrameExporter(
            known_world_dir,
            env_recorded._textures,
            str(planner_config["known_world_frames_mode"]),
        )
        if planner_config["known_world_frames_mode"] == "step":
            known_world_exporter.save_step_frame(planner, env_recorded)

    achievements: set[str] = set()
    return_ = 0.0
    planner_finished = False
    env_done = False

    if planner_config["exit_on_completion"] and planner.is_finished(env_recorded):
        planner_finished = True
        _log(worker_index, episode_index, "Exiting because all configured tasks are complete")
    else:
        while True:
            if planner_config["exit_on_completion"] and planner.is_finished(env_recorded):
                planner_finished = True
                _log(worker_index, episode_index, "Exiting because all configured tasks are complete")
                break

            action = planner.next_action(env_recorded)
            if known_world_exporter is not None:
                known_world_exporter.save_pending_task_frame(planner, env_recorded)

            _, reward, done, _ = env_recorded.step(env_recorded.action_names.index(action))
            if (
                known_world_exporter is not None
                and planner_config["known_world_frames_mode"] == "step"
            ):
                planner._refresh_local_state(env_recorded)
                known_world_exporter.save_step_frame(planner, env_recorded)

            unlocked = {
                name
                for name, count in env_recorded._player.achievements.items()
                if count > 0 and name not in achievements
            }
            for name in sorted(unlocked):
                achievements.add(name)
                total = len(env_recorded._player.achievements.keys())
                _log(worker_index, episode_index, f"Achievement ({len(achievements)}/{total}): {name}")
            if env_recorded._step > 0 and env_recorded._step % 100 == 0:
                _log(worker_index, episode_index, f"Time step: {env_recorded._step}")
            if reward:
                _log(worker_index, episode_index, f"Reward: {reward}")
                return_ += reward

            if done:
                env_done = True
                _log(worker_index, episode_index, "Episode done!")
                _log(worker_index, episode_index, f"Duration: {env_recorded._step}")
                _log(worker_index, episode_index, f"Return: {return_}")
                break

    return {
        "worker_index": worker_index,
        "episode_index": episode_index,
        "seed": seed,
        "length": int(env_recorded._step),
        "peek_count": int(planner.known_world.peek_count),
        "return": float(return_),
        "planner_finished": bool(planner_finished),
        "env_done": bool(env_done),
        "achievements_unlocked": len(achievements),
        "record_dir": str(record_dir) if record_dir is not None else None,
        "known_world_dir": str(known_world_dir) if known_world_dir is not None else None,
    }


def _run_worker(
    config: dict[str, Any],
    run_id: str,
    worker_index: int,
    episode_indices: list[int],
) -> list[dict[str, Any]]:
    """Run a batch of episodes sequentially inside one worker process."""
    return [
        run_one_episode(config, run_id, worker_index, episode_index)
        for episode_index in episode_indices
    ]


def _partition_episode_indices(episodes: int, workers: int) -> list[list[int]]:
    assignments: list[list[int]] = [[] for _ in range(workers)]
    for episode_index in range(episodes):
        assignments[episode_index % workers].append(episode_index)
    return assignments


def _run_all_episodes(config: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    episodes = int(config["episodes"])
    workers = int(config["workers"])
    assignments = _partition_episode_indices(episodes, workers)

    if workers == 1:
        return _run_worker(config, run_id, 0, assignments[0])

    ctx = multiprocessing.get_context("spawn")
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx
    ) as executor:
        futures = [
            executor.submit(_run_worker, config, run_id, worker_index, episode_indices)
            for worker_index, episode_indices in enumerate(assignments)
            if episode_indices
        ]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    return sorted(results, key=lambda item: item["episode_index"])


def _print_summary(results: list[dict[str, Any]], run_id: str) -> None:
    print("")
    print(f"Run ID: {run_id}")
    print(f"Episodes completed: {len(results)}")
    if not results:
        return
    planner_finished = sum(1 for result in results if result["planner_finished"])
    env_done = sum(1 for result in results if result["env_done"])
    avg_length = sum(result["length"] for result in results) / len(results)
    avg_return = sum(result["return"] for result in results) / len(results)
    avg_peek_count = sum(result["peek_count"] for result in results) / len(results)
    print(f"Planner-finished episodes: {planner_finished}")
    print(f"Environment-done episodes: {env_done}")
    print(f"Average length: {avg_length:.2f}")
    print(f"Average return: {avg_return:.2f}")
    print(f"Average peek count: {avg_peek_count:.2f}")


@hydra.main(version_base=None, config_path="conf", config_name="run_headless_planner")
def main(config: DictConfig):
    config_dict = OmegaConf.to_container(config, resolve=True)
    if not isinstance(config_dict, dict):
        raise TypeError("Expected DictConfig to resolve to a dictionary")
    _validate_config(config_dict)
    run_id = _run_id_from_config(config_dict)
    results = _run_all_episodes(config_dict, run_id)
    _print_summary(results, run_id)


if __name__ == "__main__":
    main()
