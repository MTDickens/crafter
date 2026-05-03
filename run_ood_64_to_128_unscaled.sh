#!/usr/bin/env bash
set -euo pipefail

INITIAL_PLANNER_RESULTS="planner_results_final/PITWI/20260501T010512/episode-00099/planner_results.json" # First seed 42 PITWI planner result

# Vanilla, w/o inference.

uv run python -m crafter.run_gui \
planner.skill_learning.enable_inference=false \
planner.skill_learning.enable_reweighting=false \
planner.skill_learning.skill_proposal.enabled=false \
planner.skill_learning.initial_patterns.add_center_only_placeholders=false \
seed=42 episodes=100 'area=[128,128]' \
planner.output_dir="planner_results_final_ood_64_to_128_unscaled/PITWI wo inference"

# OOD 64 -> 128, w/o proposal or reweighting.

uv run python -m crafter.run_gui \
planner.skill_learning.enable_inference=true \
planner.skill_learning.enable_reweighting=false \
planner.skill_learning.skill_proposal.enabled=false \
planner.skill_learning.initial_patterns.load_from_planner_results="${INITIAL_PLANNER_RESULTS}" \
planner.skill_learning.initial_patterns.add_center_only_placeholders=false \
seed=42 \
episodes=100 \
'area=[128,128]' \
planner.output_dir="planner_results_final_ood_64_to_128_unscaled/PITWI wo proposal wo reweighting"

# # OOD 64 -> 128, w/o proposal and w/ reweighting.

# uv run python -m crafter.run_gui \
# planner.skill_learning.enable_inference=true \
# planner.skill_learning.enable_reweighting=true \
# planner.skill_learning.skill_proposal.enabled=false \
# planner.skill_learning.initial_patterns.load_from_planner_results="${INITIAL_PLANNER_RESULTS}" \
# planner.skill_learning.initial_patterns.add_center_only_placeholders=false \
# seed=42 \
# episodes=100 \
# 'area=[128,128]' \
# planner.output_dir="planner_results_final_ood_64_to_128_unscaled/PITWI wo proposal"
