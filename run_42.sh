# # PITWI

# uv run python -m crafter.run_gui planner.skill_learning.enable_inference=true planner.skill_learning.enable_reweighting=true planner.skill_learning.skill_proposal.enabled=true \
# planner.skill_learning.initial_patterns.add_center_only_placeholders=true \
# seed=42

# # PITWI w/o inference

# uv run python -m crafter.run_gui planner.skill_learning.enable_inference=false planner.skill_learning.enable_reweighting=false planner.skill_learning.skill_proposal.enabled=false \
# planner.skill_learning.initial_patterns.add_center_only_placeholders=true \
# seed=42

# # PITWI w/o reweighting

# uv run python -m crafter.run_gui planner.skill_learning.enable_inference=true planner.skill_learning.enable_reweighting=false planner.skill_learning.skill_proposal.enabled=true \
# planner.skill_learning.initial_patterns.add_center_only_placeholders=true \
# seed=42

# PITWI w/o initial placeholder patterns

uv run python -m crafter.run_gui planner.skill_learning.enable_inference=true planner.skill_learning.enable_reweighting=true planner.skill_learning.skill_proposal.enabled=true \
planner.skill_learning.initial_patterns.add_center_only_placeholders=false \
seed=42

# PITWI + initial all ground truth patterns

# 这个先不用做了
