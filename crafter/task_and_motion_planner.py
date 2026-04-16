"""Task and motion planner for Crafter."""

from crafter.known_world import Task, KnownWorld


class TaskAndMotionPlanner:
  """Task and motion planner for Crafter."""

  def __init__(self):
    pass  # TODO: Add what we need. For example, it could be `task_list: list[Task]`, etc

  def plan(self, known_world: KnownWorld):
    # TODO: You have to finish tasks in the task_list one by one.
    #       For micro-perception tasks, you need to reveal the unknown grids in the known_world, until you find the target object or a place that satisfies the requirements.
    #       For micro-action tasks, it's simpler: you just need to finish them one by one.
    #
    # UPDATE: see `known_world.py` - micro-perception tasks will not be considered as "hard" tasks that must be finished in order.
    # Instead, we will just need to reveal grids until "pre-material" + "inventory" >= required.
    pass

  def reveal(self, micro_perception_task):
    # TODO: You need to reveal the unknown grids in the known_world, until you find the target object or a place that satisfies the requirements.
    # Where to reveal? We adopt a Bayesian approach:
    # - We need to reveal a place where things will mostly satisfy the micro perception task's requirements.
    #   - btw, the revealed place need to be reachable from current location (assuming unknown grids are impassable). 
    #     This method will have an argument about this. For now, it must be asserted to be true (i.e. must be reachable).
    #     - Although, for now, since passable grids on the border of the known area will never become impassable,
    #       if you always reveal an adjacent grid of a certain passable grid on the border,
    #       you can prove that: you will always be able to reach any grid of the known area (quiz: how to prove it? "Reachable" means you can get near it, not on it).
    #       In this sense, you don't have to ACTUALLY check reachability. You just need to always reveal an adjacent grid of a passable grid on the border.
    # - We will consider obtaining a tensor showing the probability distribution of the attribute of each grid. 
    #   - For now, we adopt the vanilla way: known grids have 1.0 prob of its known attribute, where unknown grids will have a uniform distribution of all attributes,
    #     though it's subject to change (we will add a pattern-based inference engine later on, and the vanilla will serve as a baseline)
    #     - If you are doing 1x1 reveal, just reveal a most likely adjacent grid (for now, since it's all uniform, you will just have to randomly select one).
    #       - For example, if you are still short of coal and iron, you will find an adjacent grid that is most probable to be either coal or iron; 
    #         if there are more than one grids that are equally most likely, you will randomly choose one among them.
    #     - If you are doing n by n reveal (e.g. 4x4), you are still trying to find a set of most likely grids. e.g. For "place table" task, you need to find a 4x4 set of grids that are jointly
    #       most likely to all be walkable (i.e. grass or path). But unlike 1x1, you will reveal only one grid and the probs will change. So TODO: you might have to carefully consider this. Like,
    #       how to formulate this problem in this way (if this way is possible) and select the next one grid to see? And can you mix 4x4 with 1x1 (you can mix 1x1 together; like "find coal" and "find wood"
    #       can be mixed into "find coal or wood". But can it be the case for 4x4?), if not, is it the right way to order these two as well, 
    #       e.g. first 1x1 (if any), then 2x2 (if any), then 3x3 (if any), then 4x4 (if any)
    pass
  
  def motion_planning(self):
    # TODO: you can change the name of this function if needed. This is to plan the actual motion of micro-action tasks,
    # it shouldn't be too hard.
    pass
