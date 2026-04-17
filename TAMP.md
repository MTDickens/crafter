# TAMP 说明书

## 1. 这份说明书是讲什么的

这份文档解释当前仓库里这套 TAMP（Task and Motion Planning，任务与运动规划）实现到底在做什么，以及它是如何和 Crafter 环境接起来的。

这里的 TAMP 不是一个“在线边玩边思考”的 planner，而是一个更偏离线的 planner：

1. 它先基于完整初始地图快照构造一个 `KnownWorld`。
2. 然后只把起点设为已知，其余都当作未知。
3. 接着在 `KnownWorld` 上做 reveal、collect、place、make 的模拟规划。
4. 最后输出一串 Crafter 底层动作名，例如 `move_left`、`do`、`place_table`、`make_wood_pickaxe`。

`run_gui.py` 在启用 planner 时，会先一次性生成整串动作，然后在环境里顺序执行。

当前实现的重点是：

- 架构清晰；
- 行为尽量贴近你之前写的 TODO；
- fail-fast；
- 先把最简可工作的版本打通，再为之后更复杂的推断留接口。

## 2. 当前代码涉及哪些文件

当前实现主要分布在下面几个文件里：

- [crafter/known_world.py](/Users/mtdickens/Programs/Research_Code/sym-img-neurips-26/crafter-v5-simplified/crafter/known_world.py)
- [crafter/task_and_motion_planner.py](/Users/mtdickens/Programs/Research_Code/sym-img-neurips-26/crafter-v5-simplified/crafter/task_and_motion_planner.py)
- [crafter/run_gui.py](/Users/mtdickens/Programs/Research_Code/sym-img-neurips-26/crafter-v5-simplified/crafter/run_gui.py)
- [crafter/conf/run_gui.yaml](/Users/mtdickens/Programs/Research_Code/sym-img-neurips-26/crafter-v5-simplified/crafter/conf/run_gui.yaml)
- [crafter/conf/tamp_planner/vanilla.yaml](/Users/mtdickens/Programs/Research_Code/sym-img-neurips-26/crafter-v5-simplified/crafter/conf/tamp_planner/vanilla.yaml)

它们的职责分别是：

- `known_world.py`：定义 planner 用的世界表示、任务表示、以及已知世界状态。
- `task_and_motion_planner.py`：定义 planner 主逻辑，包括 reveal、任务分解、路径规划、动作生成。
- `run_gui.py`：定义 GUI 启动逻辑，并在启用 planner 时调用 TAMP 生成动作。
- `run_gui.yaml`：定义 planner 开关、planner 名称、是否在 14 个任务完成后退出、以及结果 logging。
- `vanilla.yaml`：当前 planner 选择项占位，暂时没有额外参数。

## 3. 整体架构

这套实现可以分成三层。

### 3.1 `SimpleWorld`

`SimpleWorld` 是一个“轻量世界快照”。

它的目标不是完整复刻环境，而是保留 planner 真正关心的那部分状态：

- 每个格子的 material；
- 世界大小；
- 一些和原始 world 接口兼容的方法，例如 `__getitem__`、`__setitem__`、`nearby()`、`count()`。

它刻意不关心的东西包括：

- mobs；
- 饥饿、口渴、精力；
- 昼夜；
- 绝大多数对象行为。

但是，“轻量”不等于“永远不变”。

当前实现里，`SimpleWorld` 仍然会演化，只是它演化的内容非常有限，主要是这些：

- `tree` 被 `do` 之后变成 `grass`；
- `stone` / `coal` / `iron` / `diamond` 被 `do` 之后变成 `path`；
- 某个格子被 `place_*` 后 material 改变；
- 执行 `make_*` 时 inventory 改变。

也就是说，`SimpleWorld` 是“忽略复杂动态机制后的可演化世界”。

### 3.2 `KnownWorld`

`KnownWorld` 是这套 TAMP 里最核心的状态对象。

它维护两份世界：

- `_initial_world`
- `_world`

这两份世界的语义不一样。

#### `_initial_world`

`_initial_world` 是 reveal source，也就是“揭露未知格时要去查的原始真值地图”。

在当前实现里，我们假设：

- 一个格子在没有被 reveal 之前，不会发生变化；
- 所以 reveal 某个 unknown cell 时，可以直接去 `_initial_world` 里读取它最初的 material。

#### `_world`

`_world` 是“当前已知世界”。

这个世界会随着规划过程中的 micro-action 而更新，所以它表示的不是初始环境，而是 planner 眼中“已经知道的、并且已经被自己的动作影响过的世界”。

例如：

- 某棵已知树被砍掉以后，`_world` 里那一格会从 `tree` 变成 `grass`；
- 某个格子放了 table 以后，`_world` 里那一格会变成 `table`；
- 做完 `make_wood_pickaxe` 之后，inventory 也会变化。

所以真正参与 planning 的是 `_world`，不是 `_initial_world`。

### 3.3 `TaskAndMotionPlanner`

`TaskAndMotionPlanner` 是 planner 本体。

它接受一个 task list，例如：

- `collect_wood`
- `place_table`
- `make_wood_pickaxe`
- ...
- `collect_diamond`

然后把这些高层 task 依次落成底层动作串。

它做的事情可以概括成：

1. 看当前 task 需要什么材料和布局。
2. 如果知识不够，就继续 reveal。
3. 如果材料还没拿到，就去 collect。
4. 如果 task 还需要最终动作，例如 `place_*` 或 `make_*`，就执行它。

最终返回 `list[str]`。

## 4. Task 是怎么表示的

当前高层任务由 `Task` 类表示。

`Task.from_name(name)` 会把一个字符串任务名转成任务对象。

例如：

- `collect_wood`
- `place_table`
- `make_wood_pickaxe`

这些名字目前都在 `Task._TASK_DEFINITIONS` 里硬编码。

每个任务主要包含两部分：

- `material_required`
- `micro_action_tasks`

### 4.1 `material_required`

这个字段表示：为了完成这个 task，最终至少需要持有多少材料。

例如：

- `place_table` 需要 `{wood: 2}`
- `make_stone_pickaxe` 需要 `{wood: 1, stone: 1}`
- `make_iron_pickaxe` 需要 `{wood: 1, coal: 1, iron: 1}`

### 4.2 `micro_action_tasks`

这个字段表示：在材料已经满足之后，这个 task 最后还需要执行哪些动作。

例如：

- `place_table` 的 `micro_action_tasks` 包含：
  - `ensure_layout`
  - `place_named`
- `make_wood_pickaxe` 的 `micro_action_tasks` 包含：
  - `make`
- `place_stone` 的 `micro_action_tasks` 包含：
  - `place_1x1`

所以一个 task 的执行并不是“一步到位”，而是：

1. 先满足材料需求；
2. 再执行收尾性的 micro-action。

## 5. 当前 planner 的三阶段主流程

`TaskAndMotionPlanner.plan()` 对每个 task 依次做三件事：

1. `_reveal_until_ready()`
2. `_ensure_material_inventory()`
3. `_execute_micro_action()`

这是理解当前实现最关键的一部分。

### 5.1 `_reveal_until_ready()`：补知识

这个函数的职责不是去拿材料，也不是做最终任务动作，而是补充 planner 对世界的知识。

它会反复检查两类前提：

#### 前提 1：材料知识是否足够

这里检查的不是“inventory 里有没有拿到这些东西”，而是：

`inventory + known pre-material >= task requirement`

这里的 pre-material 指的是已知世界里那些“可转化成最终材料”的格子。

例如：

- `tree` 是 `wood` 的 pre-material；
- `stone` 是 `stone` 的 pre-material；
- `coal` 是 `coal` 的 pre-material；
- `iron` 是 `iron` 的 pre-material；
- `diamond` 是 `diamond` 的 pre-material。

举例：

如果任务需要 2 个 wood，而当前：

- inventory 里有 1 个 wood；
- 已知世界里还有 1 棵 reachable tree；

那么从“知识是否足够”的角度看，已经够了，因为 planner 已经知道如何再获得 1 个 wood。

#### 前提 2：布局知识是否足够

有些任务除了材料以外，还要求某种空间布局已经被发现。

当前实现里只有一个布局：

- `craft_cluster_4x4`

它用于 `place_table` / `place_furnace` / `make_*` 系列任务。

如果这个 4x4 区域还没有 fully known 且可行，那么 `_reveal_until_ready()` 会继续 reveal。

#### 为什么它返回空 `actions`

这是当前实现里一个非常重要的设计点。

`_reveal_until_ready()` 虽然“做了很多事”，但它做的是 planner 内部状态更新：

- 在 `KnownWorld` 里把某个 unknown cell 标成 known；
- 读取 `_initial_world` 里的 material；
- 更新 planner 对世界的知识。

它没有对应到环境里的真实动作。

换句话说，当前实现把 reveal 建模成了“离线推理/信息扩张”，而不是 agent 在游戏里真的迈步探索。

因此它通常返回空 `actions`，是符合当前建模的。

如果以后你希望 reveal 也对应到环境里真实发生的探索动作，那么这一层就需要改，届时 `_reveal_until_ready()` 就不该返回空列表了。

### 5.2 `_ensure_material_inventory()`：把材料真的拿到手

这个函数的职责是：既然知识已经够了，就把 task 所需材料真正收集到 inventory 里。

它会遍历 `task._materials_required`，然后对每个 item 调 `_collect_until(item, required)`。

举例：

如果某个任务要求：

- `wood: 2`
- `stone: 1`

那么 `_ensure_material_inventory()` 会先保证拿到足够的 wood，再保证拿到足够的 stone。

这一步和 `_reveal_until_ready()` 的区别是：

- `_reveal_until_ready()` 只保证“我已经知道哪里有这些资源”；
- `_ensure_material_inventory()` 才真正生成动作并收集资源。

所以这一层返回的 `actions` 往往非空。

### 5.3 `_execute_micro_action()`：执行任务的收尾动作

当前 task 所需材料都已经满足后，planner 还要执行这个 task 自己的终结动作。

例如：

- `place_table`
- `place_furnace`
- `place_stone`
- `make_wood_pickaxe`
- `make_iron_sword`

这些都在 `_execute_micro_action()` 中处理。

它会根据 `micro_action['kind']` 分发到不同逻辑。

## 6. reveal 的语义

### 6.1 `KnownWorld.reveal_cell(pos)`

这个函数现在的语义非常直接：

- 你给它一个指定格子；
- 它会检查这个格子是否合法；
- 如果合法，就把这个格子 reveal。

它不负责“选哪个格子”，只负责“揭露这个格子”。

### 6.2 `TaskAndMotionPlanner.reveal_next_cell_for_requirement(requirement)`

这个函数负责：

- 根据某个 requirement；
- 按当前 reveal policy；
- 选出“下一个最该揭露”的格子；
- 然后调用 `KnownWorld.reveal_cell()` 真正揭露它。

所以现在命名上已经区分开了：

- `reveal_cell`：揭露指定格子；
- `reveal_next_cell_for_requirement`：根据需求选择并揭露下一个格子。

## 7. 当前 reveal policy 是怎么做的

### 7.1 reveal 的候选格必须满足什么

当前只有一类合法候选格：

- 它当前是 unknown；
- 它与当前 on-reachable 区域相邻；
- 它在当前 inventory/tool 条件下是 passable 的。

这体现在 `KnownWorld.frontier_unknowns()` 里。

### 7.2 walkable / passable / on-reachable

这是整个 planner 的基础语义。

#### `walkable`

不需要任何 `do` 就能直接站上去。

当前典型例子：

- `grass`
- `path`
- `sand`

#### `passable`

在当前工具条件下，可以通过合法动作最终站上去。

典型例子：

- 如果现在有 `wood_pickaxe`，那么 `stone` 和 `coal` 就是 passable；
- 因为可以先对它 `do`，它会变成 `path`，然后就能走上去。

#### `on-reachable`

从当前玩家位置出发，存在一条完全由“当前已知 passable 格子”组成的路径。

### 7.3 1x1 reveal

对于 1x1 类型的 reveal，当前实现是最简 baseline：

- 理论上应该根据 posterior probability 选最优格；
- 当前因为 unknown 的分布被视为 uniform；
- 所以最后在合法候选中随机挑一个。

这里的“合法候选”已经经过 frontier/passable 约束筛过了。

### 7.4 4x4 reveal

4x4 reveal 用在 `craft_cluster_4x4`。

当前没有做真正复杂的联合 Bayesian 推断，而是用了一个更简单但结构合理的近似：

1. 先枚举所有当前仍然“可能合法”的 4x4 anchor；
2. 再统计每个 frontier unknown cell 被多少个合法 4x4 anchor 覆盖；
3. 分数最高的格子优先 reveal；
4. 如果并列，就随机挑一个。

这个策略的直觉是：

- 一个格子如果属于很多“仍然可能成立”的 4x4 候选；
- 那它的信息价值更高；
- reveal 它更有助于尽快排除不可能的布局，或者确认一个可行布局。

## 8. craft cluster 是什么

当前实现里，`place_table` 和 `place_furnace` 不是各找各的位置，而是共享一个固定的 4x4 区域。

这个区域的好处是：

- 保证连通性；
- 保证玩家站在里面某个格子上时，可以同时访问 table 和 furnace；
- 后续 `make_iron_*` 这种需要同时依赖 table 和 furnace 的动作会更简单。

当前固定的关键位置是：

- `furnace = (1, 1)`
- `crafting_stand = (1, 2)`
- `table = (2, 2)`

这里是相对 4x4 anchor 的局部坐标。

planner 会：

1. 先找一个 fully known 且可行的 4x4 区域；
2. 一旦找到，就 reserve 下来；
3. 后续 `place_table`、`place_furnace`、`make_*` 都复用同一个区域。

这在 `KnownWorld.find_or_reserve_craft_cluster()` 中实现。

## 9. collect 是怎么做的

`_collect_until(item, required)` 的行为可以概括成：

1. 先看已知世界里有没有 reachable 的 pre-material；
2. 如果没有，就继续 reveal；
3. 一旦有了，就找最近目标；
4. 走到目标旁边并转向；
5. 执行 `do`；
6. 更新 `KnownWorld`：
   - 地形变化；
   - inventory 增加。

例如 collect tree：

- 走到树旁边；
- 面向树；
- `do`；
- 该格在 `_world` 里从 `tree` 变为 `grass`；
- inventory 里 `wood += 1`。

## 10. place / make 是怎么做的

### 10.1 `place_named`

用于 `place_table` / `place_furnace`。

流程是：

1. 确保 craft cluster 已 reserve；
2. 找到 table 或 furnace 的目标格；
3. 走到目标格旁边并面向它；
4. 执行 `place_*`；
5. 更新 `_world` 和 inventory。

### 10.2 `place_1x1`

用于像 `place_stone` 这样的任务。

流程是：

1. 在已知世界中找最近的可放置 1x1 目标格；
2. 如果还没有合适格子，就继续 reveal；
3. 找到后走过去并面向它；
4. 执行 `place_*`；
5. 更新 `_world` 和 inventory。

### 10.3 `make`

用于 `make_wood_pickaxe`、`make_iron_sword` 等。

流程是：

1. 确保 craft cluster 已 reserve；
2. 走到 `crafting_stand`；
3. 直接执行 `make_*`；
4. 检查 `nearby` 是否满足；
5. 更新 inventory。

当前 `KnownWorld.apply_make()` 用 `self._world.nearby(tuple(self.player_pos), 1)` 来检查附近是否已有所需设施，例如：

- `wood_pickaxe` 只需要 `table`；
- `iron_pickaxe` 需要 `table` 和 `furnace`。

## 11. 路径规划是怎么做的

路径规划由 `TaskAndMotionPlanner.plan_path_to_any()` 完成。

它做的是一个 BFS。

输入是一组候选目标格，例如：

- 某个确切目标格；
- 某个资源格周围四邻格；
- 某个放置位周围四邻格。

它返回一条最短位置路径。

然后 `_execute_position_path()` 再把这条位置路径转成底层动作。

### 11.1 为什么路径中可以穿过某些不可走的资源格

因为当前 planner 的 passable 定义允许：

- 如果某格当前不是 walkable；
- 但它是 collectable 且在当前工具条件下 passable；
- 那么就可以在路径执行中：
  1. 面向它；
  2. `do`；
  3. 把它变成 leaves；
  4. 再 move 上去。

所以 `_execute_position_path()` 里如果发现 `next_pos` 的 material 不是 walkable，但仍是 passable，就会先插入一次 `do`。

## 12. `run_gui.py` 是怎么接 planner 的

`run_gui.py` 里现在多了三个关键 helper：

- `_planner_tasks_finished(...)`
- `_log_planner_results(...)`
- `_build_planner_actions(...)`

### 12.1 `_build_planner_actions(...)`

它会：

1. 从 `env_recorded._world` 构造 `SimpleWorld`；
2. 用玩家当前位置、朝向、inventory 构造 `KnownWorld`；
3. 根据 `config.planner.task_order` 构造 `TaskAndMotionPlanner`；
4. 调 `planner.plan(known_world)`；
5. 打印动作数和 planner result。

### 12.2 主循环中的执行方式

如果 `planner.enabled = true`，主循环会优先取 planner 生成的动作：

- `planner_action_idx < len(planner_actions)` 时，直接执行 planner 动作；
- 否则才会退回键盘输入/按键持续按住逻辑。

### 12.3 `exit_on_finish`

如果：

- `planner.enabled = true`
- `planner.exit_on_finish = true`

那么一旦 `planner.task_order` 中所有 task 对应的 achievement 都完成，GUI 会直接退出。

这里的“完成”不是 planner 自己内部状态，而是环境玩家的 `achievements`。

### 12.4 `result_logging`

当前只支持一个条目：

- `reveal_count`

它打印的是：

- `KnownWorld.revealed_cell_count`

也就是 planner 最终 reveal mask 中，`True` 的数量。

## 13. Hydra 配置项说明

当前 `run_gui.yaml` 里与 planner 相关的配置项是：

```yaml
planner:
  enabled: true
  name: vanilla
  exit_on_finish: false
  result_logging:
    - reveal_count
  task_order:
    - collect_wood
    - place_table
    - make_wood_pickaxe
    - make_wood_sword
    - collect_stone
    - place_stone
    - make_stone_pickaxe
    - make_stone_sword
    - place_furnace
    - collect_coal
    - collect_iron
    - make_iron_pickaxe
    - make_iron_sword
    - collect_diamond
```

说明如下：

- `enabled`
  - 是否启用 planner。
- `name`
  - 当前 planner 名称，只支持 `vanilla`。
- `exit_on_finish`
  - 是否在 14 个任务都完成后自动退出。
- `result_logging`
  - 需要打印哪些 planner 结果，目前只支持 `reveal_count`。
- `task_order`
  - planner 要依次执行的高层 task 列表。

另外：

- `defaults: - tamp_planner: vanilla` 已经在 `run_gui.yaml` 里接上了；
- 但 `crafter/conf/tamp_planner/vanilla.yaml` 暂时还是空文件，只作为选择项占位。

## 14. 为什么当前设计是 fail-fast

当前实现尽量不做“静默容错”。

一旦某个前提不满足，代码通常直接 `assert`，例如：

- task 名未知；
- reveal 候选为空；
- 想访问 unknown cell；
- 目标格不可达；
- craft cluster 不再兼容；
- planner 动作执行完了，但 14 个任务还没完成。

这样做的目的是：

- 尽早暴露建模问题；
- 不让 planner 在错误状态下继续“装作还能运行”；
- 方便之后迭代真正的设计，而不是被各种 fallback 掩盖问题。

## 15. 当前实现有哪些刻意的简化

这套实现是可以工作的第一版，但它明确做了不少简化。

### 15.1 reveal 不是环境动作

当前 reveal 只改变 `KnownWorld`，不对应环境里的真实探索动作。

所以 planner 是“离线知道地形、在线只执行动作串”的近似版本。

### 15.2 1x1 reveal 仍是 uniform baseline

虽然接口已经留成了“按 requirement 选 reveal cell”，但当前 1x1 的评分实际上还是 uniform tie-break。

### 15.3 4x4 reveal 不是严格联合 Bayesian 推断

当前 4x4 reveal 只是“看一个格子出现在多少个可行 anchor 里”，并不是在求真正的全局联合后验最优。

### 15.4 task catalog 还是代码内置

目前 task 定义没有搬到 Hydra/YAML，仍然写死在 `Task._TASK_DEFINITIONS` 里。

### 15.5 planner 一次性生成全串动作

当前不是边执行边 replanning，而是先从初始状态生成完整动作串，再交给环境执行。

这意味着：

- 如果环境在执行中发生了 planner 未建模的动态变化；
- 这版 planner 不会中途修正。

## 16. 后续自然的扩展方向

如果要继续往下做，比较自然的方向有这些：

1. 把 reveal 变成真实环境中的探索行为，而不是纯内部更新。
2. 把 task catalog 从代码搬到 Hydra/YAML。
3. 为 reveal 引入真正的 pattern-based posterior，而不是 uniform baseline。
4. 让 planner 支持执行中 replanning。
5. 扩展 `result_logging`，例如：
   - `action_count`
   - `task_finish_step`
   - `planner_inventory_trace`
6. 把 4x4 reveal 的评分从覆盖数启发式升级为更明确的联合目标。

## 17. 一句话总结

当前这版 TAMP 的核心思想是：

- 用 `KnownWorld` 把“初始真值地图”和“当前已知且会演化的地图”分开；
- 对每个高层任务先补知识，再拿材料，最后执行终结动作；
- 用最简单可工作的 reveal 和 motion planning 先把完整主线打通；
- 并且用 fail-fast 保证一旦建模假设不成立，问题会尽快暴露。
