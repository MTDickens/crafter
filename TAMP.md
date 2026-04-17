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

这里还有一个现在已经明确收紧的约束：

- 一旦某个 4x4 craft cluster 被 reserve，下游的其他 1x1 放置任务就不能再占用这个 4x4 区域里的任何格子。

注意，不只是不能占用：

- `table`
- `furnace`
- `crafting_stand`

这几个命名格，

而是整个 4x4 区域都视为“保留营地空间”。

这样做的原因是，后续 planner 会反复要求这个 reserved cluster 始终保持 layout-compatible。  
如果像 `place_stone` 这样的 1x1 任务把石头放进这个 4x4 区域中的其他 walkable 格子，那么这个 cluster 之后就不再兼容，会直接触发 fail-fast。

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

这里有一个容易忽略、但现在已经明确实现的约束：

- 如果 craft cluster 已经 reserve，那么 `place_1x1` 不允许选择这个 4x4 区域中的任何格子。

这个约束同时作用在两个层面：

1. 已知目标选择
2. future reveal 候选选择

也就是说：

- `find_placeable_1x1()` 不会返回 reserved 4x4 内的格子；
- `placeable_frontier_candidates_1x1()` 也不会把 reserved 4x4 内的 unknown 格子作为 1x1 放置 reveal 候选。

除此之外，`apply_place()` 里还有一层额外的 fail-fast 断言：

- 除了 `table` 和 `furnace` 这两个本来就属于 cluster 的命名放置外，其他 `place_*` 一律不允许写进 reserved 4x4。

这一点不是“优化”，而是为了防止 reserved craft cluster 后续失效。

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

## 12. `run_gui.py` 现在是怎么接 planner 的

这部分是这次变化最大的地方。

现在的 `run_gui.py` 不再只是“单窗口 + 单 episode + planner 预先给一串动作”，而是同时支持：

- GUI 模式；
- headless 模式；
- multiple episodes；
- worker 并发；
- planner-only（只规划不执行）；
- task-level known-world 渲染；
- 时间戳结果目录。

### 12.1 planner trace

`TaskAndMotionPlanner` 现在除了 `plan()`，还提供：

- `plan_with_trace(known_world)`

它会返回一个结构化 trace，而不是只返回动作列表。

当前 trace 里最重要的信息有：

- `actions`
- `result_metrics`
- `initial_known_mask`
- `task_boundaries`

其中 `task_boundaries` 记录每个 task 的两个关键边界：

- `pre_action_mask`
- `post_action_mask`

它们分别对应：

- `pre`：`_reveal_until_ready()` 结束之后；
- `post`：`_ensure_material_inventory()` 和 `_execute_micro_action()` 都结束之后。

这正好用于后面的 task-level 渲染。

### 12.2 单 episode 的 planner 构建流程

在单个 episode 中，如果 `planner.enabled = true`，`run_gui.py` 会：

1. `env.reset()`
2. 从 `env_recorded._world` 构造 `SimpleWorld`
3. 用玩家当前位置、朝向、inventory 构造 `KnownWorld`
4. 用 `config.planner.task_order` 构造 `TaskAndMotionPlanner`
5. 调 `planner.plan_with_trace(known_world)`
6. 打印 planner 动作数和 planner metrics

如果 `planner.plan_only = true`，流程在这里就结束，不再执行环境动作。

### 12.3 执行模式

如果 `planner.plan_only = false`，那么 `run_gui.py` 会把 planner 产出的 `actions` 顺序喂给环境。

也就是说，当前执行模式仍然是：

1. 先离线规划完整动作串；
2. 再在线顺序执行。

当前还不是边执行边 replanning。

### 12.4 `exit_on_finish`

如果：

- `planner.enabled = true`
- `planner.exit_on_finish = true`

那么一旦 `planner.task_order` 中所有 task 对应的 achievement 都完成，当前 episode 会立刻结束。

这里判断用的是环境玩家的 `achievements`，不是 planner 内部状态。

### 12.5 `result_logging`

当前 `result_logging` 仍然只支持：

- `reveal_count`

但是现在它不只是打印了，还会写进每个 episode 的 `planner_results.json`。

除此之外，在执行模式下，结果文件还会额外包含：

- `finished_task_count`
- `task_count`
- `finished_all_tasks`
- `duration_steps`
- `return`

在 `plan_only` 模式下，这些执行期字段会写成 `null`。

## 13. `run_gui.py` 的几种运行模式

### 13.1 GUI 模式

当：

- `headless = false`

时，程序会开一个 pygame 窗口。

此时允许：

- `episodes >= 1`
- 但必须 `workers == 1`

也就是说，现在已经支持“同一个窗口里顺序跑多个 episode”，但仍然不支持多窗口并发 GUI。

之所以没有支持 `headless = false, workers > 1`，不是因为完全不可能，而是因为：

- 这会把问题变成多进程/多窗口/事件循环管理；
- 和当前“极简实现”的目标不一致。

### 13.2 Headless 模式

当：

- `headless = true`

时，不会创建 GUI 窗口。

这个模式下必须：

- `planner.enabled = true`

因为当前 headless 逻辑本质上是围绕 planner 跑批设计的。

此时允许：

- `episodes >= 1`
- `workers >= 1`

### 13.3 Plan-only 模式

当：

- `planner.plan_only = true`

时，当前 episode 的流程是：

1. 仍然创建 env 并 `reset()` 一次；
2. 基于这个初始环境构造 `KnownWorld`；
3. 运行 planner；
4. 写结果；
5. 结束。

也就是说，这里的“关闭 environment”指的是：

- 不执行环境动作；
- 不是完全不创建环境。

之所以仍然需要 `reset()`，是因为 planner 还需要读取初始世界真值快照。

这个模式下必须：

- `headless = true`

并且必须禁止一切依赖环境执行过程的功能，例如：

- `record`
- `planner.render_known_world != none`

## 14. Multiple Episodes 与 Workers

### 14.1 GUI 下的 multiple episodes

如果：

- `headless = false`
- `episodes > 1`

程序会在同一个进程、同一个窗口里顺序地跑多个 episode。

每个 episode 都会：

- 单独 reset 环境；
- 单独生成 planner 结果目录；
- 单独写自己的 `planner_results.json`。

### 14.2 Headless 下的 multiple episodes

如果：

- `headless = true`
- `workers = 1`

那么会串行执行多个 episode。

### 14.3 Headless 下的 workers 并发

如果：

- `headless = true`
- `workers > 1`

那么会用 `ProcessPoolExecutor` 按 episode 粒度并发执行。

每个 worker 会独立：

- 创建 env；
- reset；
- 构造 planner；
- 写各自 episode 目录。

### 14.4 seed 派生

如果 `config.seed` 不为 `null`，那么第 `i` 个 episode 使用：

- `seed + i`

这里的 `i` 是全局 episode index，而不是某个 worker 的局部编号。

如果 `config.seed = null`，则保持环境当前的随机行为。

## 15. 结果目录与日志文件

### 15.1 根目录

当 planner 开启时，结果默认会写到：

```text
{planner.output_dir}/{timestamp}/
```

例如：

```text
planner_results/20260417T211703/
```

### 15.2 每个 episode 一个子目录

每个 episode 都有自己独立的子目录：

```text
{planner.output_dir}/{timestamp}/episode-00000/
{planner.output_dir}/{timestamp}/episode-00001/
...
```

### 15.3 `planner_results.json`

每个 episode 目录里至少会有：

- `planner_results.json`

当前它至少包含这些字段：

- `episode_index`
- `seed`
- `planner_name`
- `plan_only`
- `planned_action_count`
- `metrics`

如果是执行模式，还会额外包含：

- `finished_task_count`
- `task_count`
- `finished_all_tasks`
- `duration_steps`
- `return`

### 15.4 `result_logging`

`result_logging` 现在不仅是“打印什么”，更准确地说是“哪些 planner metrics 会被写入结果文件”。

当前只支持：

- `reveal_count`

它表示 planner 最终 reveal mask 中 `True` 的数量。

## 16. Task-level Known-World 渲染

### 16.1 三种配置值

当前 `planner.render_known_world` 有三种选择：

- `none`
- `task`
- `step`

其中：

- `none`：不保存任何 known-world 图像；
- `task`：保存 task 级别的 mask 与 png；
- `step`：当前直接 `NotImplementedError`。

### 16.2 为什么渲染是“环境真值 + planner mask”

这次实现里，known-world 渲染不是直接画 planner 自己维护的 `_world`，而是：

1. 先读取环境当前真实 world；
2. 再用 planner trace 里的 known mask 做遮罩。

这样做的好处是：

- 已知区域里显示的是环境真实执行后的状态；
- 未知区域仍然保持 unknown；
- 更贴近“执行过程中 agent 目前知道什么”的直觉。

### 16.3 未知区域怎么画

未知区域统一渲染为灰色底色。

已知区域则正常渲染：

- material 的纹理；
- 如果玩家所在格是 known，还会叠加玩家朝向对应的 player texture。

### 16.4 task-level 渲染会保存哪些时刻

当前 task-level 渲染一共会保存：

1. `initial`
2. 对每个 task，再保存：
   - `pre`
   - `post`

也就是说总数固定是：

- `1 + 2 * len(task_order)`

其中：

- `initial`：第一个 task 开始前，只有玩家起点 known；
- `pre`：该 task 的 `_reveal_until_ready()` 结束后；
- `post`：该 task 的 `_ensure_material_inventory()` 和 `_execute_micro_action()` 都结束后。

### 16.5 保存的文件

如果 `planner.render_known_world = task`，则每个 episode 目录下还会额外生成：

- `known_world_masks/*.npz`
- `known_world_images/*.png`

mask 文件当前至少包含：

- `known_mask`

png 则是可视化后的整张世界图。

## 17. Hydra 配置项说明

当前 `run_gui.yaml` 中与这套 TAMP 直接相关的配置，大致可以写成：

```yaml
headless: false
episodes: 1
workers: 1

planner:
  enabled: true
  name: vanilla
  exit_on_finish: true
  output_dir: planner_results
  plan_only: false
  render_known_world: none
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

- `headless`
  - 是否关闭 GUI 窗口。
- `episodes`
  - 要跑多少个 episode。
- `workers`
  - 并发 worker 数；仅在 `headless = true` 下允许大于 1。
- `planner.enabled`
  - 是否启用 planner。
- `planner.name`
  - 当前 planner 名称，只支持 `vanilla`。
- `planner.exit_on_finish`
  - 是否在 14 个任务都完成后提前结束当前 episode。
- `planner.output_dir`
  - planner 结果根目录。
- `planner.plan_only`
  - 是否只规划不执行。
- `planner.render_known_world`
  - 是否输出 known-world 可视化；`step` 目前未实现。
- `planner.result_logging`
  - 选择哪些 planner metrics 写入结果文件；目前只支持 `reveal_count`。
- `planner.task_order`
  - planner 要依次执行的高层 task 列表。

另外：

- `defaults: - tamp_planner: vanilla` 已经接在 `run_gui.yaml` 里；
- `crafter/conf/tamp_planner/vanilla.yaml` 目前仍然是空占位文件。

## 18. 当前有哪些 fail-fast 约束

当前实现尽量不做“静默容错”。

一旦某个前提不满足，代码通常直接 `assert`，例如：

- task 名未知；
- reveal 候选为空；
- 想访问 unknown cell；
- 目标格不可达；
- craft cluster 不再兼容；
- planner 动作执行完了，但 14 个任务还没完成；
- `headless = false` 但 `workers > 1`；
- `headless = true` 但 `planner.enabled = false`；
- `planner.plan_only = true` 但还开启了依赖环境执行期的渲染或 record；
- `planner.render_known_world = step`；
- 某个已经 reserve 的 craft cluster 由于后续放置而失去兼容性；
- 某个非 `table/furnace` 的 `place_*` 试图写入 reserved craft cluster。

这样做的目的是：

- 尽早暴露建模问题；
- 不让 planner 在错误状态下继续“装作还能运行”；
- 方便之后迭代真正的设计，而不是被各种 fallback 掩盖问题。

## 19. 当前实现有哪些刻意的简化

这套实现是可以工作的第一版，但它明确做了不少简化。

### 19.1 reveal 不是环境动作

当前 reveal 只改变 `KnownWorld`，不对应环境里的真实探索动作。

所以 planner 是“离线知道地形、在线只执行动作串”的近似版本。

### 19.2 1x1 reveal 仍是 uniform baseline

虽然接口已经留成了“按 requirement 选 reveal cell”，但当前 1x1 的评分实际上还是 uniform tie-break。

### 19.3 4x4 reveal 不是严格联合 Bayesian 推断

当前 4x4 reveal 只是“看一个格子出现在多少个可行 anchor 里”，并不是在求真正的全局联合后验最优。

### 19.4 task catalog 还是代码内置

目前 task 定义没有搬到 Hydra/YAML，仍然写死在 `Task._TASK_DEFINITIONS` 里。

### 19.5 planner 一次性生成全串动作

当前不是边执行边 replanning，而是先从初始状态生成完整动作串，再交给环境执行。

这意味着：

- 如果环境在执行中发生了 planner 未建模的动态变化；
- 这版 planner 不会中途修正。

### 19.6 task-level 渲染只跟踪 task 边界

当前只支持：

- `initial`
- 每个 task 的 `pre/post`

并不支持每一步都导出 known-world 图像。

所以 `render_known_world = step` 现在仍然明确是未实现。

### 19.7 reserved craft cluster 是硬约束，不会自动“修复”

当前实现一旦 reserve 了某个 4x4 craft cluster，就把它当作长期不变的全局布局约束。

这意味着：

- planner 不会在后面“发现营地被破坏了以后重新选一个新的 cluster”；
- 也不会自动做局部修补；
- 一旦这个 reserve 被破坏，就直接报错。

这是一种故意的 fail-fast 设计。

之前的一个真实 bug 就来自这里：

- `place_stone` 的 1x1 选点最初只避开了 `table/furnace/crafting_stand` 这些命名格；
- 但没有避开整个 reserved 4x4；
- 结果石头可能被放进营地的其他格子里；
- 后续再次检查 `craft_cluster_candidates(fully_known=True)` 时，就会报
  `Reserved craft cluster is no longer compatible`。

现在这个 bug 已经修正，规则也明确了：

- 整个 reserved 4x4 都不能被普通 1x1 放置占用。

## 20. 后续自然的扩展方向

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
7. 如果以后确实有需要，再认真支持 `headless = false, workers > 1` 的多窗口并发 GUI。
8. 如果需要更完整的可视化，再补 step-level known-world 渲染。

## 21. 一句话总结

当前这版 TAMP 的核心思想是：

- 用 `KnownWorld` 把“初始真值地图”和“当前已知且会演化的地图”分开；
- 对每个高层任务先补知识，再拿材料，最后执行终结动作；
- 用最简单可工作的 reveal 和 motion planning 先把完整主线打通；
- 再把 planner 接入 `run_gui.py`，支持 headless、batch、plan-only、结果目录和 task-level 渲染；
- 并且用 fail-fast 保证一旦建模假设不成立，问题会尽快暴露。
