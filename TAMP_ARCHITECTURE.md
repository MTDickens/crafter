# Planning-Active-Perception 架构说明

这个版本故意做成 **极简、外挂式、fail-fast** 的结构：

- 现有 GUI / env / recorder 基本不动。
- `run_gui_from_actions.py` 只多了一条输入分支：`actions_input_source=planner`。
- 真正的逻辑都放在 `crafter/task_motion_planner.py` 里，后续要换启发式、加 probabilistic inference、换 perception backend，都只需要改这一个模块。

## 核心分层

### 1. KnownWorld

负责“我已经看过哪些格子”。

- `sync(world, pos, charge_cost)`：观察一个格子。
- `reveal_all(...)`：当前设定下可直接全图初始化。
- `known_distance_map(start)`：只在当前已知且可站立的 blob 上做 BFS。

它是 perception/belief 的最小壳层，后续可以替换成概率地图，而不需要动 task/motion 层。

### 2. LazyMotionPlanner

负责“从当前位置怎么过去”。

- 图搜索跑在 **乐观地图** 上：未知格子默认可走。
- 路径代价对未知格子施加 `unknown_motion_penalty`，因此目标不是最短路，而是 **尽量少穿未知格子**。
- 路径验证是 lazy 的：只有当前路径上真的会走到的未知格子才会被 `sync(...)`。

因此，已知地图会逐渐长成：

- 一大块贴着一起的已知 blob；
- 加上若干条 LazySP 式的窄轨迹。

### 3. TaskMotionPlanner

负责“接下来干什么”。

高层 task 目前按固定 skeleton：

1. `collect_wood`
2. `place_table`
3. `make_wood_pickaxe`
4. `make_wood_sword`
5. `collect_stone`
6. `place_stone`
7. `make_stone_pickaxe`
8. `make_stone_sword`
9. `place_furnace`
10. `collect_coal`
11. `collect_iron`
12. `make_iron_pickaxe`
13. `make_iron_sword`
14. `collect_diamond`

但 task 执行时允许插入 **auxiliary excavation**：

- 比如当前任务是 `collect_iron`，如果还没有可直接够到的 iron，planner 会优先挖已经允许挖的 frontier stone/coal，往 iron 方向打洞。
- 这样就不会把“收集 stone”写死成一次性逻辑，而是让“收集目标”和“为未来目标开路”耦合起来。

## 两阶段循环

每次高层 task 都走下面的逻辑：

1. **找目标**
   - 先看已知图里有没有可达的目标资源。
   - 没有就沿着当前已知 blob 的 frontier 往外 reveal。
   - 还没有的话，选一个允许的辅助挖掘点，继续把 blob 往目标方向推进。

2. **去执行**
   - 用 `LazyMotionPlanner` 找一条“尽量少看新格子”的路径。
   - 到位后转向并执行 `do` / `place_*` / `make_*`。

然后再回到下一轮。

## 额外的小策略

### Workspace 预留

`place_table` 时会先选一个 workspace：

- 一个 stand cell；
- 一个 table slot；
- 一个 furnace slot。

这样后续 `make_wood_*` / `make_stone_*` / `make_iron_*` 都可以复用同一个 crafting spot。

### place_stone 尽量不堵路

`place_stone` 优先考虑：

- 附近的 water/lava；
- 否则选择局部连接度较低的 tile，尽量不堵主通路。

## 当前验证范围

当前重点验证的是 **当前全局视野设定**：

- `planner.initial_knowledge: global`
- `actions_input_source: planner`

也就是你现在说的这个场景。

`initial_knowledge: local` 的接口也已经留好了，但我没有把这一版当作完整的 partial-observability 成品；后面要加 probabilistic inference，建议就在 `KnownWorld` 和 frontier scoring 那一层继续长。
