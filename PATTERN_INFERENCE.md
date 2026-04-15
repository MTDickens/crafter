# planning-active-perception 的 pattern inference

这个扩展在现有 task-and-motion planner 之上增加了一个小型的 white-box 地形推断模块。

## 设计概述

实现刻意只使用了**一种**手工设计的模板族：

```python
@dataclasses.dataclass(frozen=True)
class CrossShapedPattern(Pattern):
    center: str
    top: str
    bottom: str
    left: str
    right: str
```

每个实例化后的精确十字形 pattern 都带有一个非负权重。

### 为什么第一次实现只用一种模板就够了

关键技巧在于，推断会对未观测到的槽位做 **marginalize**。

如果某个目标 tile 只有一个已观测邻居，那么库会对所有“匹配边与该观测一致”的精确十字形 pattern 的权重求和。如果有两个邻居已被观测，它就会对所有同时匹配这两个邻居的精确十字形 pattern 求和，以此类推。

因此，只用一个精确十字形模板族，就已经能形成一条清晰的证据阶梯：

- 只观测到一侧 -> 成对的局部证据
- 观测到两侧或三侧 -> 部分局部上下文
- 四侧全部观测到 -> 精确的 5-tile 证据

这样系统能保持简单且可检查，同时对地形连续性和 habitat 风格的先验也已经足够有表达力，例如：

- iron / coal 往往出现在 stone 内部
- tree 往往出现在 grassland 内部
- grass 和 stone 往往具有空间连续性
- sand 经常出现在 water 和 grass 附近

## 训练

训练是离线进行的，并且完全是 white-box。

默认策略是 `connected_frontier`：

1. 生成完全观测的历史 Crafter 地图
2. 从地图中心开始，采样一个连通的已观测 blob
3. 收集与该 blob 相邻、但处于隐藏状态的 frontier tiles
4. 在每个这样的 tile 处提取真实的精确十字形 pattern
5. 将该精确 pattern 的权重加 1

这套设计是有意与 planner 的探索机制对齐的：已观测空间通常是连通的，而 planner 也是从可达的 frontier tiles 中进行选择。

也支持一个更简单的 `all_tiles` 策略。

默认训练配置使用了一个**较小**的 `prior_scale`（`0.01`）。实践中，过重的先验会冲淡我们真正想让 frontier 选择利用到的局部 pattern 证据。

## 推断

对于一个未知位置 `u`，设 `C(u)` 表示它周围已观测邻居的集合。候选材质 `m` 的分数为：

```text
score(m | C(u))
  = prior_scale * prior_count(m)
    + sum_{p.center = m and p matches C(u)} weight(p)
```

这些分数随后会被归一化成一个覆盖所有材质的 categorical distribution。

该库可以推断：

- 单个 tile
- 一个 frontier 集合
- 完整地图中所有未知的 tiles

它还支持一种可选的递归式 **soft commit** 模式：

- 如果某个 tile 的最大 posterior 超过 `commit_threshold`
- 该推断出的材质只会在 inference engine 内部被 commit
- 真正的 `KnownWorld` 不会被修改
- 后续推断轮次可以把这些高置信度的 pseudo-label 当作额外上下文来使用

该模式默认关闭，因为 planner 应该保持保守。

## 与 Planner 的集成

新增了一种 reveal policy：

```yaml
planner:
  frontier_reveal_mode: pattern_inferred_prob_guided_gradual
  pattern_inference:
    library_path: crafter/patterns/default_connected_frontier_cross_library.json
    commit_threshold: null
    commit_max_passes: 1
```

这个策略保持了原先“只考虑可达 frontier”的约束，但会按以下标准对 frontier tiles 排序：

1. 当前目标材质的推断概率
2. 较低的 posterior entropy
3. 到可达已知区域的接近程度

因此它依然是一个 gradual frontier-expansion 策略，只不过现在由概率来引导。

## 新增 / 修改的文件

- `crafter/pattern_library.py`
  - pattern 定义
  - 加权推断
  - connected-frontier masking
  - held-out evaluation
  - JSON save / load
- `crafter/train_pattern_library.py`
  - 离线训练入口
- `crafter/conf/train_pattern_library.yaml`
  - 默认训练配置
- `crafter/task_motion_planner.py`
  - 新 reveal mode 的集成
- `crafter/conf/run_gui_from_actions.yaml`
- `crafter/conf/run_headless_planner.yaml`
- `crafter/patterns/default_connected_frontier_cross_library.json`
  - 默认训练得到的库

## 训练一个库

```bash
python -m crafter.train_pattern_library \
  output_path=crafter/patterns/my_library.json
```

## 用 pattern-guided reveal 运行 planner

```bash
python -m crafter.run_gui_from_actions \
  planner.frontier_reveal_mode=pattern_inferred_prob_guided_gradual
```

或者在 headless 模式下：

```bash
python -m crafter.run_headless_planner \
  planner.frontier_reveal_mode=pattern_inferred_prob_guided_gradual
```

## 预期的扩展点

当前代码刻意保持得很小。未来最容易做的扩展包括：

- 只有在精确十字形库明显不够用时，再加入第二种模板族
- 调整 masking 策略，使其匹配不同的探索机制
- 用别的 white-box 统计量替代硬权重，比如校准后的 log-odds
- 在更丰富的 active-perception objective 中使用推断得到的整图概率
- 只对那些在探索打分里允许被 hallucinate 的材质启用 soft commit
