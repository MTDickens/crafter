# Crafter Pattern Learning

## Overview

这套实现是在不大改现有 Crafter TAMP 架构的前提下，加进去的一条在线 pattern-learning 路径。它的目标不是替代当前 planner，而是替换 planner 在 `reveal` 阶段的打分 backend。

现在有两种 planner：

- `vanilla`
- `pattern_learning`

其中 `pattern_learning` 会在 episode 之间维护一个 replay buffer、一个 pattern library，以及一套在线重加权逻辑。

## 核心语义

这里的 `pattern`、`skill` 基本混用。当前只实现一种 pattern family：`CrossShapedPattern`。

它的语义严格是：

- `top / bottom / left / right` 用来 gate
- `center` 是被预测的类别

也就是说，它表示的是：

`P(center | top, bottom, left, right)`

而不是 FrozenLake 里那种整块 `4x4 -> 4x4` 的 patch skill。

## Numeric 表示

内部统一使用 numeric tensor，而不是字符串网格。

具体由 `CrafterTileCodec` 负责：

- worldgen material name -> integer id
- integer id -> worldgen material name
- `unknown_id`
- `boundary_id`

当前 learner 的类别空间只包含初始 worldgen 材料：

- `water`
- `grass`
- `stone`
- `path`
- `sand`
- `tree`
- `lava`
- `coal`
- `iron`
- `diamond`

`table`、`furnace` 这种执行中才出现的材料不参与 learner 的分类空间。

## Pattern 类型

### 1. 普通 cross pattern

普通 pattern 必须完整给出五个 id：

- `center_id`
- `top_id`
- `bottom_id`
- `left_id`
- `right_id`

它们都必须是正常 material id。

### 2. Ungated placeholder pattern

系统启动时，还会自动加入一组特殊 placeholder patterns。

每个 material class 都有一个：

- “预测中间格是 `water`”
- “预测中间格是 `grass`”
- ...

这些 pattern：

- 没有 gating
- 永远激活
- 只能通过专门的工厂方法创建

这样做的目的，是保证即使还没有任何结构化 pattern，library 也能表达一个最基础的“中心格属于某类”的偏好。

## 单个 skill 的输出

每个 skill 对某个 center cell 的输出，都是一个带 smoothing 的类别分布。

如果 pattern 预测的类别是 `center_id`，那么输出为：

- `P(center_id) = 1 - eps`
- 其余 `num_classes - 1` 个类别均分 `eps`

这里的 `eps` 来自配置，默认是：

- `eps = 1e-3`

这个设计刻意和 FrozenLake 那边保持一致，不做硬的 `1 / 0`。

## Gating 规则

对某个候选 center cell：

- 如果 pattern 是普通 gated pattern，就检查四邻
- 四邻里所有“当前已知”的格子，必须和 pattern 对应方向的 token 一致
- 四邻里未知的格子，不会让 pattern 失效
- center 自己不参与 gating

如果 pattern 是 ungated placeholder，那么它在所有 center cell 上都视为 gate 成功。

## 多个 pattern 的聚合

当前不使用 simplex weight。

每个 pattern 只有一个要求：

- 权重必须是正数

实现上用 `softplus(raw_weight)` 把 raw 参数映成正数权重。

对某个 cell，只收集当前 gate 成功的 active patterns，然后做局部归一化：

`posterior = sum(weight_s * output_s) / sum(weight_s)`

注意这里的归一化是每个 cell 局部做的，不是全局 simplex。

如果某个未知 cell 没有任何 active pattern，那么返回 uniform prior。

如果权重和出现 `nan` / `inf`，当前实现会直接 fail-fast，因为这被视为 bug，而不是要悄悄兜底的正常情况。

## Hard inference

系统里已经有 hard inference 机制，但当前默认基本不会触发。

原因是当前配置把：

- `enable_hard_inference: true`
- `infer_prob_threshold: 1.0`

同时打开了。

而 skill 输出由于有 `eps` smoothing，未知格的最大 posterior 几乎不可能达到严格的 `1.0`。因此这个机制现在只是“存在”，但默认不实际改写 known world。

这符合当前需求：先把机制接上，但默认关闭其效果。

## 在线训练

这套训练不是离线预训练，而是在 `run_gui.py` 的 episode loop 中在线进行。

### Replay buffer

每个 episode 结束后，会把下面这些信息放进 replay buffer：

- `full_map_ids`
- `final_known_mask`
- `start_pos`
- `episode_index`
- `seed`

这里的 `full_map_ids` 指的是初始 worldgen map，不是执行后会演化的 `_world`。

### 训练触发

当满足：

- `(episode_index + 1) % train_trigger_interval == 0`
- 并且还没到最后一个 episode

就会触发一次训练。

默认 `train_trigger_interval = 10`。

### 训练数据

训练使用 replay buffer 里 **全部** entries，而不是只取最近几条。

对每张 map，会采样若干条 connected trajectory。默认：

- `pure_samples_per_map = 20`

采样方式不是 IID monte-carlo drop，而是 connected reverse trajectory：

1. 从该 episode 的最终 known mask 出发
2. 构造一个从起点出发、覆盖所有 known cell 的随机连通 reveal 顺序
3. 把这个 reveal 顺序拆成一系列 transition

于是训练样本就是：

- 输入：`s_{t-1}`
- 目标：`s_t` 中新增出来的那一个格子的真实类别

也就是说，当前训练的任务是：

- next revealed cell classification

而不是 full-map reconstruction。

### Loss

对每个 transition，只在那个“新增的中心格”上算 multiclass loss。

目标分布同样带 smoothing：

- true class: `1 - eps`
- others: 均分 `eps`

然后对所有 sampled transitions 求平均。

### Proposal

当前实现不做 automatic mining。

pattern 来源只有三类：

1. YAML 里的 preset patterns
2. 启动时自动加入的 center-only placeholders
3. 每次训练前可选的 LLM proposals

proposal 逻辑在 Crafter 版里被拆成独立模块，结构参考了 FrozenLake 的 `proposal.py`，但输出格式改成了 Crafter cross pattern 的 JSON。

## 与 planner 的集成

集成点在 `TaskAndMotionPlanner.reveal_next_cell_for_requirement(...)`。

planner 的总体结构没改，仍然是：

1. `_reveal_until_ready`
2. `_ensure_material_inventory`
3. `_execute_micro_action`

变化只在 reveal 的打分逻辑：

- `collect`
  - 用 posterior 对 frontier cell 的目标 pre-material 概率打分
- `place_1x1`
  - 用 posterior 对可放置材料概率打分
- `layout`
  - 目前用一个极简近似：learner 给出 walkable posterior，和原先 vanilla 的 4x4 候选计数分数做简单结合

如果当前没有可用 library，或者 `enable_inference=false`，就回退到原来的 vanilla reveal。

## Hydra 配置

新的 planner preset 是：

- `crafter/conf/tamp_planner/pattern_learning.yaml`

它保留了很多 FrozenLake 风格的配置层次，但当前没有被 Crafter 主路径使用的字段，不会作为 active config 存在，而是以注释保留，例如：

- `# limit: 5`
- `# train_on_full_desc: false`
- `# exploration_extra_points_per_map: 15`
- `# pure_drop_prob: 0.5`

这样结构上仍然容易和 FrozenLake 对照，但不会误导当前 Crafter 实现真的在消费这些字段。

## 当前限制

为了保持实现最小且稳定，`pattern_learning` 当前要求：

- `workers == 1`

原因很直接：online learner 需要在 episode 之间共享 replay buffer 和当前 library。要把它正确推广到多 worker，需要再设计跨进程状态同步，这次没有做。

## 文件概览

这次新增的核心文件有：

- `crafter/utils/crafter_codec.py`
- `crafter/skills/crafter_patterns.py`
- `crafter/skills/crafter_proposal.py`
- `crafter/skills/crafter_weight_optimization.py`
- `crafter/training/crafter_pattern_learning.py`
- `crafter/conf/tamp_planner/pattern_learning.yaml`

核心接入点在：

- `crafter/task_and_motion_planner.py`
- `crafter/run_gui.py`

## 总结

这次实现的重点不是做一个“最终版”的 pattern learner，而是把以下闭环先真实接上：

1. partial known world -> numeric tensor
2. cross-shaped conditional patterns -> posterior
3. posterior -> reveal policy
4. episode replay -> online reweighting
5. optional proposal -> 扩 pattern library

后面如果要继续改，最自然的方向有三个：

- 把 layout reveal 从“简单 walkable 打分”升级成更严谨的 4x4 联合打分
- 把 connected reverse trajectory sampler 换成你更想要的连通-drop 版本
- 真正启用 hard inference，而不是把阈值固定在 1.0
