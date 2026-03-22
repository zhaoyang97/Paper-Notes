# Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents

**会议**: AAAI 2026  
**arXiv**: [2509.01022](https://arxiv.org/abs/2509.01022)  
**代码**: 无  
**领域**: 多智能体路径规划 / 仓储优化  
**关键词**: Block Rearrangement, MAPF, 符号规划, PDDL, LaCAM, 密集环境, 仓储机器人

## 一句话总结

提出 Block Rearrangement Problem (BRaP) 形式化定义，并设计五种基于配置空间搜索、PDDL 符号规划和 MAPF 的求解算法，其中 BR-LaCAM 在最大 80×80 的极端密集网格上达到 92% 成功率和毫秒级求解速度。

## 动机

1. **实际需求驱动**：Amazon Robotics 仓库使用密集存储网格提高空间利用率，但当被需要的货架（block）被深埋时，需要复杂的重排计划将其移出。
2. **现有问题定义不足**：滑块拼图、Rush Hour、Sokoban 等经典问题与仓库重排有结构相似性，但都无法完整刻画 BRaP 特有的约束（unassigned blocks、following conflict、多目标集合等）。
3. **密度极高带来组合爆炸**：网格几乎被 block 填满（空位极少），配置空间随 block 数量呈指数增长，传统方法难以扩展。
4. **Following Conflict 被忽视**：大多数 MAPF 求解器忽略 following conflict（一步后占据前者位置），而仓库场景因禁止 platooning 而必须考虑。
5. **Unassigned Agents 的协调**：大量无目标的 block 需要被主动移开为有目标 block 让路，这一协调问题在经典 MAPF 中未充分研究。
6. **缺少基准方法**：BRaP 作为新定义的问题，需要一组基线算法来建立性能参照并指引未来研究方向。

## 方法详解

### 问题形式化

在图 $G=(V,E)$ 上定义 BRaP：给定 assigned block 集合 $\mathcal{I}$（有目标）和 unassigned block 集合 $\mathcal{J}$（无目标），每个 assigned block $i$ 有初始位置 $s_i$ 和目标顶点集 $V_i$。动作集 $A=\{\text{move}, \text{wait}, \text{complete}\}$，必须避免顶点冲突、边冲突和 following conflict。目标是找到最小代价的无冲突路径方案，使所有 assigned block 到达各自目标。

### 算法一：Configuration Space Search

将 BRaP 建模为配置空间上的图搜索。每个状态包含时间、assigned block 位置、空位集合和已完成 block 集合。使用 A* 搜索，设计可容许启发式：假设每个 assigned block 沿最少阻挡路径移动，每个阻挡 block 仅需一次 move 即可清除，对多个 assigned block 的代价取平均作为联合下界。限制每步仅一个动作以降低分支因子。

### 算法二：PDDL-based Configuration Space Search

将配置空间搜索用 PDDL 描述。定义谓词 `(emp ?u)`, `(asb ?u)`, `(blk ?u)`, `(cmp ?u)`, `(goal ?u)` 表示顶点状态，三个动作 `move_blk`, `move_asb`, `complete`。使用 fast-downward 求解器求解，生成单动作顺序计划。

### 算法三：Priority-based Configuration Space Search

允许每步每个 assigned block 各执行一个动作，实现并行执行。按启发式距离分配优先级，逐一为每个 assigned block 生成计划，高优先级 block 的计划作为约束传递给低优先级 block，确保并行执行无冲突。

### 算法四：Heuristic Approach（最少阻挡路径）

纯启发式方法，适用于大规模网格。为每个 assigned block 计算最少阻挡路径 $U_i$，沿路径逐步清空前方顶点（将阻挡 block 移至最近空位），然后移动 assigned block 前进。通过 vertex blocking time 函数 $\psi$ 实现多个 assigned block 的并行移动。

### 算法五：BR-LaCAM（核心贡献）

基于 LaCAM 框架，设计 BR-PIBT 作为配置生成器：

- **优先级管理**：block 到达目标时优先级重置为 $(0,1)$ 随机值，被挤出时优先级 +1，确保被反复打断的 block 最终获得最高优先级。
- **递归规划**：按优先级降序处理 block，每个 block 尝试移向最优邻居；若被占据则递归请求占据者让路，形成位移链直到触及空位。
- **临时目标分配**：assigned block 从最近未分配目标中选取临时目标；unassigned block 优先靠近空位以保持系统流动性。
- **与 LaCAM 集成**：BR-PIBT 生成单步后继配置，若产生已探索状态则回溯并增长低层约束，逐步覆盖所有可能后继。BR-LaCAM 保持完备性，并支持 anytime 改进。

## 实验

### 实验设置

在 13,860 个测试用例上评估，网格大小从 4×10 到 80×80，assigned block 数最多占顶点数 12.5%，空位数最多占 25%。三种目标类型：Goal B（边界目标）、Goal R1（随机目标，数量=assigned block 数）、Goal R2（随机目标，数量=2倍 assigned block 数）。时间限制 10 秒/用例。

### 表1：动作代价矩阵

| Block 类型 | move | wait | complete |
|:---:|:---:|:---:|:---:|
| Assigned | 2 | 1 | 2 |
| Unassigned | 2 | 0 | N/A |

### 表2：各算法在不同网格尺寸上的成功率与代价比

| 网格大小 | BR-LaCAM 成功率 | Heuristic 成功率 | Priority 成功率 | Config 成功率 | PDDL 成功率 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 4×10 | **100%** | 93% | 91% | 87% | 89% |
| 6×10 | **100%** | 97% | 87% | 76% | 76% |
| 10×10 | **100%** | 96% | 60% | 42% | 34% |
| 20×20 | **100%** | 95% | 15% | 7% | 4% |
| 40×40 | **99%** | 91% | 8% | 3% | 0% |
| 80×80 | **92%** | 80% | 2% | 0% | 0% |

BR-LaCAM 综合代价比整体均值 1.02（标准差 0.21），Heuristic 为 1.18（标准差 0.23），其余算法在大网格上几乎完全失败。

### 表3：按目标类型分组的成功率

| 目标类型 | BR-LaCAM | Heuristic | Priority | Config | PDDL |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Goal B（边界） | **100%** | 97% | 56% | 49% | 45% |
| Goal R1（严格随机） | **98%** | 82% | 33% | 22% | 22% |
| Goal R2（宽松随机） | **99%** | 100% | 56% | 47% | 44% |

Goal R1 最具挑战性，BR-LaCAM 在此场景下代价标准差较高（0.34），反映了 BR-PIBT 短视行为在目标稀疏分布时的代价波动。

## 发现

- BR-LaCAM 在所有规模和目标类型上均表现最优，整体成功率 99%，是唯一能稳定处理 80×80 网格的算法。
- Heuristic 方法是唯一接近 BR-LaCAM 的竞争者（93% 成功率），但代价比高出约 16%。
- Configuration Space Search 和 PDDL 方法在 20×20 以上网格几乎完全失败，证实了配置空间指数爆炸的严重性。
- 空位数量增加可显著降低搜索深度、提高成功率并减少代价。
- BR-LaCAM 和 Heuristic 在 50% 以上用例中可在 1 毫秒内找到初始解，90% 以上用例在 1 秒内求解完毕。

## 亮点

- **首次形式化 BRaP**：将仓库货架重排问题严格定义为图搜索问题，建立了与滑块拼图、MAPF、MAPFUA 等经典问题的联系和区别。
- **BR-PIBT 配置生成器设计精巧**：通过优先级递增 + 递归位移链 + 临时目标分配，解决了原始 PIBT 无法处理 following conflict 的根本限制。
- **完备性保证**：BR-LaCAM 在有限配置空间上保持完备，失败仅因时间限制而非算法缺陷。
- **工业级规模验证**：在 13,860 个用例上系统评估，覆盖从 4×10 到 80×80 的网格，证明了实际可部署性。
- **Anytime 特性**：初始解快速产出后持续优化，兼顾实时性和解质量。

## 局限性

- 每步限制单动作（Config/PDDL）或单 block 单动作（Priority），未充分利用并行性。
- BR-PIBT 的短视行为在目标稀疏分布时导致代价波动大（Goal R1 场景标准差可达 0.34）。
- 未考虑动态场景：所有 block 位置和目标在规划前已知，不支持在线到达的新任务。
- 目标类型仅测试了"变为障碍物"这一种终止条件，另外两种（变为空位/变为 unassigned）未做实验。
- 启发式函数较为简单（单 block 独立估计取平均），在 assigned block 间高度耦合时可能偏松。
- 障碍物建模固定为右下角方块区域，缺少更复杂布局的测试。

## 相关工作

- **滑块拼图与 Rush Hour**：经典组合搜索问题，BRaP 在其基础上引入多 assigned block 和 unassigned block（Gozon & Yu 2024, Cian et al. 2022）。
- **PDDL 符号规划**：通过 PDDL 描述状态和动作实现自动规划，已应用于物流、机器人路径等领域（Helmert 2009, Davesa Sureda et al. 2024）。
- **经典 MAPF**：Stern et al. 2019 定义了标准 MAPF，LaCAM（Okumura 2023）实现了大规模高密度场景的快速求解。
- **MAPFUA**：Felner & Stern 2026 定义了含 unassigned agents 的 MAPF 变体，BRaP 是其在极端密集环境下的具体实例。
- **TAPF**：Ma & Koenig 2016 研究了目标分配与路径规划的联合问题，但不含 unassigned agents。
- **学习方法**：强化学习（Damani et al. 2021）和图神经网络（Li et al. 2020）已用于 MAPF，但未专门处理 BRaP 的密集约束。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次形式化 BRaP 并与 MAPF/MAPFUA 建立清晰联系
- 实验充分度: ⭐⭐⭐⭐⭐ 13,860 用例，系统覆盖多种参数组合
- 写作质量: ⭐⭐⭐⭐ 问题定义严谨，算法描述清晰
- 价值: ⭐⭐⭐⭐ 对仓储物流自动化有直接应用价值，BR-LaCAM 可实际部署
