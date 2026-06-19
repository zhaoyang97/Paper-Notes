---
title: >-
  [论文解读] Contrastive Representations for Temporal Reasoning
description: >-
  [NeurIPS 2025][自监督学习][对比学习] 提出 CRTR（Contrastive Representations for Temporal Reasoning），通过在训练批次中重复同一轨迹来引入轨迹内负样本对，消除标准时间对比学习对静态上下文特征的依赖，学习到反映时间结构的表征，在魔方等组合推理任务上首次实现无搜索求解。
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "对比学习"
  - "时间推理"
  - "组合问题"
  - "Sokoban"
  - "魔方"
---

# Contrastive Representations for Temporal Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2508.13113](https://arxiv.org/abs/2508.13113)  
**代码**: [GitHub](https://github.com/Princeton-RL/CRTR)  
**领域**: 自监督学习 / 表征学习  
**关键词**: 对比学习, 时间推理, 组合问题, Sokoban, 魔方

## 一句话总结

提出 CRTR（Contrastive Representations for Temporal Reasoning），通过在训练批次中重复同一轨迹来引入轨迹内负样本对，消除标准时间对比学习对静态上下文特征的依赖，学习到反映时间结构的表征，在魔方等组合推理任务上首次实现无搜索求解。

## 研究背景与动机

### 现有痛点

**现有痛点**：组合推理问题（如 Sokoban、魔方）通常需要昂贵的搜索算法（A*、BestFS）来求解

### 领域现状

**领域现状**：时间对比学习**（如 CRL）用于学习状态表征，正样本来自轨迹内邻近状态，负样本来自不同轨迹

### 核心矛盾

**核心矛盾**：关键失败模式**：在 Sokoban 中，不同轨迹有不同的墙壁布局（上下文），CRL 利用墙壁布局而非时间结构来区分正负样本，导致同一轨迹内所有状态被编码到极相似的表征（t-SNE 显示轨迹聚簇为小点）

### 解决思路

**解决思路**：结果：CRL 表征无法反映状态间的时间距离，对规划无用

## 方法详解

### 整体框架

CRTR 的核心改动极其简洁——仅修改对比学习中的负采样方式：

1. 将训练批次中的轨迹 ID 重复若干次（repetition factor = 2）
2. 这使得批次中有多个来自同一轨迹但不同时间步的正样本
3. 标准对比学习将其他批次元素作为负样本，因此**同一轨迹内的不同时间状态成为负样本对**
4. 模型无法再利用恒定的上下文特征（如墙壁位置）来区分，被迫学习时间结构

### 关键设计

1. **轨迹内负采样消除上下文依赖**:
    - 功能：修改数据采样，使同一轨迹的多个时间步出现在同一批次中
    - 核心思路：当负样本具有相同上下文（如相同墙壁布局）时，上下文特征对区分正负样本无用，模型被迫编码时间结构
    - 设计动机：理论上，此目标是条件互信息 $I(X;X^+|C)$ 的下界，等价于最大化 $I(X;X^+) - I(X^+;C)$，后者类似对抗性特征学习但无需对抗训练

2. **从理想化到实用化的推导**:
    - 功能：理想化方法需要知道上下文变量 $C$ 并按其条件抽负样本；实用化方法只需重复轨迹 ID
    - 核心思路：重复轨迹 ID 自然产生具有相同上下文的负样本对（同一集的不同时间步），且所有负样本都有锚点（避免表征漂移）
    - 设计动机：实际中无法事先知道哪些特征是"上下文"（如第一次看到 Sokoban 板面，如何知道墙不可移动？）

### 损失函数 / 训练策略

- 标准 InfoNCE 损失，唯一改变是批次构建方式（`traj_id = np.repeat(traj_id[:B//R], R, axis=0)`）
- 编码器架构：8 层 MLP，隐藏维度 512，表征维度 64
- 对比损失使用后向版本（backward），在魔方上优于对称版本
- Adam 优化器，学习率 0.0003，batch size 512
- repetition factor = 2 在所有测试环境中均表现良好

## 实验关键数据

### 主实验（表格）

| 环境 | CRL 成功率 | Supervised 成功率 | DeepCubeA 成功率 | **CRTR 成功率** |
|------|-----------|-----------------|----------------|---------------|
| Sokoban | ~10% | ~30% | ~35% | **~40%** |
| 魔方 | ~55% | 0% | ~60% | **~63%** |
| 15-Puzzle | ~35% | ~50% | - | **~50%** |
| Lights Out | ~30% | ~10% | - | **~80%** |
| Digit Jumper | ~5% | ~60% | - | **~70%** |

（BestFS 搜索预算 6000 节点）

### 消融实验

- Repetition factor: R=2 一致提升所有环境；过大的 R 在某些环境退化
- Spearman 相关性（表征距离 vs 真实步数距离）：CRTR > 0.8 vs CRL < 0.4（Sokoban）
- 无搜索求解：CRTR 在 4/5 任务上用贪心策略几乎解决所有实例（魔方 100% 成功！）

### 关键发现

- **最惊人的结果**：CRTR 在魔方上**不用任何搜索**即可解决所有随机打乱的配置（在 6000 步内）
- 无搜索解法虽然更长（平均 ~400 步 vs 最优 ~26 步），但展现出类似人类"块构建"的涌现行为
- CRTR 在 A* 搜索中也优于 CRL，改进不限于贪心搜索
- 平均无搜索解长度：CRTR 448.7 vs CRL 1830.3（魔方）

## 亮点与洞察

- **极简改动，巨大影响**：仅修改一行代码（重复轨迹 ID）即实现从"完全失败"到"SOTA"的跨越
- **无搜索求解组合问题**：首次仅用学习到的表征（无外部搜索算法）高效求解任意魔方状态
- 涌现的"块构建"策略呼应了人类解谜行为
- 条件互信息框架漂亮地解释了为何去除上下文可改善时间推理

## 局限与展望

- 无搜索解太长（魔方 ~400 步），远非最优
- Sokoban 成功率仍较低，可能由于死路（irreversible states）问题
- 假设动力学已知且确定性，限制了对随机/未知动力学问题的适用性
- 魔方的状态距离几乎总满足三角不等式取等，无法忠实嵌入欧氏空间

## 相关工作与启发

- CRL (Eysenbach et al.) 是直接基础，CRTR 修复了其在组合域的关键失败
- DeepCubeA 用值迭代学习启发式，CRTR 用对比学习达到可比甚至更好结果
- 与对抗性特征学习的联系：消除上下文≈对抗性去除不变特征，但无需对抗训练

## 评分

- 理论创新：⭐⭐⭐⭐⭐
- 实验验证：⭐⭐⭐⭐⭐
- 实用价值：⭐⭐⭐⭐
- 写作质量：⭐⭐⭐⭐⭐
- 综合评分：⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Contextures: Representations from Contexts](../../ICML2025/self_supervised/contextures_representations_from_contexts.md)
- [\[NeurIPS 2025\] The Complexity of Finding Local Optima in Contrastive Learning](the_complexity_of_finding_local_optima_in_contrastive_learning.md)
- [\[CVPR 2026\] Temporal Representation Enhancement (TRE): Learning to Forget Dominant Patterns for Enhanced Temporal Spiking Features](../../CVPR2026/self_supervised/temporal_representation_enhancement_tre_learning_to_forget_dominant_patterns_for.md)

</div>

<!-- RELATED:END -->
