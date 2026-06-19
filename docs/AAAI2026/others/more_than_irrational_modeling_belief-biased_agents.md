---
title: >-
  [论文解读] More Than Irrational: Modeling Belief-Biased Agents
description: >-
  [AAAI 2026][计算理性] 提出一种计算理性（CR）用户模型框架，将人类看似"不理性"的行为解释为在有限记忆（信念偏差）下的最优决策，通过嵌套粒子滤波（NPF）在线推断用户的潜在记忆界限参数 $\theta$ 和偏差信念状态 $\tilde{b}$，PM误差在45步内降低90%，并在辅助POMDP中展示自适应AI助手策略。
tags:
  - "AAAI 2026"
  - "计算理性"
  - "记忆衰退"
  - "嵌套粒子滤波"
  - "信念偏差"
  - "自适应辅助"
---

# More Than Irrational: Modeling Belief-Biased Agents

**会议**: AAAI 2026  
**arXiv**: [2511.12359](https://arxiv.org/abs/2511.12359)  
**代码**: [GitHub](https://github.com/Yifan-Zhu/More-Than-Irrational-Modeling-Belief-Biased-Agents)  
**领域**: 用户建模 / 人机协作  
**关键词**: 计算理性, 记忆衰退, 嵌套粒子滤波, 信念偏差, 自适应辅助

## 一句话总结

提出一种计算理性（CR）用户模型框架，将人类看似"不理性"的行为解释为在有限记忆（信念偏差）下的最优决策，通过嵌套粒子滤波（NPF）在线推断用户的潜在记忆界限参数 $\theta$ 和偏差信念状态 $\tilde{b}$，PM误差在45步内降低90%，并在辅助POMDP中展示自适应AI助手策略。

## 研究背景与动机

**领域现状**: 人机协作中，AI需要从用户的过去行为推断其目标、信念和未来动作。计算理性（CR）理论认为人类是在认知约束下的理性决策者，其"非理性"行为源于有限资源而非真正的随机性。

**现有痛点**: (1) 现有CR研究聚焦于视线、打字、驾驶等特定应用，缺乏对有限记忆的通用建模；(2) 已有工作假设给定不完美内部模型下的完美记忆，或将"非理性"简单归因于推理预算有限；(3) 从被动观察中在线推断用户的潜在认知界限和动态信念状态是计算上不可行的（精确推理复杂度 $O(|\mathcal{S}|^t \cdot t!)$）。

**核心矛盾**: 有限记忆导致的信念偏差使用户行为看似不理性，但AI助手如何区分"真正的不理性"和"基于错误记忆的理性决策"？

**本文目标**: 构建通用的有限记忆用户模型，并提出可行的在线推断算法，使AI可以实时追踪用户认知状态并提供自适应辅助。

**切入角度**: 将记忆衰退显式建模为认知过程 $f_\theta$，它系统性地腐蚀用户对历史观测的记忆，导致信念状态偏离真实状态。

**核心 idea**: "非理性"行为是在偏差信念下的理性决策——建模记忆腐蚀机制后，行为变得可预测和可推断。

## 方法详解

### 整体框架

框架分为三层：(1) **CR用户模型**：在标准POMDP上添加认知过程 $f_\theta$，将真实历史 $h_t$ 映射为腐蚀后的内部记忆 $\tilde{h}_t$，用户基于 $\tilde{h}_t$ 做贝叶斯滤波得到偏差信念 $\tilde{b}_t$，并在 $\tilde{b}_t$ 上执行最优策略；(2) **在线推断**：嵌套粒子滤波（NPF）同时维护外层 $\theta$ 粒子和内层 $\tilde{h}$ 粒子，根据观测到的用户动作更新权重；(3) **辅助POMDP**：AI助手利用推断得到的 $\theta$ 和 $\tilde{b}$ 选择最优干预策略（不干预/记忆提示/动作提示）。

### 关键设计

1. **显式记忆腐蚀过程 $f_\theta$**:

    - 功能：将用户内部记忆建模为随时间动态腐蚀的过程
    - 核心思路：在每个时步 $t$，内部记忆按 $\tilde{h}_t \sim f_\theta(\tilde{h}_{t-1}, o_t, a_{t-1})$ 更新。以记忆衰退为例，$\theta$ 表示遗忘概率——每步中每个历史观测有 $p=\theta$ 的概率被替换为默认值。偏差信念需全量重算：$\tilde{b}_t(s_t) \propto \sum_{s_{:t-1}} p(s_0)\mathcal{O}(\tilde{o}_t^0|s_0) \prod_{i=1}^t \mathcal{O}(\tilde{o}_t^i|s_i)\mathcal{T}(s_i|s_{i-1},\tilde{a}_t^{i-1})$
    - 设计动机：马尔可夫性不再成立——当记忆修改了 $\tilde{o}_t^i$ 时信念需要"回溯性"重新评估，这正是人类"回忆重放"的数学表达

2. **嵌套粒子滤波推断**:

    - 功能：从被动观察的动作流在线推断用户的 $\theta$ 和 $\tilde{h}$
    - 核心思路：维护 $N_\theta$ 个外层粒子（不同 $\theta$ 假设），每个外层粒子下 $N_{\tilde{h}}$ 个内层粒子（$\tilde{h}$ 的采样）。权重更新：$w^{(i,j)} \leftarrow w^{(i,j)} \cdot \pi_*(a_{t-1}|\tilde{b}_{t-1}^{(i,j)};\theta^i)$。外层权重通过内层似然聚合。总复杂度 $O(N_\theta N_{\tilde{h}} t |\mathcal{S}|)$
    - 设计动机：NPF天然适合"静态参数+动态状态"的联合估计结构；策略 $\pi_*(\cdot;\theta)$ 可以预计算，避免在线策略学习的高昂开销

### 损失函数 / 训练策略

采用PPO训练各 $\theta$ 对应的最优策略 $\pi_*(\cdot;\theta)$。NPF推断阶段不涉及训练——纯在线贝叶斯更新。辅助POMDP中AI助手的策略同样用PPO训练，奖励函数为"用户成功到达目标"减去"干预成本"（动作提示成本>记忆提示成本>不干预）。

## 实验关键数据

### 主实验

在线推断精度（100步内，所有 $\theta_{true}$ 上平均，$\tau=3.0$）：

| 指标 | 初始误差 | 45步 | 78步 | 最终(100步) |
|------|:-------:|:----:|:----:|:----------:|
| PM误差 | ~0.15 | 降低90% | 降低95% | **0.0087±0.0035** |
| MAP误差 | ~0.2 | - | - | 接近0 |

### 消融实验

CR模型行为合理性验证（T迷宫任务）：

| 记忆界限 $\theta$ | 行为模式 | 说明 |
|:-----------------:|---------|------|
| 0.0（完美记忆） | 最短路径直达目标 | 一步下探→直行→正确终点 |
| 0.4（中等衰退） | 多次观察标的物 | 收集冗余观测以增强记忆鲁棒性 |
| 0.7（严重衰退） | "遗忘-复查"模式 | 返回已访问位置重新确认目标 |
| 1.0（无记忆） | 随机猜测 | 不浪费时间探索，直接随机选择 |

### 关键发现

1. **不同 $\theta$ 产生直觉合理的行为谱**: 从最优到随机，中间过渡行为高度可解释
2. **推断收敛快速**: 大部分误差缩减在前20-30步（2-3个episode）内完成
3. **自适应辅助策略合理**: AI学会对低衰退用户不干预、对中等衰退用户给记忆提示、对高衰退用户给动作提示
4. **干预时机精准**: AI在关键决策点（如转弯前）提供辅助，而非持续干预

## 亮点与洞察

- **"非理性即偏差信念下的理性"**: 将认知科学洞察形式化为可计算的数学框架
- **通用性框架**: $f_\theta$ 可替换为任意记忆模型（衰退、干扰、容量限制等）
- **推断效率实际可行**: $O(N_\theta N_{\tilde{h}} t |\mathcal{S}|)$ 相比精确推理的指数级复杂度，支持实时应用
- **辅助POMDP闭环验证**: 不仅建模和推断，还展示了下游应用价值

## 局限与展望

- 假设AI可访问环境动力学模型（$\mathcal{T}$, $\mathcal{O}$），真实场景中通常未知
- 实验仅在简单T迷宫上验证，复杂连续状态空间有待测试
- 记忆衰退模型（每步独立遗忘）过于简化，真实人类记忆有更复杂的衰退曲线
- 策略需要对每个 $\theta$ 预训练，$\theta$ 空间连续化后计算量可能不可行
- 未与其他用户建模方法（如逆强化学习、Boltzmann理性模型）进行定量对比

## 相关工作与启发

- **vs Kwon et al. 2020**: 维护不完美内部模型但假设完美记忆——本文明确建模记忆有限性
- **vs Jacob et al. 2023**: 用推理预算约束解释非理性——本文聚焦记忆约束，互补
- **vs CRTypist (Shi et al. 2024)**: 特定应用的记忆衰退——本文提供通用框架
- **启发**: 建模"为什么用户做了'错误'选择"比仅预测行为更有价值——理解认知限制是真正的个性化基础

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将有限记忆的计算理性模型与高效在线推断结合成通用框架
- 实验充分度: ⭐⭐⭐ 仅T迷宫一个环境，行为合理性和推断精度验证充分但缺乏大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，动机论证优雅
- 价值: ⭐⭐⭐⭐ 对人机协作和自适应系统有方法论贡献，但实际验证受限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[ICML 2026\] Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](../../ICML2026/others/return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)
- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](../../ACL2025/others/are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[CVPR 2026\] More Than Meets the Eye: A Unified Image Fusion Framework via Semantic-Pixel Entropy Trade-off for Zero-Shot Generalization](../../CVPR2026/others/more_than_meets_the_eye_a_unified_image_fusion_framework_via_semantic-pixel_entr.md)
- [\[AAAI 2026\] OR-R1: Automating Modeling and Solving of Operations Research Optimization Problems](or-r1_automating_modeling_and_solving_of_operations_research_optimization_proble.md)

</div>

<!-- RELATED:END -->
