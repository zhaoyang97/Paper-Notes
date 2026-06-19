---
title: >-
  [论文解读] CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries
description: >-
  [ICML 2025][自监督学习][偏好强化学习] 提出 CLARIFY 方法，通过对比学习构建融合偏好信息的轨迹嵌入空间，利用拒绝采样选择更清晰可区分的偏好查询，从而提升离线 PbRL 在非理想反馈下的标注效率和策略性能。 偏好强化学习（PbRL）通过人类对轨迹片段对的偏好比较来推断奖励函数，避免了显式奖励工程的复杂性…
tags:
  - "ICML 2025"
  - "自监督学习"
  - "偏好强化学习"
  - "对比学习"
  - "模糊查询"
  - "轨迹嵌入"
  - "离线RL"
---

# CLARIFY: Contrastive Preference Reinforcement Learning for Untangling Ambiguous Queries

**会议**: ICML 2025  
**arXiv**: [2506.00388](https://arxiv.org/abs/2506.00388)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: 偏好强化学习, 对比学习, 模糊查询, 轨迹嵌入, 离线RL

## 一句话总结

提出 CLARIFY 方法，通过对比学习构建融合偏好信息的轨迹嵌入空间，利用拒绝采样选择更清晰可区分的偏好查询，从而提升离线 PbRL 在非理想反馈下的标注效率和策略性能。

## 研究背景与动机

偏好强化学习（PbRL）通过人类对轨迹片段对的偏好比较来推断奖励函数，避免了显式奖励工程的复杂性。然而，当两个轨迹片段高度相似时，人类难以给出明确的偏好判断，导致**模糊查询（ambiguous queries）**问题。这一问题不仅降低了标注效率，还限制了 PbRL 在实际场景中的应用。

现有方法的核心矛盾在于：大多数 PbRL 方法（如 PEBBLE、PT、OPRL 等）要么忽略了模糊查询的存在，要么仅在在线设置中解决该问题（如 Mu et al., 2024），无法直接迁移到离线场景。离线 PbRL 中数据固定、无法与环境交互，如何在有限的偏好预算内最大化选择"清晰可区分"的查询对，成为关键瓶颈。

本文的切入角度是：利用对比学习将偏好信息编码到轨迹嵌入空间中，使得"清晰可区分"的片段在嵌入空间中距离远，"模糊"的片段距离近。基于这一嵌入空间，通过拒绝采样策略选择更多无歧义的查询，从而提升标注效率。核心 idea：**用对比学习建模偏好结构，在嵌入空间中通过距离区分查询的清晰度，再用拒绝采样筛选高质量查询。**

## 方法详解

### 整体框架

CLARIFY 分为两个阶段：
1. **表示学习阶段**：使用对比学习训练轨迹编码器 $z = f_\phi(\tau)$，将轨迹映射到固定维度的嵌入空间，同时融合偏好信息（清晰/模糊标签）。
2. **查询选择阶段**：基于学到的嵌入空间，通过拒绝采样选择嵌入距离较大（即更清晰可区分）的查询对，送给人类标注。

具体流程为：先随机采样一批查询预训练编码器和奖励模型 → 基于嵌入空间选择新查询 → 更新偏好数据集和奖励模型 → 重新训练嵌入 → 最终用 IQL 等离线 RL 算法训练策略。

### 关键设计

1. **Ambiguity Loss $\mathcal{L}_{\text{amb}}$**：核心思路是最大化清晰可区分片段对的嵌入距离，同时最小化模糊片段对的嵌入距离。对于偏好数据集中标记为 $p \in \{0, 1\}$ 的"清晰"查询，拉远两个片段的嵌入；对于 $p = \text{no\_cop}$ 的"模糊"查询，拉近两个片段的嵌入。设计动机是直接实现"清晰远、模糊近"的嵌入空间目标。但仅用此损失会导致过拟合和表示坍塌（模糊片段映射到同一点）。

2. **Quadrilateral Loss $\mathcal{L}_{\text{quad}}$**：为解决 $\mathcal{L}_{\text{amb}}$ 单独使用的问题，引入四边形损失建模偏好关系。对于两组清晰查询 $(\sigma_+, \sigma_-)$ 和 $(\sigma_+', \sigma_-')$，鼓励"好"片段之间 $(z_+, z_+')$ 和"差"片段之间 $(z_-, z_-')$ 的距离小于跨组距离 $(z_+, z_-')$ 等。关键公式为最小化：$-\mathbb{E}[\ell(z^+, z^{-\prime}) + \ell(z^{+\prime}, z^-) - \ell(z^+, z^{+\prime}) - \ell(z^-, z^{-\prime})]$。通过配对使用查询，训练数据从 $O(n)$ 扩展到 $O(n^2)$，缓解小样本过拟合问题，同时作为正则化防止表示坍塌。

3. **拒绝采样查询选择**：计算查询对的嵌入距离 $d_{\text{emb}}$，估计清晰和模糊查询的密度函数 $\rho_{\text{clr}}$ 和 $\rho_{\text{amb}}$，构造加权密度 $\rho(d) = 0.5(\rho_1 + \rho_2)$，其中 $\rho_1$ 基于差值、$\rho_2$ 基于比值。最终的采样分布 $q(d) = p(d) \cdot \rho(d)$ 提高了选择清晰查询的概率。设计动机是不仅选择距离最大的查询（那样会降低多样性），而是在保持多样性的同时增大清晰查询的比例。

### 损失函数 / 训练策略

总损失为四项加权和：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{amb}}\mathcal{L}_{\text{amb}} + \lambda_{\text{quad}}\mathcal{L}_{\text{quad}} + \lambda_{\text{norm}}\mathcal{L}_{\text{norm}}$$

其中 $\mathcal{L}_{\text{recon}}$ 是基于 Bi-directional Decision Transformer 的重建损失，$\mathcal{L}_{\text{norm}}$ 约束嵌入的 L2 范数接近 1 以稳定训练。嵌入距离的连续分布被离散化为 $n_{\text{bin}}$ 个区间以处理拒绝采样。

## 实验关键数据

### 主实验

在 Metaworld 和 DMControl 总共 9 个任务上，比较 CLARIFY 与 MR、OPRL、PT、OPPO、LiRE 等基线方法：

| 任务 | 指标 | CLARIFY | 之前 SOTA (OPRL/LiRE) | 提升 |
|------|------|---------|----------------------|------|
| dial-turn (ε=0.5) | 成功率 | 77.50 ± 7.37 | 57.33 ± 25.02 (OPRL) | +20.17 |
| drawer-open (ε=0.5) | 成功率 | 83.50 ± 7.40 | 72.67 ± 2.87 (OPRL) | +10.83 |
| handle-pull-side (ε=0.5) | 成功率 | 95.00 ± 1.22 | 89.75 ± 6.07 (PT) | +5.25 |
| walker-walk (ε=0.5) | 回报 | 796.34 ± 12.87 | 789.18 ± 28.77 (LiRE) | +7.16 |
| cheetah-run (ε=0.5) | 回报 | 617.31 ± 14.43 | 553.61 ± 43.16 (LiRE) | +63.70 |
| dial-turn (ε=0.7) | 成功率 | 79.40 ± 3.83 | 63.40 ± 9.46 (OPRL) | +16.00 |
| walker-walk (ε=0.7) | 回报 | 816.54 ± 11.08 | 795.02 ± 22.80 (LiRE) | +21.52 |

### 消融实验

| 配置 | dial-turn | sweep-into | 说明 |
|------|-----------|------------|------|
| 无 $\mathcal{L}_{\text{amb}}$，无 $\mathcal{L}_{\text{quad}}$ | 63.20 ± 4.79 | 40.00 ± 11.29 | 等同 OPRL |
| 有 $\mathcal{L}_{\text{amb}}$，无 $\mathcal{L}_{\text{quad}}$ | 69.00 ± 11.20 | 52.80 ± 17.01 | 不稳定，易过拟合 |
| 无 $\mathcal{L}_{\text{amb}}$，有 $\mathcal{L}_{\text{quad}}$ | 71.25 ± 8.81 | 62.20 ± 4.92 | 收敛较慢 |
| 两者均有（CLARIFY） | **77.50 ± 3.01** | **68.00 ± 3.19** | 最佳且最稳定 |

### 关键发现

- **查询清晰度**：CLARIFY 在 skip rate ε=0.5 下，dial-turn 的清晰查询比例达 76.33%，远高于 MR (46.95%)、OPRL (31.67%) 和 PT (43.90%)。
- **人类实验验证**：在 walker-walk 真人标注实验中，CLARIFY 回报 420.75 vs OPRL 265.91，查询清晰度 63.33% vs 53.33%，标注准确率 87.08% vs 66.67%。
- **查询效率**：即使仅有 100 个查询，CLARIFY 也显著优于 MR (dial-turn: 59.50 vs 49.50)。
- 密度直接选择（选最清晰的查询）反而效果差，因为缺乏多样性；拒绝采样的方式兼顾了清晰度和多样性。

## 亮点与洞察

- 将 PbRL 中长期被忽视的"模糊查询"问题形式化，并提供了系统解决方案
- 四边形损失的设计非常巧妙：利用查询配对将样本量从 O(n) 扩展到 O(n²)，同时建模偏好的全局结构
- 理论保证扎实（Proposition 5.1 的 margin 分离和 Proposition 5.2 的凸可分性）
- 嵌入空间的 t-SNE 可视化直观展示了方法的有效性
- 真人实验与模拟实验结果一致，增强了方法的可信度

## 局限与展望

- 目前仅在离线 PbRL 中验证，在线场景的扩展尚未探讨
- 嵌入训练依赖 BDT 架构，对不同任务的适应性有待考察
- 拒绝采样中密度估计的离散化引入了额外超参数 $n_{\text{bin}}$
- 人类实验规模较小（每轮仅 20 或 100 个反馈），大规模人类反馈场景下的表现未知
- 可考虑将此方法应用于 LLM 对齐（RLHF）中的偏好数据筛选

## 相关工作与启发

- 与 Mu et al. (2024) 最相关，但后者仅解决在线设置的模糊查询问题
- 对比学习在 RL 中的应用（图像表示学习、时间距离学习）提供了方法论基础
- LiRE (Choi et al., 2024) 通过列表级比较增强反馈，是互补的方向
- 对 RLHF 的启发：LLM 对齐中类似的"模糊偏好"问题（两个回复质量相近时人类难以判断）可能受益于类似方法

## 评分

- 新颖性: ⭐⭐⭐⭐ — 四边形损失和拒绝采样查询选择的组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ — 9 个任务、模拟+真人实验、充分的消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论分析和实验安排合理
- 价值: ⭐⭐⭐⭐ — 解决了 PbRL 的实际痛点，有 RLHF 应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Collapse-Proof Non-Contrastive Self-Supervised Learning](collapse-proof_non-contrastive_self-supervised_learning.md)
- [\[CVPR 2026\] VideoSSR: Video Self-Supervised Reinforcement Learning](../../CVPR2026/self_supervised/videossr_video_self-supervised_reinforcement_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](generalization_analysis_for_supervised_contrastive_representation_learning_under.md)

</div>

<!-- RELATED:END -->
