---
title: >-
  [论文解读] MAGR: Manifold-Aligned Graph Regularization for Continual Action Quality Assessment
description: >-
  [ECCV 2024][LLM安全][持续学习] 提出 MAGR 方法，通过流形对齐投影器和 Intra-Inter-Joint 图正则化器，解决持续动作质量评估（CAQA）中特征回放导致的旧特征与当前特征流形不对齐问题，在四个数据集上显著超越现有基线。 领域现状： 动作质量评估（AQA）用于定量评价运动、康复等场景中的动作…
tags:
  - "ECCV 2024"
  - "LLM安全"
  - "持续学习"
  - "动作质量评估"
  - "特征回放"
  - "流形对齐"
  - "图正则化"
---

# MAGR: Manifold-Aligned Graph Regularization for Continual Action Quality Assessment

**会议**: ECCV 2024  
**arXiv**: [2403.04398](https://arxiv.org/abs/2403.04398)  
**代码**: [GitHub](https://github.com/ZhouKanglei/MAGR_CAQA)  
**领域**: LLM安全  
**关键词**: 持续学习, 动作质量评估, 特征回放, 流形对齐, 图正则化

## 一句话总结

提出 MAGR 方法，通过流形对齐投影器和 Intra-Inter-Joint 图正则化器，解决持续动作质量评估（CAQA）中特征回放导致的旧特征与当前特征流形不对齐问题，在四个数据集上显著超越现有基线。

## 研究背景与动机

**领域现状**: 动作质量评估（AQA）用于定量评价运动、康复等场景中的动作表现，但现有方法在静态小规模数据集上训练，无法适应技能随时间的动态变化。

**现有痛点**: 传统 AQA 模型需要在全量数据上重新训练才能更新；持续学习（CL）可以解决非平稳性问题，但面临灾难性遗忘。CL 领域主要研究离散分类任务，而 AQA 涉及连续的质量分数回归，带来独特挑战。

**核心矛盾**: 特征回放方法可以保护隐私（不存储原始视频），但更新 backbone 后，旧特征与当前特征流形之间产生严重失配（misalignment），导致灾难性遗忘加剧；固定 backbone 则丧失适应能力。

**本文目标**: 在不访问原始数据的前提下，如何在更新 backbone 时修正旧特征的偏移，并保证特征分布与质量分数分布的一致性。

**切入角度**: 两步走策略——先将旧特征投影到当前流形，再通过图正则化从局部和全局视角对齐特征空间与质量空间。

**核心 idea**: 通过学习 session 间的流形偏移来修正旧特征，并用角度距离矩阵分块正则化来确保特征空间与质量分数空间的一致性。

## 方法详解

### 整体框架

MAGR 框架包含两个核心模块：Manifold Projector (MP) 和 Intra-Inter-Joint Graph Regularizer (IIJ-GR)。在 session $t$ 的训练中：
- 一个分支用更新后的 encoder $f$ 学习新数据
- 另一个分支从记忆库 $\mathcal{M}^{t-1}$ 中取出旧特征，经 MP 修正后进行回放
- IIJ-GR 同时对新旧特征施加正则化

总体目标函数为：

$$\min_{\Theta} \mathcal{L}_D + \mathcal{L}_M + \lambda_P \mathcal{L}_P + \lambda_R \mathcal{L}_R$$

其中 $\mathcal{L}_D$ 是新数据的回归损失，$\mathcal{L}_M$ 是记忆回放损失，$\mathcal{L}_P$ 是投影学习损失，$\mathcal{L}_R$ 是图正则化损失。

### 关键设计

1. **Manifold Projector (MP)**: 学习从旧流形到当前流形的映射。

    - **投影学习（Projector Learning）**: 在 session $t$ 开始时冻结上一个 session 的 encoder $f'$，利用当前 session 数据的初始特征 $\bar{\boldsymbol{h}}_j^t = f'(\mathbf{x}_j^t)$ 和更新后特征 $\boldsymbol{h}_j^t = f(\mathbf{x}_j^t)$ 的差异来学习流形偏移。投影器采用带残差连接的 MLP：
    $\hat{\boldsymbol{h}}_j^t = \bar{\boldsymbol{h}}_j^t + p(\bar{\boldsymbol{h}}_j^t)$
    - **特征投影（Feature Projection）**: 对记忆库中每个旧特征，用学到的投影器进行修正：
    $\tilde{\boldsymbol{h}}_i^s = \tilde{\boldsymbol{h}}_i^s + p(\tilde{\boldsymbol{h}}_i^s)$
    - **学习损失**: $\mathcal{L}_P = \frac{1}{|\mathcal{D}^t|}\sum_j \|\boldsymbol{h}_j^t - \hat{\boldsymbol{h}}_j^t\|_2^2$
    - **设计动机**: 不需要访问旧的原始数据，仅用当前 session 数据就能估计流形偏移；残差连接使学习更稳定（消融证实去除残差链接 $\rho_{avg}$ 下降 7%）。

2. **Intra-Inter-Joint Graph Regularizer (IIJ-GR)**: 对齐特征分布与质量分数分布。

    - **角距离矩阵**：归一化特征后用角度距离替代欧氏距离，满足测地距特性：
    $\mathbf{A} = \arccos(\tilde{\mathbf{H}}\tilde{\mathbf{H}}^\top), \quad \tilde{\mathbf{H}} = \mathbf{H}/\|\mathbf{H}\|$
    - **距离矩阵分块（DMP）**: 将距离矩阵分成 4 个子矩阵 $\mathbf{A}_{11}, \mathbf{A}_{12}, \mathbf{A}_{21}, \mathbf{A}_{22}$，分别对应旧-旧、旧-新、新-旧、新-新之间的关系。
    - **图正则化**: 用质量分数距离矩阵 $\mathbf{S}$ 作为监督，通过 KL 散度约束：
    $\mathcal{L}_R = L(\mathbf{A}, \mathbf{S}) + \sum_{i=1}^{2}\sum_{j=1}^{2} L(\mathbf{A}_{ij}, \mathbf{S}_{ij})$
   其中 $L(\mathbf{P}, \mathbf{Q}) = \frac{1}{N}\sum_{i=1}^{N} \text{KL}(\sigma(\mathbf{P}_i), \sigma(\mathbf{Q}_i))$
    - **设计动机**: 欧氏距离不满足质量分数的测地性质；KL 散度比 MSE 更松弛、更适合相关性指标（消融证实用 MSE 则 $\rho_{avg}$ 下降 6%）。

3. **Ordered Uniform Sampling (OUS)**: 用于在每个 session 结束时选择代表性特征存入记忆库。先按质量分数排序再均匀采样，确保覆盖完整的分数范围，相比随机采样 $\rho_{avg}$ 高 4%。

### 损失函数 / 训练策略

- 总损失：$\mathcal{L}_D + \mathcal{L}_M + \lambda_P \mathcal{L}_P + \lambda_R \mathcal{L}_R$，其中 $\lambda_P = \lambda_R = 1$
- 优化器：Adam，学习率和权重衰减均为 $10^{-4}$
- 每个 session 最多训练 50 轮，batch size 为 5，mini-batch 为 3
- Backbone: I3D（预训练权重），BatchNorm 冻结
- MP 模块使用两层 MLP
- 每个 session 存储 10 个代表性样本

## 实验关键数据

### 主实验

| 数据集 | 指标 ($\rho_{avg}$↑) | MAGR | 最强基线 | 提升 |
|--------|------|------|------|------|
| MTL-AQA | $\rho_{avg}$ | 0.8979 | 0.8720 (MER) | +2.59% |
| FineDiving | $\rho_{avg}$ | 0.8580 | 0.8309 (GEM) | +2.71% |
| UNLV-Dive | $\rho_{avg}$ | 0.7668 | 0.7397 (MER) | +2.71% |
| JDM-MSA | $\rho_{avg}$ | 0.7166 | 0.6689 (MER) | +4.77% |

Joint Training 上界分别为 0.9360 / 0.9075 / 0.8460 / 0.7556。

### 消融实验

| 配置 | $\rho_{avg}$ | 说明 |
|------|------|------|
| MAGR (完整) | 0.8979 | 基准 |
| w/o MP | 0.6949 (↓23%) | MP 是最关键组件 |
| w/o MP 残差连接 | 0.8391 (↓7%) | 残差连接提升稳定性 |
| w/o IIJ-GR | 0.7362 (↓18%) | 完整正则化不可或缺 |
| w/o II-GR (仅去除局部) | 0.8463 (↓6%) | 局部正则化有独立贡献 |
| w/o J-GR (仅去除全局) | 0.7839 (↓13%) | 全局正则化贡献更大 |
| w/o KL (用 MSE) | 0.8447 (↓6%) | KL 优于 MSE |
| w/o OUS (随机采样) | 0.8619 (↓4%) | OUS 采样策略有效 |

### 关键发现

- 特征偏移程度越大，MAGR 的优势越明显：UNLV-Dive 偏移 MSE 为 51.75，相关性增益高达 15.64%
- 在标签稀缺和噪声条件下，MAGR 表现出更强的鲁棒性
- t-SNE 可视化显示 MAGR 能在不同 session 间保持特征的有序分布

## 亮点与洞察

- **问题定义新颖**: 首次提出 CAQA（持续动作质量评估）任务，将 CL 从分类拓展到连续回归
- **流形投影巧妙**: 仅用当前 session 数据即可学习流形偏移，无需旧数据，兼顾隐私和适应性
- **角距离替代欧氏距离**: 基于测地线性质的洞察，使特征距离更好地反映质量分数关系
- **全面的 benchmark 建设**: 提出 grade-incremental 设置和定制评估指标，为后续 CAQA 研究奠定基础

## 局限与展望

- 记忆库仍需额外存储开销，极端低存储场景可能受限
- OUS 在非常小的样本量（如每 session 仅 3 个）时效果下降
- 仅在 GCN-based AQA 模型上验证，未探索其他 AQA 架构
- 未考虑跨领域的 AQA 场景（如从跳水迁移到体操）

## 相关工作与启发

- **经验回放** (MER, DER++) 有效但有隐私问题；特征回放克服隐私问题但面临流形偏移
- **NC-FSCIL** 通过固定 backbone 避免偏移但牺牲适应性
- **SLCA** 的生成式回放在 AQA 场景质量不稳定
- 启发：对于涉及隐私的连续回归任务，流形对齐是比简单特征存储更关键的设计

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义 CAQA 任务，流形对齐思路创新
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集 + 消融 + 鲁棒性 + 可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，motivation 阐述充分
- 价值: ⭐⭐⭐⭐ 为持续学习在回归任务的应用开辟新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](../../ACL2025/llm_safety/real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[ICML 2025\] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](../../ICML2025/llm_safety/federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)

</div>

<!-- RELATED:END -->
