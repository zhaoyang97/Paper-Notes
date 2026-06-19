---
title: >-
  [论文解读] Interlocking-free Selective Rationalization Through Genetic-based Learning
description: >-
  [ACL 2025][选择性合理化] 本文提出 GenSPP，首个完全消除 interlocking 问题的选择性合理化框架，通过遗传算法对生成器和预测器进行分离优化，在合成数据集和仇恨言论检测任务上显著提升了高亮质量（Hl-F1 提升 6.5%–10.3%），同时保持了可比的分类性能。 领域现状：选择性合理化（select…
tags:
  - "ACL 2025"
  - "选择性合理化"
  - "遗传算法"
  - "interlocking问题"
  - "自解释模型"
  - "高亮提取"
---

# Interlocking-free Selective Rationalization Through Genetic-based Learning

**会议**: ACL 2025  
**arXiv**: [2412.10312](https://arxiv.org/abs/2412.10312)  
**代码**: 有（论文提及开源数据和代码）  
**领域**: 其他  
**关键词**: 选择性合理化、遗传算法、interlocking问题、自解释模型、高亮提取

## 一句话总结

本文提出 GenSPP，首个完全消除 interlocking 问题的选择性合理化框架，通过遗传算法对生成器和预测器进行分离优化，在合成数据集和仇恨言论检测任务上显著提升了高亮质量（Hl-F1 提升 6.5%–10.3%），同时保持了可比的分类性能。

## 研究背景与动机

**领域现状**：选择性合理化（selective rationalization）是可解释 AI 中一种流行的范式，其核心架构是 select-then-predict（SPP）流水线——由一个生成器从输入文本中提取高亮（rationale），再喂给一个预测器进行分类。这类模型天然具有忠实自解释的特性，在事实核查、法律分析等高风险场景中广泛应用。

**现有痛点**：SPP 架构存在一个严重的优化问题叫 interlocking：由于生成器产生的是离散二值掩码，而预测器通过连续梯度更新学习，两者的更新速率不匹配。当生成器被困在次优掩码上时，预测器会在该掩码上过拟合，反过来又强化了生成器维持该选择，形成恶性循环。现有方法（如 Gumbel softmax 采样、权重共享、软合理化引导等）只能缓解而无法消除这一问题，且引入了额外的超参数调优负担。

**核心矛盾**：interlocking 的根源在于 SGD 联合优化中生成器离散化函数（rounding）导致的策略分段常数特性——即使应用平滑技术，生成的掩码可能在多个梯度步骤中保持不变，导致预测器过拟合。

**本文目标**：设计一个完全无 interlocking 的选择性合理化框架，不需要额外启发式、采样技巧或架构修改。

**切入角度**：作者观察到 interlocking 本质上是联合优化问题 $\min_\theta \min_\omega \mathcal{L}$ 中两个模块耦合导致的。如果将双重最小化拆分成分离优化——先独立定义生成器，再从头训练预测器——就能从结构上打破 interlocking。

**核心 idea**：用遗传算法（GA）的全局搜索替代 SGD 联合训练，每个个体代表一个生成器参数配置，对每个个体独立训练一个预测器来评估适应度，从而实现完全分离的优化。

## 方法详解

### 整体框架

GenSPP 沿用经典 SPP 架构（生成器 $g_\theta$ + 预测器 $f_\omega$），但训练方式完全不同。首先初始化一个包含 $I$ 个个体的群体 $\mathcal{P}$，每个个体代表一组生成器参数。每一代中：(1) 对每个个体，从头训练一个预测器最小化分类损失（保持生成器冻结）；(2) 用适应度函数 $h$ 评估每个个体；(3) 通过选择、交叉、变异产生下一代。迭代 $G$ 代后输出最优个体。

### 关键设计

1. **分离训练（Disjoint Training）**:

    - 功能：从结构上消除 interlocking
    - 核心思路：将联合优化问题重新形式化为约束优化 $\min_\theta \Omega(m)$ s.t. $\min_\omega \mathcal{L}(f_\omega(g_\theta(x) \odot x), y) \leq l + \epsilon$，其中 $l$ 是在完整输入上训练的最优预测器的损失。这意味着找到正则化最优的高亮，使得预测器能达到接近在原始输入上的性能。生成器和预测器的依赖关系是单向的：$f_\omega$ 依赖 $g_\theta$，反之则不成立。
    - 设计动机：SGD 联合训练中两个模块的不对等学习速率（连续 vs 离散）是 interlocking 的根因，分离优化从根本上斩断了这种耦合。

2. **非可微适应度函数设计**:

    - 功能：同时评估分类性能和高亮质量，无需权重平衡
    - 核心思路：定义 $\tilde{h} = 1 - \mathcal{L}$（当 $\mathcal{L}_t < l + \epsilon$ 时），其中 $\mathcal{L} = (1 - \Omega(m)) \times (1 - \min(\mathcal{L}_t, 1))$。当分类损失未达到阈值时直接设为 0，这使学习过程先聚焦分类性能，再逐步优化高亮质量。最终适应度 $h = 1/(\tilde{h} + \hat{\epsilon})$。
    - 设计动机：传统加权 $\mathcal{L}_t + \Omega(m)$ 会将 $(0.0, 1.0)$ 和 $(0.5, 0.5)$ 映射为相同代价，但前者分类完美、高亮极差，后者两方面都一般——应该被区别对待。非可微适应度函数解决了这一问题。

3. **遗传搜索策略（GA Operations）**:

    - 功能：实现参数空间的局部和全局搜索
    - 核心思路：使用轮盘赌选择（roulette-wheel）配对个体，one-point 交叉生成新个体，高斯噪声变异实现局部探索，半精英（half-elitism）生存选择保留最优个体。群体大小 $I=50$，进化 $G=100$ 代。
    - 设计动机：GA 的群体搜索天然减少了陷入局部最优的风险，交叉提供全局探索，变异提供局部探索。且 GA 不需要梯度计算，避免了采样带来的方差问题。

### 损失函数 / 训练策略

每个个体评估时，预测器使用分类交叉熵 $\mathcal{L}_{ce}$ 训练 3 个 epoch，学习率 $10^{-2}$。高亮正则化采用稀疏约束 $\mathcal{L}_s$ 和（可选的）连续性约束 $\mathcal{L}_c$。适应度评估在验证集上进行。

## 实验关键数据

### 主实验

在合成 Toy 数据集和真实 HateXplain 仇恨言论数据集上的对比：

| 方法 | Toy Clf-F1 | Toy Hl-F1 | HateXplain Clf-F1 | HateXplain Hl-F1 |
|------|-----------|-----------|-------------------|-----------------|
| FR | 基线 | 基线 | 基线 | 基线 |
| MGR | 多生成器 | 较高方差 | 可比 | 中等 |
| MCD | 因果引导 | 中等 | 可比 | 中等 |
| G-RAT | 注意力引导 | 较高 | 可比 | 较高 |
| **GenSPP** | **可比** | **+10.3%** | **可比** | **+6.5%** |

GenSPP 在高亮质量上显著优于所有竞争方法（Wilcoxon 检验 $p \leq 0.01$），同时分类性能可比。

### 消融实验（Synthetic Skewing 恢复实验）

| 配置 | Toy Hl-F1 | HateXplain Hl-F1 | 说明 |
|------|-----------|-------------------|------|
| GenSPP (G=100) | 高 | 高 | 标准配置 |
| GenSPP (G=150) | 最优 | 最优 | 增加预算，完全恢复 |
| GenSPP_sk (偏斜初始化) | 接近标准 | 接近标准 | 从偏斜状态恢复 |
| 基线模型 (skew) | 高方差 | 不稳定 | 多数 seed 无法恢复 |

### 关键发现

- GenSPP 的方差显著低于所有基线，说明遗传搜索的稳健性。在 Toy 数据集上 MGR 和 G-RAT 表现出显著的不稳定性。
- 在 HateXplain 上，GenSPP 学会了对负例（正常文本）不选择任何高亮，而对正例保留有价值的选择——这种灵活性是基线模型无法实现的。
- GenSPP 是最小的模型（与 FR 同等大小，比其他基线小 2-4 倍），但性能最好。
- 计算代价：GenSPP 单次 seed 运行在 Toy 上约 36 分钟、HateXplain 上约 78 分钟，远高于基线（8/4 分钟），但低方差意味着不需要多次运行。

## 亮点与洞察

- **从优化视角根治 interlocking**：不是修补症状（采样、引导、正则化），而是从优化结构上消除耦合，思路非常干净。这种"换优化器"的思路可以迁移到其他存在离散-连续耦合的双组件系统中。
- **非可微适应度函数设计巧妙**：用阈值机制先保证分类达标、再优化高亮，避免了传统多目标加权中的权重调优问题。
- **合成数据集的设计**：构建了一个可控的字符串匹配任务用于评估合理化框架，每个类别有唯一高亮模式且带有干扰片段，这是社区中首个此类评估基准。

## 局限与展望

- **计算开销**：遗传搜索需要在每一代为每个个体训练一个预测器，计算量比 SGD 方法高约 10-20 倍。作者指出可通过并行评估和更高效的 GA（如 CMA-ES）来缓解。
- **模型规模受限**：实验仅在 GRU 级别的轻量模型上验证，未扩展到 Transformer 等大模型。遗传搜索的参数空间会随模型增大指数增长。
- **数据集规模较小**：Toy 仅 10k 样本、HateXplain 约 20k，未在大规模 NLP 任务上验证。
- **仅关注无监督合理化**：未探索有监督合理化设置下的效果。

## 相关工作与启发

- **vs FR (Liu et al., 2022)**：FR 用 Gumbel softmax + 权重共享来缓解 interlocking，但本质上仍是联合 SGD 优化，interlocking 只是被缓解。GenSPP 的分离训练从结构上消除了问题。
- **vs G-RAT (Hu & Yu, 2024)**：G-RAT 引入额外的注意力引导模块提供软合理化信号。虽然效果不错，但增加了模型复杂度（参数量是 GenSPP 的 4 倍）。GenSPP 不需要任何额外模块。
- **vs Li et al., 2022 (3-stage)**：三阶段方法试图通过迭代冻结来打破 interlocking，但第一阶段仍存在 interlocking。GenSPP 完全避免了联合训练。

## 评分

- 新颖性: ⭐⭐⭐⭐ 用遗传算法替代 SGD 联合训练解决 interlocking，思路独特但遗传算法本身不新
- 实验充分度: ⭐⭐⭐⭐ 包含合成数据集、真实数据集、偏斜恢复实验、多种基线对比，但数据集规模和模型规模受限
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，方法描述详尽，数学形式化完整
- 价值: ⭐⭐⭐ 有理论贡献但实用性受限于计算开销和可扩展性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](../../CVPR2025/others/training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[NeurIPS 2025\] What Does It Take to Build a Performant Selective Classifier?](../../NeurIPS2025/others/what_does_it_take_to_build_a_performant_selective_classifier.md)
- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](../../AAAI2026/others/forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)

</div>

<!-- RELATED:END -->
