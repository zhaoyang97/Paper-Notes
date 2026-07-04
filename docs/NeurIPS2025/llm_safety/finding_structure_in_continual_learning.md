---
title: >-
  [论文解读] Finding Structure in Continual Learning
description: >-
  [NeurIPS 2025][LLM安全][持续学习] 提出基于Douglas-Rachford Splitting (DRS)的持续学习优化框架，将稳定性与可塑性解耦为两个独立的近端子问题，并结合Rényi散度替代KL散度实现更鲁棒的先验对齐，从而在无需回放缓冲区或额外模块的条件下有效缓解灾难性遗忘。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "持续学习"
  - "灾难性遗忘"
  - "Douglas-Rachford Splitting"
  - "Rényi散度"
  - "贝叶斯推断"
  - "近端算子"
---

# Finding Structure in Continual Learning

**会议**: NeurIPS 2025  
**arXiv**: [2602.04555](https://arxiv.org/abs/2602.04555)  
**代码**: 待确认  
**领域**: LLM安全  
**关键词**: 持续学习, 灾难性遗忘, Douglas-Rachford Splitting, Rényi散度, 贝叶斯推断, 近端算子  

## 一句话总结
提出基于Douglas-Rachford Splitting (DRS)的持续学习优化框架，将稳定性与可塑性解耦为两个独立的近端子问题，并结合Rényi散度替代KL散度实现更鲁棒的先验对齐，从而在无需回放缓冲区或额外模块的条件下有效缓解灾难性遗忘。

## 背景与动机
持续学习（Continual Learning）要求模型在顺序学习多个任务时，既能习得新知识（可塑性/plasticity），又能保留旧知识（稳定性/stability），这一矛盾被称为"稳定性-可塑性困境"。现有方法主要有三大流派：

1. **回放方法**（Replay）：存储过去数据进行重放，但内存开销随任务数线性增长
2. **架构扩展方法**：为每个任务添加新模块，导致模型不可持续增长
3. **正则化方法**：在损失函数中添加正则项惩罚重要参数的变化（如EWC用Fisher信息矩阵），但本质上仍是将两个目标耦合为单一损失 $\mathcal{L}_{CL} = L_{\text{new}} + R_{\text{reg}}$

作者指出，上述方法的根本问题不在于目标函数本身，而在于**优化策略**——用标准SGD优化耦合目标会导致梯度冲突，使稳定性和可塑性陷入零和博弈。解决之道不是平衡冲突，而是**改变交互方式**。

## 核心问题
如何从优化层面将持续学习中的稳定性和可塑性目标解耦，避免梯度干扰，实现两者的协同而非对抗？

## 方法详解

### 整体框架
模型采用变分自编码器（VAE）架构：编码器 $\phi$ 将输入映射到潜空间后验分布 $q_\phi(z|x)$，解码器 $\theta$ 从潜变量预测输出。关键设计是**后验到先验传播**：任务 $t-1$ 学完后，其后验分布成为任务 $t$ 的先验。

训练目标为：

$$\mathcal{L}(\phi, \theta) = \underbrace{\mathbb{E}_{z \sim q_\phi}\left[\sum_{n=1}^{N} \log p_\theta(y_n | x_n, z)\right]}_{f:\ \text{可塑性}} - \underbrace{\lambda \sum_{i=1}^{d} w_i D_\alpha(q_\phi^i \| p^i)}_{g:\ \text{稳定性}}$$

其中 $w_i = (\sigma_p^i)^2 / \sum_j (\sigma_p^j)^2$ 为自适应权重，先验方差大的维度约束更松（允许更多可塑性），方差小的维度约束更紧（保护已学特征）。

### DRS优化流程（核心贡献）
将上述目标中的 $f$（可塑性）和 $g$（稳定性）分别交给各自的近端算子处理。算法维护辅助变量 $u$，每轮迭代包含三步：

**Step 1 — 可塑性近端步（Task-Fitting）**：
$$x_i = \text{prox}_f(u_{i-1}) = \arg\min_{\phi,\theta}\left[f(\phi,\theta) + \frac{1}{2\gamma}\|(\phi,\theta) - u_{i-1}\|^2\right]$$
用Adam做若干步梯度下降近似求解，同时更新编码器和解码器。

**Step 2 — 稳定性反射步（Prior-Alignment）**：
$$y_i = \text{prox}_g(2x_i - u_{i-1})$$
仅更新编码器 $\phi$，将后验对齐到先验。解码器参数直接从Step 1传递，保留其在新任务上的专门化。

**Step 3 — 松弛更新**：
$$u_i = u_{i-1} + \lambda_r(y_i - x_i)$$
在可塑性和稳定性之间做插值，驱动辅助变量向两者的共识点移动。

### Rényi散度替代KL散度
作者通过Proposition 3.1证明了KL散度在DRS框架中的缺陷：当可塑性步骤提出的参数远离先验支撑时，KL的"零回避"（zero-forcing）性质会使稳定性步骤完全忽略可塑性提议，导致学习停滞。而Rényi散度的"零避免"（zero-avoiding）性质保证惩罚始终有限，允许在先验和新提议之间做有意义的插值。

### 收敛性保证
Proposition 3.2证明：(i) DRS的不动点对应持续学习目标的稳定点，满足 $0 \in \nabla f(\omega^*) + \partial g(\omega^*)$；(ii) 迭代过程中可塑性和稳定性步骤的差异趋于零 $\lim_{k\to\infty}\|x_k - y_k\| = 0$，即两个目标最终达成协调。

## 实验关键数据
在6个基准上与14种方法对比（ResNet-18骨干）：

| 指标 | 设定 | 本文方法 | 最佳对比方法 |
|------|------|---------|------------|
| 平均准确率 | 不相交任务(4个) | **65.7%** | SB-MCL 64.9% |
| 平均准确率 | 联合任务(2个) | **88.2%** | SPG/SB-MCL 87.5% |
| 后向迁移BWT | 不相交任务 | **-1.9** | TAG -1.2 (遗忘更少但准确率低很多) |
| 后向迁移BWT | 联合任务 | **+3.2** | UCL/UPGD +2.0 |
| 前向迁移FWT | 不相交任务 | **+7.9** | SB-MCL +7.1 |
| 前向迁移FWT | 联合任务 | **+10.4** | POCL +9.1 |

- 在CASIA-100的100个顺序任务上，遗忘率始终低于4%，而KL方法超过13%
- 消融实验中最佳 $\alpha=2.0$，此时准确率约77%；$\alpha=0$（无稳定项）时降至约72%
- 去掉随机采样（用确定性潜变量）训练时间减少9%，但准确率从79.1%降至76.3%
- 计算开销与基线方法相当或更快

## 亮点
1. **视角新颖**：将持续学习重新定义为优化问题而非目标函数设计问题，用算子分裂方法DRS将稳定性和可塑性从"拔河"变成"协商"
2. **无需回放**：完全不需要存储历史数据，靠后验传播实现知识保留，内存效率高
3. **理论扎实**：提供了收敛性证明和Rényi散度优于KL的理论论证，不是纯经验性方法
4. **简洁高效**：不新增架构模块，不需要外部记忆，计算开销与标准方法相当
5. **前向迁移显著**：不仅减少遗忘，还能用旧知识加速新任务学习（FWT +10.4），实现真正的"协同"

## 局限与展望
1. **高斯假设的局限**：所有分布限制为高斯族以获取Rényi散度闭式解，对更复杂的多模态后验可能不够灵活
2. **近端算子的近似**：$\text{prox}_f$ 用梯度下降近似，实际中近似精度对最终性能的影响未充分讨论
3. **超参数 $\gamma, \lambda, \alpha$ 的敏感性**：虽然做了 $\alpha$ 的消融，但 $\gamma$ 和 $\lambda$ 的交互影响未详细分析
4. **实验场景有限**：主要在图像分类上验证，缺少NLP、强化学习等更多样的持续学习场景
5. **与大模型的结合**：在预训练模型（如CLIP、ViT-Large）上的效果未验证，实际应用中持续学习常基于大模型微调
6. **任务边界假设**：仍需明确的任务边界来触发先验更新，对无监督/在线持续学习场景不直接适用

## 与相关工作的对比

| 方法 | 类别 | 是否需要回放 | 优化方式 | 散度类型 | 特点 |
|------|------|:---:|------|------|------|
| EWC | 正则化 | 否 | SGD | - | Fisher信息矩阵加权正则 |
| VCL/UCL | 贝叶斯 | 是(可选) | SGD | KL | 后验传播+KL约束 |
| SB-MCL | 贝叶斯 | 是 | SGD | KL | 集合方法+联合训练 |
| UPGD | 梯度修正 | 否 | 修正梯度 | - | 直接修改梯度方向 |
| POCL | 近端 | 是 | 近端点 | KL | 对组合损失用单一近端 |
| **本文** | **算子分裂** | **否** | **DRS** | **Rényi** | **双近端解耦+Rényi** |

与同样使用近端方法的POCL相比，本文的关键区别在于：POCL对组合损失（任务+回放）施加单一近端算子，仍需回放数据；本文将目标**分裂**为两个独立近端子问题，完全无需回放。

## 启发与关联
1. **优化视角的启示**：很多ML问题中的"目标冲突"（如多任务学习、公平性约束）都可以尝试用算子分裂方法解耦，而非简单加权求和
2. **Rényi散度的更广泛应用**：在变分推断、生成模型中用Rényi替代KL可能也有类似的鲁棒性优势
3. **与联邦学习的联系**：联邦学习中的本地更新-全局聚合也存在类似的可塑性-稳定性矛盾，DRS框架值得引入
4. **DRS在深度学习优化中的潜力**：目前DRS主要用于凸优化和信号处理，本文展示了其在非凸深度学习中的可行性，打开了一扇新大门

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 从优化算法层面解决持续学习问题，视角独特
- 实验充分度: ⭐⭐⭐⭐ — 6个数据集+14个对比方法+消融实验充分，但场景集中在图像分类
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、理论推导严谨，图例直观
- 价值: ⭐⭐⭐⭐ — 提供了新范式，但实际落地需验证在大模型场景下的可扩展性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](../../ICML2025/llm_safety/improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)
- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](../../ICML2025/llm_safety/unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)

</div>

<!-- RELATED:END -->
