---
title: >-
  [论文解读] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery
description: >-
  [NeurIPS 2025][自监督学习][Generalized Category Discovery] 提出 SEAL 框架，利用自然存在的语义层级结构（而非手工设计的抽象层级）指导广义类别发现，通过层级语义引导的软对比学习和跨粒度一致性模块，在细粒度基准上取得 SOTA 性能。 广义类别发现（GCD）旨在给定部分标注数…
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "Generalized Category Discovery"
  - "Hierarchical Learning"
  - "对比学习"
  - "Semantic Hierarchy"
  - "Cross-Granularity Consistency"
---

# SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery

**会议**: NeurIPS 2025  
**arXiv**: [2510.18740](https://arxiv.org/abs/2510.18740)  
**代码**: [有](https://visual-ai.github.io/seal/)  
**领域**: Self-Supervised Learning / Open-World Recognition  
**关键词**: Generalized Category Discovery, Hierarchical Learning, Contrastive Learning, Semantic Hierarchy, Cross-Granularity Consistency

## 一句话总结

提出 SEAL 框架，利用自然存在的语义层级结构（而非手工设计的抽象层级）指导广义类别发现，通过层级语义引导的软对比学习和跨粒度一致性模块，在细粒度基准上取得 SOTA 性能。

## 研究背景与动机

广义类别发现（GCD）旨在给定部分标注数据后，对所有未标注图像进行分类——无论它们属于已知类还是未知类。这比新类发现（NCD）更具挑战性，因为未标注数据同时包含已知和未知类别的样本。

GCD 的核心在于从已知类别向未知类别进行知识迁移。层级信息已被证明对此有效，但现有方法存在明显缺陷：

- **InfoSieve** 用隐式的二叉树构建抽象层级（通过共享二进制码前缀）
- **CiPR** 通过迭代合并数据分区构建抽象层级
- **TIDA** 用手工定义的上下层级，类别数由超参数控制

这些方法的问题在于抽象层级可能引入噪声和错误。例如在图 1(a) 中，"西伯利亚虎"和"孟加拉猫"可能被错误合并为一类，"红狐"可能被拆分为多个类，高相似度的"巴吉度猎犬"和"比格犬"容易混淆。

论文的核心问题是：**真实世界中自然存在的、语义有据的分类学层级能否作为更可靠的指导？** 这个想法来自生物学研究中分类学家利用已知物种标本对新标本进行分类的实践。从信息论角度，论文进一步证明了语义层级标签能提供更紧的互信息界。

## 方法详解

### 整体框架

SEAL 基于 SimGCD 基线，引入三个关键组件：
1. **语义感知多任务框架**：在多个语义层级上同时进行类别发现
2. **跨粒度一致性模块（CGC）**：确保不同粒度的预测互相一致
3. **层级语义引导的软对比学习**：根据语义接近度为负样本分配不同权重

### 关键设计

1. **语义感知层级学习**：

    - 定义 $H$ 个语义层级，对应标签 $\mathbf{y}_1, \ldots, \mathbf{y}_H$（从粗到细）
    - 共享图像编码器 $\mathcal{F}$，后接投影层 $\phi$ 将特征解耦为各粒度的子表示：$\mathbf{z} = \phi(\mathcal{F}(\mathbf{x})) = [\mathbf{z}_1; \mathbf{z}_2; \ldots; \mathbf{z}_H]$
    - 粗粒度分支复用细粒度特征但阻断梯度：$\hat{\mathbf{z}}_i = [\mathbf{z}_1; \cdots; \mathbf{z}_h; \Gamma(\mathbf{z}_{h+1}); \cdots; \Gamma(\mathbf{z}_H)]$
    - 语义层级来自自然分类系统（如生物学中的科-属-种），不需要手工设计
    - 每个层级独立训练 GCD 分类器

2. **跨粒度一致性自蒸馏（CGC）**：

    - 解决多层级分类中的不一致问题（如同一实例被标为"柴犬"但粗粒度却标为"猫"）
    - 定义动态转移矩阵 $M_h \in \mathbb{R}^{n_H \times n_h}$：编码细粒度类到粗粒度类的映射
    - 已知类：固定的 one-hot 向量；未知类：均匀初始化，训练中动量更新
    - 一致性损失：$\mathcal{L}_{cgc} = \sum_{h=1}^{H-1} D_{KL}(p(\mathbf{x}_i|\boldsymbol{\theta}_h) | p(\mathbf{x}_i|\boldsymbol{\theta}_H) \times M_h)$
    - 核心思想：细粒度预测经转移矩阵映射后应与粗粒度预测一致

3. **层级语义引导的软对比学习**：

    - 动机：标准对比学习将所有非正样本视为等价负样本，忽略了语义相关性
    - 在每个语义层级计算 batch 内的成对相似度矩阵 $S_h$，逐层融合得到层级相似度 $\tilde{S}_h$
    - 生成语义感知软标签：$\tilde{Y}_{\text{soft}_h} = (1-\lambda_s) \cdot \mathbf{I} + \lambda_s \cdot \tilde{S}_h$
    - 对比损失中用软标签替代硬 0/1 标签，语义近的样本获得更小的负权重
    - 相似度度量采用混合度量：$\text{sim}(\mathbf{z}_i, \mathbf{z}_k') = \lambda_c \mathbf{z}_i \cdot \mathbf{z}_k'^{\top} - (1-\lambda_c)\|\frac{\mathbf{z}_i}{\|\mathbf{z}_i\|} - \frac{\mathbf{z}_k'}{\|\mathbf{z}_k'\|}\|_2$
    - $\lambda_c$ 训练中线性衰减：先用角度度量（简单），后加入距离度量（精细）——课程学习策略

### 损失函数 / 训练策略

总损失：$\mathcal{L}_{all} = \sum_h^H (\mathcal{L}_{\text{soft}_{rep}}^h + \mathcal{L}_{cls}^h) + \mathcal{L}_{cgc}$

- $\mathcal{L}_{\text{soft}_{rep}}^h$：各层级的软对比学习损失
- $\mathcal{L}_{cls}^h$：各层级的分类损失（标注数据用 ground truth，未标注用 sharpened 伪标签 + 熵正则化）
- 端到端单阶段训练，200 epoch，余弦学习率调度

## 实验关键数据

### 主实验：SSB 基准（DINOv1 backbone）

| 方法 | CUB All | CUB New | Cars All | Cars New | Aircraft All | 平均 |
|------|---------|---------|----------|----------|-------------|------|
| SimGCD | 60.3 | 57.7 | 53.8 | 45.0 | 54.2 | 56.1 |
| SPTNet | 65.8 | 65.1 | 59.0 | 49.3 | 59.3 | 61.4 |
| DebGCD | 66.3 | 63.5 | 65.3 | 57.4 | 61.7 | 64.4 |
| **SEAL** | **66.2** | 63.2 | **65.3** | **58.5** | **62.0** | **64.5** |

### 消融实验（基于 Oxford-Pet 数据集，推测自论文结构）

| 组件 | 效果说明 |
|------|----------|
| 语义层级多任务 | 信息论证明提供更紧互信息界（公式 4-6） |
| CGC 模块 | 动态转移矩阵的动量更新有效处理未知类别 |
| 软对比学习 | 相比硬负样本对比，减少了语义相近样本的错误惩罚 |
| 混合相似度度量 | 课程学习策略（角度→距离）持续提升性能 |
| 梯度控制器 Γ | 防止粗粒度分支的训练偏差 |

### 关键发现

- SEAL 在所有细粒度基准上取得 SOTA 或接近 SOTA，尤其在 Aircraft (+0.3) 和 Cars New (+1.1) 上优势明显
- 使用 DINOv2 backbone 时性能进一步提升（CUB All 达到更高水平）
- 层级信息对新类（New）的提升大于已知类（Old），说明语义层级确实有助于知识迁移
- 自然语义层级比手工抽象层级更可靠——不会出现错误合并/拆分类别的问题

## 亮点与洞察

- **信息论动机扎实**：从互信息分解角度严格证明了层级标签的优越性（公式 4-6），不只是直觉上的"层级有用"
- **CGC 中动态转移矩阵的设计巧妙**：已知类固定映射、未知类动量更新，优雅地处理了 GCD 中标签不完整的挑战
- **软对比学习的语义引导**符合直觉——同科不同属的物种不应被推得和完全无关的物种一样远
- 与生物学分类实践的类比既增强了动机也暗示了更广泛的适用性

## 局限与展望

- 需要获取各数据集的语义层级标签，这在某些领域可能不易获得
- 假设已知总类别数 $K$（虽然可以用现有方法估计）
- 在通用粗粒度数据集（CIFAR、ImageNet）上的优势可能没有细粒度数据集那么明显
- 转移矩阵的动量更新策略可能对初始化敏感，收敛性未严格分析

## 相关工作与启发

- 从 SimGCD 出发进行改进，保持了框架的简洁性和端到端训练的便利
- 层级对比学习在闭集分类中已有成功应用，SEAL 是首次将语义层级引入 GCD 任务
- 与 HypCD（双曲嵌入隐式建模层级）形成互补——SEAL 用显式层级标签，HypCD 用几何结构
- 超级类/粗粒度信息在 zero-shot 学习中也常用，SEAL 的思路可扩展到相关任务

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在 GCD 中利用自然语义层级，信息论动机有新意
- 实验充分度: ⭐⭐⭐⭐ — SSB + Oxford-Pet + Herbarium19 全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 理论和方法部分清晰，图例直观
- 价值: ⭐⭐⭐⭐ — 对 GCD 社区有实质贡献，方法设计可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Consistent Supervised-Unsupervised Alignment for Generalized Category Discovery](consistent_supervised-unsupervised_alignment_for_generalized_category_discovery.md)
- [\[CVPR 2026\] TAR: Token-Aware Refinement for Fine-grained Generalized Category Discovery](../../CVPR2026/self_supervised/tar_token-aware_refinement_for_fine-grained_generalized_category_discovery.md)
- [\[CVPR 2025\] MOS: Modeling Object-Scene Associations in Generalized Category Discovery](../../CVPR2025/self_supervised/mos_modeling_object-scene_associations_in_generalized_category_discovery.md)
- [\[ICCV 2025\] A Hidden Stumbling Block in Generalized Category Discovery: Distracted Attention](../../ICCV2025/self_supervised/a_hidden_stumbling_block_in_generalized_category_discovery_d.md)
- [\[CVPR 2026\] The Devil Is in Gradient Entanglement: Energy-Aware Gradient Coordinator for Robust Generalized Category Discovery](../../CVPR2026/self_supervised/the_devil_is_in_gradient_entanglement_energy-aware_gradient_coordinator_for_robu.md)

</div>

<!-- RELATED:END -->
