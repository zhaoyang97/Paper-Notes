---
title: >-
  [论文解读] Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)
description: >-
  [ICML 2025][域增量学习] DCE 提出频率感知专家组 + 动态专家选择器的双阶段训练框架，同时解决域增量学习中域内类别不平衡和跨域类别分布偏移两个难题，在四个 benchmark 上达到 SOTA。 领域现状：域增量学习（DIL）要求模型在数据分布不断变化的非平稳环境中持续学习，同时保留历史知识…
tags:
  - "ICML 2025"
  - "域增量学习"
  - "类别不平衡"
  - "专家混合"
  - "持续学习"
  - "预训练模型"
---

# Addressing Imbalanced Domain-Incremental Learning through Dual-Balance Collaborative Experts (DCE)

**会议**: ICML 2025  
**arXiv**: [2507.07100](https://arxiv.org/abs/2507.07100)  
**代码**: [https://github.com/Lain810/DCE](https://github.com/Lain810/DCE)  
**领域**: LLM评测  
**关键词**: 域增量学习, 类别不平衡, 专家混合, 持续学习, 预训练模型

## 一句话总结

DCE 提出频率感知专家组 + 动态专家选择器的双阶段训练框架，同时解决域增量学习中域内类别不平衡和跨域类别分布偏移两个难题，在四个 benchmark 上达到 SOTA。

## 研究背景与动机

**领域现状**：域增量学习（DIL）要求模型在数据分布不断变化的非平稳环境中持续学习，同时保留历史知识。当前基于预训练模型（PTM）的 DIL 方法主要采用两种范式：共享 prompt（如 L2P、CODA-Prompt）和域特定 prompt（如 S-iPrompt），通过冻结 PTM 骨干参数并引入轻量级适配模块来缓解灾难性遗忘。

**现有痛点**：现实数据中类别不平衡是普遍现象（如自动驾驶中极端天气下的交通标志样本远少于正常场景），但现有 DIL 方法几乎没有考虑这一问题。类别不平衡在 DIL 中表现为两个维度：(a) **域内类别不平衡**——单个域内不同类别的样本数量差异巨大，导致模型过拟合多样本类、欠学少样本类；(b) **跨域类别分布偏移**——不同域中同一类别的样本频率不同，导致类别的"多/少"身份随域变化。

**核心矛盾**：共享 prompt 方法通过共享特征空间促进知识迁移，能改善少样本类的性能，但会引发多样本类的灾难性遗忘；域特定 prompt 方法通过参数隔离减少遗忘，但阻断了跨域知识共享，少样本类无法从新域数据中受益。这形成了一个**知识共享 vs. 遗忘抵抗**的根本矛盾。

**本文目标**：(a) 如何在单个域的训练过程中平衡不同频率类别的学习？(b) 如何在跨域场景下同时保持多样本类的知识并利用新域数据提升少样本类的泛化？

**切入角度**：作者观察到不同损失函数天然偏向不同频率的类别——CE loss 偏向多样本类，balanced softmax loss 追求平衡，inverse frequency loss 偏向少样本类。通过让多个专家分别使用不同损失函数，可以覆盖所有频率组的学习需求。同时，通过高斯采样生成平衡的伪特征来训练专家选择器，可以在不依赖真实数据的情况下实现跨域知识平衡融合。

**核心 idea**：用不同损失函数训练频率感知专家组解决域内不平衡，再用基于历史类统计的平衡高斯采样训练动态专家选择器解决跨域分布偏移。

## 方法详解

### 整体框架

DCE 采用双阶段训练范式。每到达一个新域时：

- **第一阶段（频率感知专家训练）**：在冻结的预训练 ViT 编码器之后部署三个并行的专家模块，每个专家使用不同的损失函数独立训练，分别擅长多样本类、平衡类和少样本类的特征学习。Visual Prompt 在第一个任务时学习，之后冻结。
- **第二阶段（动态专家选择器训练）**：用冻结的特征编码器计算每个类的均值和协方差统计量，通过**均衡高斯采样**生成合成伪特征数据集，用这些平衡数据训练一个 MLP 选择器，为每个专家分配软权重。

推理时，输入经过编码器提取特征后，所有专家并行计算输出，由选择器加权融合得到最终预测。

### 关键设计

1. **频率感知专家组（Frequency-Aware Expert Group）**：

    - 功能：三个并行的 MLP+分类器模块，分别偏向不同频率组的类别学习。
    - 核心思路：第一个专家 $e_b$ 使用标准 CE loss $\ell_{CE}$，天然偏向多样本类；第二个专家 $e_{b+1}$ 使用 balanced softmax loss $\ell_{Bal} = -\log \frac{\exp(v_y^2 + \log p_b^y)}{\sum_j \exp(v_j^2 + \log p_b^j)}$，通过引入类频率先验校正偏差，实现平衡预测；第三个专家 $e_{b+2}$ 使用 inverse distribution loss $\ell_{Rev} = -\log \frac{\exp(v_y^3 + 2\log p_b^y)}{\sum_j \exp(v_j^3 + 2\log p_b^j)}$，反转训练分布以强调少样本类。三个损失独立优化，总损失为 $\ell_{exp} = \ell_{CE} + \ell_{Bal} + \ell_{Rev}$。
    - 设计动机：单一损失函数无法兼顾所有频率组的类别。CE 和 inverse 损失提供互补的极端偏向，balanced 损失居中，三者协同覆盖完整的频率谱。这种设计将类别不平衡问题"分解"给不同专家，每个专家只需在自己偏好的频率组上做好就行。

2. **动态专家选择器（Dynamic Expert Selector）**：

    - 功能：一个 MLP 网络 $s(\cdot) : \mathbb{R}^d \to \mathbb{R}^{3b}$，为当前积累的所有 $3b$ 个专家分配重要性权重。
    - 核心思路：(a) 在每个域训练完专家后，用冻结编码器计算每个类的特征均值 $\mu_b^c$ 和协方差 $\Sigma_b^c$，存入全局统计库 $G$；(b) 对 $G$ 中的每个 (域, 类) 对均匀采样 $K$ 个伪特征：$\tilde{D} = \bigcup_{i=1}^{b}\bigcup_{c=1}^{|\mathcal{Y}|} \{(\tilde{x}, c) \sim \mathcal{N}(\mu_i^c, \Sigma_i^c)\}_{k=1}^K$，关键在于 $K$ 对所有域-类对保持一致，确保平衡采样；(c) 用合成数据训练选择器：$\mathcal{L}_{Select} = \frac{1}{|\hat{D}|}\sum_{(\tilde{x},y)\in\hat{D}} \ell_{CE}(\sum_{i=1}^{3b} s(\tilde{x})_i \cdot e_i(\tilde{x}), y)$。
    - 设计动机：S-iPrompt 等硬分配方法将测试样本分配给单个域专家，无法利用跨域知识。动态选择器通过软加权实现自适应的跨域专家融合，等值采样确保少样本类和多样本类获得同等的训练信号，避免选择器继承数据不平衡的偏差。

3. **协方差估计的 OAS 收缩（Oracle Approximating Shrinkage）**：

    - 功能：对少样本类使用 OAS 收缩机制估计更稳定的协方差矩阵。
    - 核心思路：当类别样本数 $n \geq 10$ 时，计算 $\hat{\Sigma} = (1-\rho)\hat{\Sigma}_{emp} + \rho \cdot \frac{\text{tr}(\hat{\Sigma}_{emp})}{d} \cdot I_d$，收缩系数 $\rho$ 由 OAS 公式自动确定。同时在每个域内对类特定协方差取平均，降低存储开销。
    - 设计动机：少样本类的样本量不足以可靠估计高维协方差矩阵，OAS 通过引入正则化先验提升估计的稳定性。

### 损失函数/训练策略

- **第一阶段**：各专家独立优化 $\ell_{exp} = \ell_{CE} + \ell_{Bal} + \ell_{Rev}$，Visual Prompt 仅在第一个域时通过 VPT 优化，之后冻结
- **第二阶段**：选择器通过 $\mathcal{L}_{Select}$ 在合成特征上优化
- 优化器：SGD，batch size 128，学习率 0.001，cosine 衰减
- 第一阶段训练 20/30 epochs（数据集而定），第二阶段训练 10 epochs

## 实验关键数据

### 主实验

在四个 benchmark 上使用 ViT-B/16-IN1K，5 种随机域序列取平均：

| 数据集 | 指标 | DCE | S-iPrompt | CODA-Prompt | L2P | 提升(vs 次优) |
|--------|------|-----|-----------|-------------|-----|-------------|
| Office-Home | $\bar{\mathcal{A}}$ | **84.6** | 81.4 | 82.4 | 78.7 | +1.2 |
| Office-Home | $\mathcal{A}_B$ | **84.4** | 80.8 | 83.3 | 80.5 | +1.1 |
| Office-Home | $\mathcal{A}_{few}$ | **79.4** | 66.0 | 73.2 | 73.7 | +5.7 |
| DomainNet | $\bar{\mathcal{A}}$ | **64.3** | 59.0 | 47.6 | 48.5 | +5.3 |
| DomainNet | $\mathcal{A}_B$ | **63.5** | 57.9 | 45.1 | 45.2 | +5.6 |
| DomainNet | $\mathcal{A}_{few}$ | **50.8** | 31.5 | 38.2 | 37.3 | +12.6 |
| CORe50 | $\bar{\mathcal{A}}$ | **80.1** | 62.7 | 72.8 | 72.3 | +7.3 |
| CDDB-Hard | $\bar{\mathcal{A}}$ | **74.6** | 64.2 | 67.9 | 67.3 | +6.7 |

少样本类提升尤其显著：DomainNet 上少样本类准确率从次优的 38.2% 提升到 50.8%，提升 12.6 个百分点。

### 消融实验

| 配置 | $\bar{\mathcal{A}}$ (Office-Home) | 说明 |
|------|-----------------------------------|------|
| 1 expert ($\ell_{CE}$) | ~80 | 仅多样本偏向，少样本差 |
| 2 experts ($\ell_{CE} + \ell_{Bal}$) | ~82 | 中等改善 |
| 3 experts (DCE完整) | **84.6** | 最优 |
| 4 experts (+额外损失) | ~84.7 | 增益边际递减 |

### 关键发现

- **三个专家是最优选择**：从1→2→3专家持续提升，4专家增益微乎其微。三专家在效果和效率间取得最佳平衡。
- **少样本类受益最大**：在所有配置中，$\ell_{Rev}$ 专家对少样本类的贡献最显著，去掉后少样本准确率大幅下降。
- **Class Performance Drift (CPD) 分析**：DCE 在多样本类上的遗忘低于共享 prompt 方法，在少样本类上的改善优于域特定 prompt 方法，整体 CPD 仅次于不训练的 SimpleCIL。
- **计算效率**：DCE 在第一个任务后只更新专家参数（不经过编码器反向传播），且推理只需一次前向传播，训练和推理时间均优于 L2P 和 DualPrompt。
- **baseline 加 balanced loss 仍不如 DCE**：将 baseline 方法换成 balanced softmax loss 后部分方法反而降性能，DCE 仍保持明显优势。

## 亮点与洞察

- **损失函数即专家分工**：不同损失函数天然对不同频率类别有偏好，将这一特性显式利用来构建多专家系统，思路简洁优雅。这个 idea 可以推广到任何需要处理类别不平衡的增量学习场景。
- **伪特征合成解耦了数据不平衡**：通过高斯采样在特征空间合成平衡数据训练选择器，巧妙绕过了真实数据不平衡的限制。类似思路可用于任何需要平衡训练信号但受限于真实数据分布的场景。
- **分析框架有价值**：提出的 Class Performance Drift (CPD) 指标比传统遗忘度量更适合不平衡 DIL 场景，因为少样本类的准确率可能随训练提升而非下降。
- **共享 vs. 域特定的第三条路**：DCE 不是简单取折中，而是通过训练阶段分离（第一阶段域特定训练 + 第二阶段跨域融合）实现了"两全其美"。

## 局限与展望

- **仅限视觉领域**：实验全部在图像分类上，未验证在 NLP 或多模态场景下的有效性
- **高斯假设可能不成立**：对 PTM 特征分布建模为单峰高斯是基于经验观察，对于复杂分布的类别可能不够准确
- **专家数量固定为三**：三个专家对应三种频率组（多/中/少），但缺乏理论指导——不同不平衡程度是否需要不同数量的专家？
- **存储开销随域数增长**：每个域需要存储三个专家和类统计信息，长序列域增量场景下存储成本不可忽视
- **缺少与 MoE-Adapter 等更新方法的对比**：虽然提到了 Yu et al. 2024 的 MoE Adapter 框架，但主实验中未直接比较

## 相关工作与启发

- **vs L2P / DualPrompt / CODA-Prompt**（共享prompt方法）：它们通过共享 prompt pool 实现知识迁移，但在不平衡场景下多样本类遗忘严重。DCE 的域特定专家训练避免了特征空间纠缠。
- **vs S-iPrompt**（域特定prompt方法）：S-iPrompt 通过 KNN 硬分配隔离域知识，遗忘少但跨域迁移差。DCE 的动态选择器实现了软分配，既保持域隔离又允许知识共享。
- **vs RIDE / TADE**（长尾学习中的多专家方法）：它们在静态长尾数据上使用多专家，DCE 将这一思路扩展到增量学习的动态场景，增加了跨域知识平衡融合的挑战。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究不平衡 DIL，双平衡设计有新意，但各组件（多专家、高斯采样）单独看不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 benchmark、5 种域序列、多种 backbone、详细消融和分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰（Figure 2 的共享vs域特定对比很直观），问题定义严谨
- 价值: ⭐⭐⭐⭐ 填补了 DIL+不平衡的空白，框架设计实用，但适用范围限于视觉分类

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] C4D: 4D Made from 3D through Dual Correspondences](../../ICCV2025/others/c4d_4d_made_from_3d_through_dual_correspondences.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICML 2025\] Access Controls Will Solve the Dual-Use Dilemma](access_controls_will_solve_the_dual-use_dilemma.md)
- [\[ECCV 2024\] Foster Adaptivity and Balance in Learning with Noisy Labels](../../ECCV2024/others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)
- [\[ACL 2025\] Interlocking-free Selective Rationalization Through Genetic-based Learning](../../ACL2025/others/interlocking-free_selective_rationalization_through_genetic-based_learning.md)

</div>

<!-- RELATED:END -->
