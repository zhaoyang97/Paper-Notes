---
title: >-
  [论文解读] Anytime Continual Learning for Open Vocabulary Classification
description: >-
  [ECCV2024][模型压缩][continual learning] 提出 AnytimeCL 框架，通过部分微调 CLIP 最后一个 transformer block 并动态加权融合微调模型与原始模型的预测，实现任意时刻接收样本、任意标签集推理的开放词汇持续学习。 - 传统持续学习在离散标签空间上操作…
tags:
  - "ECCV2024"
  - "模型压缩"
  - "continual learning"
  - "open vocabulary"
  - "CLIP"
  - "online learning"
  - "feature compression"
---

# Anytime Continual Learning for Open Vocabulary Classification

**会议**: ECCV2024  
**arXiv**: [2409.08518](https://arxiv.org/abs/2409.08518)  
**代码**: [GitHub](https://github.com/jessemelpolio/AnytimeCL)  
**领域**: 模型压缩  
**关键词**: continual learning, open vocabulary, CLIP, online learning, feature compression  
**作者**: Zhen Zhu, Yiming Gong, Derek Hoiem (UIUC)

## 一句话总结

提出 AnytimeCL 框架，通过部分微调 CLIP 最后一个 transformer block 并动态加权融合微调模型与原始模型的预测，实现任意时刻接收样本、任意标签集推理的开放词汇持续学习。

## 背景与动机

- 传统持续学习在离散标签空间上操作，新增标签/任务会改变问题定义，增加学习难度
- 开放词汇设定下，分类是连续特征与标签嵌入的比较，学习仅是在已有问题上的改进，更适合持续改进
- 即使 CLIP 在互联网规模数据上训练，在很多具体任务上性能仍不理想
- 先前方法 Zhu et al. 使用线性分类器 + 层次聚类，需存储全部训练样本，且 AIM 加权策略仅考虑样本是否来自已见类别，早期阶段表现不佳
- **核心需求**：系统能在任意时刻高效接收新样本更新，并能在任意标签集上进行推理

## 核心问题

1. 如何在仅收到部分类别训练数据时，仍能对完整标签集进行准确预测？
2. 如何以毫秒级速度高效整合每个新训练样本？
3. 如何在持续学习过程中不丢失开放词汇能力，避免灾难性遗忘？
4. 如何压缩中间层特征以降低存储和计算开销？

## 方法详解

### 整体架构

系统包含两个模型：**冻结的原始 CLIP 模型**和**可微调模型**（仅微调最后一个 transformer block）。两个模型对同一图像分别产生预测概率 $P_o(y|x)$ 和 $P_t(y|x)$，最终预测通过在线逐类加权融合：

$$P(y|x) = \alpha_o(y) P_t(y|x) + \alpha_t(y) P_o(y|x)$$

### 部分微调策略

- 仅微调 CLIP ViT 的最后一个 transformer block（decoder），保持文本标签嵌入固定
- 微调部分以原始权重初始化
- 保持标签嵌入固定有助于特征与文本模态保持相关性，减少对已见标签的过拟合
- 收到新样本时，将其与存储样本组成类别平衡的 mini-batch（大小 B=32），执行单步更新

### 类别平衡采样

对于 batch size B 和已见标签集 $\mathcal{Y}_t$，均匀选择 $\min(B-1, |\mathcal{Y}_t|)$ 个类别，再从每个类别中均匀采样等量实例。实验表明类别平衡采样和均匀采样效果相近，均优于 FIFO 等复杂策略。

### "Other" 正则化损失

引入"以上都不是"选项来处理真实标签不在候选集中的情况：

$$\mathcal{L}(x,y,\mathcal{Y}) = \mathcal{L}_{ce}(x, y, \mathcal{Y} \cup \text{other}) + \beta \mathcal{L}_{ce}(x, \text{other}, (\mathcal{Y} \cup \text{other}) \setminus y)$$

其中 "other" 仅用一个可学习的 bias 项建模，$\beta=0.1$。该正则化稳定了开放词汇训练过程。

### Online Class-wise Weighting (OCW)

核心创新之一。使用指数移动平均（EMA，衰减 $\eta=0.99$）在线估计每个标签上两个模型的准确率：

$$c_t(y) = \eta \hat{c}_t(y) + (1-\eta) \mathbb{1}[y_t(x)=y]$$

关键设计：在模型更新之前，用当前样本更新准确率估计，避免了训练样本对微调模型准确率的有偏估计。两个模型的权重按其估计准确率比例分配：

$$\alpha_t(y) = \frac{c_t(y)}{c_t(y) + c_o(y) + \epsilon}$$

对于微调模型未见过的标签，$\alpha_t(y)=0$，完全使用原始模型预测。

### 注意力加权 PCA 特征压缩

- 部分微调需存储中间层特征（50 个 768 维 token），每样本约 153KB
- 全局 PCA 或 VQ-VAE 压缩效果差（网络已学到高效表示）
- **逐图像 PCA**：对每张图像的 token 计算 PCA 向量，仅存储 5 个 PCA 向量及系数
- **CLS 注意力加权**：用 CLS token 与各 patch token 的注意力作为 PCA 权重
- **量化**：进一步将浮点值量化为 8/16 位无符号整数
- 最终实现 **30 倍压缩**（153KB → 5KB），精度损失 < 1%

## 实验关键数据

**设置**：8 个目标任务（CIFAR100, SUN397, FGVCAircraft 等）+ 3 个新颖任务（ImageNet, UCF101, DTD），共 226,080 训练样本，1,034 类。

| 指标 | 场景 | AnytimeCL vs. 前 SOTA |
|------|------|----------------------|
| 所有阶段 | Data Incremental | 每阶段均优于 CLIP+LinProbe (AIM) |
| 所有阶段 | Class Incremental | 每阶段均优于前方法，早期优势尤其明显 |
| 所有阶段 | Task Incremental | 持续优于前方法，在线/离线均有提升 |
| Transfer | MTIL Task Inc. | 69.4（零遗忘，与 CLIP 持平） |
| Avg. | MTIL Task Inc. | 77.0（+11.7 vs. CLIP） |

**特征压缩对比（CIFAR100）**：

| 方法 | 存储/样本 | 时间/batch | 准确率 |
|------|----------|-----------|--------|
| 完整图像 | 150.5 KB | 43.9 ms | 77.8% |
| 完整特征 | 153.6 KB | 25.6 ms | 77.8% |
| Per-instance PCA + CLS权重 + 量化 | 5.3 KB | 13.9 ms | 77.5% |

**扩展性**：用 DINOv2 替换微调模型后，在后期阶段表现更陡的提升曲线，且保持零样本性能。

## 亮点

1. **真正的 "anytime"**：毫秒级整合新样本，支持 data/class/task incremental 三种场景
2. **OCW 加权策略精巧**：利用更新前样本无偏估计模型准确率，比 AIM 在早期阶段优势显著
3. **固定标签嵌入 + 部分微调**：简单有效地保持开放词汇能力，零遗忘
4. **30 倍特征压缩**：逐图像注意力加权 PCA 是新颖且实用的思路，兼顾存储、速度和隐私
5. **系统设计完整**：从训练、推理到存储全链路考虑，支持联邦学习等扩展场景

## 局限与展望

- 仅验证了分类任务，未扩展到检测、分割等更复杂任务
- 树聚类的可扩展性实验规模有限，缺少百万级样本的验证
- 特征压缩的隐私保护程度未做定量分析（是否可从压缩特征反推原图）
- 仅使用 ViT-B/32 backbone，未验证更大模型（ViT-L 等）的表现
- 单步更新的学习率需根据离线训练超参缩放，对新任务类型的泛化性需进一步验证

## 与相关工作的对比

| 方法 | 开放词汇 | 在线更新 | 特征压缩 | 动态加权 |
|------|---------|---------|---------|---------|
| WiSE-FT / ZSCL | ✓（但泛化逐步退化） | ✗ | ✗ | 固定权重混合 |
| Zhu et al. (AIM) | ✓ | ✓ | ✗ | AIM（仅考虑是否已见类） |
| CLS-ER | ✗ | ✓ | ✗ | EMA 权重平均 |
| **AnytimeCL** | **✓（零遗忘）** | **✓（毫秒级）** | **✓（30x）** | **OCW（逐类动态）** |

核心优势在于 OCW 在早期训练阶段的鲁棒性——当仅有少量类别有样本时，AIM 会错误地将样本路由到微调模型，而 OCW 根据实际准确率分配权重。

## 启发与关联

- OCW 动态加权思路可推广到任意多专家模型的融合，不限于两个模型
- 逐图像 PCA 压缩思路对联邦学习中特征通信有直接应用价值
- "other" 正则化损失对开放集识别任务有借鉴意义
- 部分微调 + 固定标签嵌入的范式可应用于 open-vocabulary detection/segmentation
- 与 complementary learning systems 理论的对应关系值得在更大规模上验证

## 评分

- 新颖性: ⭐⭐⭐⭐ — OCW 和注意力加权 PCA 压缩是好的创新点，整体方法组合得当
- 实验充分度: ⭐⭐⭐⭐ — 三种持续学习场景 + 灵活推理 + 丰富消融实验，但缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法描述完整，图表信息丰富
- 价值: ⭐⭐⭐⭐ — 提出了实用的开放词汇持续学习框架，离实际部署仍有距离但方向正确

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery](category_adaptation_meets_projected_distillation_in_generalized_continual_catego.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[CVPR 2026\] Back to Source: Open-Set Continual Test-Time Adaptation via Domain Compensation](../../CVPR2026/model_compression/back_to_source_open-set_continual_test-time_adaptation_via_domain_compensation.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](../../ICML2026/model_compression/energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)

</div>

<!-- RELATED:END -->
