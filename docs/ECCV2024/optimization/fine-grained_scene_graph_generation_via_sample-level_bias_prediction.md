---
title: >-
  [论文解读] Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction
description: >-
  [ECCV2024][优化/理论][Scene Graph Generation] 提出样本级偏置预测方法 SBP，通过 Bias-Oriented GAN 利用物体对 union region 的上下文信息预测样本特异性纠偏向量，将粗粒度关系修正为细粒度关系，在 VG/GQA/VG-1800 上相比数据集级纠偏方法平均提升 5.6%/3.9%/3.2% 的 Average@K。
tags:
  - "ECCV2024"
  - "优化/理论"
  - "Scene Graph Generation"
  - "Long-Tailed Distribution"
  - "Bias Correction"
  - "GAN"
  - "Fine-Grained Relationships"
---

# Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction

**会议**: ECCV2024  
**arXiv**: [2407.19259](https://arxiv.org/abs/2407.19259)  
**代码**: [Zhuzi24/SBG](https://github.com/Zhuzi24/SBG)  
**领域**: 图学习  
**关键词**: Scene Graph Generation, Long-Tailed Distribution, Bias Correction, GAN, Fine-Grained Relationships

## 一句话总结

提出样本级偏置预测方法 SBP，通过 Bias-Oriented GAN 利用物体对 union region 的上下文信息预测样本特异性纠偏向量，将粗粒度关系修正为细粒度关系，在 VG/GQA/VG-1800 上相比数据集级纠偏方法平均提升 5.6%/3.9%/3.2% 的 Average@K。

## 背景与动机

场景图生成（Scene Graph Generation, SGG）从图像中提取物体及其关系，构建结构化语义图 $\mathcal{G}=\{(o_i, r_{i\to j}, o_j)\}$，广泛支持视觉问答、图像检索、图像描述等下游任务。然而 SGG 面临严重的**长尾分布问题**：以 Visual Genome（VG）数据集为例，粗粒度头部类别（如 "on"、"has"、"wearing"）样本极其丰富，而细粒度尾部类别（如 "walking on"、"part of"、"flying in"）样本稀缺，导致模型预测严重偏向头部的粗粒度关系，生成的场景图缺乏信息量。

现有纠偏方法可归为**数据集级纠偏**：DLFE 估计每类标签频率 $\mathbf{c}$，将有偏概率除以 $\mathbf{c}$ 恢复无偏概率；RTPB 学习抗性偏置 $\mathbf{b}$，从分类 logit 中减去以增强尾部检测。二者的共同局限在于：对所有物体对统一应用同一个全局纠偏向量，忽略了不同物体对的上下文差异。例如 ⟨man, beach⟩ 和 ⟨cat, table⟩ 的 union region 包含截然不同的视觉语义，应支持差异化的纠偏策略。

本文核心洞察：每个物体对的 union region 蕴含丰富且专属的上下文信息，可据此预测**样本特异性的纠偏偏置**，实现从数据集级到样本级的纠偏粒度升级。

## 核心问题

如何为每个物体对预测其专属的纠偏偏置向量 $\mathbf{b}_s$，使模型能针对性地将粗粒度关系（如 "on"）修正为细粒度关系（如 "walking on"），同时避免过度抑制头部类别、实现 R@K 与 mR@K 的最佳平衡？

## 方法详解

整体框架 SBG（Sample-Level Bias Prediction for Fine-Grained SGG）遵循标准两阶段 SGG 流程，核心创新在于提出 SBP 模块实现样本级纠偏。分为三个阶段：

### 第一阶段：经典 SGG 模型训练

采用 Faster R-CNN（ResNeXt-101-FPN backbone）检测物体，再用 Motif/VCtree/Transformer 等经典模型预测关系。此阶段获得有偏的关系预测向量 $\mathbf{z}=[z_1, z_2, \ldots, z_M]$，其中 $M$ 为关系类别数。由于长尾效应，$\mathbf{z}$ 中头部类别得分偏高。

### 第二阶段：构建纠偏偏置集合 $\mathcal{S}$

理论动机：理想情况下，无偏预测 $\mathbf{z}_u$ 与有偏预测 $\mathbf{z}$ 之间存在差值 $\mathbf{b}_u = \mathbf{z}_u - \mathbf{z}$，但 $\mathbf{z}_u$ 无法获得。退一步，只要找到 $\hat{\mathbf{z}}$ 满足 $\text{argmax}(\text{softmax}(\hat{\mathbf{z}})) = \text{argmax}(\text{softmax}(\mathbf{z}_u))$（即预测正确即可，不必完全无偏），就可以通过 $\hat{\mathbf{z}} = \mathbf{z} + \mathbf{b}_s$ 实现纠偏。

具体构建流程：

1. 利用物体对的 union region 特征 $f_{uni}$（通过与 Faster R-CNN 共享权重的特征提取器获取视觉特征，再融合空间特征）
2. 通过单层 Transformer 编码器 $\phi$ 将高维特征映射到一维，与全局偏置融合：$\mathbf{b}^{tru} = \phi(f_{uni}) + \mathbf{b}^{glo}$
3. 全局偏置 $\mathbf{b}^{glo} = -\log(w^a / \sum_{j \in M} w_j^a + \epsilon)$，由数据集统计的关系类别权重计算
4. 验证修正后预测 $r_{pre} = \text{argmax}(\text{softmax}(\mathbf{z} + \mathbf{b}^{tru}))$ 是否等于 $r_{tru}$
5. 若满足直接加入 $\mathcal{S}$；否则计算差值 $d = \hat{\mathbf{z}}[r_{tru}] - \hat{\mathbf{z}}[r_{pre}]$，更新 $\mathbf{b}^{tru}[r_{pre}] = \mathbf{b}^{tru}[r_{pre}] + d + \varepsilon$ 后加入

### 第三阶段：Bias-Oriented GAN（BGAN）训练

冻结第一阶段的经典 SGG 模型参数 $\Lambda^O$，训练 BGAN 学习预测纠偏偏置。

**生成器 G**（5 层一维卷积）：

- 输入：union 特征 $f_{uni}$、全局偏置 $\mathbf{b}^{glo}$、原始预测 $\mathbf{z}$
- 输出：预测偏置 $\mathbf{b}^{pre} = \Upsilon(\Lambda^{B_G}; f_{uni}, \mathbf{b}^{glo}, \mathbf{z})$
- 损失：$\mathcal{L}_G = -\text{mean}(\mathcal{T}_G) + \alpha \cdot \mathcal{L}_{CE}(\hat{\mathbf{z}})$，GAN 损失引导拟合真实偏置分布，交叉熵损失约束修正结果正确

**判别器 D**（3 层一维卷积）：

- 输入：预测偏置 $\mathbf{b}^{pre}$ 或真实偏置 $\mathbf{b}^{tru}$
- 输出判别分数 $\mathcal{T}_G$ 和 $\mathcal{T}_S$
- 损失：$\mathcal{L}_D = -\text{mean}(\mathcal{T}_S) + \text{mean}(\mathcal{T}_G)$

训练策略：每次迭代判别器更新 5 次、生成器更新 1 次。优化器为 RMSProp，学习率分别为 0.0001（G）和 0.0005（D）。推理时仅使用经典 SGG 模型 + 生成器 G，修正公式为：

$$\hat{\mathbf{z}} = \mathbf{z} + \mathbf{b}_s$$

## 实验关键数据

### VG 数据集主要结果（Average@K，三种 backbone）

| Backbone | PredCls A@50/100 | SGCls A@50/100 | SGDet A@50/100 |
|---|---|---|---|
| Motif+SBG | **43.8 / 45.9** | **26.2 / 27.1** | **20.4 / 23.7** |
| VCtree+SBG | **44.0 / 45.9** | **31.3 / 32.5** | 19.4 / 22.4 |
| Transformer+SBG | **44.6 / 46.7** | **27.1 / 28.0** | 20.1 / 23.3 |

三种 backbone 上均达到或接近最优 A@K，且性能表现一致性强——其他方法往往仅在某个 backbone 上有竞争力。

### 样本级 vs 数据集级纠偏的详细对比（Motif 模型，VG，R@100/mR@100/A@100）

| 方法 | R@100 变化 | mR@100 变化 | A@100 变化 |
|---|---|---|---|
| DLFE | −12.9 | +9.4 | −1.8 |
| RTPB | −24.6 | +18.1 | −3.3 |
| **SBG** | **−9.9** | **+15.1** | **+2.6** |

关键发现：DLFE 和 RTPB 虽然提升了 mR@K，但 R@K 下降严重导致 A@K 反而降低；SBG 是唯一能使 A@K 正增长的方法，实现了头部与尾部类别的最佳平衡。PredCls/SGCls/SGDet 三任务平均提升 **5.6%/3.9%/3.2%**。

### 效率对比（Motif PredCls）

| 对比项 | vs DLFE | vs RTPB |
|---|---|---|
| A@100 提升 | +10.6% | +14.5% |
| 训练时间 | −45.0%（12.6h vs 22.9h） | +4.1% |
| 推理速度增加 | +1.2% | +2.2% |
| 参数量增加 | +0.5% | +0.3% |

### 跨数据集泛化（Motif PredCls）

| 数据集 | vs CFA 的 A@50 提升 | vs CFA 的 A@100 提升 | 特点 |
|---|---|---|---|
| GQA | +0.2% | +0.1% | 长尾效应较弱 |
| VG | +2.7% | +1.8% | 中等长尾 |
| VG-1800 | F-Acc Top-10 +10.48% | - | 1800 类关系，重度长尾 |

数据量越大、长尾效应越严重，SBG 优势越明显。

### 消融实验（Transformer PredCls）

union region vs 整张图像：使用 union region 特征的 A@50/100 为 44.6/46.7，优于整张图像的 40.3/42.0，表明全图信息引入了噪声干扰。

全局偏置 $\mathbf{b}^{glo}$ 的作用：仅用 $f_{uni}$ 时 A@50/100 为 42.6/44.8，加入 $\mathbf{b}^{glo}$ 后提升至 44.6/46.7；但单独使用 $\mathbf{b}^{glo}$（不用 $f_{uni}$）效果极差（42.1/43.5），证明样本特异性的 union 特征是核心。

### 可扩展性验证

**单阶段 SGG 方法**（VG SGDet）：ISG 上 A@50/100 从 25.2/29.5 提升至 26.7/30.9；SGTR 上从 18.3/21.8 提升至 20.2/23.6；SS R-CNN 上从 21.1/24.4 提升至 23.7/27.3。

**目标检测**（COCO）：Faster R-CNN + SBP 的 mAP 从 36.4 提升至 37.6（+1.2%），尾部类别 mAP 从 41.6 提升至 44.4（+2.8%）。

## 亮点

- **纠偏粒度的本质升级**：首次将样本级偏置修正引入 SGG，从"对所有样本用同一偏置"到"为每个样本预测专属偏置"，清晰的思路演进
- **GAN 用于偏置预测而非数据生成**：利用对抗训练的强拟合能力逼近真实纠偏向量，比全连接网络等更强
- **R@K 与 mR@K 的最佳平衡**：是唯一使 Average@K 实现正增长的纠偏方法，避免了以牺牲头部为代价提升尾部的常见问题
- **即插即用的通用框架**：适配三种主流两阶段 backbone（Motif/VCtree/Transformer）、三种单阶段方法、甚至目标检测任务
- **极低的推理开销**：参数量仅增加 0.3-0.5%，推理速度增加 1.2-2.2%，可实际部署

## 局限与展望

- GAN 训练本身存在不稳定性，需要仔细调参（D/G 更新比 5:1，学习率敏感），增加了调参成本
- 纠偏偏置集合 $\mathcal{S}$ 的构建过程较复杂，涉及条件判断和迭代修正，不够简洁
- 仍为两阶段训练（先训 SGG 模型冻结参数 → 再训 BGAN），未实现端到端联合优化
- 在长尾效应较弱的数据集（如 GQA）上提升有限（A@K 仅 +0.1-0.2%）
- 全局偏置 $\mathbf{b}^{glo}$ 仍依赖数据集统计先验，未完全脱离数据集级信息
- 未探索 Transformer-based 生成器或扩散模型替代 GAN 的潜力

## 与相关工作的对比

| 方法 | 类型 | 核心思路 | 局限 |
|---|---|---|---|
| DLFE | 数据集级纠偏 | 估计类别频率，除以有偏概率恢复无偏 | 所有样本共享同一 $\mathbf{c}$，R@K 下降严重 |
| RTPB | 数据集级纠偏 | 学习抗性偏置，减去分类 logit | 所有样本共享同一 $\mathbf{b}$，A@K 反降 |
| CFA | 数据增强 | 三元组特征组合增强尾部多样性 | 不涉及直接纠偏，间接缓解 |
| HML | 层次学习 | 分层逐步学习关系 | 计算开销大，泛化性弱 |
| GCL | 分组学习 | 分组协作提升尾部关注度 | R@K 牺牲大 |
| IETrans | 数据增强 | 关系转换增强尾部训练样本 | 仍属数据集层面操作 |
| **SBG（本文）** | **样本级纠偏** | **GAN 预测样本特异偏置向量** | **需额外训练 BGAN** |

## 启发与关联

- **纠偏粒度升级范式**：类别级→数据集级→样本级的推进思路可迁移到任何存在长尾问题的任务（检测、分割、NLP 分类等）
- **GAN 作为修正量预测器**：传统 GAN 生成图像/文本，此处用于生成修正信号，扩展了 GAN 的应用场景
- **union region 的信息价值**：物体对的 union region 不仅包含关系语义，还能支撑偏置预测，启发后续工作更多挖掘这一区域的潜力
- **"可预测准确但不必完美"的松弛思想**：放弃追求理想无偏预测 $\mathbf{z}_u$，转而寻求"预测结果正确"的 $\hat{\mathbf{z}}$，降低了问题难度却不损失实用性
- **极小开销换显著收益**的模式：推理阶段仅增加 <3% 开销却获得 10%+ 的 A@K 提升，是一种值得推广的后处理策略

## 评分

- 新颖性: ⭐⭐⭐⭐ (样本级纠偏思路清晰且首创，GAN 用于偏置预测有新意)
- 实验充分度: ⭐⭐⭐⭐⭐ (3 数据集 × 3 backbone × 3 任务 + 单阶段方法 + 目标检测泛化 + 全面消融)
- 写作质量: ⭐⭐⭐⭐ (动机阐述逐层递进，公式推导完整，与相关工作对比清晰)
- 价值: ⭐⭐⭐⭐ (即插即用的通用长尾纠偏框架，低开销高收益，泛化性已验证)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](../../NeurIPS2025/optimization/learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/optimization/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[ICML 2025\] Nonparametric Teaching for Graph Property Learners](../../ICML2025/optimization/nonparametric_teaching_for_graph_property_learners.md)
- [\[CVPR 2026\] Learning to Learn Weight Generation via Local Consistency Diffusion](../../CVPR2026/optimization/learning_to_learn_weight_generation_via_local_consistency_diffusion.md)

</div>

<!-- RELATED:END -->
