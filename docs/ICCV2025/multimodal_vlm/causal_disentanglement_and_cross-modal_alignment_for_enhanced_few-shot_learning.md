---
title: >-
  [论文解读] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning
description: >-
  [ICCV 2025][多模态VLM][少样本学习] 提出 Causal CLIP Adapter (CCA)，利用 ICA 对 CLIP 视觉特征进行因果解纠缠，并通过单向微调文本分类器和双向交叉注意力增强跨模态对齐，在 11 个基准数据集上实现了少样本分类 SOTA。 问题定义 少样本学习（FSL）需要模型在有限标注数据…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "少样本学习"
  - "CLIP"
  - "因果解纠缠"
  - "独立成分分析"
  - "跨模态对齐"
---

# Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning

**会议**: ICCV 2025  
**arXiv**: [2508.03102](https://arxiv.org/abs/2508.03102)  
**代码**: [github.com/tianjiao-j/CCA](https://github.com/tianjiao-j/CCA)  
**领域**: 多模态VLM  
**关键词**: 少样本学习, CLIP, 因果解纠缠, 独立成分分析, 跨模态对齐

## 一句话总结

提出 Causal CLIP Adapter (CCA)，利用 ICA 对 CLIP 视觉特征进行因果解纠缠，并通过单向微调文本分类器和双向交叉注意力增强跨模态对齐，在 11 个基准数据集上实现了少样本分类 SOTA。

## 研究背景与动机

### 问题定义

少样本学习（FSL）需要模型在有限标注数据下快速适应新任务。当前主流方法大多基于 CLIP 等预训练视觉-语言模型进行迁移学习，但直接使用 CLIP 提取的特征面临一个核心挑战：**特征是纠缠的**。

### 已有方法的不足

**纠缠特征导致过拟合**：CLIP 视觉编码器输出的特征 $\mathbf{z}'$ 是真实潜变量 $\mathbf{z}$ 的线性混合。现有方法（如 Tip-Adapter、TaskRes 等）直接使用这些纠缠特征，分类器需要隐式学习一个密集权重矩阵 $\mathbf{W}_{\text{dense}} = \mathbf{W}_{\text{unmix}} \cdot \mathbf{W}'_{\text{sparse}}$，参数量大，在少样本场景下容易过拟合

**仅依赖 Cache Model 不够**：Tip-Adapter 等方法仅在图像模态内构建相似度匹配，未充分利用 CLIP 的跨模态对齐能力

**Prompt-tuning 方法训练开销大**：CoOp 等需要通过 CLIP 文本编码器反向传播，训练时间显著高于 adapter 方法

### 核心动机

理论研究（Daunhawer et al.）证明 CLIP 等多模态对比学习模型可以将真实潜变量恢复到线性变换的程度，即 $\mathbf{z}' = \mathbf{A}\mathbf{z} + \mathbf{c}$。既然线性变换已知存在，那么可以直接用 ICA 去除它，获得解纠缠特征，从而大幅减少下游任务需要学习的参数。

**核心 idea**：先用 ICA 解纠缠 CLIP 特征降低参数量，再通过跨模态对齐弥补解纠缠带来的模态内对齐损失。

## 方法详解

### 整体框架

CCA 包含三个核心组件：(1) 基于 FastICA 的解纠缠 Cache Model，用于模态内图像匹配；(2) 微调的 CLIP 文本分类器，实现单向跨模态对齐；(3) 交叉注意力融合模块，实现双向跨模态信息交互。三者的 logits 通过线性组合得到最终预测。

### 关键设计

#### 1. **FastICA 解纠缠**

- **功能**：从 ImageNet 子集中提取 unmixing matrix $\mathbf{U} \in \mathbb{R}^{C \times M}$，对所有 CLIP 视觉特征进行解纠缠
- **核心思路**：CLIP 特征在超球面上，$\mathbf{z}' = \mathbf{A}\mathbf{z} + \mathbf{c}$，ICA 通过最大化非高斯性恢复独立潜变量。解纠缠后的 Cache Keys：

$$\mathbf{F}_{\text{cache}}^d = \mathbf{F}_{\text{cache}} \mathbf{U} \mathbf{W}_c, \quad \mathbf{F}_{\text{query}}^d = \mathbf{F}_{\text{query}} \mathbf{U}$$

其中 $\mathbf{W}_c$ 是可训练的 cache adapter（初始化为单位矩阵，加 $\ell_1$ 正则化诱导稀疏性）

- **设计动机**：解纠缠后分类器只需学习稀疏权重矩阵，参数量大幅减少，有效缓解少样本过拟合。$\mathbf{U}$ 从 ImageNet 提取后所有数据集共用，计算开销小

#### 2. **单向跨模态对齐（文本分类器微调）**

- **功能**：微调 CLIP 文本分类器 $\mathbf{W}_t \in \mathbb{R}^{N \times C}$（由类别名经文本编码器获得），使文本特征更好地对齐图像特征
- **核心思路**：以较小的学习率（0.0001）微调文本权重，保留 CLIP 先验知识的同时增强下游对齐
- **设计动机**：ICA 解纠缠可能破坏 CLIP 原有的模态内对齐，通过文本分类器微调可以从跨模态角度补偿这种损失

#### 3. **双向交叉注意力融合**

- **功能**：通过交叉注意力机制生成两种混合特征——图像增强的文本分类器和文本增强的图像特征
- **核心公式**：

$$\mathbf{W}_t^* = (\text{softmax}(\mathbf{W}_t \mathbf{F}_{\text{query}}^\top) \mathbf{F}_{\text{query}})^\top$$

$$\mathbf{F}_{\text{query}}^* = \text{softmax}(\mathbf{F}_{\text{query}} \mathbf{W}_t^\top) \mathbf{W}_t$$

最终跨模态 logits：$\mathbf{l}_2 = \mathbf{F}_{\text{query}} \mathbf{W}_t^\top + \gamma \mathbf{F}_{\text{query}} \mathbf{W}_t^* + \eta \mathbf{F}_{\text{query}}^* \mathbf{W}_t^\top$

- **设计动机**：单向微调只能从文本到图像对齐，双向交叉注意力允许两个模态相互补充信息，捕捉更丰富的语义关系

### 损失函数 / 训练策略

- **损失函数**：标准交叉熵损失 + cache adapter 的 $\ell_1$ 正则化
- **最终 logits**：$\mathbf{l} = \alpha \mathbf{l}_1 + \mathbf{l}_2$
- **训练细节**：SGD 优化，cache adapter 学习率 0.001，文本分类器学习率 0.0001，大多数数据集仅训练 20 个 epoch（EuroSAT 100 个）
- **超参数**：$\alpha, \gamma, \eta, \beta$ 在验证集上通过网格搜索优化

## 实验关键数据

### 主实验

11 个数据集的平均分类准确率（16-shot）：

| 方法 | 类型 | ImageNet | 11数据集均值 |
|------|------|----------|-------------|
| Zero-shot CLIP | 零样本 | 60.33 | - |
| CoOp | Prompt-tuning | 62.95 | - |
| Tip-Adapter | 无训练 | 61.80 | - |
| Tip-Adapter-F | 微调 | 65.45 | - |
| CCA | 无训练 | 63.00 | - |
| **CCA-FT** | **微调** | **66.04** | **最优** |

CCA-FT 在所有 11 个数据集的所有 shot 设定上均超越 Tip-Adapter-F，且在平均准确率上超越所有 SOTA 方法。

### 消融实验

| 配置 | 1-shot | 4-shot | 16-shot | 说明 |
|------|--------|--------|---------|------|
| 完整模型 | 66.00 | 72.10 | 77.60 | 所有组件齐全 |
| 无 ICA | 64.94 | 70.34 | 75.95 | 解纠缠贡献最大 |
| 固定 cache adapter | 65.16 | 70.78 | 75.75 | 模态内对齐重要 |
| 固定文本分类器 | 65.10 | 71.32 | 77.00 | 文本微调有帮助但较小 |
| 无融合特征 | 65.81 | 72.02 | 77.43 | 交叉注意力补充较小 |

### 关键发现

- **ICA 解纠缠是最关键组件**：移除后所有 shot 数下性能下降最大（16-shot 下降 1.65%）
- **分布外鲁棒性强**：在 ImageNet-V2 和 ImageNet-Sketch 上，CCA-FT 均优于所有对比方法
- **对高斯噪声和对抗攻击鲁棒**：CCA/CCA-FT 在两种设定下大幅优于 Tip-Adapter/-F（对抗攻击下 16-shot 准确率 35.48% vs 33.21%）
- **计算效率高**：20 epoch，4.9 min 训练（vs CoOp 的 200 epoch，7.5h），且准确率更高
- **跨骨干网络一致有效**：在 ResNet50/101、ViT-B/16、ViT-B/32 上均优于 Tip-Adapter-F

## 亮点与洞察

1. **理论驱动的实用方法**：从因果解纠缠理论出发，利用 ICA 的线性解纠缠能力解决 FSL 中的过拟合问题，理论与实践结合紧密
2. **单/双向对齐互补设计**：意识到解纠缠会破坏模态内对齐，通过两种互补的跨模态对齐机制弥补，整体设计逻辑严谨
3. **unmixing matrix 跨数据集共用**：从 ImageNet 提取一次即可用于所有下游数据集，实用性强
4. **稀疏性归纳偏置**：$\ell_1$ 正则化诱导 cache adapter 稀疏，与解纠缠表示的理论预期一致

## 局限与展望

1. **ICA 假设的局限**：假设至多一个潜变量为高斯分布，在某些数据分布下可能不满足
2. **超球面约束**：CLIP 特征在超球面上，ICA 的有效维度少一维（$M-1$），可能损失信息
3. **文本分类器设计简单**：仅使用线性微调，未探索更灵活的文本适配方式
4. **未探索更大规模 CLIP**：仅在 ResNet50/101 和 ViT-B 上验证，未测试 ViT-L 等更大骨干
5. **交叉注意力训练/测试不一致**：训练时双向均用 query 特征，测试时反向路径改用 cache 特征

## 相关工作与启发

- CLIP 的因果解纠缠理论（Daunhawer et al.）为本文提供了理论基础，这一理论视角值得在其他 CLIP 应用中探索
- FastICA 的轻量级特性使其特别适合少样本场景，类似的无监督解纠缠方法可能也能应用于其他迁移学习任务
- 单/双向对齐的互补设计思路可以推广到其他多模态融合场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将因果解纠缠理论引入 CLIP 少样本学习是新颖的角度，ICA + 跨模态对齐的组合设计合理
- **实验充分度**: ⭐⭐⭐⭐ — 11 个数据集 + 多个 shot 设定 + OOD 测试 + 噪声/对抗鲁棒性 + 消融研究，实验全面
- **写作质量**: ⭐⭐⭐⭐ — 理论动机清晰，方法描述详细，图示直观
- **价值**: ⭐⭐⭐⭐ — 提供了一种轻量高效的 CLIP adapter 方法，理论洞察对社区有启发价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](../../NeurIPS2025/multimodal_vlm/vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](../../CVPR2025/multimodal_vlm/unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[CVPR 2026\] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment](../../CVPR2026/multimodal_vlm/noise-aware_few-shot_learning_through_bi-directional_multi-view_prompt_alignment.md)

</div>

<!-- RELATED:END -->
