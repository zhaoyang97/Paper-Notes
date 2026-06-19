---
title: >-
  [论文解读] In-Context Compositional Learning via Sparse Coding Transformer
description: >-
  [NeurIPS 2025][多模态VLM][稀疏编码] 受稀疏编码启发，将 Transformer 注意力机制重新解释为在编码字典和解码字典上的投影，通过稀疏系数显式表示组合规则，并利用提升方案（lifting scheme）将上下文任务的组合规则迁移到目标任务。 In-context compositional lear…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "稀疏编码"
  - "注意力机制"
  - "组合学习"
  - "上下文学习"
  - "Transformer"
---

# In-Context Compositional Learning via Sparse Coding Transformer

**会议**: NeurIPS 2025  
**arXiv**: [2511.20194](https://arxiv.org/abs/2511.20194)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 稀疏编码, 注意力机制, 组合学习, 上下文学习, Transformer

## 一句话总结

受稀疏编码启发，将 Transformer 注意力机制重新解释为在编码字典和解码字典上的投影，通过稀疏系数显式表示组合规则，并利用提升方案（lifting scheme）将上下文任务的组合规则迁移到目标任务。

## 研究背景与动机

In-context compositional learning 要求模型从上下文示例中推断组合规则并应用到目标问题。标准 Transformer 在这类任务上面临两个根本性限制：

**Softmax 产生稠密注意力权重**：导致信息无差别全局混合，无法表示输入中的组合结构

**缺乏规则迁移机制**：无法从上下文示例中有效提取和复用局部组合规则

核心观察：当目标任务输入为零（未观测）时，标准 Transformer 的 softmax 注意力退化为均匀分布 $\frac{1}{N}\mathbf{1}$，输出变为所有输入的简单平均——这直接导致了模糊的预测。

## 方法详解

### 整体框架

将 MHA（多头注意力）分解为三个步骤：

1. **编码**：将输入投影到编码字典 $\phi(\mathbf{X})$ 上获取系数
2. **稀疏化**：对系数施加稀疏约束
3. **解码**：用稀疏系数线性组合解码字典 $\psi(\mathbf{X})$ 生成输出

数学形式：

$$\text{MHA}(\mathbf{X}) = \sum_{h=1}^{H} \sigma(\mathbf{X} \underbrace{\phi^{(h)}(\mathbf{X})}_{\text{编码字典}}) \underbrace{\psi^{(h)}(\mathbf{X})}_{\text{解码字典}}$$

其中编码字典 $\phi^{(h)}(\mathbf{X}) = \mathbf{W}_{qk}^{(h)} \mathbf{X}^\top$，解码字典 $\psi^{(h)}(\mathbf{X}) = \mathbf{X} \mathbf{W}_{vo}^{(h)}$。

### 关键设计

#### 1. 稀疏系数

用 soft-thresholding 替代 softmax 作为非线性函数 $\sigma(\cdot)$：

$$\text{prox}(\mathbf{S}) = \text{sign}(\mathbf{S}) \odot \max(|\mathbf{S}| - \xi, 0)$$

稀疏系数 $\boldsymbol{\alpha} = \sigma(\mathbf{X} \phi(\mathbf{X}))$ 保留最有信息量的成分，抑制冗余交互，使表示与底层组合规则更对齐。

关键分析——当目标输入 $\mathbf{X}_L = \mathbf{0}$ 时：

- **标准 Transformer**: $\boldsymbol{\alpha}_L = \text{softmax}(\mathbf{0}) = \frac{1}{N}\mathbf{1}$ → 输出为全局平均 → 模糊
- **本方法**: $\boldsymbol{\alpha}_L = \text{prox}(\mathbf{0}) = \mathbf{0}$ → 诚实地输出零 → 需要系数迁移

#### 2. 目标任务系数估计（Coefficient Transfer）

受 lifting scheme 启发，通过上下文系数的线性组合估计目标系数：

$$\boldsymbol{\alpha}_L \leftarrow \boldsymbol{\alpha}_L + \sum_{i=1}^{L-1} \lambda_i \boldsymbol{\alpha}_i$$

其中 $\lambda_i$ 为可学习参数。更新后的输出：

$$\mathbf{Z}_L = \boldsymbol{\alpha}_L \psi(\mathbf{X}) + \sum_{i=1}^{L-1} \lambda_i \boldsymbol{\alpha}_i \psi(\mathbf{X})$$

参数效率极高：每层仅增加 $L-1$ 个可学习参数（上下文任务数量）。

#### 3. 基函数变体

编码/解码字典的基函数 $\phi(\cdot), \psi(\cdot)$ 可引入非线性（如 ReLU）来增强表达能力：

$$\phi(\mathbf{X}) = \text{ReLU}(\mathbf{W}_{qk}^{(h)} \mathbf{X}), \quad \psi(\mathbf{X}) = \text{ReLU}(\mathbf{W}_{vo}^{(h)} \mathbf{X})$$

实验表明双端 ReLU 效果最佳（73.6% vs 71.7%）。

### 损失函数 / 训练策略

- 合成数据集：MSE 损失，Adam 优化器，lr=0.001，200 epochs，batch size 128
- S-RAVEN：Cross-entropy 损失，Adam，lr=0.001，weight decay 0.1，1 epoch，batch size 128
- RAVEN：MSE 损失，Adam，lr=0.0001，2000 epochs，batch size 256
- 所有模型使用 RMS normalization

## 实验关键数据

### 主实验（S-RAVEN）

| 方法 | 4层/10M | 4层/20M | 4层/40M | 8层/10M | 8层/20M | 8层/40M |
|------|---------|---------|---------|---------|---------|---------|
| Transformer | 51.6 | 55.7 | 58.1 | 59.8 | 63.3 | 65.1 |
| HYLA | 55.0 | 68.6 | 73.2 | 72.5 | 77.1 | 79.3 |
| **Ours** | **63.1** | **73.9** | **76.3** | **72.6** | **78.2** | **82.7** |

### RAVEN 数据集

在 PSNR > 40 的严格标准下：标准 Transformer 几乎 0% 的测试样本达标，而本方法保持约 40%——差异巨大。

### 消融实验

| 阈值 $\xi$ | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 |
|------------|-------|------|------|-----|-----|
| 稀疏度 (%) | 18.53 | 57.82 | 90.45 | 97.82 | 99.38 |

| 基函数配置 | 准确率 |
|-----------|--------|
| 线性 $\phi$, 线性 $\psi$ | 71.7 |
| ReLU $\phi$, 线性 $\psi$ | 72.3 |
| 线性 $\phi$, ReLU $\psi$ | 72.9 |
| ReLU $\phi$, ReLU $\psi$ | **73.6** |

### 关键发现

1. **稀疏性是组合学习的关键**：标准 Transformer 的稠密注意力在组合泛化任务上彻底失败（输出模糊），而稀疏注意力产生清晰、结构化的预测
2. **系数迁移不可或缺**：没有系数迁移时，即使有稀疏性，目标任务输出仍为零
3. **少量数据即有效**：4 层模型在 10M 数据上就比 8 层标准 Transformer 在 40M 数据上表现更好
4. **应用于语言模型**：集成到 Llama-7B 后，在常识推理上超越 base model（仅需约百个参数 vs LoRA 的 50M+）

## 亮点与洞察

1. **理论优雅**：建立了注意力机制与稀疏编码的严格对应关系，不只是类比而是数学重构
2. **问题分析深刻**：精确指出标准 Transformer 在目标输入为零时退化为均匀平均的根本原因
3. **参数极其高效**：每层仅增加 $L-1$ 个参数（通常 < 10），系数迁移几乎零开销
4. **可控稀疏性**：通过阈值 $\xi$ 可精确控制注意力稀疏度

## 局限与展望

1. 仅在小规模合成数据集（S-RAVEN、RAVEN）上验证，未在大规模实际任务上证明
2. 语言模型实验（Llama-7B）结果尚不及 LoRA/DoRA，但参数量极少
3. Soft-thresholding 在 backpropagation 中的梯度特性可能限制深层网络训练
4. 系数迁移的线性组合假设可能对复杂非线性组合规则不足

## 相关工作与启发

- **HYLA**：同在 S-RAVEN 上的强基线，使用混合线性注意力，但未显式建模组合规则
- **稀疏注意力**（Longformer, BigBird 等）：目标是降低计算复杂度，而本文目标是增强组合推理——动机完全不同
- **启发**：稀疏编码视角可能为理解大模型的 in-context learning 能力提供新框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 从稀疏编码角度重构注意力机制的思路非常原创
- 实验充分度: ⭐⭐⭐ — 合成数据集验证充分，但缺乏大规模真实任务验证
- 写作质量: ⭐⭐⭐⭐ — 理论推导和分析清晰
- 价值: ⭐⭐⭐⭐ — 为理解和改进 Transformer 组合推理提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer](../../ICML2025/multimodal_vlm/rollingq_reviving_the_cooperation_dynamics_in_multimodal_transformer.md)
- [\[CVPR 2025\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](../../CVPR2025/multimodal_vlm/hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)
- [\[CVPR 2026\] Bridging the Modality Gap in Compositional Zero-Shot Learning via Sparse Alignment and Unimodal Memory Bank](../../CVPR2026/multimodal_vlm/bridging_the_modality_gap_in_compositional_zero-shot_learning_via_sparse_alignme.md)
- [\[NeurIPS 2025\] Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models](sparse_autoencoders_learn_monosemantic_features_in_visionlan.md)

</div>

<!-- RELATED:END -->
