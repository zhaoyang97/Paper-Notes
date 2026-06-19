---
title: >-
  [论文解读] Flexible Concept Bottleneck Model
description: >-
  [AAAI 2026][可解释性][概念瓶颈模型] 本文提出Flexible Concept Bottleneck Model (FCBM)，通过引入超网络动态生成概念权重和可学习温度的sparsemax模块，实现了概念池的动态适配（包括完全替换），并在5个公开数据集上以相似的有效概念数达到了与SOTA基线可比的精度，仅需单个epoch微调即可适应全新概念集。
tags:
  - "AAAI 2026"
  - "可解释性"
  - "概念瓶颈模型"
  - "超网络"
  - "稀疏激活"
  - "VLM"
---

# Flexible Concept Bottleneck Model

**会议**: AAAI 2026  
**arXiv**: [2511.06678](https://arxiv.org/abs/2511.06678)  
**代码**: [https://github.com/deepopo/FCBM](https://github.com/deepopo/FCBM)  
**领域**: 可解释性  
**关键词**: 概念瓶颈模型, 可解释性, 超网络, 稀疏激活, VLM

## 一句话总结
本文提出Flexible Concept Bottleneck Model (FCBM)，通过引入超网络动态生成概念权重和可学习温度的sparsemax模块，实现了概念池的动态适配（包括完全替换），并在5个公开数据集上以相似的有效概念数达到了与SOTA基线可比的精度，仅需单个epoch微调即可适应全新概念集。

## 研究背景与动机

### 领域现状
概念瓶颈模型 (CBM) 是一种通过引入中间概念层来增强神经网络可解释性的方法——模型先预测人类可理解的概念，再基于概念进行最终任务预测。基于视觉语言模型 (VLM) 的 CBM 已经能通过 LLM 自动生成概念集并利用 CLIP 进行自动标注，大大降低了对专家标注的依赖。

### 现有痛点
现有 VLM-based CBM 的关键问题在于**概念集固定**：

**模型重训练代价高**：当需要引入新概念或更新概念时（如医学领域发现新的生物标志物），必须对整个模型进行端到端重训练

**基础模型快速迭代**：VLM 基础模型（如 CLIP）频繁更新，底层语义表示变化后需要重新对齐概念嵌入

**灵活性受限**：固定概念池无法适应不同部署场景中对不同概念子集的偏好

### 核心矛盾
VLM-based CBM 利用了 VLM 强大的对齐能力来自动构建概念池，但概念到标签的映射（线性层）在训练和推理时都是固定的。这意味着概念数量 $m$ 和权重矩阵的形状是绑定的——一旦 $m$ 变化，必须重新训练。

### 切入角度
用超网络 (hypernetwork) 将概念的文本特征映射为权重，使得权重的生成**独立于概念数量**。同时用 sparsemax 控制稀疏性以保证可解释性。

## 方法详解

### 整体框架
FCBM 包含四个核心组件：
1. 两阶段学习框架（概念预测器 + 标签预测器）
2. LLM 生成概念集 + CLIP 特征提取
3. 超网络动态生成概念权重
4. 可学习温度的 sparsemax 模块

### 关键设计

#### 1. **两阶段学习框架**
- **第一阶段（概念预测器训练）**：优化映射 $g$，使图像经CLIP编码后的特征与概念文本特征的余弦立方相似度最大化：
  $$g^* = \arg\min_g \sum_{j=1}^m [-\text{sim}(\mathbf{c}_{:,j}, \mathbf{q}_{:,j})]$$
- **第二阶段（标签预测器训练）**：优化超网络 $h$，最小化交叉熵损失：
  $$h^* = \arg\min_h \sum_{i=1}^N \text{CE}(g^* \circ \omega(\mathbf{x}_i) \cdot \mathring{h}(\mathbf{t}), \mathbf{y}_i)$$
- 其中 $\mathring{h}(\mathbf{t}) \triangleq \mathcal{S}_{\max}^\tau(h(\mathbf{t}))$ 是经过 sparsemax 处理的稀疏权重

#### 2. **超网络 (Hypernetwork)**
- 映射 $h: \mathbb{R}^d \rightarrow \mathbb{R}^n$，将文本特征维度映射到类别维度
- **核心优势**：$h$ 的参数规模**不依赖于概念数量** $m$，可以处理任意大小的概念集
- 输出 $h(\mathbf{t}) \in \mathbb{R}^{m \times n}$ 具有与传统线性投影相同的形状，可理解为每个概念对各类别的贡献权重
- **推理时特征分布对齐**：当使用新概念集 $\mathbf{t}'$ 时，通过统计量标准化保证分布一致性：
  $$\tilde{\mathbf{t}}' \triangleq \frac{\sigma_\mathbf{t}}{\sigma_{\mathbf{t}'}}(\mathbf{t}' - \overline{\mathbf{t}'}) + \bar{\mathbf{t}}$$
  $$\tilde{h}(\mathbf{t}') \triangleq \frac{\sigma_{h(\mathbf{t})}}{\sigma_{h(\tilde{\mathbf{t}}')}}\left(h(\tilde{\mathbf{t}}') - \bar{h}(\tilde{\mathbf{t}}')\right) + \bar{h}(\mathbf{t})$$
- **设计动机**：固定线性层 $f$ 的权重矩阵维度与 $m$ 绑定，无法适应概念数量变化；超网络从文本特征动态生成权重，自然解耦了概念数量和模型参数

#### 3. **可学习温度的 Sparsemax**
- 标准 sparsemax 生成稀疏输出（不同于 softmax 的全部非零输出），使模型聚焦于最相关概念
- 引入**可学习温度参数** $\tau$ 来动态控制稀疏程度：$\tau$ 越高激活概念越少，越低考虑概念越多
- 温度梯度推导：$\frac{\partial \mathcal{L}}{\partial \tau} = \sum_{i \in P(\mathbf{s})} \frac{1}{|P(\mathbf{s})|} \cdot \frac{\partial \mathcal{L}}{\partial \tilde{\mathbf{s}}_i}$
- **设计动机**：超网络输出通常非稀疏，直接使用会损害可解释性。可学习温度让模型自动平衡预测精度和概念激活稀疏度

### 概念生成与CLIP特征
- 使用 LLM（GPT-3 / DeepSeek-V3 / GPT-4o）按三类提示生成概念：关键特征、常见关联特征、超类
- CLIP 同时编码图像和概念文本到共享特征空间（维度 $d$）
- CLIP-derived 特征 $\mathbf{c} = \mathbf{z} \cdot \mathbf{t}^\top \in \mathbb{R}^{N \times m}$

## 实验关键数据

### 主实验（NEC≈30时的预测精度）

| 骨干网络 | 方法 | CIFAR10 | CIFAR100 | CUB | Places365 | ImageNet |
|----------|------|---------|----------|-----|-----------|---------|
| ResNet50 | Standard (非稀疏) | 88.55 | 70.19 | 71.00 | 53.28 | 73.14 |
| ResNet50 | LF-CBM | 86.16 | 64.62 | 56.91 | 48.88 | 66.03 |
| ResNet50 | CF-CBM | 85.42 | 64.31 | 64.23 | 46.39 | 65.95 |
| ResNet50 | **FCBM (本文)** | **85.59** | **64.77** | 63.46 | **49.13** | **66.34** |
| ViT-L/14 | Standard (非稀疏) | 98.02 | 86.99 | 85.22 | 55.66 | 84.11 |
| ViT-L/14 | LF-CBM | 97.18 | 81.98 | 75.44 | 50.51 | 79.70 |
| ViT-L/14 | CF-CBM | 96.35 | 82.33 | 79.56 | 48.55 | 79.16 |
| ViT-L/14 | **FCBM (本文)** | **97.21** | **83.63** | **80.52** | **51.39** | **80.62** |

### 消融实验（不同模块的零样本泛化能力，ViT-L/14）

| 方法 | CIFAR10 训练 | CIFAR10 DS零样本 | CIFAR100 训练 | CIFAR100 DS零样本 | ImageNet 训练 | ImageNet DS零样本 |
|------|-------------|-----------------|--------------|------------------|--------------|------------------|
| Hard截断 | 97.27 | 78.78 | 65.15 | 23.61 | 75.22 | 15.07 |
| FCBM-无温度 | 89.05 | 75.58 | 62.42 | 38.54 | 49.13 | 23.65 |
| **FCBM (完整)** | **97.21** | **94.89** | **83.63** | **62.27** | **80.62** | **51.70** |

### 关键发现
1. **精度持平SOTA**：在5个数据集上FCBM在半数以上超过所有基线，其余持平
2. **零样本概念泛化**：替换整个概念集（由DeepSeek-V3或GPT-4o生成）后，仅需1个epoch微调即可恢复大部分性能
3. **稀疏度分析**：NEC从30增至full，精度仅小幅提升且基本稳定，证明稀疏选择有效
4. **Hard截断零样本最差**：硬截断在训练概念上有时表现不错，但零样本泛化能力最弱
5. **可学习温度至关重要**：去掉可学习温度后，模型无法有效控制稀疏度，精度显著下降

## 亮点与洞察
- **动态概念适配的首创方案**：首次实现整个概念池的无缝替换而无需重训练全模型
- **超网络的巧妙应用**：从文本特征生成权重的设计自然解决了可变概念数问题
- **统计量标准化的泛化技巧**：训练/推理时的分布对齐公式是一个优雅的工程解决方案
- **概念贡献可视化**：Places365的"campus"类案例清晰展示了不同概念集的语义等价性
- **应用前景**：适合知识快速迭代的场景（如医疗生物标志物更新、VLM模型切换）

## 局限与展望
- 零样本泛化在细粒度类别（如CUB鸟类）和专业领域上仍有较大性能差距
- 仅验证了分类任务，未扩展到检测、分割等其他视觉任务
- 超网络增加了推理时的计算开销（需要对每个概念前向传播）
- 仅使用CLIP作为VLM骨干，未探索其他VLM（如SigLIP、EVA-CLIP）
- 概念生成仍依赖LLM的提示工程，不同提示策略可能产生显著差异
- 当新旧概念集语义差距过大时，统计量标准化可能不足以弥合分布差异

## 相关工作与启发
- Label-free CBM (Oikarinen et al., 2023)：首次使用GPT-3自动生成概念集，FCBM在此基础上解决了概念固定的限制
- VLG-CBM (Srivastava et al., 2024)：提出NEC指标量化有效概念数，FCBM采用了相同指标保证公平对比
- OpenCBM (Tan et al., 2024)：支持测试时灵活添加/移除概念，但无法完全替换概念池
- 超网络 (Ha et al., 2016)：经典的"用网络生成网络权重"思想，FCBM将其引入CBM领域实现动态概念映射
- 启发：超网络+稀疏选择的组合可推广到其他需要动态特征/属性映射的任务

## 评分
- 新颖性: ⭐⭐⭐⭐ （超网络+sparsemax的组合有新意，但核心思路并不复杂）
- 实验充分度: ⭐⭐⭐⭐ （5个数据集+多组消融，但缺少下游应用验证）
- 写作质量: ⭐⭐⭐⭐ （公式推导清晰，结构完整）
- 价值: ⭐⭐⭐⭐ （解决了CBM的实际部署问题，有明确应用场景）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](partially_shared_concept_bottleneck_models.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[CVPR 2025\] Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](../../CVPR2025/interpretability/albm_attribute_concept_space.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)

</div>

<!-- RELATED:END -->
