---
title: >-
  [论文解读] Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?
description: >-
  [NeurIPS 2025 Spotlight][预训练][目标绑定] 通过定义 IsSameObject 谓词并设计二次探针，证明大规模预训练 ViT（尤其是 DINO、CLIP）自然涌现了目标绑定能力，该信号编码在低维子空间中并主动引导注意力机制，挑战了认知科学界认为 ViT 缺乏绑定能力的观点。
tags:
  - "NeurIPS 2025 Spotlight"
  - "预训练"
  - "目标绑定"
  - "Transformer"
  - "IsSameObject"
  - "自监督学习"
  - "探针分析"
---

# Does Object Binding Naturally Emerge in Large Pretrained Vision Transformers?

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.24709](https://arxiv.org/abs/2510.24709)  
**代码**: [GitHub](https://github.com/liyihao0302/vit-object-binding)  
**领域**: LLM预训练  
**关键词**: 目标绑定, Vision Transformer, IsSameObject, 自监督学习, 探针分析

## 一句话总结

通过定义 IsSameObject 谓词并设计二次探针，证明大规模预训练 ViT（尤其是 DINO、CLIP）自然涌现了目标绑定能力，该信号编码在低维子空间中并主动引导注意力机制，挑战了认知科学界认为 ViT 缺乏绑定能力的观点。

## 研究背景与动机

**目标绑定（Object Binding）** 是认知科学中的核心概念：大脑将分散在不同皮层区域的低级特征（颜色、形状、运动等）整合为统一的物体表征。这一能力支撑了人类对物体的高效存储、组合式记忆和推理。

在 AI 领域，这个问题具有重要意义但研究不足：

**认知科学的质疑**：研究者认为 ViT 缺乏动态灵活分组特征的机制、缺乏迭代精炼的循环连接、作为纯连接主义模型无法进行真正的符号处理

**目标中心学习的局限**：Slot Attention 等方法通过外部模块强制绑定，但引入了额外的扩展和训练挑战

**关键问题**：ViT 是否能在没有显式架构归纳偏置的情况下，仅通过大规模预训练就自然习得目标绑定能力？

作者的核心洞察：自注意力机制的二次性质为 ViT 表征"两个 patch 是否属于同一物体"提供了计算基础。

## 方法详解

### 整体框架

定义 **IsSameObject** 谓词：对于层 $\ell$ 的两个 token 嵌入 $(x_i^{(\ell)}, x_j^{(\ell)})$，判断它们是否属于同一物体：

$$\text{IsSameObject}(x_i^{(\ell)}, x_j^{(\ell)}) = \phi(x_i^{(\ell)}, x_j^{(\ell)}), \quad \phi: \mathbb{R}^d \times \mathbb{R}^d \to [0,1]$$

关键研究假设：
- IsSameObject 编码可能是**线性**的还是根本上**二次**的
- 信号是**成对关系**还是**逐点映射**（先映射到物体ID再比较）
- 模型依赖**类别标签**还是**物体实例**来区分
- 信号存储在**少数专化维度**还是**分布在多个维度**

### 关键设计

设计了四种探针架构来检验上述假设：

**1. 线性探针**：
$$\text{IsSameObject}_{lin}(x,y) = \sigma(Wx + Wy + b), \quad W \in \mathbb{R}^{1 \times d}$$

**2. 对角二次探针**（专化维度）：
$$\text{IsSameObject}_{diag}(x,y) = \sigma(x^\top W y + b), \quad W \text{ 为对角矩阵}$$

**3. 全二次探针**（分布式）：
$$\text{IsSameObject}_{quad}(x,y) = \sigma(x^\top W_1^\top W_2 y + b)$$

其中 $W_1, W_2 \in \mathbb{R}^{k \times d}$，$k \ll d$，设 $W_2 = SW_1$（$S$ 为符号对角矩阵）保证对称性。

**4. 物体类别/实例探针**（逐点）：先将嵌入映射为概率分布，再计算内积。

**绑定信号的分解**：假设每个 token 嵌入可分解为特征部分和绑定部分：

$$h^{(\ell)}(x_t) = f^{(\ell)}(x_t, c) + b^{(\ell)}(x_t)$$

其中 $f$ 编码纹理、形状等属性，$b$ 编码与哪些其他 token 属于同一物体的信息。训练好的二次探针可视为将激活投影到 IsSameObject 子空间。

### 损失函数 / 训练策略

在 ADE20K 数据集上训练探针，使用交叉熵损失对同物体/不同物体 patch 对进行分类。基线准确率为 72.6%（总是预测"不同"），反映了大多数 patch 对不属于同一物体的类别不平衡。

消融实验设计：
- **非知情消融**：随机打乱绑定向量 $b(x_i)$
- **知情消融（注入）**：利用真实实例掩码注入 IsSameObject 信号

## 实验关键数据

### 主实验

**跨模型 IsSameObject 解码准确率**：

| 模型 | 最高准确率 | 超过基线 (pp) | 峰值层 (0-1) |
|------|-----------|--------------|-------------|
| DINOv2-Small | 86.7% | +14.1 | 1.00 |
| DINOv2-Base | 87.5% | +14.9 | 0.82 |
| DINOv2-Large | **90.2%** | **+17.6** | 0.78 |
| DINOv2-Giant | 88.8% | +16.2 | 0.77 |
| Supervised (ViT-L) | 84.2% | +11.6 | 0.39 |
| CLIP (ViT-L) | 82.9% | +10.3 | 0.65 |
| MAE (ViT-L) | 76.3% | +3.7 | 0.13 |

**探针对比**（DINOv2-Large）：二次探针 > 对角二次探针 > 物体实例探针 > 物体类别探针 > 线性探针

### 消融实验

**IsSameObject 消融对下游任务的影响**（DINOv2-Large 第 18 层）：

| 指标 | 原始 | 随机打乱50% | 随机打乱100% | 注入α=0.5 | 注入α=0 |
|------|------|-------------|-------------|-----------|---------|
| 语义分割 mIoU | 44.14% | 41.03% | 39.20% | 44.91% | 43.59% |
| 实例分割 mIoU | 35.14% | 31.39% | 28.19% | 36.37% | 37.02% |
| DINO Loss | 0.6182 | 0.6591 | 0.6749 | — | — |

**注意力相关性**：中间层注意力权重与 IsSameObject 得分存在正相关（Pearson r=0.163~0.201），表明模型确实利用绑定信号分配注意力。

### 关键发现

1. **绑定是习得的而非架构固有的**：DINO、CLIP 和有监督 ViT 都显示出强绑定信号，但 MAE 几乎没有（仅 +3.7pp），说明绑定能力依赖于特定的预训练目标
2. **信号是二次分布式的**：全二次探针显著优于线性和对角探针，与自注意力的二次形式一致
3. **层级演变规律**：早中层逐步识别局部物体；深层转向基于类别的分组，位置信息在深层被丢弃
4. **消融验证因果性**：打乱绑定信号降低分割性能并增加预训练损失，注入真实信号则提升实例分割
5. **低维子空间**：IsSameObject 编码在低维投影空间中，不同物体实例在前几个主成分上线性可分

## 亮点与洞察

1. **连接认知科学与深度学习**：将心理学中的目标绑定概念与 ViT 的涌现行为联系起来，提供了 AI 系统中类人认知能力的证据
2. **训练目标决定绑定能力**：对比实验揭示了重要的归纳偏置来源——DINO 的对比学习要求跨增强视图的一致性，自然促进了物体级特征学习；而 MAE 的重建目标不需要这种能力
3. **类似大脑的层级组织**：ViT 中层关注局部物体、深层关注语义类别的模式，与大脑腹侧通路的视网膜拓扑组织相呼应
4. **对 Slot Attention 的重新思考**：解决绑定问题可能不需要外部模块，而是可以通过定制训练目标或最小架构修改来加强 ViT 内在的绑定机制

## 局限与展望

1. 探针将 patch 嵌入分解为"特征"和"绑定"部分的假设过于简化，需要进一步经验验证
2. 未建立目标绑定与下游任务性能之间的因果关系
3. 下游评估仅限于分割任务，视觉推理等其他任务待验证
4. 仅研究了 patch 级别的绑定，更一般形式的绑定（如属性绑定）未探索
5. 为何 MAE 不涌现绑定信号的机制解释不够深入

## 相关工作与启发

- **Slot Attention**：显式目标中心方法，通过可学习 slot 竞争 token 特征来强制绑定；本文证明这种能力可以自然涌现
- **Feng & Steinhardt (2023)**：在语言模型中发现绑定是通过低维 binding-ID 编码实现的；本文将此扩展到视觉领域
- **Dai et al. (2024)**：研究 LLM 中的绑定表征分析，发现属性通过低维代码链接到主体
- **DINO/DINOv2**：自监督 ViT 的涌现特性（如注意力图对应显著区域），本文进一步揭示了其物体绑定能力
- 对多模态理解很有启发意义：如果 ViT 已经内在地编码了"哪些部分属于一起"，可以被 VLM 利用来改善组合理解

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (从认知科学角度提出全新研究问题，IsSameObject 定义精妙)
- 实验充分度: ⭐⭐⭐⭐ (跨模型、跨探针、消融完整，但下游验证有限)
- 写作质量: ⭐⭐⭐⭐⭐ (概念定义清晰，论证逻辑严密，连接认知科学与AI)
- 价值: ⭐⭐⭐⭐ (深化了对 ViT 表征的理解，对目标中心学习有重要指导意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] One Prompt Fits All: Universal Graph Adaptation for Pretrained Models](one_prompt_fits_all_universal_graph_adaptation_for_pretrained_models.md)
- [\[NeurIPS 2025\] How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?](how_does_sequence_modeling_architecture_influence_base_capabilities_of_pre-train.md)
- [\[CVPR 2025\] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](../../CVPR2025/llm_pretraining/bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)
- [\[ICML 2025\] Does Data Scaling Lead to Visual Compositional Generalization?](../../ICML2025/llm_pretraining/does_data_scaling_lead_to_visual_compositional_generalization.md)
- [\[ICML 2025\] Counting in Small Transformers: The Delicate Interplay between Attention and Feed-Forward Layers](../../ICML2025/llm_pretraining/counting_in_small_transformers_the_delicate_interplay_between_attention_and_feed.md)

</div>

<!-- RELATED:END -->
