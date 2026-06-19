---
title: >-
  [论文解读] DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models
description: >-
  [CVPR 2025][模型压缩][Mixture-of-Experts] 提出DeRS（Decompose-Replace-Synthesis）范式，利用upcycled MoE专家间的极高相似性（余弦相似度>0.999），将N个专家分解为1个共享基础权重+N个轻量delta权重，通过稀疏化/量化/低秩表示压缩delta权重，在MoE层参数减少65%的同时性能不降，或训练时额外参数减少2270倍。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "Mixture-of-Experts"
  - "参数高效"
  - "upcycling"
  - "稀疏矩阵"
---

# DeRS: Towards Extremely Efficient Upcycled Mixture-of-Experts Models

**会议**: CVPR 2025  
**arXiv**: [2503.01359](https://arxiv.org/abs/2503.01359)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: Mixture-of-Experts、模型压缩、参数高效、upcycling、稀疏矩阵

## 一句话总结
提出DeRS（Decompose-Replace-Synthesis）范式，利用upcycled MoE专家间的极高相似性（余弦相似度>0.999），将N个专家分解为1个共享基础权重+N个轻量delta权重，通过稀疏化/量化/低秩表示压缩delta权重，在MoE层参数减少65%的同时性能不降，或训练时额外参数减少2270倍。

## 研究背景与动机

**领域现状**：Upcycled MoE将预训练dense模型的FFN层复制N份初始化为N个专家，训练后各专家微调差异化。这种方式比从头训练MoE省资源，已广泛用于NLP、视觉和多模态任务。

**现有痛点**：N个专家引入了大量参数（如MoE-LLaVA-Phi的5B总参数中3.4B被专家占据），但由于都从同一FFN初始化，训练后的专家权重之间余弦相似度极高（>0.999），存在巨大冗余。

**核心矛盾**：upcycled MoE的专家虽然功能上有差异化，但权重空间中的差异极其微小（delta权重相比基础权重可以忽略不计），当前没有方法利用这一特性减少参数。

**本文目标** 如何利用upcycled MoE专家间的高相似性特征，在训练和推理两个阶段都实现极致的参数压缩。

**切入角度**：将每个专家$W_i$分解为共享基础$W_{base}$加专家特有的微小调整$\Delta_i$，然后用轻量形式表示$\Delta_i$。

**核心 idea**：upcycled MoE的专家差异只在微小的delta权重上，用稀疏/量化/低秩压缩delta即可在几乎不损性能的情况下极大减少参数。

## 方法详解

### 整体框架
DeRS范式三步走：Decompose（将N个专家权重分解为$W_{base} + \Delta_i$）→ Replace（用轻量形式$\mathcal{F}(\Delta_i)$替代原始$\Delta_i$）→ Synthesis（推理时将$W_{base} + \mathcal{F}(\Delta_i)$在线合成专家权重）。基于此框架提出两个应用：推理阶段的DeRS Compression和训练阶段的DeRS Upcycling。

### 关键设计

1. **DeRS Compression（推理阶段压缩）**:

    - 功能：压缩已训练好的vanilla upcycled MoE模型
    - 核心思路：两种方式——（a）稀疏化：随机丢弃$\Delta_i$中p比例的元素（如p=0.9），用紧凑向量存储，MoE层参数从$N \cdot d \cdot d_h$降到$(1 + N(1-p)) \cdot d \cdot d_h$；（b）量化：将$\Delta_i$从16bit量化到k bit（如2bit），存储成本从$N \times 16$降到$16 + N \times k$。实验发现丢弃90%元素或量化到2bit几乎不影响性能
    - 设计动机：delta权重极小且冗余，随机稀疏化即可工作说明其信息密度很低

2. **DeRS Upcycling - 稀疏矩阵（训练阶段）**:

    - 功能：在训练时就以参数高效方式构建MoE专家
    - 核心思路：不复制N份FFN，而是保持1份共享$W_{shared}$（初始化自原FFN）+ N个稀疏增量$\mathcal{F}(\Delta_i)$。每个增量用index向量$I_i$和value向量$V_i$表示，$I_i$随机生成并固定，$V_i$可训练且零初始化。稀疏率p=0.9999时额外参数仅0.26M（vs vanilla的1.24B，减少4770倍）
    - 设计动机：既然训练后delta本来就很小，不如直接在训练时就限制其参数量，让模型在受限空间中学习差异化

3. **DeRS Upcycling - 低秩矩阵（训练阶段）**:

    - 功能：另一种参数高效的专家差异化表示
    - 核心思路：类似LoRA，用$\Delta_i = A_i \cdot B_i$（$A_i \in \mathbb{R}^{d \times r}$，$B_i \in \mathbb{R}^{r \times d_h}$）。$A_i$随机初始化，$B_i$零初始化。参数量从$N \cdot d \cdot d_h$降到$d \cdot d_h + N \cdot r \cdot (d + d_h)$
    - 设计动机：低秩分解是另一种成熟的参数高效方式，与稀疏矩阵互补

### 损失函数 / 训练策略
与原始MoE-LLaVA的训练策略一致，使用LLaVA-mix-665k数据集微调，每隔一层的FFN替换为MoE层（4专家top-2激活）。DeRS Upcycling中$W_{shared}$和$V_i$（或$A_i, B_i$）联合训练。

## 实验关键数据

### 主实验

| 模型 | 方法 | 额外参数↓ | 整体性能 |
|------|------|----------|---------|
| MoE-LLaVA-StableLM | Vanilla | 1.24B | 57.4 |
| MoE-LLaVA-StableLM | DeRS-SM | 0.26M (↓4770x) | 57.7 (+0.3) |
| MoE-LLaVA-StableLM | DeRS-LM | 1.20M (↓1033x) | 57.5 (+0.1) |
| MoE-LLaVA-Phi | Vanilla | 2.52B | 60.8 |
| MoE-LLaVA-Phi | DeRS-SM | 1.11M (↓2270x) | 61.1 (+0.3) |
| MoE-LLaVA-Phi | DeRS-LM | 2.42M (↓1041x) | 61.0 (+0.2) |

### 消融实验

| DeRS Compression配置 | MoE层参数变化 | 性能影响 |
|---------------------|-------------|---------|
| 稀疏化 drop=0.9 | 4→1.4个专家等价 | 无损 |
| 稀疏化 drop=0.99 | 4→1.04个专家等价 | 极微弱下降 |
| 量化 2-bit delta | 存储降低$\frac{16+4 \times 2}{4 \times 16}$=37.5% | 无损 |
| 量化 1-bit delta | 最极端压缩 | 轻微下降 |

### 关键发现
- 丢弃delta权重90%元素后性能不降反微升，证实了delta的极高冗余性
- DeRS-SM（稀疏矩阵）在极高稀疏率（99.99%）下仍能工作，说明专家间的差异化只需极少参数
- DeRS Upcycling不仅压缩参数，还能略微提升性能（+0.3），可能因为正则化效果
- 三个任务（通用多模态、医学多模态、代码生成）和六种MoE架构上结论一致
- 在Med-MoE上同样有效，医学任务中专家的冗余模式与通用任务一致

## 亮点与洞察
- **利用upcycling的独特结构性质**：不同于通用MoE压缩方法（剪枝专家、合并专家），DeRS精准利用了upcycled MoE中专家共享初始化这一独特属性，是问题特异性的巧妙解法
- **参数减少2270倍仍能工作**：这个数字令人震惊，说明upcycled MoE中专家差异化的信息量远比大家想象的少得多
- **训练和推理双阶段适用**：DeRS Compression（后训练压缩）和DeRS Upcycling（参数高效训练）涵盖了MoE的全生命周期

## 局限与展望
- 仅适用于upcycled MoE，对从头训练的MoE（如Switch Transformer）不适用，因为专家间没有共享初始化
- 稀疏矩阵的索引$I_i$是随机固定的，探索学习索引位置可能进一步提升性能
- DeRS Compression中的在线合成$W_{base} + \mathcal{F}(\Delta_i)$引入额外计算，未分析推理延时影响
- 实验主要在3B级别模型上，更大模型（如70B级MoE）上的效果未验证

## 相关工作与启发
- **vs LoRA**: DeRS-LM在结构上类似LoRA，但DeRS的关键创新在于先分解出共享基础然后只低秩化delta，而非直接对整个权重做低秩适配
- **vs MC-SMoE等专家剪枝**: 这些方法通过合并/删除整个专家来压缩，可能丢失专家特有信息；DeRS保留所有专家但压缩其差异表示
- **vs 通用MoE**: 实验发现upcycled MoE的专家余弦相似度>0.999，这比从头训练的MoE高得多，DeRS正是利用了这一gap

## 评分
- 新颖性: ⭐⭐⭐⭐ 观察敏锐（余弦相似度>0.999），范式设计自然优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三任务六架构、训练+压缩双场景、消融极其详细
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Decompose-Replace-Synthesis三步走命名直观
- 价值: ⭐⭐⭐⭐ 对upcycled MoE部署有直接价值，2270倍压缩令人印象深刻

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](../../ICLR2026/model_compression/unveiling_super_experts_in_mixture-of-experts_large_language_models.md)
- [\[ICLR 2026\] Efficient Quantization of Mixture-of-Experts with Theoretical Generalization Guarantees](../../ICLR2026/model_compression/efficient_quantization_of_mixture-of-experts_with_theoretical_generalization_gua.md)
- [\[ICLR 2026\] Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss](../../ICLR2026/model_compression/coupling_experts_and_routers_in_mixture-of-experts_via_an_auxiliary_loss.md)
- [\[NeurIPS 2025\] Dense Backpropagation Improves Training for Sparse Mixture-of-Experts](../../NeurIPS2025/model_compression/dense_backpropagation_improves_training_for_sparse_mixture-of-experts.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](../../NeurIPS2025/model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
