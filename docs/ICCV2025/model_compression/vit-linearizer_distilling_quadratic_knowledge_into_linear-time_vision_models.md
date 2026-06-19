---
title: >-
  [论文解读] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models
description: >-
  [ICCV 2025][模型压缩][跨架构蒸馏] 提出 ViT-Linearizer，一种跨架构蒸馏框架，通过**激活匹配**和**掩码预测**两个核心机制，将 ViT 自注意力中学习到的"二次知识"高效迁移到线性复杂度的循环模型（Mamba-based Adventurer），在 ImageNet 上达到 84.3% 准确率，同时在高分辨率任务中实现最高 4.2× 的推理加速。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "跨架构蒸馏"
  - "ViT"
  - "Mamba"
  - "线性复杂度"
  - "激活匹配"
  - "掩码预测"
---

# ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models

**会议**: ICCV 2025  
**代码**: 未公开  
**领域**: 模型压缩  
**关键词**: 跨架构蒸馏, ViT, Mamba, 线性复杂度, 激活匹配, 掩码预测  
**作者**: Guoyizhe Wei, Rama Chellappa (Johns Hopkins University)

## 一句话总结

提出 ViT-Linearizer，一种跨架构蒸馏框架，通过**激活匹配**和**掩码预测**两个核心机制，将 ViT 自注意力中学习到的"二次知识"高效迁移到线性复杂度的循环模型（Mamba-based Adventurer），在 ImageNet 上达到 84.3% 准确率，同时在高分辨率任务中实现最高 4.2× 的推理加速。

## 研究背景与动机

Vision Transformers (ViTs) 凭借全局自注意力机制在视觉表征学习中取得了卓越性能，但其 $\mathcal{O}(L^2)$ 的二次复杂度在处理高分辨率输入时会成为严重的计算瓶颈。随着对高分辨率、高保真度视觉输入的需求快速增长，如何高效利用 ViTs 学到的"二次知识"变得越来越重要。

另一方面，Mamba、RWKV、xLSTM 等 RNN 风格的 token mixer 在视觉任务中展现出了有竞争力的预测结果和更优的精度-计算权衡。这些循环视觉模型的计算成本和内存需求随序列长度**线性增长**，是应对自注意力二次复杂度爆炸问题的潜在方案。然而，与已有大量研究投入的 ViTs 不同，循环视觉模型的探索目前仍局限于较小的数据规模和模型尺寸。

这些局限性促使作者开发了一种**跨架构蒸馏方法**，将 ViTs 的能力有效转移到线性时间的循环模型中。核心发现是：朴素的蒸馏方法在 ViT 和 Mamba 之间失败了；关键在于**激活匹配**和**掩码预测**两个机制的结合。

## 方法详解

### 整体框架

ViT-Linearizer 的整体流程如 Figure 2 所示：将完整输入图像喂给冻结的教师模型（ViT），将随机掩码后的图像喂给学生模型（基于 Mamba-2 的 Adventurer）。在 $K$ 个中间阶段执行 token 级的激活匹配，在最终层让学生预测教师对未见（被掩码）token 的表征。仅训练学生网络，教师全程冻结。

教师模型采用 CLIP 的 ViT-Base/16，学生模型采用 Adventurer（搭载 Mamba-2 token mixer）。两者的 token mixer 公式对比如下：

**自注意力**（$\mathcal{O}(L^2)$）：

$$\mathbf{y} = \text{softmax}(\mathbf{q}\mathbf{k}^T/\sqrt{d})\mathbf{v}$$

**Mamba-2**（$\mathcal{O}(L)$）：

$$\mathbf{y} = \left(\exp(-\text{softplus}(\delta)\alpha)\mathbf{S} + \mathbf{v}\mathbf{k}^T\right)\mathbf{q}$$

其中 $\mathbf{S}_t = \exp(-\text{softplus}(\delta_t)\alpha)\mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^T$ 是以循环方式累积 token 依赖的隐状态。

### 激活匹配 (Activation Matching)

**核心洞察**：ViT 模型通常在中间层的激活图中捕获比最终层输出更多的信息内容。这些激活图直接反映了在自注意力二次计算成本下学到的 token 级依赖关系。

具体做法：将教师和学生模型的 blocks 划分为 $K$ 个阶段（默认 $K=4$），在每个阶段计算 $\mathbb{R}^{L \times L}$ 的激活图——所有 token 之间的成对余弦相似度：

$$\mathbf{A}_{\text{tea}}^k = \frac{\mathbf{f}_{\text{tea}}^k(i) \cdot \mathbf{f}_{\text{tea}}^k(j)}{\|\mathbf{f}_{\text{tea}}^k(i)\|_2 \|\mathbf{f}_{\text{tea}}^k(j)\|_2}$$

对每一行做 $\ell_2$ 归一化后，定义激活匹配损失：

$$\mathcal{L}_{\text{act}} = \frac{1}{KL}\sum_{k=1}^{K}\sum_{i=1}^{L}\left[1 - \langle\bar{\mathbf{A}}_{\text{tea}}^k(i,:), \bar{\mathbf{A}}_{\text{stu}}^k(i,:)\rangle\right]$$

此损失本身涉及 $\mathcal{O}(L^2)$ 计算——作者称之为"二次约束"，并实验证明它是蒸馏二次知识的必要组件。

### 掩码预测 (Masked Prediction)

采用标准的非对称结构：教师接收完整图像，学生接收掩码输入（随机将一部分 patch token 替换为可学习的 [mask] token），使用 MAE 的 75% 掩码比例。学生需要预测教师在被掩码位置的输出表征：

$$\mathcal{L}_{\text{mask}} = \frac{1}{aL}\sum_{i \in \Omega}\text{Smooth}\ell_1(\mathbf{Y}_{\text{tea}}(i,:), \mathbf{Y}_{\text{stu}}(i,:))$$

### 激活匹配与掩码的整合

关键设计：掩码机制改变了学生的有效表征空间——被掩码位置的中间特征是对未见信息的预测，而非对应输入 token 的表征。**对未见 token 应用激活匹配会导致直接的信息泄漏**，使最终层的掩码预测轻易坍塌。

因此，激活匹配仅在学生可见的 token 上执行，最终的激活图尺寸为 $\mathbb{R}^{(1-a)L \times (1-a)L}$。

**总损失函数**：$\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{mask}}$，默认 $\lambda = 1$。

## 实验关键数据

### 主实验：ImageNet-1k 分类

| 模型 | Token Mixer | 输入尺寸 | 内存 | 吞吐量 | 准确率(%) |
|------|------------|---------|------|--------|----------|
| CLIP ViT-B/16 | Self-attention | 224² | 14.4GB | 613 | 84.7 |
| Vim-Base | Mamba | 224² | 20.0GB | 180 | 81.9 |
| Adventurer-Base (supervised) | Mamba-2 | 224² | 13.0GB | 736 | 82.6 |
| **Adventurer-Base (ours)** | Mamba-2 | 224² | 13.0GB | 736 | **84.3** |
| CLIP ViT-B/16 | Self-attention | 448² | >80GB | 95 | 85.3 |
| **Adventurer-Base (ours)** | Mamba-2 | 448² | 45.2GB | 199 | **85.0** |

448×448 输入时，蒸馏模型实现 **2.1× 推理加速**，仅损失 0.3% 准确率。

### 语义分割结果

| 数据集 | 骨干网络 | 参数量 | 吞吐量 | mIoU(%) |
|--------|---------|--------|--------|---------|
| **ADE20k** | CLIP ViT-B/16 | 119M | 1.00× | 51.0 |
| | Adventurer-Base (ours) | 115M | **2.74×** | **51.3** |
| **Cityscapes** | CLIP ViT-B/16 | 122M | 1.00× | 81.8 |
| | Adventurer-Base (ours) | 118M | **4.21×** | **82.0** |

在 Cityscapes 上实现 **4.21× 加速**且 mIoU 超过教师模型。

### 消融实验

| 模式 | 掩码预测 | 激活匹配 | IN1k 准确率 | ADE mIoU |
|------|---------|---------|------------|----------|
| supervised | ✗ | ✗ | 82.6 | 47.8 |
| no act. match | ✓ | ✗ | 83.6 | 49.7 |
| no mask pred. | ✗ | ✓ | 83.8 | 50.1 |
| **default** | ✓ | ✓ | **84.3** | **51.3** |

| 激活匹配范围 | IN1k 准确率 | ADE mIoU |
|-------------|------------|----------|
| Class token only | 83.7 | 50.0 |
| **Visible tokens only** | **84.3** | **51.3** |
| All tokens | 83.4 | 49.0 |

- 两个机制各自贡献显著增益，结合使用最优
- 匹配所有 token（含 mask）会导致信息泄漏，性能下降
- 仅匹配 class token（$\mathcal{O}(L)$ 约束）不足以充分转移二次知识

### 跨模型尺寸蒸馏

| 教师 | 学生 | 准确率(%) |
|------|------|----------|
| CLIP ViT-B/16 (86M) | Adventurer-S (44M) | 83.1 |
| CLIP ViT-B/16 (86M) | Adventurer-B (99M) | 84.3 |
| CLIP ViT-B/16 (86M) | **Adventurer-L (346M)** | **85.0** |
| CLIP ViT-L/14 (307M) | Adventurer-L (346M) | 85.2 |

发现"逆向蒸馏"现象：大学生也能从小教师中获益，Adventurer-L 达到 85.0%（此前 supervised 只有 83.4%），刷新了 Mamba 架构的 SOTA。

## 亮点与洞察

1. **跨架构蒸馏的可行性验证**：首次系统性地将 ViT 的知识蒸馏到 Mamba 架构，证明循环模型可以继承 ViT 的"注意力知识"
2. **激活匹配是核心**：不同于简单的输出层蒸馏，通过中间层的 token-wise 依赖关系匹配实现了有效的知识迁移
3. **掩码与匹配的巧妙整合**：发现信息泄漏问题并提出仅在可见 token 上匹配的解决方案
4. **效率优势随分辨率增长**：从 224² 的 1.2× 到 Cityscapes 的 4.21×，线性复杂度在长序列下优势愈发显著
5. **逆向蒸馏**：大学生可以从小教师获益，表明 ViT-Linearizer 不仅降低推理成本，还赋予循环模型"注意力知识"和掩码建模能力

## 局限性

1. 蒸馏阶段的激活匹配损失本身是 $\mathcal{O}(L^2)$——虽然仅在训练时使用，但增加了训练成本
2. 实验主要在 Base/Large 规模进行，更大规模的 ViT-Linearizer 效果有待探索
3. 目前仅验证了分类和语义分割任务，在生成、多模态推理等任务上的效果需要进一步验证
4. 依赖于强教师模型（CLIP ViT），教师质量直接决定上限

## 相关工作与启发

- **与 Transformers are SSMs (Dao & Gu, 2024)** 的理论联系：自注意力和 Mamba-2 在公式上有高度相似性，为蒸馏提供了理论基础
- **与 DeiT (Touvron et al., 2021)** 的对比：DeiT 是 CNN→ViT 的蒸馏先驱，本文是 ViT→Mamba 的反向探索
- **对高分辨率视觉的启示**：未来超细粒度 patchification（50K+ token/image）场景中，线性复杂度模型的优势将更加关键

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 跨架构蒸馏框架设计巧妙，激活匹配+掩码预测的组合有明确的技术洞察
- **实验**: ⭐⭐⭐⭐ — 分类、分割多任务验证，消融全面，跨教师/跨尺寸实验充实
- **写作**: ⭐⭐⭐⭐ — 思路清晰，动机充分，公式简洁
- **价值**: ⭐⭐⭐⭐ — 为 ViT 推理加速提供了一条新的技术路线，对循环视觉模型的发展有推动意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICML 2025\] Distilling Tool Knowledge into Language Models via Back-Translated Traces](../../ICML2025/model_compression/distilling_tool_knowledge_into_language_models_via_back-translated_traces.md)
- [\[CVPR 2026\] Masking Teacher and Reinforcing Student for Distilling Vision-Language Models](../../CVPR2026/model_compression/masking_teacher_and_reinforcing_student_for_distilling_vision-language_models.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](../../NeurIPS2025/model_compression/reordering_patches_improves_vision_models.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)

</div>

<!-- RELATED:END -->
