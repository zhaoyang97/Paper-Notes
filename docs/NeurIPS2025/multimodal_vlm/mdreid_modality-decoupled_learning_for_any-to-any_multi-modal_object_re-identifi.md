---
title: >-
  [论文解读] MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification
description: >-
  [NeurIPS 2025][多模态VLM][多模态ReID] 提出MDReID框架，通过将模态特征解耦为模态共享（modality-shared）和模态特有（modality-specific）两部分，实现任意模态组合下的目标重识别（any-to-any ReID），在模态匹配和模态不匹配场景下均大幅超越现有方法。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "多模态ReID"
  - "模态解耦"
  - "跨模态检索"
  - "any-to-any匹配"
  - "度量学习"
---

# MDReID: Modality-Decoupled Learning for Any-to-Any Multi-Modal Object Re-Identification

**会议**: NeurIPS 2025  
**arXiv**: [2510.23301](https://arxiv.org/abs/2510.23301)  
**代码**: [GitHub](https://github.com/stone96123/MDReID)  
**领域**: 多模态VLM  
**关键词**: 多模态ReID, 模态解耦, 跨模态检索, any-to-any匹配, 度量学习

## 一句话总结

提出MDReID框架，通过将模态特征解耦为模态共享（modality-shared）和模态特有（modality-specific）两部分，实现任意模态组合下的目标重识别（any-to-any ReID），在模态匹配和模态不匹配场景下均大幅超越现有方法。

## 研究背景与动机

**领域现状**：多模态目标重识别（ReID）利用RGB、NIR、TIR等多种光谱互补信息，在复杂场景下显著提升识别鲁棒性。

**现有痛点**：现有方法（如TOP-ReID、EDITOR）假设查询和图库的模态严格对齐，但实际部署中摄像头类型、部署环境不同，导致模态不一致。

**核心矛盾**：模态缺失时，试图从可用模态重建缺失模态是一个病态问题（ill-posed），不可预测的模态特有信息会导致次优学习。

**本文目标**：设计一个灵活的框架，支持任意查询-图库模态组合的检索，包括模态匹配和模态不匹配两种场景。

**切入角度**：将模态信息分解为可预测可迁移的共享特征和不可预测的特有特征，分别处理。

**核心 idea**：通过在ViT中引入模态共享和模态特有的可学习token，显式解耦表示，并用正交性损失和知识差异损失增强解耦效果。

## 方法详解

### 整体框架

MDReID基于Vision Transformer (ViT) backbone，包含两个核心组件：
- **Modality Decoupled Learning (MDL)**：将每个模态的表示拆分为模态共享和模态特有两部分
- **Modality-aware Metric Learning (MML)**：通过度量学习进一步增强特征解耦

### 关键设计

1. **Modality Decoupled Learning (MDL)**：

    - **功能**：为每个模态提取共享特征和特有特征
    - **为什么**：共享特征用于跨模态检索（模态不匹配场景），特有特征保留模态独有的判别信息（模态匹配场景）
    - **怎么做**：在ViT中，为每个模态的patch embedding序列前置两个可学习token：$I_{sp}^M$（模态特有）和 $I_{sh}^M$（模态共享），经ViT编码后得到解耦特征。构建统一特征向量：
    $v_{full} = [I_{sp}^R, I_{sp}^N, I_{sp}^T, I_{sh}^R, I_{sh}^N, I_{sh}^T]$
      对于缺失模态，对应位置用零向量填充，并用二值可用性掩码标记
    - **区别**：不同于TOP-ReID试图重建缺失模态表示，MDReID避免了病态重建问题

2. **相似度计算**：

    - **模态特有相似度** $Sim_{sp}$：仅比较相同模态的特有特征，通过可用性掩码处理缺失情况
    - **模态共享相似度** $Sim_{sh}$：计算所有可用共享特征对之间的相似度矩阵
    $Sim_{total}(v_q, v_g) = (Sim_{sp} + Sim_{sh}) / 2$

3. **Representation Orthogonality Loss (ROL)**：

    - **功能**：在通道级别促进模态共享特征的聚合，并强制共享和特有特征正交
    - **怎么做**：定义6×6的理想目标相似度矩阵$A$，其中特有特征之间为单位矩阵（正交），共享特征之间全为1（一致），两组之间全为0（正交），然后最小化实际相似度与目标的平方误差：
    $L_{ROL} = \sum_{i,j} (V_{sim}(i,j) - A(i,j))^2$

4. **Knowledge Discrepancy Loss (KDL)**：

    - **功能**：确保共享+特有特征的组合比单独使用任一类特征更具判别力
    - **怎么做**：利用triplet loss思想，要求组合特征的最大正样本距离更小、最小负样本距离更大：
    $L_{KDL} = \|D_p - 0\|_1 + \|D_n - 1\|_1$

### 损失函数 / 训练策略

总损失函数：
$$L = L_{ce} + L_{tri} + L_{MML}$$
其中 $L_{MML} = w_1 \times L_{ROL} + w_2 \times L_{KDL}$，$w_1=1.5$，$w_2=5.25$。

训练使用Adam优化器，batch size 64，基础学习率 $3.5 \times 10^{-4}$，ViT微调学习率 $5 \times 10^{-6}$，训练50个epoch。backbone采用CLIP-Base视觉编码器。

## 实验关键数据

### 主实验

**模态匹配场景（RNT-to-RNT）**：

| 方法 | RGBNT201 mAP | RGBNT201 R-1 | RGBNT100 mAP | RGBNT100 R-1 | MSVR310 mAP | MSVR310 R-1 |
|------|-------------|-------------|-------------|-------------|------------|------------|
| TOP-ReID | 72.3 | 76.6 | 81.2 | 96.4 | 35.9 | 44.6 |
| EDITOR | - | - | 82.1 | 96.4 | 39.0 | 49.3 |
| **MDReID** | **82.1** | **85.2** | **85.3** | **95.6** | **51.0** | **68.9** |

**模态不匹配场景（平均，4种场景）**：

| 方法 | RGBNT201 Avg mAP | RGBNT100 Avg mAP | MSVR310 Avg mAP |
|------|-----------------|-----------------|-----------------|
| TOP-ReID | 18.2 | 26.8 | 11.2 |
| EDITOR | 8.5 | 11.9 | 2.5 |
| **MDReID** | **21.6** | **38.6** | **22.1** |

### 消融实验

| 配置 | MDL | $L_{ROL}$ | $L_{KDL}$ | mAP | R-1 |
|------|-----|-----------|-----------|-----|-----|
| 1（单分类器） | ✕ | ✕ | ✕ | 27.8 | 27.1 |
| 2（MDL only） | ✓ | ✕ | ✕ | 39.4 | 38.2 |
| 3（+ROL） | ✓ | ✓ | ✕ | 41.2 | 40.8 |
| 5（完整） | ✓ | ✓ | ✓ | **43.2** | **42.3** |

### 关键发现

- MDL（模态解耦学习）贡献最大，引入模态特有分类器使mAP从27.8%提升到39.4%（+11.6%）
- ROL和KDL分别额外带来约2%和2%的提升，验证了解耦约束的有效性
- 在模态不匹配场景中，MDReID相比TOP-ReID在MSVR310上提升了10.9% mAP，展示了强鲁棒性

## 亮点与洞察

- **问题定义清晰**：首次系统化地定义了image-level的any-to-any多模态ReID问题
- **解耦思路优雅**：通过共享/特有token + ViT自注意力机制自然地实现特征解耦，避免了复杂的重建模块
- **相似度设计合理**：基于可用性掩码的灵活相似度计算，自适应处理任意模态缺失
- **ROL的目标矩阵设计**：6×6的理想相似度矩阵直觉清晰——特有特征互相正交、共享特征互相一致

## 局限与展望

- 仅在三种模态（RGB/NIR/TIR）上验证，扩展到更多模态（如深度、事件相机）的表现未知
- 训练阶段假设所有模态可用，未探索训练时模态缺失的情况
- 模态共享和模态特有的解耦程度依赖于损失函数的超参数调优（$w_1, w_2$较敏感）
- 未讨论在open-set ReID或大规模数据集上的扩展性

## 相关工作与启发

- **TOP-ReID**：通过循环token排列聚合多光谱特征，但受限于模态对齐假设
- **RLE**：指出跨光谱特征预测是病态问题，启发了本文的解耦思路
- **CLIP backbone**：利用预训练视觉-语言模型的强表征能力，提升特征质量
- **启发**：模态解耦的思想可推广到其他多模态任务（如视觉-语言、视觉-音频融合）

## 评分

- 新颖性: ⭐⭐⭐⭐ 解耦思路本身不新，但在any-to-any ReID中的应用和实现较优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、模态匹配/不匹配/缺失多种场景、详细消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式完整，图示直观
- 价值: ⭐⭐⭐⭐ 解决了多模态ReID中重要的实际部署问题，性能提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Generalizable Object Re-Identification via Visual In-Context Prompting](../../ICCV2025/multimodal_vlm/generalizable_object_re-identification_via_visual_in-context_prompting.md)
- [\[ICCV 2025\] MAVias: Mitigate Any Visual Bias](../../ICCV2025/multimodal_vlm/mavias_mitigate_any_visual_bias.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2026\] Describe Anything Anywhere At Any Moment](../../CVPR2026/multimodal_vlm/describe_anything_anywhere_at_any_moment.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](mmwalk_towards_multi-modal_multi-view_walking_assistance.md)

</div>

<!-- RELATED:END -->
