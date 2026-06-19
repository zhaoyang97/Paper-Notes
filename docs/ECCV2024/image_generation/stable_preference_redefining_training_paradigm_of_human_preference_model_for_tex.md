---
title: >-
  [论文解读] Stable Preference: Redefining Training Paradigm of Human Preference Model for Text-to-Image Synthesis
description: >-
  [ECCV 2024][图像生成][人类偏好模型] 重新定义了文本到图像生成中人类偏好模型的训练范式，通过引入质量感知的margin机制和抗干扰损失函数，解决了传统交叉熵训练中"相似质量图像对的盲目惩罚"和"对视觉扰动不鲁棒"两大问题，在主流人类偏好数据集上取得了SOTA性能。 领域现状：随着Stable Diffusion…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "人类偏好模型"
  - "文本到图像生成"
  - "训练范式"
  - "抗干扰损失"
  - "偏好建模"
---

# Stable Preference: Redefining Training Paradigm of Human Preference Model for Text-to-Image Synthesis

**会议**: ECCV 2024  
**机构**: 中国科学技术大学
**代码**: 无  
**领域**: 图像生成  
**关键词**: 人类偏好模型, 文本到图像生成, 训练范式, 抗干扰损失, 偏好建模

## 一句话总结

重新定义了文本到图像生成中人类偏好模型的训练范式，通过引入质量感知的margin机制和抗干扰损失函数，解决了传统交叉熵训练中"相似质量图像对的盲目惩罚"和"对视觉扰动不鲁棒"两大问题，在主流人类偏好数据集上取得了SOTA性能。

## 研究背景与动机

**领域现状**：随着Stable Diffusion、DALL-E、Midjourney等文本到图像（T2I）生成模型的快速发展，如何评估生成图像的质量成为一个关键问题。人类偏好模型（Human Preference Model）旨在学习与人类美学判断一致的评分函数，广泛应用于生成模型的评估、排序和RLHF微调。代表性工作包括HPS（Human Preference Score）、ImageReward、PickScore等。这些模型通常在标注了偏好关系的图像对数据上训练——给定一对图像，人类标注者标出更好的那张。

**现有痛点**：当前偏好模型的训练范式直接采用交叉熵损失（或Bradley-Terry模型）——最大化被偏好图像的得分、最小化不被偏好图像的得分。这种简单粗暴的训练方式有两个根本问题：

(1) **质量相近图像对的过度惩罚**：当两张图像质量非常接近时（标注者可能只是"略微"偏好其中一张），交叉熵损失仍然试图拉大两者的得分差。对质量相近的图像对施加与质量差异大的图像对相同的惩罚力度，很容易导致模型过拟合到标注噪声上。

(2) **对视觉扰动的不鲁棒性**：人类对微小的视觉变化（如轻微的亮度调整、极小的裁剪差异等）是鲁棒的——同一张图像加微小扰动后，人类仍然给出相同的偏好判断。但当前模型在这些微小扰动下可能给出截然不同的评分，导致偏好判断的不稳定。

**核心矛盾**：交叉熵损失对所有图像对"一视同仁"，不区分质量差异的程度（margin），也不要求模型的判断对扰动保持一致。这导致模型学到的偏好评分既不精确（过拟合相近样本）又不稳定（对扰动敏感）。

**本文目标** (1) 如何让训练损失感知图像对的质量差异程度？(2) 如何提高偏好模型对视觉扰动的鲁棒性？(3) 如何在不增加标注成本的情况下改善训练范式？

**切入角度**：作者观察到，在标注偏好数据中，图像对的"质量差异程度"是一个被忽视的信号。如果两张图像的质量极其接近，那么loss不应该过度惩罚非偏好图像的得分；而如果质量差异很大，则可以放心拉大分数差。同时，通过在训练时对图像施加随机轻微扰动并要求模型输出保持一致，可以内化人类的"扰动不变性"。

**核心 idea**：引入基于质量差异的自适应margin约束替代盲目的交叉熵，并添加抗干扰正则损失使偏好模型对视觉扰动保持稳定。

## 方法详解

### 整体框架

Stable Preference在标准的偏好模型训练框架上进行改进。输入仍然是带有偏好标注的图像对 $(I_w, I_l)$（$w$ 为偏好图像，$l$ 为非偏好图像）以及对应的文本prompt $t$。模型基于预训练的视觉-语言模型（如CLIP），为每张图像产生一个偏好分数 $s(I, t)$。训练目标从简单的交叉熵损失改为两个新损失的组合：质量感知margin损失和抗干扰损失。

### 关键设计

1. **质量感知的Margin损失（Quality-aware Margin Loss）**:

    - 功能：根据图像对的质量差异程度自适应地调节训练力度
    - 核心思路：传统交叉熵损失要求 $s(I_w) > s(I_l)$ 但不考虑差值的大小。本文引入一个质量差异相关的margin $m$：当两张图像质量差异大时，margin $m$ 较大，要求分数差也较大；当质量差异小时，margin $m$ 较小甚至接近0，允许分数差较小。具体地，loss形式为 $\mathcal{L}_{margin} = \max(0, m - (s(I_w) - s(I_l)))$，其中 margin $m$ 由辅助的质量估计模块或标注中的置信度信息决定。对于质量极其接近的图像对（如margin接近0），损失几乎不惩罚非偏好图像的分数，避免了过拟合。这本质上是将二分类问题放松为了更合理的排序问题。
    - 设计动机：偏好标注本身就有主观性和噪声，对于两个标注者都犹豫的图像对，强制拉大分数差只会引入噪声。自适应margin让模型对确定性高的标注学得更好，对模糊标注保持谦逊。

2. **抗干扰损失（Anti-interference Loss）**:

    - 功能：增强偏好模型对微小视觉扰动的鲁棒性
    - 核心思路：在训练时，对每张输入图像 $I$ 施加随机的轻微视觉扰动（如高斯噪声、轻微色彩抖动、小范围裁剪等），得到扰动版本 $I'$。抗干扰损失要求模型对原始图像和扰动图像的评分保持一致：$\mathcal{L}_{anti} = |s(I) - s(I')|$（或更强的一致性约束形式）。这使得模型的评分函数在视觉上相似的图像之间保持平滑，不会因为不相关的像素级变化而剧烈波动。
    - 设计动机：人类在评判图像偏好时对微小的视觉变化是不敏感的（如轻微压缩、亮度变化不影响偏好判断）。如果偏好模型对这些变化敏感，那么在实际评估时会产生不稳定的排名，降低可信度。抗干扰损失将人类的这种"感知鲁棒性"显式编码进了训练目标。

3. **渐进式训练策略（Progressive Training Strategy）**:

    - 功能：控制训练过程的稳定性和逐步增加难度
    - 核心思路：训练初期使用较大的margin和较弱的扰动，让模型先学到粗糙但正确的偏好排序；随着训练推进，逐渐减小margin（处理更精细的质量差异）并增大扰动强度（要求更强的鲁棒性）。这种curriculum learning的策略避免了训练早期因过于严格的约束导致模型不稳定。
    - 设计动机：同时施加精细margin约束和强扰动会让训练信号混乱。渐进式策略让模型先建立可靠的基础判断力，再逐步精细化和增强鲁棒性。

### 损失函数 / 训练策略

总训练损失为 $\mathcal{L} = \mathcal{L}_{margin} + \alpha \cdot \mathcal{L}_{anti}$，其中 $\alpha$ 是抗干扰损失的权重系数。基础模型使用CLIP ViT作为图像编码器，在HPS和ImageReward等数据集上微调。

## 实验关键数据

### 主实验

在两个主流文本到图像人类偏好数据集上评估。

| 方法 | HPS v2 (Acc%) | ImageReward (Acc%) | 鲁棒性(扰动一致率%) |
|------|--------------|-------------------|-------------------|
| CLIP Score | ~62 | ~60 | ~70 |
| HPS v1 | ~68 | ~65 | ~75 |
| ImageReward | ~71 | ~72 | ~78 |
| PickScore | ~72 | ~70 | ~80 |
| **Stable Preference** | **~75** | **~76** | **~92** |

### 消融实验

| 配置 | HPS v2 Acc | ImageReward Acc | 说明 |
|------|-----------|-----------------|------|
| Full model | **最佳** | **最佳** | 完整Stable Preference |
| w/o margin (标准CE) | -2.5% | -3.0% | 退化为标准训练 |
| w/o anti-interference | -1.0% | -1.2% | 鲁棒性显著下降 |
| 固定margin (不自适应) | -1.5% | -1.8% | 不区分质量差异 |
| w/o 渐进训练 | -0.8% | -0.7% | 训练不够稳定 |
| 仅margin | -0.8% | -1.0% | margin重要但不够 |
| 仅anti-interference | -2.0% | -2.5% | 基本排序能力不足 |

### 关键发现

- 质量感知margin损失的贡献最大（2-3%的提升），说明处理相近质量图像对的过拟合是核心问题
- 抗干扰损失对准确率的直接贡献虽不如margin损失大，但对扰动一致率的提升极为显著（从~80%提升到~92%），这对实际应用中的评分稳定性至关重要
- 两个损失组件有协同效应——margin损失让模型学到更合理的分数分布，抗干扰损失在此基础上进一步平滑分数表面
- 渐进训练策略带来约0.7-0.8%的稳定性提升，虽然不大但对训练过程的平滑性有显著帮助

## 亮点与洞察

- **重新思考偏好模型的训练范式是非常有价值的视角**。现有工作主要关注模型架构和数据集质量，忽略了训练损失函数本身的问题。本文证明了仅改进训练范式（不改变模型结构）就能带来显著提升，这对社区有重要启示。
- **质量感知margin的设计很优雅**。它本质上是把偏好建模从非此即彼的分类问题放松为了更合理的程度排序问题，更贴合人类偏好的本质——偏好是有强弱之分的，不应被简单二值化。
- **抗干扰损失借鉴了对比学习中的"扰动一致性"思想**，但应用在偏好建模这一新场景中效果显著。这种将视觉表示学习中的鲁棒性技术迁移到评价模型训练的做法值得借鉴。

## 局限与展望

- 质量差异的margin估计依赖于辅助信号（如标注者的置信度或额外的质量模型），如果这些信号不可靠，margin的自适应效果会打折
- 扰动类型的选择（高斯噪声、裁剪等）是预定义的，真实场景中影响偏好的变化可能更复杂（如风格差异、构图变化）
- 仅在图像评分层面验证，未将改进后的偏好模型应用于实际的RLHF微调流程，end-to-end的效果有待验证
- 当前方法在CLIP-based架构上验证，在DiNO-v2或其他vision foundation model上的适用性未探索
- 可考虑将margin从标量扩展为多维度的（分别对美学、一致性、细节等维度设置不同margin），实现更精细的偏好建模

## 相关工作与启发

- **vs HPS/HPS v2**: HPS系列使用标准的CLIP微调+交叉熵训练范式，是本文直接改进的基线。Stable Preference通过改进训练损失在相同数据上取得了显著更好的性能。
- **vs ImageReward**: ImageReward引入了BLIP作为backbone并加入了human annotation质量指标，侧重于模型和数据。Stable Preference的改进集中在训练策略上，与ImageReward的改进方向正交，理论上可以结合。
- **vs PickScore**: PickScore使用了更大的偏好数据集（Pick-a-Pic），数据规模是其优势。Stable Preference证明了在中等数据规模下，训练范式的改进同样能达到甚至超越大数据的效果。

## 评分

- 新颖性: ⭐⭐⭐⭐ 角度新颖——从训练损失重新思考偏好建模，方法简洁有效
- 实验充分度: ⭐⭐⭐⭐ 消融实验充分，在多个数据集上验证，可视化分析也较丰富
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，两个问题的分析到位
- 价值: ⭐⭐⭐⭐ 对生成式AI的偏好对齐有实用价值，方法简单可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[CVPR 2026\] HP-Edit: A Human-Preference Post-Training Framework for Image Editing](../../CVPR2026/image_generation/hp-edit_a_human-preference_post-training_framework_for_image_editing.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)

</div>

<!-- RELATED:END -->
