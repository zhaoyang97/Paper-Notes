---
title: >-
  [论文解读] Attention Prompting on Image for Large Vision-Language Models
description: >-
  [ECCV 2024][多模态VLM][提示学习] 提出Attention Prompting on Image（API），通过辅助VLM（CLIP或LLaVA）根据文本查询生成注意力归因图，将其作为热力图叠加到原始图像上引导LVLM聚焦相关区域，在MM-Vet上提升LLaVA-1.5达3.8%，跨多种LVLM（包括GPT-4V）通用有效。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "提示学习"
  - "注意力机制"
  - "LVLM"
  - "self-reflection"
  - "model ensemble"
---

# Attention Prompting on Image for Large Vision-Language Models

**会议**: ECCV 2024  
**arXiv**: [2409.17143](https://arxiv.org/abs/2409.17143)  
**代码**: [GitHub](https://github.com/yu-rp/apiprompting)  
**领域**: 大视觉语言模型 / 视觉提示  
**关键词**: visual prompting, attention heatmap, LVLM, self-reflection, model ensemble

## 一句话总结

提出Attention Prompting on Image（API），通过辅助VLM（CLIP或LLaVA）根据文本查询生成注意力归因图，将其作为热力图叠加到原始图像上引导LVLM聚焦相关区域，在MM-Vet上提升LLaVA-1.5达3.8%，跨多种LVLM（包括GPT-4V）通用有效。

## 研究背景与动机

**领域现状**：视觉提示（visual prompting）通过在图像上添加圈圈、箭头、遮罩等标注来引导LVLM关注特定区域。已有方法（如FGVP、SoM）依赖分割模型生成标注，无需训练且直觉有效。

**现有痛点**：(1) 已有视觉提示技术仅处理图像本身，不考虑文本查询内容——无论问什么问题，同一张图的视觉提示结果相同；(2) 这导致提示区域与实际问题所需关注的区域不匹配；(3) 基于分割的方法（FGVP/SoM）本质是实例级proposal，不适用于通用VQA任务。

**核心矛盾**：如何让视觉提示随文本查询动态变化，使模型根据不同问题关注图像的不同区域？

**切入角度**：利用LVLM自身的视觉-文本对齐能力生成query-aware的注意力归因图，将其作为视觉提示叠加到图像上。

## 方法详解

### 整体框架

给定图像$I$和文本查询$T^i$，API分两步：(1) **归因图生成**：用辅助VLM $g$（可以是CLIP或LLaVA自身）计算文本查询对图像各patch的重要性得分，生成归因图$\Psi \in \mathbb{R}^{P \times P}$；(2) **热力图叠加**：将归因图上采样到像素空间，经均值滤波平滑后作为alpha通道与原图混合，得到标注图像$I^a$送入推理VLM $f$。若$g=f$则为self-reflection，若$g \neq f$则为模型集成。

### 关键设计

1. **CLIP的cls token分解归因图**

    - 利用ViT的残差连接，将CLIP的图像级相似度分解到各patch的贡献
    - 对深层MSA输出逐patch计算与文本嵌入$\hat{T}$的相似度，得到$\Psi^{cls}$——直接定位与查询实体相关的patch
    - 互补归因图$\Psi^{comp}$：最后一层非cls token与$\hat{T}$的相似度取反——低信息量的"寄存器"token相似度高，有实际内容的patch相似度低
    - 最终归因图 $\Psi = \Psi^{cls} + \Psi^{comp} - \Psi^{cls} \cdot \Psi^{comp}$（软OR操作），兼顾显式实体定位和隐式相关区域保留

2. **LLaVA的注意力权重归因图**

    - 直接取LLaVA深层decoder的cross-attention权重（输出token对图像token的注意力值）
    - 在所有生成token和所有注意力头上取平均，得到每个图像patch的平均被关注程度
    - 比CLIP方案更简单，但需要先做一次推理生成输出序列

### 损失函数 / 训练策略

API是一种**无需训练**的推理时技术，不涉及损失函数或训练过程。核心超参数包括：归因图使用的起始层$L'$、均值滤波核大小$k$。

## 实验关键数据

### 主实验

| 模型 | 提示方法 | MM-Vet | LLaVA-Bench | MMMU |
|------|---------|--------|------------|------|
| LLaVA-1.5 | 无提示 | 32.8 | 71.9 | 35.2 |
| LLaVA-1.5 | FGVP (Mask) | 31.0 (-1.8) | 57.4 (-14.5) | 36.1 (+1.0) |
| LLaVA-1.5 | SoM | 26.4 (-6.4) | 56.1 (-15.8) | 35.6 (+0.4) |
| LLaVA-1.5 | **API (CLIP)** | **35.3 (+2.5)** | **74.1 (+2.2)** | **37.5 (+2.4)** |
| LLaVA-1.5 | **API (LLaVA)** | **36.6 (+3.8)** | **74.8 (+2.9)** | **37.0 (+1.8)** |
| GPT-4V | 无提示 | 67.0 | 102.0 | 50.6 |
| GPT-4V | **API (CLIP)** | **67.7 (+0.7)** | **103.3 (+1.3)** | **51.0 (+0.4)** |

### 消融实验

| 消融项 | MM-Vet | 说明 |
|--------|--------|------|
| 仅$\Psi^{cls}$ | 34.1 | 缺少隐式相关patch |
| 仅$\Psi^{comp}$ | 33.8 | 缺少显式实体定位 |
| $\Psi^{cls} + \Psi^{comp}$（软OR） | 35.3 | 两者互补最优 |
| 无均值滤波 | 33.5 | 矩形mask与物体形状不匹配 |
| 不同起始层$L'$ | $L'=L-2$最优 | 浅层信息不够判别 |

### 关键发现

- API在LLaVA-1.5上比FGVP和SoM显著更好——关键在于query-aware（后两者不看问题）
- FGVP和SoM在大多数模型-数据集组合中反而降低性能——query-agnostic的视觉提示可能造成mismatch
- API在闭源模型GPT-4V和Gemini上同样有效（+0.7%到+11.6%），证明了通用性
- 当辅助模型$g$与推理模型$f$相同时（self-reflection），效果最好（API-LLaVA on LLaVA: +3.8%）

## 亮点与洞察

- 将视觉提示从"query-agnostic"升级到"query-aware"是一个关键的范式转变——同一图，不同问题应highlight不同区域
- cls token分解是一种巧妙的归因方法——利用残差连接的可加性将全局相似度拆解到patch级
- 发现非cls token的高相似度实际是"寄存器"功能——与register token的最新研究一致

## 局限性 / 可改进方向

- 需要额外的一次辅助模型前向推理，推理成本翻倍
- 热力图叠加为像素级乘法，可能丢失被压暗区域的信息——对需要全局理解的任务可能有害
- CLIP的归因图对非实体类查询（如"这幅画的风格是什么"）效果可能有限
- 未测试在视频VQA或多图对比等复杂场景中的效果

## 相关工作与启发

- **vs FGVP/SoM**：FGVP/SoM用分割模型生成固定标注，与文本查询无关；API根据查询动态生成归因热力图
- **vs Self-Reflection**：传统self-reflection在文本空间迭代（反复回答+修改），API在像素空间做self-reflection——更直接
- **vs GradCAM**：GradCAM需要梯度回传，API仅需前向推理；关键创新是cls token分解替代了梯度方法
- **启发**：LVLM的注意力权重是宝贵的"免费"信号——可用于更多场景如注意力蒸馏、token剪枝、hallucination检测

## 评分

- 新颖性: ⭐⭐⭐⭐ query-aware视觉提示是重要的范式升级，cls分解方法有创意
- 实验充分度: ⭐⭐⭐⭐⭐ 5个LVLM×6个基准+详细消融+闭源模型验证
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，两种归因图方案对比全面
- 价值: ⭐⭐⭐⭐ 无训练的即插即用方法，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](../../ICML2026/multimodal_vlm/large_vision-language_models_get_lost_in_attention.md)
- [\[ECCV 2024\] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)
- [\[ICML 2026\] Neutral-Reference Prompting for Vision-Language Models](../../ICML2026/multimodal_vlm/neutral-reference_prompting_for_vision-language_models.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)

</div>

<!-- RELATED:END -->
