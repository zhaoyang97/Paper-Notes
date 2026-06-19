---
title: >-
  [论文解读] Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][图像生成][概念移除] 提出 Six-CD 基准，包含六类不良概念（有害、裸露、名人、版权角色、物体、艺术风格）和新评估指标 in-prompt CLIP score，首次系统性地对比评估文生图扩散模型的概念移除方法。 文生图扩散模型（如 Stable Diffusion）能生成高质量图像…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "概念移除"
  - "扩散模型安全"
  - "基准评测"
  - "NSFW过滤"
  - "文生图模型"
---

# Six-CD: Benchmarking Concept Removals for Text-to-Image Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2406.14855](https://arxiv.org/abs/2406.14855)  
**代码**: 无  
**领域**: 图像生成 / AI安全 (Image Generation / AI Safety)  
**关键词**: 概念移除, 扩散模型安全, 基准评测, NSFW过滤, 文生图模型

## 一句话总结

提出 Six-CD 基准，包含六类不良概念（有害、裸露、名人、版权角色、物体、艺术风格）和新评估指标 in-prompt CLIP score，首次系统性地对比评估文生图扩散模型的概念移除方法。

## 研究背景与动机

文生图扩散模型（如 Stable Diffusion）能生成高质量图像，但也可被恶意利用——生成暴力、裸露图像或在不当场景中生成公众人物肖像等。概念移除方法（Concept Removal）通过修改模型参数来阻止特定不良概念的生成，是确保模型安全性的关键技术。

然而，现有研究面临三个核心问题：（1）**缺乏一致的全面对比**：不同方法仅评估有限类别（如 ESD 只测物体和艺术风格，SPM 只测名人），缺乏跨方法的统一基准。（2）**无效 prompt 问题**：现有数据集（如 I2P）中大量 prompt 本身就很难触发恶意内容生成，在这些"无效 prompt"上评估概念移除毫无意义，还会导致与特定概念（如名人发出有效 prompt）的不公平比较。（3）**忽视了 in-prompt 保留能力**：当移除"Mickey Mouse is eating a burger"中的 Mickey Mouse 概念后，生成的图像应仍包含"eating a burger"。现有评估完全忽略了这种保留良性语义的能力。

Six-CD 通过构建全面数据集、筛选有效 prompt 子集、引入新评估指标来系统性地解决上述问题。

## 方法详解

### 整体框架

Six-CD 基准包含三个核心组件：（1）覆盖六类不良概念的全面数据集；（2）针对通用概念（有害/裸露）的有效 prompt 筛选子集；（3）新评估指标 in-prompt CLIP score 衡量概念移除后的良性语义保留能力。评估覆盖梯度下降（ESD、SPM、SDD、FMN）、闭式解（UCE、MACE、EMCID）和推理时（SLD、SEGA）三类方法。

### 关键设计

1. **六类概念分类体系**:
    - 功能：建立全面且有层次的不良概念分类
    - 核心思路：将不良概念分为**通用概念**（对所有用户有害）和**特定概念**（涉及特定实体的权利侵犯）两大类。通用概念包括有害内容（暴力、自杀、仇恨等）和裸露内容；特定概念包括名人身份、版权角色（如 Mickey Mouse）、物体（如特定品牌）和艺术风格。数据来源合并 I2P、MMA、SD-uncensored、Unsafe Diffusion 四个 NSFW 资源，使用 NudeNet 和 Q16 分类器自动标注细粒度标签
    - 设计动机：现有工作各自评估不同类别导致无法跨方法比较；六类分类覆盖了所有文献中出现的不良概念类型

2. **有效 Prompt 筛选（Effective Prompt Filtering）**:
    - 功能：为通用概念构建高有效性 prompt 子集，提高评估效率和公平性
    - 核心思路：定义 prompt 有效性为使用该 prompt 触发模型生成恶意内容的概率。通过实验发现通用概念（有害/裸露）的 prompt 有效性远低于特定概念（名人等），因为通用概念的 prompt 通常更隐晦和多样化。为通用概念提供额外的高有效性 prompt 子集，仅保留确实能高概率触发恶意生成的 prompt
    - 设计动机：在无效 prompt 上评估概念移除是无意义的（原模型就不会生成恶意内容），且会导致通用概念与特定概念之间的不公平比较

3. **In-prompt CLIP Score 评估指标**:
    - 功能：衡量概念移除后模型对 prompt 中良性语义的保留能力
    - 核心思路：构建 Dual-Version 数据集——每个 prompt 有恶意版本和良性版本（去除不良概念但保留其余语义）。对恶意版本应用概念移除方法生成图像，然后计算生成图像与良性版本文本之间的 CLIP score。成功的概念移除方法应获得高 in-prompt CLIP score，表明保留了良性语义
    - 设计动机：太激进的概念移除可能连良性语义也一并移除，导致模型输出与 prompt 毫无关联——这同样是不可接受的

### 损失函数 / 训练策略

- 本文是基准评测工作，不提出新的训练方法
- 评估的概念移除方法包括：
    - 梯度下降类：ESD、SPM、SDD、FMN
    - 闭式解类：UCE、MACE、EMCID
    - 推理时类：SLD、SEGA
- 评估指标：概念移除效果（FID、检测器准确率）+ 良性概念保留（FID）+ in-prompt CLIP score

## 实验关键数据

### 主实验

通用概念移除效果（有效 prompt 子集）：

| 方法 | 类型 | 有害移除率↑ | 裸露移除率↑ | 良性保留↑ | In-prompt CLIP↑ |
|------|------|-----------|-----------|---------|----------------|
| ESD | 梯度下降 | 高 | 高 | 中 | 低 |
| SPM | 梯度下降 | 中 | 中 | 高 | 高 |
| UCE | 闭式解 | 中 | 中 | 中 | 中 |
| MACE | 闭式解 | 高 | 高 | 低 | 低 |
| SLD | 推理时 | 中 | 低 | 高 | 高 |

### 消融实验

| 分析维度 | 发现 |
|---------|------|
| 有效vs无效prompt | 无效prompt上概念移除"看起来"很好但实际无意义 |
| 通用vs特定概念 | 特定概念更易移除（prompt更精确），通用概念更难 |
| 单概念vs多概念移除 | 同时移除多类概念时性能显著下降 |
| 移除激进度 | 过于激进的方法（如 ESD）严重损害 in-prompt 保留能力 |

### 关键发现

- 概念移除效果与良性语义保留之间存在**根本性权衡**——移除越彻底，保留越差
- 通用概念（有害/裸露）比特定概念（名人/版权角色）更难移除，因为 prompt 更加隐晦多样
- 闭式解方法（UCE、MACE）在特定概念上效果好但在通用概念上不稳定
- 推理时方法（SLD）虽然跳过微调但可能被开源用户轻松禁用
- 在有效 prompt 子集和全集上的评估结果差异巨大，凸显了 prompt 筛选的重要性

## 亮点与洞察

- **In-prompt 保留能力的提出填补了评估盲区**：之前只关注"移除了多少恶意内容"和"保留了多少良性 prompt 生成能力"，完全忽视了"含有恶意概念的 prompt 中的良性语义是否保留"。这才是实际部署中用户体验的关键
- **有效 prompt 筛选的洞察非常实用**：发现 I2P 数据集中大量 prompt 根本不会触发恶意生成，在这些上做评估是自欺欺人。这一发现促使研究社区重新审视现有评估结果

## 局限与展望

- 仅在 Stable Diffusion 系列模型上评估，未覆盖 DALL-E、Midjourney 等闭源模型
- 概念类别的划分（有害/裸露）依赖自动分类器（NudeNet, Q16），可能存在标注噪声
- 特定概念使用模板化 prompt，可能不够反映真实用户的多样化表达
- 未探索概念移除方法对模型整体生成质量（如 FID）的长期影响

## 相关工作与启发

- **vs I2P 数据集 (SLD)**: I2P 包含大量无效 prompt 且仅覆盖裸露/有害两类；Six-CD 覆盖六类+筛选有效 prompt
- **vs ESD**: ESD 在本基准上移除效果好但 in-prompt 保留差，说明其方法过于激进
- **vs SPM**: SPM 在保留能力上表现最佳，但移除效果相对温和，适合需要平衡的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个全面的概念移除基准，in-prompt CLIP score 填补评估空白
- 实验充分度: ⭐⭐⭐⭐⭐ 9种方法×6类概念×多指标评估，分析极其详尽
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，三个gap逐一解决的结构好
- 价值: ⭐⭐⭐⭐ 为概念移除研究提供了急需的标准化评估框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](../../ICCV2025/image_generation/trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2025\] Scaling Down Text Encoders of Text-to-Image Diffusion Models](scaling_down_text_encoders_of_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] Everything in Its Place: Benchmarking Spatial Intelligence of Text-to-Image Models](../../ICLR2026/image_generation/everything_in_its_place_benchmarking_spatial_intelligence_of_text-to-image_model.md)

</div>

<!-- RELATED:END -->
