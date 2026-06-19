---
title: >-
  [论文解读] Make Me Happier: Evoking Emotions Through Image Diffusion Models
description: >-
  [ICCV 2025][图像生成][情感图像编辑] EmoEditor 提出首个系统性的**情感驱动图像生成**框架，通过双分支扩散模型（全局情感条件 + 局部语义特征）实现仅输入源图和目标情感即可生成具有目标情感的图像，无需手工文本指令或参考图，并构建了 340K 情感标注图对的 EmoPair 数据集。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "情感图像编辑"
  - "扩散模型"
  - "情感分析"
  - "双分支架构"
  - "迭代情感推理"
---

# Make Me Happier: Evoking Emotions Through Image Diffusion Models

**会议**: ICCV 2025

**arXiv**: [2403.08255](https://arxiv.org/abs/2403.08255)

**领域**: 图像生成/情感编辑

**关键词**: 情感图像编辑, 扩散模型, 情感分析, 双分支架构, 迭代情感推理

## 一句话总结

EmoEditor 提出首个系统性的**情感驱动图像生成**框架，通过双分支扩散模型（全局情感条件 + 局部语义特征）实现仅输入源图和目标情感即可生成具有目标情感的图像，无需手工文本指令或参考图，并构建了 340K 情感标注图对的 EmoPair 数据集。

## 研究背景与动机

- **情感编辑的空白**：尽管图像生成/编辑技术飞速发展（InstructPix2Pix、SDEdit、ControlNet），但**情感条件下的图像编辑**几乎未被探索。这在心理治疗（自闭症/精神分裂）、商业广告、艺术设计中有重要应用
- **全局方法的不足**：传统方法如颜色迁移、风格迁移只能进行全局属性调整（亮度、色调），无法识别和编辑引发特定情感的**局部区域**。例如仅提亮着火山坡无法传达"敬畏"感——需要去除火焰并添加彩云
- **情感的局部性**：图像中的情感由全局因素（整体色调）和局部因素（面部表情、特定物体如墓地引发悲伤、气球引发欢乐）共同决定。人类对负面刺激比正面更敏感，简单的全局调整难以改变强烈的负面情绪
- **缺乏数据集**：不存在带有源-目标情感标注和编辑指令的图像对数据集

## 方法详解

### 整体框架

EmoEditor = EmoPair 数据集构建 + 双分支潜在扩散模型 + 迭代情感推理（IEI）。核心流程：输入源图 + 目标情感 -> 情感编辑方向计算 -> 全局情感条件化 + 局部源图特征交互 -> 反向扩散生成 -> 情感预测器迭代验证。

### 关键设计

**1. EmoPair 数据集构建**

使用 Mikels 8 类情感模型：正面（欢乐、敬畏、满足、兴奋）和负面（愤怒、厌恶、恐惧、悲伤）。

- **EPAS（标注子集，331,595 对）**：取 InstructPix2Pix 的图像对，用预训练情感预测器（在 EmoSet 上训练）标注源/目标图像的情感类别
- **EPGS（生成子集，6,949 对）**：取 EmoSet 中富有情感的图像 -> GPT-3 为每类情感生成 50 条通用编辑指令 -> 人工标注员排名保留 top 10 -> 用 Ip2p 按指令编辑 EmoSet 源图 -> 质量筛选

**2. 双分支扩散模型架构**

- **全局分支（情感条件）**：
    - 目标情感编码为 one-hot 向量
    - 用预训练情感预测器预测源图情感概率分布（不使用离散标签，因为真实图可能同时承载多种情感）
    - 计算情感编辑方向 = 目标 one-hot - 源情感概率分布
    - 通过可训练的情感编码器映射到 latent 空间

- **局部分支（源图条件）**：
    - 预训练 VAE 编码器提取源图特征 z
    - 前向扩散对 z 加噪得到 z_t
    - 反向过程中通过 cross-attention 将情感条件和源图条件融入去噪网络

**3. 对齐损失（Neuro-Symbolic Alignment）**

关键创新：让情感 embedding 与 EmoPair 中文本编辑指令的 CLIP embedding 对齐，使模型隐式学习"情感到具体编辑操作"的映射。损失为余弦相似度的互补值。这不是强制预测精确文本，而是在神经符号空间中的软对齐，保留模型创造力。

**4. 迭代情感推理（IEI）**

推理时引入情感预测器作为 critic：

- 生成图像后评估两个条件：(1) SSIM 在 0.5-0.8 之间；(2) 预测的目标情感置信度超过 0.6
- 不满足则：将生成图像作为新输入 -> 重新预测情感状态 -> 更新编辑方向 -> 重新生成
- 最多迭代 30 次，若均未达标则选置信度最高的结果

### 损失函数

总损失 = 潜在扩散噪声预测损失 + 0.5 x 情感-文本对齐损失。

## 实验关键数据

### 主实验（Cross-Valence 情感编辑）

| 方法 | ESMI (CAM) | ESMI (HA) |
|------|-----------|----------|
| Color Transfer | 28.33 | 26.68 |
| Neural Style Transfer | 47.48 | 46.00 |
| CLIP-Styler | 34.33 | 32.77 |
| ControlNet | 32.94 | 31.53 |
| SDEdit | 35.91 | 34.27 |
| InstructPix2Pix | 27.37 | 25.72 |
| AIF | 29.71 | 28.31 |
| EmoEdit | 32.94 | 37.53 |
| LMS (大模型级联) | 35.40 | 33.76 |
| **EmoEditor (Ours)** | **51.56** | **49.99** |

- 人类心理物理实验（136 名被试 x 180 试次 = 24,480 试次）：EmoEditor 在与所有基线的两两对比中均超过 50% 偏好率

### 消融实验

| 变体 | 输入 | IEI | L_emb | ESMI |
|------|------|-----|-------|-------|
| 仅文本条件 | text | 无 | 无 | 5.85 |
| 文本 + IEI | text | 有 | 无 | 50.24 |
| One-hot + L_emb | e_oh | 无 | 有 | 7.36 |
| 编辑方向 + L_emb | e_dir | 无 | 有 | 8.51 |
| 编辑方向 + IEI | e_dir | 有 | 无 | 47.36 |
| 编辑方向 + IEI + L_emb | e_dir | 有 | 有 | **51.56** |

### 关键发现

- IEI 是性能提升的**最大贡献因素**（从 8.51 到 47.36）
- 情感编辑方向优于固定 one-hot，因为它考虑了源图的情感状态
- 对齐损失提供额外改进（47.36 到 51.56）
- CAM-based 和 HA-based ESMI 评分一致性高，验证了评估指标的可靠性
- LMS（GPT-4o + GPT-o4 + Ip2p 级联大模型）也无法超越端到端训练的 EmoEditor

## 亮点与洞察

1. **问题定义的新颖性**：首次将"情感驱动的图像编辑"完整形式化——给定源图和目标情感，生成保持场景结构但唤起目标情感的图像
2. **情感编辑方向的巧妙设计**：编辑方向 = 目标 - 源，让模型自适应调整编辑幅度——源图已有部分目标情感时少编辑，完全相反时大幅更改
3. **Neuro-Symbolic 对齐**：不强制预测精确文本，而在 embedding 空间软对齐，既约束模型学到情感-编辑的对应关系，又保留生成多样性
4. **IEI 的实用性**：迭代生成 + 自动评估的闭环设计，在推理时无需人工介入
5. **严谨的人体评估**：136 名被试的大规模心理物理实验 + MTurk 质量控制，在 HCI/心理学标准下也是高质量评估
6. **有趣的发现**：同一情感触发元素（火焰引发愤怒）在不同场景下有不同的最优编辑方案（室内变可爱灯具/室外变宁静草地），说明模型确实学到了上下文感知的情感编辑

## 局限性

- 输入图像固定为 224x224 分辨率，限制了实际应用
- IEI 最多迭代 30 次，推理成本较高
- 情感预测器的准确性直接影响 IEI 和编辑方向计算的质量
- 8 类情感分类（Mikels 模型）较粗粒度，无法捕捉更细微的情感状态
- EmoPair 数据集中 EPAS 子集（来自 Ip2p）的情感标注是自动标注而非人工标注
- 跨文化情感差异未考虑

## 相关工作

- **InstructPix2Pix** [Brooks et al., 2023]：文本指令驱动的图像编辑，EmoEditor 的技术基线
- **EmoEdit** [Yang et al., 2024]：情感编辑先驱，但依赖固定查询字典限制了编辑多样性
- **AIF** [Li et al., 2023]：从文本反映情感到图像，但只能全局滤镜效果
- **EmoSet** [Yang et al., 2023]：最大的视觉情感分析数据集（120K 图 + 情感标签）
- **SDEdit** [Meng et al., 2022]：基于 SDE 的图像编辑基线

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | 5/5 |
| 有效性 | 4/5 |
| 实用性 | 4/5 |
| 清晰度 | 4/5 |
| 综合 | 4/5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EmoEdit: Evoking Emotions through Image Manipulation](../../CVPR2025/image_generation/emoedit_evoking_emotions_through_image_manipulation.md)
- [\[ICCV 2025\] DiffDoctor: Diagnosing Image Diffusion Models Before Treating](diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)

</div>

<!-- RELATED:END -->
