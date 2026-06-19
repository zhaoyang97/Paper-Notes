---
title: >-
  [论文解读] IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark
description: >-
  [ICCV 2025][图像生成][红外图像] 提出 IRGPT，首个基于真实红外图像的多模态大语言模型，构建了包含 260K+ 图像-文本对的大规模红外-文本数据集 IR-TD，并设计了双跨模态课程迁移学习策略（Bi-cross-modal Curriculum），在 9 个红外任务基准上取得 SOTA 性能，零样本 psum 比基线 InternVL2-8B 提升 76.35。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "红外图像"
  - "多模态大语言模型"
  - "课程学习"
  - "跨模态迁移"
  - "视觉-语言模型"
---

# IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark

**会议**: ICCV 2025  
**arXiv**: [2507.14449](https://arxiv.org/abs/2507.14449)  
**代码**: 数据集已开源  
**领域**: 图像生成  
**关键词**: 红外图像, 多模态大语言模型, 课程学习, 跨模态迁移, 视觉-语言模型

## 一句话总结

提出 IRGPT，首个基于真实红外图像的多模态大语言模型，构建了包含 260K+ 图像-文本对的大规模红外-文本数据集 IR-TD，并设计了双跨模态课程迁移学习策略（Bi-cross-modal Curriculum），在 9 个红外任务基准上取得 SOTA 性能，零样本 psum 比基线 InternVL2-8B 提升 76.35。

## 研究背景与动机

**为什么现有 VLM 无法处理红外图像？** 当前主流的视觉-语言模型（LLaVA、Qwen2-VL、InternVL2 等）主要针对可见光图像训练，在处理红外图像时会产生严重的幻觉问题。例如，金属招牌会被错误识别为非金属材质物体。这一问题的根源在于两方面：

**数据稀缺**：红外领域缺乏大规模的图像-文本对齐数据集。可见光图像可以通过网络爬取获得海量描述文本，但红外图像无法采用同样的方式

**合成数据的局限性**：先前方法（如 Infrared-LLaVA）依赖风格迁移从可见光图像生成合成红外图像，但合成数据受限于生成模型的训练分布，无法捕捉真实红外图像的独特特性（如热信息主导的目标显著性、不同光谱波段的成像差异）

**为什么从可见光迁移学习也不简单？** 虽然红外和可见光图像存在一定相似性，但直接迁移面临红外图像语义信息稀疏、近红外/热红外等不同波段差异巨大、图像-文本对齐质量参差不齐等挑战。需要一种从简单到复杂、循序渐进的学习策略。

## 方法详解

### 整体框架

IRGPT 以 InternVL2-8B 为基础模型，包含视觉编码器、基础 LLM、文本分词器和可训练视觉投影器。训练策略分为两个阶段：

- **增量预训练**：仅训练视觉编码器和投影层，目标是适应红外图像输入并与文本对齐
- **监督指令微调**：同时微调视觉投影器和 LLM（使用 LoRA 防止表达能力损失）

### IR-TD 数据集构建

为解决红外-文本数据稀缺问题，作者汇集了 63 个公开数据集，探索了 5 种数据生成路径：

- **路径 IV（LLM 辅助）**：利用可见光-红外图像对，用 LLM 生成可见光图像描述，再人工适配为红外图像标注
- **路径 V（规则生成）**：根据标注信息基于规则生成大规模图像-文本对

最终数据集包含 190K 预训练样本、33K 指令样本和 37K 基准测试样本，覆盖识别、定位、关系判断、行人重识别、安全监控、航拍计数等 9 类任务。与 Infrared-LLaVA 的闭源 IID 数据集相比，IR-TD 具有开源、图像真实、无数据冗余的优势。

### 双跨模态课程迁移学习

核心创新在于设计了两个"课程"来量化样本难度，指导由简到难的训练：

**Lesson 1: IR-VIS（红外-可见光距离度量）**

利用特征提取器在红外和可见光灰度图像上重训练，通过域内距离投影计算每个红外样本到可见光域的距离：

$$d_i = (\phi(x_i^{ir}) - c^{ir}) \cdot \frac{c^{vis} - c^{ir}}{\|c^{vis} - c^{ir}\|} + \text{MMD}$$

其中 $c^{ir}$、$c^{vis}$ 为两个域的中心向量，MMD 为最大均值差异。越接近可见光的样本越容易学习（近红外 > 热红外）。

**Lesson 2: IR-T（红外-文本对齐度量）**

使用预热的 CLIP 计算红外图像与文本的对齐损失。为区分"困难但有价值的样本"和"对齐错误的样本"，引入**损失变化率**：

$$\alpha = \frac{l' - l}{l}$$

其中 $l$ 和 $l'$ 分别为预热前后的损失值。$\alpha > 0$ 表示对齐变差（可能为噪声样本），$\alpha \leq 0$ 表示正常收敛。

**课程调度**

将两个 Lesson 的排名整合为综合排名，将数据划分为 M 个难度层级。训练时按层级顺序组织，层级内随机采样，兼顾结构化学习和随机性。损失变化率作为自适应权重：

$$w_i = \begin{cases} 1 - \sigma(\alpha_i / \text{Median}(\{\alpha_j | \alpha_j > 0\})) & \alpha_i > 0 \\ 1 + \sigma(-\alpha_i / \text{Median}(\{-\alpha_k | \alpha_k \leq 0\})) & \alpha_i \leq 0 \end{cases}$$

最终加权交叉熵损失为 $\mathcal{L} = \frac{1}{N}\sum_{i=1}^N w_i \cdot [-\sum_c y_{i,c} \log p_{i,c}]$。

## 实验关键数据

### 主实验：9 任务性能对比

| 模型 | Scene | Rec. | Gro. | Rel. | ReID | Sec. | psum↑ | Loc. | A.C. | P.C. | nsum↓ |
|------|-------|------|------|------|------|------|-------|------|------|------|-------|
| LLaVA1.5-7B (zero-shot) | 35.89 | 23.11 | 11.17 | 23.68 | 2.61 | 18.63 | 115.09 | 83.12 | 42.39 | 117.22 | 242.73 |
| Qwen2-VL-7B (zero-shot) | 49.80 | 59.42 | 25.74 | 56.19 | 4.62 | 30.08 | 225.85 | 50.41 | 18.05 | 89.12 | 157.58 |
| InternVL2-8B (zero-shot) | 46.02 | 77.69 | 30.05 | 53.02 | 7.33 | 38.19 | 252.30 | 43.29 | 21.75 | 60.39 | 125.43 |
| InternVL2-26B (zero-shot) | 62.24 | 72.88 | 39.48 | 55.42 | 8.16 | 43.72 | 281.90 | 38.25 | 21.68 | 64.65 | 124.58 |
| **IRGPT (CL, zero-shot)** | **65.12** | **86.28** | **36.68** | **58.33** | **33.55** | **48.69** | **328.65** | **33.32** | **13.25** | **47.30** | **93.87** |
| InternVL2-26B (fine-tune) | 84.03 | 98.94 | 48.77 | 95.89 | 47.09 | 99.06 | 473.78 | 12.50 | 1.38 | 5.68 | 19.57 |
| **IRGPT (CL, fine-tune)** | **85.12** | **99.79** | **51.58** | **98.69** | **50.79** | **99.82** | **485.79** | **3.32** | **0.25** | **0.82** | **4.39** |

零样本场景下，IRGPT 比 InternVL2-8B 基线 psum 提升 76.35，nsum 降低 31.56；微调后超越更大的 InternVL2-26B，psum 达 485.79（+12.01），nsum 仅 4.39（-15.18）。

### 消融实验：课程学习各组件效果

| L1 | anti-L1 | L2 | anti-L2 | α | psum | nsum |
|----|---------|----|---------|----|------|------|
| - | - | - | - | - | 464.49 | 26.85 |
| ✓ | - | - | - | - | 470.23 | 17.29 |
| - | - | ✓ | - | - | 466.81 | 19.33 |
| ✓ | - | ✓ | - | - | 481.36 | 7.08 |
| ✓ | - | ✓ | - | ✓ | **485.79** | **4.39** |
| - | ✓ | - | ✓ | ✓ | 433.24 | 26.50 |

Lesson 2（图文对齐）对 psum 更敏感，Lesson 1（图像质量）对 nsum 更敏感；反向课程（anti-CL）严重降低性能，验证了由简到难策略的有效性。

### 采样调度消融

| 调度策略 | zero-shot psum | fine-tune psum | fine-tune nsum |
|---------|----------------|----------------|----------------|
| Difficulty-Descending | 286.84 | 433.24 | 26.50 |
| Difficulty-Ascending | 309.63 | 474.29 | 19.24 |
| Random | 307.18 | 479.46 | 12.94 |
| **Ascending-Stratified Random** | **328.65** | **485.79** | **4.39** |

分层随机上升策略最优，兼顾了结构化学习进程和随机变异性。

## 亮点与洞察

1. **真实数据 vs 合成数据**：首次证明基于真实红外图像的大规模训练显著优于合成红外方法，IR-TD 的真实性是关键优势
2. **课程学习的有效设计**：双视角（跨域距离 + 文本对齐度）的难度度量互补性强——L1 关注图像质量，L2 关注语义正确性
3. **损失变化率的实用价值**：通过区分"困难但可学习"和"对齐错误"样本，自适应权重既保护困难样本又抑制噪声，比纯静态损失更鲁棒
4. **ReID 任务的突破**：零样本 ReID 从 7.33 提升到 33.55，说明课程学习帮助模型真正理解了红外行人身份特征

## 局限性

- 模型基于 InternVL2-8B，相对轻量，在极复杂推理任务上可能不如更大模型
- 域投影距离计算是近似方法，精确度依赖特征提取器质量
- IR-TD 的预训练子集中 LLM 生成的描述经过人工适配，规模化存在瓶颈
- 计数类任务（Loc/A.C./P.C.）在 zero-shot 下仍有较大误差，预训练阶段偏重对齐而非精确理解

## 相关工作与启发

- **Infrared-LLaVA**：最相关工作，但使用合成红外数据，存在模态幻觉问题
- **InfMAE**：红外掩码图像建模基础模型，本文用其特征提取器计算域距离
- **课程学习（Baby-step）**：分层随机采样策略直接继承自经典课程学习理论
- **启发**：跨模态迁移中的难度量化思路可推广到其他模态（如医学影像→自然图像、卫星图→航拍图等）

## 评分 ⭐⭐⭐⭐

- 创新性：⭐⭐⭐⭐ — 首个真实红外 MLLM，双跨模态课程学习设计新颖
- 实验充分度：⭐⭐⭐⭐⭐ — 9 任务基准、零样本/微调双评、完整消融
- 实用价值：⭐⭐⭐⭐ — 数据集开源且有实际安防/自动驾驶应用场景
- 写作质量：⭐⭐⭐⭐ — 结构清晰，数据集构建和方法动机阐述充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ECCV 2024\] XPSR: Cross-modal Priors for Diffusion-based Image Super-Resolution](../../ECCV2024/image_generation/xpsr_crossmodal_priors_for_diffusionbased_image_superresolut.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](../../CVPR2025/image_generation/orida_object-centric_real-world_image_composition_dataset.md)
- [\[CVPR 2025\] UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](../../CVPR2025/image_generation/unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)

</div>

<!-- RELATED:END -->
