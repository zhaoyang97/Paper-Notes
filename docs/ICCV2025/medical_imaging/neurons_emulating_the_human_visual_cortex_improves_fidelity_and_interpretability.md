---
title: >-
  [论文解读] NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction
description: >-
  [ICCV 2025][医学图像][fMRI-to-Video] 提出 NEURONS 框架，受人类视觉皮层层级结构启发，将 fMRI 到视频的重建解耦为四个子任务（关键物体分割、概念识别、场景描述、模糊视频重建），模拟 V1/V2/V4/ITC 等脑区的功能特化，在视频一致性（26.6%）和语义准确度（19.1%）上显著超越 SOTA。
tags:
  - "ICCV 2025"
  - "医学图像"
  - "fMRI-to-Video"
  - "Brain Decoding"
  - "Visual Cortex"
  - "扩散模型"
  - "Neuroscience"
---

# NEURONS: Emulating the Human Visual Cortex Improves Fidelity and Interpretability in fMRI-to-Video Reconstruction

**会议**: ICCV 2025  
**arXiv**: [2503.11167](https://arxiv.org/abs/2503.11167)  
**代码**: [https://github.com/xmed-lab/NEURONS](https://github.com/xmed-lab/NEURONS)  
**领域**: 医学图像  
**关键词**: fMRI-to-Video, Brain Decoding, Visual Cortex, diffusion model, Neuroscience

## 一句话总结

提出 NEURONS 框架，受人类视觉皮层层级结构启发，将 fMRI 到视频的重建解耦为四个子任务（关键物体分割、概念识别、场景描述、模糊视频重建），模拟 V1/V2/V4/ITC 等脑区的功能特化，在视频一致性（26.6%）和语义准确度（19.1%）上显著超越 SOTA。

## 研究背景与动机

从脑活动解码视觉刺激是理解人脑的关键路径。fMRI-to-image 已取得成功（利用 CLIP + Stable Diffusion），但 fMRI-to-video 仍面临巨大挑战：

**时空动态复杂**：视频需要捕获物体运动、场景转换、时间一致性，远超静态图像

**现有方法的隐式对齐问题**：
   - MinD-Video：用视觉 fMRI 特征条件化扩散模型，但缺乏低级视觉细节
   - NeuroClips：引入语义和感知重建器，但主要依赖在 CLIP 隐藏空间的隐式对齐
   - CLIP 隐藏空间高度语义化，但 fMRI 体素编码了多粒度信息，隐式对齐不鲁棒

核心 insight：人类视觉皮层有明确的功能分区（V1/V2 处理边缘/形状，V4/ITC 处理物体/面部识别等）。直接模拟这种层级结构，将学习分解为多个显式子任务，可以更好地解码不同粒度的视觉信息。

## 方法详解

### 整体框架

NEURONS 由三部分组成：
1. **Brain Model**：将 fMRI 表示映射到运动嵌入（预训练阶段）
2. **Decoupler**：将运动嵌入解耦为四个显式子任务的渐进式学习
3. **Aggregated Video Reconstruction**：推理时整合所有子任务输出引导 T2V 扩散模型

### 关键设计

1. **解耦任务构建（利用现成模型自动生成标签）**：

    - **场景描述生成**：Qwen2.5-VL-72B 为每帧生成描述性标题
    - **概念名称生成**：Qwen 识别帧中主要物体，归类到 WordNet 总结的 51 个概念类别
    - **概念分割掩码**：Grounded-SAM 用物体名称作为文本 prompt 生成每帧的二值掩码
    - **关键物体发现**：基于运动动态（帧间位移）、物体大小、语义重要性的多标准方法。排除背景类别，优先选择人/动物等语义重要类别

   设计动机：由粗到细，从最少语义信息的分割开始，逐步引入概念、描述、运动信息，模拟视觉皮层的层级处理。

2. **Brain Model**：

    - 基于预训练的 MindEyeV2 backbone
    - 引入 motion projection $\mathcal{P}_{vid}(\cdot)$ 将图像嵌入 $e^i \in \mathbb{R}^{B \times N \times C}$ 映射到时空空间 $e^v \in \mathbb{R}^{B \times F \times N \times C}$（与 NeuroClips 不同，考虑帧间时序关系）
    - 用 BiMixCo 对比损失对齐 $e^v$ 与 CLIP 视觉编码器的目标视频嵌入
    - 额外将 $e^v$ 嵌入为文本嵌入 $e^t$，与 CLIP 文本嵌入做对比学习

   设计动机：预训练阶段产生合适的视觉和文本嵌入，为后续解耦任务提供基础。

3. **Decoupler - 四个子任务**：

    - **Key Object Segmentation**：设计文本驱动的 VAE 视频解码器，以视频嵌入 $e^v$ 和关键物体类名的 CLIP 文本嵌入为输入，通过交叉注意力激活对应 patch，上采样后用分割头生成二值掩码。BCE 损失 $\mathcal{L}_{seg}$
    - **Concept Recognition**：多标签分类器 $\mathcal{D}_{cls}$ 识别帧中的概念，用视觉嵌入的帧维均值作为输入。交叉熵损失 $\mathcal{L}_{cls}$
    - **Scene Description**：微调 GPT-2 文本解码器，以文本嵌入 $e^t$ 为前缀生成标题。前缀语言建模损失 $\mathcal{L}_{txt}$
    - **Blurry Video Reconstruction**：复用分割的 VAE 解码器但替换为重建头，生成模糊视频 $y_c^{rec}$，与 SD VAE 的潜在嵌入对齐。MAE 损失 $\mathcal{L}_{rec}$

4. **渐进式学习策略**：
   正弦权重调度 $w = 1 + 9 \cdot |\sin(\frac{C}{T} \cdot \pi)|$，四个损失函数的"上升期"交错排列，确保平滑过渡：先强调分割 → 概念识别 → 场景描述 → 视频重建

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = w_1 \mathcal{L}_{seg} + w_2 \mathcal{L}_{cls} + w_3 \mathcal{L}_{txt} + w_4 \mathcal{L}_{rec}$$

其中 $w_1$-$w_4$ 按正弦调度交错变化（范围 [1, 10]），确保从简单到复杂的渐进学习。

**推理**：关键物体掩码缩放到 [0.5, 1] 并乘到控制图像和模糊视频上强调关键物体 → T2V 扩散模型（AnimateDiff）生成最终视频。

**数据集**：cc2017 开源 fMRI-视频数据集，3T MRI，2s 时间分辨率。8640 训练样本 + 1200 测试样本。

## 实验关键数据

### 主实验 (表格)

**视频级定量对比：**

| 方法 | 2-way ↑ | 50-way ↑ | CLIP-pcc ↑ |
|------|---------|----------|------------|
| Wen | - | 0.166 | - |
| Wang | 0.773 | - | 0.402 |
| Kupershmidt | 0.771 | - | 0.386 |
| MinD-Video | 0.839 | 0.197 | 0.408 |
| MindAnimator | 0.830 | - | 0.428 |
| NeuroClips | 0.834 | 0.220 | 0.738 |
| **NEURONS** | **0.863** | **0.262** | **0.934** |

**各被试结果：**

| 被试 | 2-way ↑ | 50-way ↑ | CLIP-pcc ↑ |
|------|---------|----------|------------|
| Subject 1 | 0.862 | 0.254 | 0.932 |
| Subject 2 | 0.860 | 0.252 | 0.933 |
| Subject 3 | 0.868 | 0.278 | 0.937 |

### 消融实验 (表格)

**关键组件消融（Subject 1）：**

| Brain Model | $\mathcal{L}_{seg}$ | $\mathcal{L}_{cls}$ | $\mathcal{L}_{txt}$ | $\mathcal{L}_{rec}$ | PL | AVR | 2-way ↑ | 50-way ↑ | CLIP-pcc ↑ |
|:-----------:|:---:|:---:|:---:|:---:|:--:|:---:|---------|----------|------------|
| ✓ |  |  |  | ✓ |  |  | 0.814 | 0.164 | 0.894 |
| ✓ | ✓ |  |  | ✓ |  |  | 0.834 | 0.225 | 0.926 |
| ✓ | ✓ | ✓ |  | ✓ |  |  | 0.836 | 0.234 | 0.911 |
| ✓ | ✓ | ✓ | ✓ | ✓ |  |  | 0.847 | 0.213 | 0.923 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |  | 0.856 | 0.235 | 0.937 |
| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **0.862** | **0.254** | **0.932** |

**概念识别准确率（Top-5 类别）：**

| 排名 | 类别 | 准确率 |
|------|------|--------|
| 1 | human | 0.735 |
| 2 | food | 0.600 |
| 3 | animal | 0.450 |
| 4 | water body | 0.310 |
| 5 | fish | 0.292 |

**场景描述对比：**

| 方法 | BLEU-1 | BLEU-4 | CIDEr | Verb Acc |
|------|--------|--------|-------|----------|
| NeuroClips | 0.227 | 0.022 | 0.156 | 0.116 |
| **NEURONS** | **0.238** | **0.036** | **0.239** | **0.243** |

### 关键发现

1. **CLIP-pcc 提升 26.6%**（0.738 → 0.934）：时空一致性大幅改善，解耦学习有效捕获运动信息
2. **50-way 准确率提升 19.1%**（0.220 → 0.262）：语义精度显著提高
3. **分割任务贡献最大**：加入 $\mathcal{L}_{seg}$ 后 50-way 从 0.164 跳到 0.225（+37%），CLIP-pcc 从 0.894 到 0.926
4. **渐进式学习关键**：加入 PL 后 2-way 从 0.847 到 0.856，50-way 从 0.213 到 0.235
5. **动词准确率翻倍**（0.116 → 0.243）：直接从 fMRI 生成描述比从生成关键帧再 caption 更可靠
6. **功能映射验证设计灵感**：分割嵌入对应 V1/V2/MT，概念识别对应 V4/EBA/LOC，场景描述对应 PPA/FFA，完美匹配设计初衷

## 亮点与洞察

- **神经科学启发 → 工程设计 → 反过来验证神经科学**：从视觉皮层层级结构出发设计子任务，最终脑图映射验证了功能对齐，形成美妙闭环
- **显式解耦 vs 隐式对齐**：将多粒度信息显式分离到不同子任务，比在 CLIP 隐藏空间隐式对齐更鲁棒
- **渐进式正弦权重调度**：简单优雅的训练策略，确保从简单到复杂的平滑过渡
- **利用 VLM 生成训练标签**：Qwen2.5-VL-72B + Grounded-SAM 自动构建所有子任务标签
- **关键物体掩码增强**：推理时将掩码乘到条件信号上强调关键物体，简单有效

## 局限与展望

1. 关键物体分割的 Dice 分数较低（35.63%），主要因为测试集中有大量未见类别
2. fMRI 时间分辨率仅 2s，难以捕获快速视觉变化
3. 依赖预训练的 MindEyeV2 backbone 和 AnimateDiff T2V 模型的能力
4. 被试训练样本有限（每人 8640 样本），跨被试泛化仍是挑战
5. 生成的模糊视频分辨率较低（$H/8 \times W/8$），细节恢复依赖 T2V 模型
6. 51 个概念类别可能不足以覆盖所有视觉内容

## 相关工作与启发

- **fMRI-to-image 系列**：Mind-Reader、MindEye、BrainDiffuser 等为静态重建奠定基础
- **NeuroClips**：直接前身，提出语义+感知双重建器，但隐式对齐不鲁棒
- **视觉皮层功能分区研究**：V1/V2 (边缘/形状) → V4/ITC (物体识别) → PPA/FFA (场景/面部) 的神经科学发现直接启发了子任务设计
- 对脑机接口和临床应用（如帮助失语症患者表达视觉体验）有潜在价值

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将视觉皮层层级结构显式映射为解耦子任务，设计理念具有启发性
- **实验充分度**: ⭐⭐⭐⭐ 视觉重建+子任务评估+脑图映射，多角度验证；但仅一个数据集
- **写作质量**: ⭐⭐⭐⭐⭐ 从神经科学动机到工程设计到验证的叙事逻辑极佳
- **价值**: ⭐⭐⭐⭐ 在 fMRI-to-video 领域取得显著突破，且脑图映射结果对神经科学研究也有价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Meta-Learning an In-Context Transformer Model of Human Higher Visual Cortex](../../NeurIPS2025/medical_imaging/meta-learning_an_in-context_transformer_model_of_human_higher_visual_cortex.md)
- [\[CVPR 2026\] Bridging Brain and Semantics: A Hierarchical Framework for Semantically Enhanced fMRI-to-Video Reconstruction](../../CVPR2026/medical_imaging/bridging_brain_and_semantics_a_hierarchical_framework_for_semantically_enhanced_.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](../../ICLR2026/medical_imaging/lavca_llm-assisted_visual_cortex_captioning.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)

</div>

<!-- RELATED:END -->
