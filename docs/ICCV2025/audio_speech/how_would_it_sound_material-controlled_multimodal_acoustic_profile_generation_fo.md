---
title: >-
  [论文解读] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects
description: >-
  [ICCV 2025][音频/语音][房间脉冲响应] 提出材质可控的声学特征生成任务（M-CAPA），给定室内场景的音视觉观测和用户定义的新材质配置，生成反映材质变化的目标房间脉冲响应（RIR），并构建了配套的 Acoustic Wonderland 数据集。 声音传播受房间几何结构和 物体/表面材质：的显著影响——同一房间…
tags:
  - "ICCV 2025"
  - "音频/语音"
  - "房间脉冲响应"
  - "材质控制"
  - "音视觉学习"
  - "RIR生成"
  - "声学模拟"
---

# How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects

**会议**: ICCV 2025  
**arXiv**: [2508.02905](https://arxiv.org/abs/2508.02905)  
**代码**: [项目页面](https://mahnoor-fatima-saad.github.io/m-capa.html)  
**领域**: 音频语音  
**关键词**: 房间脉冲响应, 材质控制, 音视觉学习, RIR生成, 声学模拟

## 一句话总结

提出材质可控的声学特征生成任务（M-CAPA），给定室内场景的音视觉观测和用户定义的新材质配置，生成反映材质变化的目标房间脉冲响应（RIR），并构建了配套的 Acoustic Wonderland 数据集。

## 研究背景与动机

声音传播受房间几何结构和 **物体/表面材质** 的显著影响——同一房间中，木墙和混凝土墙产生截然不同的混响特征。准确的房间脉冲响应（RIR）建模对 AR/VR、游戏、建筑声学设计至关重要。

现有 RIR 预测方法的局限：

**物理仿真法**（如光线追踪）需要精细的 3D mesh 和材质标注，获取成本高、扩展性差。

**数据驱动法** 多从图像/音频/房间尺寸预测 RIR，但通常 **忽略材质属性**，将房间简化为矩形盒子或仅用 RGB 隐式推断材质。
3. 少数考虑材质的方法 [AV-RIR, Listen2Scene] 要么需要密集采样和 3D 重建，要么使用固定的语义类-材质映射（如"所有墙壁=砖头"），**无法在推理时灵活修改材质配置**。

本文提出的新任务：给定场景的原始音视觉观测 $(V, A_S)$ 和用户指定的目标材质 mask $\mathcal{M}_T$，生成新 RIR $A_T$。用户可在推理时动态调整材质（如将地板换成地毯、墙壁换成玻璃），无需实际改造房间。

## 方法详解

### 整体框架

M-CAPA 模型由三部分组成：**多模态场景编码器** $f^E$（编码音视觉特征）→ **目标材质编码器** $f^M$（编码新材质配置）→ **条件 RIR 生成器** $f^T$（融合两者生成目标 RIR）。

### 关键设计

1. **多模态场景编码器**  

    - 视觉编码器 $f^V$: 四层卷积 UNet 编码器处理 256×256 RGB 图像 $V_n$，输出视觉嵌入 $e_v$。
    - 语义编码器 $f^G$: 相同结构处理语义分割 mask $G_n$，输出语义嵌入 $e_g$。
    - 声学编码器 $f^A$: 四层卷积 UNet 编码器处理双声道频谱图 $A_S \in \mathbb{R}^{2 \times F \times T}$（STFT 变换），输出声学嵌入 $e_a$。
    - 三者拼接得到多模态嵌入 $e_m = [e_v; e_g; e_a]$。

   设计洞察：仅使用 90° FoV 的 RGB 即可，因为 **回声响应本身已捕获整个房间的声学信息**（包括视野外的区域）。

2. **目标材质编码器**  
   将目标材质 mask $\mathcal{M}_T \in \mathbb{R}^{H \times W}$（每个像素为材质类别索引）通过卷积编码器映射为嵌入 $e_t$。用户只需在语义分割图上点选物体并分配材质类别即可生成 $\mathcal{M}_T$。

3. **条件 RIR 生成器（核心创新）**  
   融合层 $\mathcal{F}$ 合并 $e_m$ 和 $e_t$，通过四层转置卷积解码器（带 $f^A$ 的 skip connection）输出两个张量：
    - **加权 mask** $W_T \in \mathbb{R}^{2 \times F \times T}$: 控制源 RIR 中哪些频率/时间 bin 的混响需要增强/抑制
    - **材质残差** $B_T \in \mathbb{R}^{2 \times F \times T}$: 引入源 RIR 中不存在的新混响模式

   最终生成：$\hat{A}_T = W_T \odot A_S + B_T$

   **关键动机**：传统 masking 方法只能调整已有频率-时间 bin 的强度，但新材质可能在之前"沉默"的 bin 上引入全新混响。残差项 $B_T$ 解决了这一问题。消融实验证实 $B_T$ 对 RTE 和 CTE 指标贡献显著。

### 损失函数 / 训练策略

$$L_n = \lambda_1 \|{\hat{A}_T - A_T}\|_2 + \lambda_2 \|{\hat{A}_T - A_T}\|_1 + \lambda_3 L_D(\hat{A}_T, A_T)$$

- L2 损失 + L1 损失：捕获频谱细节误差
- 能量衰减损失 $L_D$：对齐预测和真实 RIR 的时间能量衰减曲线，提升混响质量
- $\lambda_1 = \lambda_2 = 0.5$, $\lambda_3 = 5 \times 10^{-3}$
- Adam 优化器，学习率 $10^{-3}$，batch size 64，单 GPU 训练

## 实验关键数据

### 主实验（表格）

**未见场景上的 RIR 生成性能（Du_u 测试集，×10⁻²）**

| 方法 | 输入 | L1↓ | STFT↓ | RTE(ms)↓ | CTE(dB)↓ |
|------|------|-----|-------|---------|---------|
| Direct Mapping | $A_S$ | 7.47 | 7.10 | 119.7 | 12.78 |
| Image2Reverb | $V$ | 14.13 | 7.59 | 223.4 | 19.15 |
| FAST-RIR++ | $A_S$ | 14.81 | 28.39 | 231.8 | 16.83 |
| Material Aware | $V$ | 8.91 | 11.29 | 98.06 | 11.75 |
| AV-RIR | $A_S$+$V$ | 7.59 | 7.17 | 99.10 | 11.35 |
| **M-CAPA (ours)** | **$A_S$+$V$** | **5.27** | **3.87** | **91.44** | **8.44** |

M-CAPA 在所有指标上大幅优于全部基线和 SOTA 方法。即使是仅用视觉的 M-CAPA 变体（L1=6.06）也超过使用音视觉输入的 AV-RIR。

### 消融实验（表格）

**模型组件消融（Du_u 测试集）**

| 配置 | L1↓ | STFT↓ | RTE(ms)↓ | CTE(dB)↓ |
|------|-----|-------|---------|---------|
| M-CAPA 完整模型 | **5.27** | **3.87** | **91.44** | **8.44** |
| 去掉 $\mathcal{M}_T$ | 5.61 | 4.06 | 109.46 | 9.19 |
| 去掉 $B_T$ (仅masking) | 5.75 | 4.93 | 105.19 | 10.83 |
| 使用推断的 $G_n$ | 5.63 | 3.99 | 97.63 | 9.10 |
| 仅提供变化材质 | 5.47 | 4.00 | 96.36 | 9.04 |

去掉残差项 $B_T$ 导致 CTE 恶化 2.39 dB，证实了"纯 masking 不够"的论点。

### 关键发现

- 材质变化覆盖面积在 50%-70% 时误差最低（对应墙壁、地板等大平面），极小面积变化（如一把椅子）反而更难预测。
- 不同材质的难度差异大：布料和吸音砖较易预测，钢铁和木材较难（可能因其在多频段的复杂反射/吸收特性）。
- 真实场景用户研究：5 名用户从语音+预测 RIR 中辨识目标材质的准确率为 61.1%（随机基线 33%），验证了模型在真实场景的泛化能力。
- 模型极其轻量：仅 10.56M 参数、17.98 GFLOPs，推理时间 114ms，而 AV-RIR 为 390.66M 参数。

## 亮点与洞察

- 首个允许推理时任意修改材质配置的 RIR 生成方法，填补了声学模拟中的交互式编辑空白。
- $W_T \odot A_S + B_T$ 的生成公式简洁优雅，将已有混响的调整和新混响的引入解耦。
- Acoustic Wonderland 数据集（168 万数据点，含 2673 种材质配置）为社区提供了系统评估材质-声学关系的新基准。
- 用户交互设计直觉：通过语义分割 mask 点选物体分配材质，无需像素级标注。

## 局限与展望

- 对形状高度不规则的物体（如穹顶、复杂柱形结构），材质变化的声学影响预测较差。
- 推理时无法引入训练时未见过的新材质类别（12 种固定材质类别）。
- 仅验证了仿真数据，真实场景用户研究规模较小（2 个场景、5 人）。
- 噪声鲁棒性不足：源 RIR 加噪后性能下降，未来可用带噪训练增强。
- 材质的频率相关特性（不同频段吸收率差异）可进一步建模。

## 相关工作与启发

- AV-RIR [CVPR 2024] 是本文最直接的对比方法，但其使用固定材质-语义映射且需从训练集检索晚期混响。
- Image2Reverb [ICCV 2021] 首次从 RGB+深度生成完整 RIR，但忽略材质信息。
- 本任务有广泛应用前景：室内设计声学预览、VR/AR 沉浸式体验、录音棚材质规划等。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （首创材质可控 RIR 生成任务，配套数据集和方法完整）
- 实验充分度: ⭐⭐⭐⭐ （多基线对比、多分割评估、消融详尽、含用户研究）
- 写作质量: ⭐⭐⭐⭐ （问题定义清晰，方法描述规范）
- 价值: ⭐⭐⭐⭐ （对 AR/VR 声学渲染和室内声学设计有直接应用价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](../../CVPR2025/audio_speech/video-guided_foley_sound_generation_with_multimodal_controls.md)
- [\[CVPR 2026\] How Far Can We Go With Synthetic Data for Audio-Visual Sound Source Localization?](../../CVPR2026/audio_speech/how_far_can_we_go_with_synthetic_data_for_audio-visual_sound_source_localization.md)
- [\[ICCV 2025\] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)
- [\[NeurIPS 2025\] Resounding Acoustic Fields with Reciprocity](../../NeurIPS2025/audio_speech/resounding_acoustic_fields_with_reciprocity.md)
- [\[ECCV 2024\] Beat-It: Beat-Synchronized Multi-Condition 3D Dance Generation](../../ECCV2024/audio_speech/beat-it_beat-synchronized_multi-condition_3d_dance_generation.md)

</div>

<!-- RELATED:END -->
