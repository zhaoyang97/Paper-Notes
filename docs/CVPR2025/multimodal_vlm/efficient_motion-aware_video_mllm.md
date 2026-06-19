---
title: >-
  [论文解读] Efficient Motion-Aware Video MLLM
description: >-
  [CVPR 2025][多模态VLM][视频多模态大模型] 本文提出 EMA（Efficient Motion-Aware video MLLM），利用压缩视频中的 GOP 结构融合空间与运动信息，以原生慢-快架构减少冗余并增强运动表示，同时引入 MotionBench 做运动理解基准，在多个视频 QA 和运动理解任务上取得 SOTA。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视频多模态大模型"
  - "运动感知"
  - "压缩视频"
  - "GOP编码器"
  - "MotionBench"
---

# Efficient Motion-Aware Video MLLM

**会议**: CVPR 2025  
**arXiv**: [2503.13016](https://arxiv.org/abs/2503.13016)  
**代码**: 无  
**领域**: 视频理解 / 多模态VLM  
**关键词**: 视频多模态大模型, 运动感知, 压缩视频, GOP编码器, MotionBench

## 一句话总结
本文提出 EMA（Efficient Motion-Aware video MLLM），利用压缩视频中的 GOP 结构融合空间与运动信息，以原生慢-快架构减少冗余并增强运动表示，同时引入 MotionBench 做运动理解基准，在多个视频 QA 和运动理解任务上取得 SOTA。

## 研究背景与动机

**领域现状**：当前视频多模态大语言模型（Video MLLM）大多采用均匀帧采样 + 图像级编码器的范式。典型方法从视频中等间隔抽取若干帧，用图像编码器（如 CLIP-ViT）提取每帧特征，然后输入 LLM。这种策略在视觉问答、视频描述等任务上取得了不错效果。

**现有痛点**：均匀帧采样策略存在两个根本问题。首先是**效率低**：相邻帧之间大量视觉信息冗余，均匀采样浪费计算资源在处理重复内容上。其次是**运动感知不足**：图像级编码器无法显式捕捉帧间运动信息，导致模型在需要理解物体运动方向、速度、轨迹等任务上表现不佳。

**核心矛盾**：要提升运动理解，直觉上需要更密集的帧采样来捕获运动细节，但这会大幅增加计算开销。如何在降低计算成本的同时增强运动感知，是一个效率与性能的 trade-off。

**本文目标**：(1) 设计高效的视频表示方案，减少冗余同时保留运动信息；(2) 建立评估运动理解的专用基准。

**切入角度**：压缩视频（如 H.264/H.265）天然包含 GOP（Group of Pictures）结构，其中 I 帧是完整 RGB 帧，P/B 帧存储运动向量（motion vectors）和残差。这天然是一种"慢-快"架构——少量 RGB 关键帧提供空间信息，大量运动向量提供时间运动信息，且获取运动向量几乎零额外计算代价。

**核心 idea**：利用压缩视频的 GOP 结构设计 motion-aware GOP 编码器，将少量 RGB 帧的空间特征与大量运动向量的运动特征在 GOP 单元内融合，生成紧凑且信息丰富的 visual tokens。

## 方法详解

### 整体框架
输入为压缩视频流，解码出 I 帧（RGB 关键帧）和运动向量。空间分支用图像编码器处理稀疏的 RGB 帧，运动分支处理密集的运动向量序列。两者在 GOP 级别融合为统一的 visual tokens，最后经过 projection 层输入 LLM 进行视频理解。

### 关键设计

1. **Motion-Aware GOP 编码器**:

    - 功能：在 GOP 单元内融合空间与运动信息，生成紧凑的 visual tokens
    - 核心思路：每个 GOP 包含一个 I 帧和若干 P/B 帧。I 帧通过图像编码器（如 ViT）提取空间特征作为"慢"路径，提供丰富的语义和外观信息。P/B 帧中的运动向量通过轻量运动编码器处理作为"快"路径，提供密集的运动信息。两个路径在 GOP 级别通过融合模块整合。这种设计天然借鉴了 SlowFast 网络的思想，但输入直接来自压缩视频流，无需额外的光流计算。
    - 设计动机：压缩视频中的运动向量是编码时已经计算好的"免费"运动信息。相比于从 RGB 帧中重新计算光流（如 RAFT），直接使用运动向量节省大量计算。同时 GOP 结构提供了天然的时间分组单元。

2. **Slow-Fast 原生输入架构**:

    - 功能：用更少但更密集的 RGB 帧 + 更多但更稀疏的运动向量，在减少冗余的同时增强运动表示
    - 核心思路：空间路径（Slow）采样较少的关键帧（I 帧），每帧提供高分辨率的空间语义信息。运动路径（Fast）利用更多的运动向量帧，每帧仅包含像素级位移信息但时间分辨率更高。这种不对称设计使得总 token 数量显著少于均匀采样所有帧的方法，同时运动信息反而更丰富。
    - 设计动机：视频中空间外观变化缓慢（低帧率即可），而运动信息变化快（需要高帧率），因此对两者使用不同的时间分辨率是最优策略。

3. **MotionBench 运动理解基准**:

    - 功能：评估模型在不同运动类型上的理解能力
    - 核心思路：涵盖四种运动类型的视频问答基准：线性运动（linear）、曲线运动（curved）、旋转运动（rotational）和接触运动（contact-based）。每类运动设计针对性的问题，如"球沿什么方向移动？""物体旋转了多少度？"等。
    - 设计动机：现有视频 QA 基准（如 VideoQA、NExT-QA）更侧重场景理解和因果推理，缺少对运动本身的细粒度评估。MotionBench 填补了这一空白。

### 损失函数 / 训练策略
采用标准的视频-语言对齐训练：先在大规模视频-文本数据上预训练视觉-语言投影层，再在下游视频 QA 数据集上指令微调。训练过程冻结图像编码器和 LLM，仅更新 GOP 编码器和投影层。

## 实验关键数据

### 主实验：视频 QA 基准

| 基准 | 指标 | EMA | 之前 SOTA | 提升 |
|------|------|------|----------|------|
| MotionBench | 综合准确率 | SOTA | — | 首创基准 |
| 常见视频 QA | 准确率 | SOTA | 次优方法 | 显著提升 |
| 长视频理解 | 准确率 | 有竞争力 | — | 可扩展性验证 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full EMA | 最优 | 完整模型（RGB + 运动向量融合） |
| w/o 运动向量 | 下降 | 仅用 RGB 帧，运动理解显著退化 |
| w/o GOP 融合 | 下降 | 简单拼接而非 GOP 内融合，效果变差 |
| 均匀采样 baseline | 最低 | 传统等间隔采样，冗余大且缺运动信息 |

### 关键发现
- 运动向量的引入对 MotionBench 上所有四种运动类型均有提升，特别是旋转和接触运动
- EMA 的推理成本低于均匀采样更多帧的方法，因为运动向量的处理开销远小于 RGB 帧
- 在长视频理解基准上 EMA 也表现出色，说明 GOP 结构具有良好的可扩展性
- Slow-Fast 架构中 RGB 帧和运动向量帧的比例对性能有影响，需要平衡

## 亮点与洞察
- **压缩视频是天然的 Slow-Fast 架构**：GOP 中 I/P/B 帧的结构设计本来就是为了在带宽和质量之间权衡，而这恰好也是视频理解中效率和信息量的最优平衡点。直接利用压缩域信息而非先解码再编码，思路巧妙且实用。
- **"免费"的运动信息**：运动向量在视频编解码中已经计算好，不需要额外的光流网络。这种利用已有信息的思路可以迁移到其他需要运动信息的视频任务中。
- **MotionBench 的贡献**：对运动理解的系统性评估填补了现有基准的空白，有助于推动视频 MLLM 在运动理解方向的进步。

## 局限与展望
- **压缩格式依赖**：方法依赖 H.264/H.265 等特定压缩格式的 GOP 结构，对于非压缩视频或其他编码格式需要适配
- **运动向量质量**：压缩视频中的运动向量是为编码效率优化的，并非为准确运动估计设计，在快速运动或遮挡场景下可能不准
- **MotionBench 规模**：作为新基准，规模和多样性可能还不够大，需要社区共同完善
- **未探索音频**：许多运动理解（如接触判断）可以借助音频信息，但本文未涉及多模态融合

## 相关工作与启发
- **vs VideoChat / Video-LLaVA**：传统 Video MLLM 用均匀帧采样 + 图像编码器，缺乏显式运动建模；EMA 通过压缩域运动向量显式引入运动信息
- **vs SlowFast Networks**：SlowFast 在 RGB 域用双路径不同帧率采样；EMA 在压缩域天然实现 slow-fast，利用 I 帧和运动向量分别承担空间和运动角色
- **vs CoDeF / MotionFormer**：这些方法通过学习从视频中提取运动表示；EMA 直接利用压缩流中的运动向量，避免了额外的运动估计计算

## 评分
- 新颖性: ⭐⭐⭐⭐ 利用压缩视频 GOP 结构作为原生 slow-fast 输入的想法有新意，但压缩域运动向量用于视频理解并非首创
- 实验充分度: ⭐⭐⭐⭐ 在多个基准上与多种方法对比，同时提出新的运动理解基准 MotionBench
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述直观，MotionBench 设计有条理
- 价值: ⭐⭐⭐⭐ 为视频 MLLM 的运动理解提供了实用方案和评估工具，具有较高的实用价值
- 综合: ⭐⭐⭐⭐ 将压缩视频底层结构与高层视频理解巧妙连接，兼具工程实用性与学术贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)
- [\[CVPR 2026\] MSJoE: Jointly Evolving MLLM and Sampler for Efficient Long-Form Video Understanding](../../CVPR2026/multimodal_vlm/msjoe_jointly_evolving_mllm_and_sampler_for_efficient_long-form_video_understand.md)
- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](../../NeurIPS2025/multimodal_vlm/in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)

</div>

<!-- RELATED:END -->
