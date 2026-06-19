---
title: >-
  [论文解读] DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding
description: >-
  [ICCV 2025][视频理解][视频表示] DynImg 提出了一种新颖的视频表示方法，将非关键帧作为"时序视觉提示"叠加在关键帧下方形成动态图像，在视觉编码器内部实现细粒度时空交互（而非高层token级交互），配合4D旋转位置编码维护正确的时空序列关系，在多个视频理解基准上以更少的视觉token超越SOTA约2%。
tags:
  - "ICCV 2025"
  - "视频理解"
  - "视频表示"
  - "时空交互"
  - "视觉提示"
  - "位置编码"
  - "多模态LLM"
---

# DynImg: Key Frames with Visual Prompts are Good Representation for Multi-Modal Video Understanding

**会议**: ICCV 2025  
**arXiv**: [2507.15569](https://arxiv.org/abs/2507.15569)  
**代码**: [https://dynimg.github.io/](https://dynimg.github.io/)  
**领域**: 视频理解 / 多模态大语言模型  
**关键词**: 视频表示, 时空交互, 视觉提示, 位置编码, 多模态LLM

## 一句话总结

DynImg 提出了一种新颖的视频表示方法，将非关键帧作为"时序视觉提示"叠加在关键帧下方形成动态图像，在视觉编码器内部实现细粒度时空交互（而非高层token级交互），配合4D旋转位置编码维护正确的时空序列关系，在多个视频理解基准上以更少的视觉token超越SOTA约2%。

## 研究背景与动机

多模态大语言模型（MLLM）在视频理解中的应用日益广泛，但如何有效融合时序信息仍是关键挑战：

- **现有方法的时空解耦问题**：传统方法将空间和时序信息分离处理——先用预训练图像编码器提取各帧空间特征，再在高层query/token级别进行时空交互（如Video-ChatGPT的并行时空池化、Video-LLaMA的Q-former+时序模块）
- **细粒度信息丢失**：在视觉特征提取和空间合并过程中（k-means、池化、Q-former、卷积），空间信息被持续抽象压缩，许多细粒度细节在平均化中模糊和丢失
- **快速运动区域被忽略**：由于运动模糊等因素，快速移动物体的区域在空间特征提取阶段未能获得准确的细粒度表示。这些时序上重要的区域若在初始阶段被忽略，后续的token级交互效果将大打折扣

核心矛盾是：**时空交互发生得太晚**——在高层抽象token上进行交互时，关键的细粒度运动信息已经丢失。本文的核心idea：**将时空交互前移到视觉特征提取阶段的像素级别**，利用非关键帧作为"时序提示"引导编码器关注快速运动区域。

## 方法详解

### 整体框架

视频被分解为关键帧和非关键帧。每个DynImg由1个高分辨率关键帧（上方）和4个缩小的非关键帧（下方一排，时序排列）组成。DynImg连同4D位置编码一起输入视觉编码器，编码器内部通过self-attention实现关键帧与时序提示之间的细粒度时空交互。输出特征经投影层后送入LLM。

### 关键设计

1. **Temporal Prompts（时序视觉提示）**:

    - 功能：在像素级别将时序信息融入空间特征提取过程
    - 核心思路：使用MPEG-4方法均匀采样4个I-frame作为关键帧 $K$。对每个关键帧 $K_i$，从前后区间各随机选取2个I-frame作为非关键帧 $N$。非关键帧缩小后拼接在关键帧下方形成DynImg。在ViT的self-attention中，关键帧的patch可以attend到时序提示中高度相似的patch，识别运动趋势
    - 设计动机：借鉴多模态图像理解中的视觉提示思想（如在图像上画框标注），但将其扩展到视频——用非关键帧提供运动变化信息，在编码器内部通过ViT的长程建模能力实现时空交互

2. **4D Video Rotary Position Embedding（4D视频旋转位置编码）**:

    - 功能：为DynImg中的视觉token维护正确的四维时空序列关系
    - 核心思路：在标准1D RoPE基础上扩展为4个维度——高度(H)、宽度(W)、时间(T)、序列(S)。角度计算为加权和：$x\cdot\theta = x_h\cdot\theta_h + x_w\cdot\theta_w + x_t\cdot\theta_t + x_s\cdot\theta_s$。其中 $\theta_s$ 保留LLM预训练的正弦编码，$\theta_h, \theta_w, \theta_t$ 初始化为0（可学习），确保训练初期不破坏预训练LLM的效果
    - 设计动机：DynImg的组合形式对LLM来说很陌生，直接使用会导致时空关系混乱。需要显式的时间维度坐标帮助建立跨帧空间特征的关联。关键帧时间坐标为0，前后非关键帧对称递增/递减

3. **DynImg的组成细节**:

    - 功能：确保patch级特征提取不会跨越帧边界
    - 核心思路：控制非关键帧缩放尺寸，使其为patch size的整数倍，避免拼接后的patch横跨两帧边界
    - 设计动机：跨帧边界的patch会混合不同帧的像素信息，引入噪声

### 损失函数 / 训练策略

- 视觉编码器：SigLip-so400m-384
- LLM：Qwen2.5-7B-Instruct，所有参数可训练
- 投影层：前馈层 + PLLaVA的自适应平均池化模块（pooling shape 16×12×12）
- 每个视频合成4个DynImg作为输入
- 训练数据：约737K视频-文本对（TextVR 39K, YouCook2 8K, VideoChat 7K, WebVid 400K, Kinetics-710 40K, SSv2 40K等）
- 训练方法遵循PLLaVA的训练配方

## 实验关键数据

### 主实验

**开放式视频QA（5个基准平均）：**

| 方法 | 编码器 | LLM | MSVD Acc | MSRVTT Acc | ActivityNet Acc | TGIF Acc | Video-ChatGPT Avg |
|------|--------|-----|----------|-----------|-----------------|----------|-------------------|
| PLLaVA | ViT-L | 7B | 76.6 | 62.0 | 56.3 | 77.5 | 3.12 |
| IG-VLM | Unk | GPT-4V | 76.3 | 63.8 | 57.0 | 65.3 | 3.17 |
| **DynImg** | **SigLip** | **7B** | **78.6** | **64.1** | **57.9** | **77.5** | **3.25** |

**MVBench多选QA（20类平均）：DynImg 55.8 vs PLLaVA 46.6 vs ST-LLM 54.9**

运动相关任务的突出提升：Moving Direction +21.0%，Moving Count +15.0%，Moving Attribute +26.5%

### 消融实验

**DynImg关键组件消融（MSVD）：**

| 时序提示 | 融合阶段 | 4D-RoPE | Acc | Score |
|---------|---------|---------|-----|-------|
| ✗ | - | - | 74.9 | 3.8 |
| ✓ | 编码器后 | - | 75.2 | 3.9 |
| ✓ | 编码器前 | ✗ | 77.3 | 4.0 |
| ✓ | 编码器前 | ✓ | **78.6** | **4.2** |

**非关键帧数量消融：**

| N-frame数 | 1 | 2 | 4 | 6 | 12 |
|-----------|---|---|---|---|-----|
| Acc | 71.9 | 72.3 | **78.6** | 78.1 | 77.5 |

4帧最优；太少无法提供有效时序变化，太多则帧尺寸过小损失信息。

**DynImg数量消融：4个DynImg最优（78.6），1个77.0，16个77.7。**

### 关键发现
- **编码器前 vs 编码器后**：提示在编码器前融合（+2.4%）远优于编码器后融合，证明了像素级时空交互的关键性——编码器后的提示仅增加LLM信息量，无法实现真正的时空交互
- **4D-RoPE至关重要**：从77.3到78.6（+1.3%），帮助LLM理解DynImg的组成逻辑
- **Token效率突出**：DynImg仅需576个视觉token（4帧），而PLLaVA需2304个token（16帧），在token数减少75%的情况下准确率更高

## 亮点与洞察
- **"时空交互前移"的核心洞察**：将交互从高层token推到底层像素，避免了特征压缩过程中的运动信息丢失
- **视觉提示的巧妙延伸**：从图像级视觉提示（标注框）到视频级时序提示（非关键帧叠加），思路自然而新颖
- **4D位置编码设计优雅**：初始化为0的可学习维度确保渐进训练，不破坏预训练权重
- **效率与效果兼得**：更少token达到更高精度，实际部署价值显著

## 局限与展望
- MPEG-4解码增加了数据加载时间（0.06s→0.32s），虽然相对训练时间可接受但仍需优化
- 仅探索了4个DynImg输入，对长视频场景的扩展性未验证
- 非关键帧缩小后分辨率受限，极细致的运动细节可能仍会丢失
- 当前仅在7B LLM上验证，更大模型的效果未知
- 关键帧选择依赖MPEG-4的I-frame，可能不是内容感知的最优选择

## 相关工作与启发
- **Video-ChatGPT**：时空并行池化架构，时空交互在高层token级别进行
- **Video-LLaVA**：统一图像和视频输入，使用对齐投影策略
- **PLLaVA**：提出自适应池化减少token冗余，是本文的主要基线
- **IG-VLM**：使用类漫画风格的图像网格，利用GPT-4V实现提升
- **Follow-your-pose / AniPortrait**：视觉提示在图像任务中的成功应用

## 评分
- 新颖性: ⭐⭐⭐⭐ 时序视觉提示的概念新颖，4D-RoPE设计精巧，但基本思路是视觉提示的延伸
- 实验充分度: ⭐⭐⭐⭐⭐ 5个开放式QA基准+MVBench+多维度消融，实验非常全面
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，图示直观，定量和定性分析完整
- 价值: ⭐⭐⭐⭐ 提供了视频理解中时空交互的新思路，对MLLM视频架构设计有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VideoMiner: Iteratively Grounding Key Frames of Hour-Long Videos via Tree-based Group Relative Policy Optimization](videominer_iteratively_grounding_key_frames_of_hour-long_videos_via_tree-based_g.md)
- [\[ICCV 2025\] AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning](aim_adaptive_inference_multimodal_llms_token_merging_pruning.md)
- [\[NeurIPS 2025\] MUVR: A Multi-Modal Untrimmed Video Retrieval Benchmark with Multi-Level Visual Correspondence](../../NeurIPS2025/video_understanding/muvr_a_multi-modal_untrimmed_video_retrieval_benchmark_with_multi-level_visual_c.md)
- [\[ICCV 2025\] Multi-modal Multi-platform Person Re-Identification: Benchmark and Method](multi-modal_multi-platform_person_re-identification_benchmark_and_method.md)
- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](4dbench_benchmarking_multimodal_large_language_models_for_4d.md)

</div>

<!-- RELATED:END -->
