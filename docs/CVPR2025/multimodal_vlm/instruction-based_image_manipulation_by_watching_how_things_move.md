---
title: >-
  [论文解读] Instruction-based Image Manipulation by Watching How Things Move
description: >-
  [CVPR 2025][多模态VLM][指令图像编辑] 本文提出 InstructMove，通过从视频中采样帧对并用多模态大模型生成编辑指令来构建大规模真实图像编辑数据集，结合空间条件化策略微调 T2I 模型，在姿态调整、视角变换等非刚性编辑任务上实现了 SOTA 效果。 1. 领域现状：文本驱动图像编辑领域中…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "指令图像编辑"
  - "视频帧对"
  - "空间条件化"
  - "非刚性变换"
  - "多模态大模型"
---

# Instruction-based Image Manipulation by Watching How Things Move

**会议**: CVPR 2025  
**arXiv**: [2412.12087](https://arxiv.org/abs/2412.12087)  
**代码**: [https://github.com/mingdeng-cao/InstructMove](https://github.com/mingdeng-cao/InstructMove)  
**领域**: 扩散模型 / 图像编辑  
**关键词**: 指令图像编辑, 视频帧对, 空间条件化, 非刚性变换, 多模态大模型

## 一句话总结
本文提出 InstructMove，通过从视频中采样帧对并用多模态大模型生成编辑指令来构建大规模真实图像编辑数据集，结合空间条件化策略微调 T2I 模型，在姿态调整、视角变换等非刚性编辑任务上实现了 SOTA 效果。

## 研究背景与动机

1. **领域现状**：文本驱动图像编辑领域中，InstructPix2Pix 等方法通过语言模型生成编辑指令、再用 tuning-free 方法生成目标图像来构建训练数据集，但目标图像是合成的，质量受限。
2. **现有痛点**：现有数据集中目标图像由 Prompt-to-Prompt 等方法合成生成，存在严重的外观偏差和伪影，导致模型难以处理复杂的非刚性编辑（如姿态变化、视角调整）和精细的内容保持。
3. **核心矛盾**：合成目标图像的质量天花板限制了编辑模型的能力上限——合成图像无法捕获真实世界中自然的形变和运动。
4. **本文目标** (a) 如何构建包含真实目标图像的大规模编辑数据集？(b) 如何在编辑时既灵活改变结构又保持原图内容？
5. **切入角度**：视频帧天然捕获了物体运动（姿态变化、元素移动、相机运动），且帧间保持了主体身份一致性，是理想的编辑数据源。
6. **核心 idea**：用视频帧对作为真实的 source-target 图像对，用 MLLM 生成编辑指令，并通过空间维度拼接（而非通道拼接）来条件化参考图像，实现灵活且高保真的指令驱动图像编辑。

## 方法详解

### 整体框架
整个方法分为两大部分：数据构建和模型训练。数据构建从互联网视频中采样帧对，经过光流筛选后用 MLLM 生成编辑指令，得到 600 万规模的 (source, target, instruction) 三元组。模型训练以预训练的 Stable Diffusion 为基础，将参考图像在空间维度（宽度方向）与噪声图拼接作为输入，用去噪损失训练。推理时输入原图和编辑指令即可生成编辑结果。

### 关键设计

1. **视频帧对采样与筛选流水线**:

    - 功能：从视频中提取具有适度变换的高质量帧对
    - 核心思路：首先对视频进行字幕描述并过滤不适合编辑的视频（风景、静态等）；以固定3秒间隔采样两帧；用 RAFT 计算光流进行运动筛选——保留运动适中的帧对，丢弃运动过大或过小的；通过反向 warp 计算背景遮挡掩码，过滤背景变化过大的帧对
    - 设计动机：确保帧对之间有足够但不过度的变换，避免静态帧（无编辑信息）和剧烈运动帧（对应关系丢失）

2. **MLLM 驱动的指令生成**:

    - 功能：自动为帧对生成高质量的编辑指令
    - 核心思路：将 source 和 target 帧对输入 GPT-4o 或 Pixtral-12B，提示模型分析两帧之间的差异（主体变化、相对位置、相机角度、背景变化），生成以动作动词开头的绝对性描述指令（如"将蜜蜂移到花朵中心"而非相对描述）；允许 MLLM 拒绝难以准确描述的帧对
    - 设计动机：相比纯文本 LLM 生成指令的方法（如 InstructPix2Pix），多模态 LLM 能直接观察图像差异，生成更多样化和准确的指令，覆盖非刚性变换和视角变化

3. **空间条件化策略 (Spatial Conditioning)**:

    - 功能：让模型在编辑时既能灵活改变结构又能保持原图内容
    - 核心思路：将参考图像潜码 $z^s$ 和带噪目标潜码 $z^e_t$ 在宽度维度拼接形成双倍宽度输入 $z_t = \text{Concat}_{width}([z^s, z^e_t])$，送入 U-Net 去噪后裁剪右半部分计算损失 $\mathcal{L}_{\text{Edit}} = \mathbb{E}[\|\epsilon - \text{Crop}(\epsilon_\theta(z_t, C, t))\|^2]$。与传统通道拼接不同，空间拼接不强制参考图与目标图空间对齐，让网络通过 cross-attention 自由关注参考图任意区域
    - 设计动机：通道拼接会强制空间对齐，限制非刚性变换的灵活性；空间拼接保留了预训练 T2I 模型的原始架构,因此可无缝集成 ControlNet 等额外控制机制

### 损失函数 / 训练策略
使用标准扩散去噪损失，仅对输出的右半部分（对应目标图像）计算 MSE 损失。训练基于 SD 1.5，分辨率 512×512，学习率 $1 \times 10^{-4}$，Adam 优化器，8 块 A100 上训练 100K 步，总 batch size 256。推理使用 DDIM 50 步采样。支持 mask 引导的局部编辑（通过 latent blending）和 ControlNet 集成。

## 实验关键数据

### 主实验

| 方法 | CLIP-D ↑ | CLIP-Inst ↑ | CLIP-I ↑ |
|------|----------|-------------|----------|
| NullTextInversion | 0.0660 | 0.7648 | 0.9063 |
| MasaCtrl | 0.0436 | 0.8527 | 0.9160 |
| InstructPix2Pix | 0.0887 | 0.8569 | 0.9380 |
| MagicBrush | 0.0972 | 0.8648 | 0.9318 |
| UltraEdit | 0.0824 | 0.8571 | 0.9184 |
| **InstructMove (Ours)** | **0.1361** | **0.8724** | 0.9275 |

| 方法 | 人类偏好率 |
|------|-----------|
| Imagic | 5.0% |
| InstructPix2Pix | 3.25% |
| MagicBrush | 4.13% |
| **InstructMove (Ours)** | **87.62%** |

### 消融实验

| 配置 | CLIP-D ↑ | CLIP-Inst ↑ | CLIP-I ↑ | 说明 |
|------|----------|-------------|----------|------|
| SC + IP2P data | 0.1277 | 0.8414 | 0.9094 | 用合成数据集训练 |
| CC + Our data | 0.0853 | 0.8679 | 0.8552 | 通道条件化 |
| **SC + Our data** | **0.1361** | **0.8724** | **0.9275** | 完整模型 |

### 关键发现
- 数据集贡献最大：使用真实视频帧对 vs IP2P 合成数据，CLIP-D 提升了 6.6%，CLIP-I 提升了 1.8%
- 空间条件化 vs 通道条件化：SC 在指令对齐和内容保持上均大幅优于 CC，CC 的 CLIP-I 仅 0.8552 说明通道拼接严重损害了内容保持
- 人类评估中以 87.62% 的偏好率碾压所有基线，说明非刚性编辑能力的巨大优势
- 现有方法在非刚性编辑上 CLIP-I 分数高是因为它们几乎不改变原图（编辑失败导致的假阳性）

## 亮点与洞察
- **视频作为编辑数据源的洞察**非常巧妙：视频帧天然提供了身份一致的 source-target 对，且涵盖了丰富的自然动态变换，这比合成数据质量高一个数量级。这个思路可以迁移到任何需要 paired 数据的任务。
- **空间拼接替代通道拼接**是一个极简但高效的设计：不修改任何网络架构，仅改变输入拼接方式就解锁了非刚性编辑能力，同时保持了与 ControlNet 等现有工具的兼容性。
- **MLLM 作为自动标注器**的流水线具有很好的通用性：可以扩展到其他需要图像对描述的任务中。

## 局限与展望
- MLLM 有时生成不准确的指令或遗漏帧间细微变化，导致编辑模型可能引入非预期的视角偏移
- 仅限于视频中能捕获到的真实变换，无法处理风格迁移、物体替换等艺术性编辑（作者通过混合其他数据集缓解）
- T2I 预训练模型的文本理解能力限制了复杂编辑指令的执行
- 可探索：将此方法扩展到视频编辑（利用连续帧）或 3D 一致性编辑

## 相关工作与启发
- **vs InstructPix2Pix**: IP2P 用 GPT-3 生成文本指令 + P2P 生成合成目标图像，数据质量受限于合成方法；本文用真实视频帧+MLLM，数据质量大幅提升
- **vs UltraEdit/EmuEdit**: 这些方法也试图提升编辑数据集质量，但仍依赖合成目标图像；本文是首个使用真实图像的大规模编辑数据集
- **vs 零样本编辑方法（NullText, MasaCtrl）**: 这些方法无需训练数据但速度慢且不稳定，本文通过训练获得了更好的性能和效率

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 视频帧对+MLLM指令生成的数据构建思路非常创新，空间条件化策略简洁有效
- 实验充分度: ⭐⭐⭐⭐ 定量指标、人类评估、消融实验齐全，但自建的 50 张测试集规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，图表质量高，motivation 阐述得很好
- 价值: ⭐⭐⭐⭐⭐ 开辟了利用视频数据训练图像编辑模型的新范式，数据集和方法都有广泛影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](../../ICCV2025/multimodal_vlm/hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[CVPR 2025\] ASAP: Advancing Semantic Alignment Promotes Multi-Modal Manipulation Detecting and Grounding](asap_advancing_semantic_alignment_promotes_multi-modal_manipulation_de.md)
- [\[CVPR 2025\] Reasoning to Attend: Try to Understand How \<SEG\> Token Works](reasoning_to_attend_try_to_understand_how_seg_token_works.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
