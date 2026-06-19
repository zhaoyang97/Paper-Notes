---
title: >-
  [论文解读] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation
description: >-
  [ICCV 2025][图像生成][图像编辑评估] 本文提出 ADIEE，一种自动化构建指令引导图像编辑评估数据集的方法，并基于超过 10 万样本微调 LLaVA-NeXT-8B 模型作为评分器，在多个基准上超越所有开源 VLM 和 Gemini-Pro 1.5，同时可作为奖励模型提升图像编辑模型性能。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "图像编辑评估"
  - "自动数据集生成"
  - "VLM评分器"
  - "奖励模型"
  - "指令引导编辑"
---

# ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation

**会议**: ICCV 2025  
**arXiv**: [2507.07317](https://arxiv.org/abs/2507.07317)  
**代码**: [https://github.com/SherryXTChen/ADIEE.git](https://github.com/SherryXTChen/ADIEE.git)  
**领域**: 图像生成/编辑  
**关键词**: 图像编辑评估, 自动数据集生成, VLM评分器, 奖励模型, 指令引导编辑

## 一句话总结
本文提出 ADIEE，一种自动化构建指令引导图像编辑评估数据集的方法，并基于超过 10 万样本微调 LLaVA-NeXT-8B 模型作为评分器，在多个基准上超越所有开源 VLM 和 Gemini-Pro 1.5，同时可作为奖励模型提升图像编辑模型性能。

## 研究背景与动机
指令引导图像编辑近年来发展迅猛，用户通过自然语言即可对图像进行修改。然而，如何自动化且准确地评估编辑效果成为一个关键瓶颈。传统指标如 MSE、PSNR、SSIM 和 LPIPS 无法捕捉语义层面的对齐度；CLIP 相关指标虽有所进步，但 CLIP 文本编码器对长文本和组合关系的理解有限。

利用视觉-语言模型（VLM）作为评判者是一个有前景的方向，但存在显著问题：开源 VLM 与人类判断对齐度差，闭源模型（如 GPT-4o）虽表现更好，但缺乏透明性且成本高。关键障碍在于缺乏公开的训练数据集来微调开源 VLM——现有的只有小规模基准，评估方案各异。

核心矛盾在于：构建大规模高质量的图像编辑评估训练数据通常依赖人工标注（成本高、不可扩展）或闭源模型生成标签（受限于闭源模型性能）。本文巧妙地利用图像编辑空间中已有的资源——编辑训练数据集和多种文本引导编辑模型——来自动生成评估训练数据。

核心 idea：图像编辑训练数据集本身就隐含着评估训练数据——ground-truth 编辑图像代表成功编辑（高分），原始输入图像代表失败编辑（低分），多模型生成的结果则提供中间质量的样本。

## 方法详解

### 整体框架
ADIEE 由四部分组成：（1）利用多种图像编辑模型生成不同质量的输出来构建训练数据；（2）利用多轮编辑序列构建带有连续分数的训练样本；（3）微调 LLaVA-NeXT-8B 作为评分器；（4）将训练好的评分器用作奖励模型来微调编辑模型。

### 关键设计

1. **基于编辑模型的数据生成**:

    - 功能：利用 9 种文本引导编辑方法（CycleDiffusion、DiffEdit、Prompt-to-Prompt、Pix2Pix-Zero、SDEdit、Text2LIVE、InstructPix2Pix、MagicBrush、AURORA）对 MagicBrush 数据集生成编辑结果
    - 核心思路：利用 CLIP 方向相似性（CLIP-D）和 CLIP/DINO 图像相似性来识别负样本。若 $\text{CLIP-D}(I^o, \hat{I^e}, p^o, p^e) < 0$，则编辑结果比原图更偏离 GT，标记为 0 分。对于需要 input/target caption 的模型，微调 Qwen2-VL-7B 生成与编辑指令对齐的 caption 对。正样本中，对 MagicBrush 和 AURORA 的输出使用 $\tau_{\text{CLIP-D}} = 0.2$ 阈值区分 0.5 分（部分成功）和 1 分（完全成功）
    - 设计动机：不直接将 CLIP/DINO 分数作为评估分数（因其对细节不敏感），而是利用它们的整体趋势来识别"容易的负样本"，避免依赖这些模型的局限性

2. **多轮编辑序列数据生成**:

    - 功能：利用多轮编辑数据集（MagicBrush、SEED-Data-Edit）中的编辑序列，通过随机采样输入-输出对来构建连续分数的训练样本
    - 核心思路：对于编辑序列 $\mathcal{S} = [I_0, p_1, I_1, \ldots, p_l, I_l]$，随机选择 $I_{j_1}$ 和 $I_{j_2}$（$j_1 < j_2$）作为输入和 GT 输出。任意中间图像 $I_k$ 的评分按比例计算：$f(I_{j_1}, P_{j_1 \to j_2}, I_k) = \frac{k - j_1}{j_2 - j_1}$（对于 $k \in [j_1, j_2]$），超编辑图像得 0.5 分，未编辑图像得 0 分
    - 设计动机：多轮序列天然提供了从"完全未编辑"到"部分完成"再到"完全完成"再到"过度编辑"的连续分数分布，丰富了训练数据的多样性

3. **评分器架构（VLM 微调）**:

    - 功能：微调 LLaVA-NeXT-8B，引入特殊 [SCORE] token，其 embedding 通过 MLP 解码为最终的编辑质量分数
    - 核心思路：将输入图像和输出图像拼接后与文本查询一起送入模型，扩展词汇表添加 [SCORE] token，该 token 的 hidden state 经 MLP 解码器映射为数值分数。使用 LoRA（rank=8）微调，在 6×H100 上训练 30 epoch，batch size=192
    - 设计动机：相比让 VLM 直接输出数值文本，使用专用 token + MLP 解码器能更好地隔离任务相关信息，减少幻觉，产生更精确的分数

### 损失函数 / 训练策略
评分器训练使用标准的 VLM 训练损失。作为奖励模型时有两种用法：（1）**奖励条件微调**：为每个训练样本预测编辑质量分数并附加到编辑指令后，推理时使用最高质量分数 prompt；（2）**奖励反馈学习**：从评分器反向传播梯度，损失为 $\mathcal{L}_{reward} = 10 - s$，与扩散模型 MSE 损失联合优化。

## 实验关键数据

### 主实验：评分器与人类判断的相关性

| 方法 | ImagenHub (Spearman) | AURORA-Bench (Spearman) | GenAI-Bench (Acc%) | AURORA-Bench (Acc%) |
|------|---------------------|------------------------|-------------------|-------------------|
| Human-to-Human | 0.4184 | - | - | - |
| GPT-4o | 0.3821 | 0.4038 | 53.54 | 50.81 |
| Gemini-Pro 1.5 | 0.2728 | 0.1052 | 55.93 | 28.13 |
| Qwen2.5-VL | 0.1859 | 0.2351 | 32.10 | 30.69 |
| LLaVA-OneVision | 0.0829 | 0.0555 | 22.31 | 33.25 |
| **ADIEE (Ours)** | **0.3450** | **0.4734** | **59.96** | **55.56** |

### 消融实验

| 配置 | ImagenHub | AURORA (point) | GenAI-Bench | AURORA (pair) |
|------|-----------|---------------|-------------|--------------|
| ADIEE (完整) | 0.3450 | 0.4744 | 59.41% | 52.38% |
| w/o score token & MLP | 0.2931 | 0.3558 | 58.32% | 20.50% |

| 奖励条件微调效果 | ImagenHub 平均分 | 说明 |
|----------------|-----------------|------|
| MagicBrush 基线 | 5.90 | 原模型 |
| + 奖励条件微调 | 6.43 (+8.98%) | 使用评分器作为奖励 |
| + 奖励反馈学习 | 6.30 (+6.78%) | 反向传播梯度 |

### 关键发现
- ADIEE 在 AURORA-Bench 上的分数相关性比此前 SOTA 提高 17.24%（绝对提升 0.0696），在配对比较准确率上分别提升 7.21%（GenAI）和 9.35%（AURORA）
- ADIEE 超越所有开源 VLM 且超过 Gemini-Pro 1.5，在 ImagenHub 上仅略低于 GPT-4o
- [SCORE] token + MLP 解码器设计至关重要，移除后配对比较准确率从 52.38% 暴降到 20.50%
- 人类偏好研究（56 名参与者）证实微调编辑模型被选为更优的比例为 41.67% vs 基线 32.00%

## 亮点与洞察
- **数据工程的巧妙设计**：不依赖人工标注或闭源 API 生成训练标签，而是利用编辑数据集本身的结构性（GT=高分，输入=低分）和多模型输出的质量差异自动构建多样化数据
- **多轮编辑序列**的利用非常聪明，用编辑进度比例自然产生连续分数
- **评分器 → 奖励模型**的闭环设计展示了评估指标的实际应用价值

## 局限与展望
- 分数分配依赖启发式规则（如 CLIP-D 阈值 0.2），可能引入偏差
- 评分器缺乏可解释性，无法指出编辑失败的具体原因
- 目前仅在 MagicBrush 数据集上验证了奖励模型效果，更多编辑模型的泛化有待验证

## 补充细节

训练数据组成：共使用 93,915 个由 9 种编辑方法生成的样本 + 21,382 个多轮编辑序列，总计超过 10 万样本。负样本来自 4 个一致性较差的模型（DiffEdit、Pix2Pix-Zero、SDEdit、Text2LIVE），正样本取自 MagicBrush 和 AURORA。人类偏好研究中 56 名参与者各比较了 50 对随机采样的输出。

## 相关工作与启发
- **vs VIEScore**: VIEScore 使用现成 VLM 零样本打分，性能受限于模型本身的对齐能力；ADIEE 通过微调在目标任务上大幅超越
- **vs OmniEdit**: OmniEdit 使用 GPT-4o 生成评估标签来训练 InternVL2，性能上界受限于 GPT-4o 本身且成本高；ADIEE 完全不依赖闭源模型标签
- **vs RewardEdit20K**: RewardEdit20K 使用 GPT-4o 按多维度打分，同样受限于闭源模型成本和性能天花板

## 评分
- 新颖性: ⭐⭐⭐⭐ 自动化数据集构建方法新颖且实用，多轮序列利用巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准全面评估，用户研究加持，奖励模型应用完整
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ 开源评分器 + 数据集对社区有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing](superedit_rectifying_and_facilitating_supervision_for_instruction-based_image_ed.md)
- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[ICCV 2025\] CAP: Evaluation of Persuasive and Creative Image Generation](cap_evaluation_of_persuasive_and_creative_image_generation.md)

</div>

<!-- RELATED:END -->
