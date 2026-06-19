---
title: >-
  [论文解读] ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance
description: >-
  [ICCV 2025][图像生成][统一多模态模型] 提出 ILLUME，一个通过统一的下一 token 预测范式将多模态理解和生成能力整合进单个 LLM 的统一 MLLM。通过**语义视觉分词器**（减少4倍预训练数据量至15M）和**自增强多模态对齐方案**（让模型自评自生成图像与文本的一致性），在多种理解、生成和编辑任务上达到了State-of-the-art统一模型的竞争力甚至超越。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "统一多模态模型"
  - "视觉理解与生成"
  - "语义视觉分词器"
  - "自增强对齐"
  - "下一token预测"
---

# ILLUME: Illuminating Your LLMs to See, Draw, and Self-Enhance

**会议**: ICCV 2025  
**arXiv**: [2412.06673](https://arxiv.org/abs/2412.06673)  
**代码**: 无公开  
**领域**: 图像生成  
**关键词**: 统一多模态模型, 视觉理解与生成, 语义视觉分词器, 自增强对齐, 下一token预测

## 一句话总结

提出 ILLUME，一个通过统一的下一 token 预测范式将多模态理解和生成能力整合进单个 LLM 的统一 MLLM。通过**语义视觉分词器**（减少4倍预训练数据量至15M）和**自增强多模态对齐方案**（让模型自评自生成图像与文本的一致性），在多种理解、生成和编辑任务上达到了State-of-the-art统一模型的竞争力甚至超越。

## 研究背景与动机

### 核心问题

如何构建一个**高效**的统一 MLLM，在单一框架内同时支持视觉理解、图像生成和图像编辑？

### 现有方法的不足

**工具调用方案**（如 LLaVA + DALL-E）：解耦架构限制了模型潜力

**回归式统一模型**（如 Emu、Emu2）：需要联合训练 LLM 和扩散模型，工程成本高、稳定性差

**VQ 分词方案**（如 Chameleon、AnyGPT）：统一了 next-token prediction，但需要**海量数据**进行图文对齐——Chameleon 需 1.4B 图文对，Janus 需 65M

### 关键洞察

现有 VQ 分词器（如 VQGAN）以图像重建损失训练，量化表示聚焦低层纹理，缺乏语义信息，导致 LLM 中图文对齐**极其缓慢**。若在**语义特征空间**中进行量化，可大幅加速对齐过程。

### 第二个问题

统一模型的理解和生成能力能否**互相促进**？作者实验发现（Table 2）简单联合训练并**没有明显互利**，需要更精细的方法。

## 方法详解

### 整体架构

ILLUME 基于 Vicuna-7B，扩展了视觉词汇表以支持离散视觉 token 的生成：

- **理解端**：UNIT 视觉编码器 → 视觉适配器 → LLM 文本空间（保留连续特征，避免 VQ 信息损失）
- **生成端**：LLM 预测离散视觉 token → 语义视觉分词器解码 → Stable Diffusion 重建高分辨率图像
- **统一优化目标**：$\mathcal{L} = -\sum_{i=1} \log P_\theta(y_i | y_{\leq i})$，$y_i$ 为文本或视觉 token

### 关键设计 1：语义视觉分词器

与传统 VQGAN（用图像重建损失训练）不同：

- 利用预训练 UNIT 视觉编码器提取**语义特征**
- 通过**特征重建损失**监督量化过程和码本学习
- 码本大小 16384，每张图用 256 个离散 token
- 利用 **Stable Diffusion** 从语义特征重建图像（高压缩比 32×），补偿量化丢失的低层细节
- 效果：支持从固定数量 token 生成更高分辨率（512×512）图像

### 关键设计 2：三阶段渐进训练

| 阶段 | 目标 | 可训练部分 | 数据量 | 训练步数 |
|------|------|-----------|--------|---------|
| Stage-1: 视觉嵌入初始化 | 初始化视觉表示 | 视觉适配器 + 视觉嵌入/分类头 | 558K (LLaVA-Pretrain) + 图像重建任务 | 5000 |
| Stage-2: 统一图文对齐 | 学习理解+生成 | LLM + 视觉适配器 | **15M** 多模态数据 | 15000 |
| Stage-3: 监督微调 | 任务特定能力 | 全模型 | 指令调优 + 高质量图文对 + 混合模态数据 | 8000 |

Stage-1 的创新：引入**图像重建任务**——让 LLM 生成原始图像——来快速初始化新引入的视觉嵌入权重。

Stage-3 支持高分辨率输入：采用 image patchfy 策略（最大 9 切片，基础分辨率 448），每个切片下采样到 256 token。

### 关键设计 3：自增强多模态对齐方案

核心思想：让 MLLM 学习自评自生成图像质量，形成理解↔生成的正反馈循环。

**Step 1：自生成语料** — 从训练集文本子集自生成图像

**Step 2：评估数据生成** — 用 GPT-4o 评估自生成图像与文本的一致性（评估维度：物体准确性、数量、颜色、空间关系），生成评分 + 分析理由

**Step 3：SFT 对齐训练** — 将评估数据格式化为对话：
- 好的生成 → 单轮评估对话
- 差的生成 → 两轮（评估 + 修正）

共生成 **50K** 评估数据，融入 Stage-3 训练。

**互利机制**：
- **生成辅助理解**：通过分析自生成负样本，理解失败原因，提升图像解读准确性
- **理解辅助生成**：利用判别能力评估生成图像是否对齐文本，避免生成错误

## 实验关键数据

### 主实验：多模态理解（Table 3）

| 模型 | 类型 | POPE | MMBench | SEED | MME-P | MM-Vet | MMMU | AI2D |
|------|------|------|---------|------|-------|--------|------|------|
| LLaVA-1.5 (7B) | 理解专用 | 85.9 | 64.3 | 58.6 | 1510.7 | 31.1 | 35.4 | 54.8 |
| LLaVA-NeXT (7B) | 理解专用 | 86.5 | 67.4 | 64.7 | - | 43.9 | 35.1 | 66.6 |
| Emu3-Chat (8B) | 理解专用 | 85.2 | 58.5 | 68.2 | - | 37.2 | 31.6 | 70.0 |
| Janus (1.3B) | 统一 | 87.0 | 69.4 | 63.7 | 1338.0 | 34.3 | 30.5 | - |
| **ILLUME (7B)** | **统一** | **88.5** | **75.1** | **72.9** | **1445.3** | 37.0 | **38.2** | **71.4** |

ILLUME 在 12 个基准中的 10 个获得第一或第二名。相比 Janus，MMMU 提升 25%，SEED 提升 14%。

### 主实验：图像生成（Table 4）

| 模型 | 类型 | MJHQ FID↓ | GenAI Overall | GenEval Overall |
|------|------|-----------|---------------|-----------------|
| SDXL (2.6B) | 扩散 | 9.55 | 0.55 | 0.55 |
| Janus (1.3B) | 统一 | 10.10 | - | 0.61 |
| Show-o (1.5B) | 统一 | 15.18 | 0.53 | 0.53 |
| **ILLUME (7B)** | **统一** | **7.76** | 0.61 | **0.61** |

### 消融实验：自增强对齐的效果（Table 7）

| 设置 | POPE | MME-P | MMBench | SEED | MM-Vet | MMMU | GenEval Overall |
|------|------|-------|---------|------|--------|------|----------------|
| baseline | 86.4 | 1358.6 | 61.7 | 65.0 | 27.4 | 31.2 | 0.56 |
| **+ assessment** | 86.1 | **1446.7** | **63.1** | **66.0** | **29.0** | **32.0** | **0.59** |

仅增加 50K 评估数据，理解和生成均有提升。MME-P 提升近 90 分，GenEval 提升 0.03。

### 关键发现

1. 语义分词器 vs 重建分词器：训练Loss收敛速度在20M数据下差异显著，重建分词器在同数据量下生成效果不佳
2. 联合训练本身无显著互利（Table 2），但自增强方案有效促进了理解↔生成的协同
3. ILLUME 仅需 15M 数据即达竞争力，是 Janus 的 1/4、Chameleon 的 1/93

## 亮点与洞察

1. **数据效率的本质原因**：语义信息是加速 LLM 中图文对齐的关键——VQ 分词器的设计应面向 LLM 而非图像重建
2. **自增强方案的巧妙之处**：利用模型自身的不完美输出作为学习信号，无需额外标注数据
3. **架构对称性**：理解端用连续特征（保留精度），生成端用离散 token（统一范式），两端共享编码器但各取所需
4. **推理灵活性**：利用 CFG（classifier-free guidance）进行图像生成推理，支持交错图文数据的 any-to-any 任务

## 局限性

1. **基座模型规模**：仅在 Vicuna-7B 上验证，尚未验证在更大/更新 LLM 上的表现
2. **生成分辨率**：固定 512×512，相比 SDXL（1024×1024）仍有差距
3. **扩散模型依赖**：生成端仍需 SD 模型重建图像，并非端到端的纯 AR 生成
4. **训练成本**：32 节点 × 8 NPU × 3 天，对学术实验室仍然昂贵
5. **自增强仅用 50K**：未探索更大规模评估数据或更多评估维度的效果

## 相关工作与启发

- **与 Janus 的区别**：Janus 使用独立编码器解耦理解和生成表示，ILLUME 共享编码器但通过连续/离散分流
- **与 Emu3 的对比**：Emu3 是纯 AR 架构，ILLUME 在生成端仍依赖扩散解码器，但数据效率更高
- **与 Show-o 的对比**：Show-o 用 Phi-1.5B 较小基座；ILLUME 用更大 LLM + 更高效分词器
- **启发**：自增强方案可推广到更多模态（视频、音频、3D），语义分词器可成为通用设计原则

## 评分 ⭐⭐⭐⭐

**创新性**: ⭐⭐⭐⭐ — 语义分词器 + 自增强对齐方案双管齐下  
**实用性**: ⭐⭐⭐⭐ — 大幅降低数据需求，统一多任务框架  
**实验深度**: ⭐⭐⭐⭐ — 覆盖理解/生成/编辑三大领域，消融充分  
**写作质量**: ⭐⭐⭐⭐ — 架构图清晰，动机明确，实验组织有序

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Learning to See in the Extremely Dark](learning_to_see_in_the_extremely_dark.md)
- [\[ICCV 2025\] Your Text Encoder Can Be An Object-Level Watermarking Controller](your_text_encoder_can_be_an_object-level_watermarking_controller.md)
- [\[ICCV 2025\] PanoLlama: Generating Endless and Coherent Panoramas with Next-Token-Prediction LLMs](panollama_generating_endless_and_coherent_panoramas_with_next-token-prediction_l.md)
- [\[CVPR 2025\] See Further When Clear: Curriculum Consistency Model](../../CVPR2025/image_generation/see_further_when_clear_curriculum_consistency_model.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)

</div>

<!-- RELATED:END -->
