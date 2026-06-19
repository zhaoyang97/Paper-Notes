---
title: >-
  [论文解读] IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling
description: >-
  [ICML2025][音频/语音][text-to-audio] 提出 IMPACT 框架，将迭代掩码并行解码（MGM）与潜在扩散模型（LDM）结合，在连续潜在空间中进行文本到音频生成，以轻量 MLP 扩散头替代重型注意力层，同时引入无条件预训练阶段，在 AudioCaps 上取得 FD/FAD 指标 SOTA 且推理速度与最快的 MAGNET-S 相当。
tags:
  - "ICML2025"
  - "音频/语音"
  - "text-to-audio"
  - "扩散模型"
  - "mask-based generative modeling"
  - "iterative parallel decoding"
  - "连续潜在空间"
---

# IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling

**会议**: ICML2025  
**arXiv**: [2506.00736](https://arxiv.org/abs/2506.00736)  
**代码**: [audio-impact.github.io](https://audio-impact.github.io/)  
**领域**: 图像生成  
**关键词**: text-to-audio, diffusion models, mask-based generative modeling, iterative parallel decoding, 连续潜在空间

## 一句话总结
提出 IMPACT 框架，将迭代掩码并行解码（MGM）与潜在扩散模型（LDM）结合，在连续潜在空间中进行文本到音频生成，以轻量 MLP 扩散头替代重型注意力层，同时引入无条件预训练阶段，在 AudioCaps 上取得 FD/FAD 指标 SOTA 且推理速度与最快的 MAGNET-S 相当。

## 研究背景与动机

### 文本到音频生成的现状
文本到音频生成旨在根据自然语言提示合成语义匹配的高质量音频，应用场景涵盖音频内容创作、视频游戏、营销广告等。当前 SOTA 方法主要分为两大类：

**扩散模型系列**：以 Tango（Ghosal et al., 2023; Kong et al., 2024）和 AudioLDM（Liu et al., 2023, 2024）为代表，在音频保真度和质量上表现最优，但采用带有注意力层的重型网络架构作为扩散模型骨干，加之迭代去噪采样过程，导致推理延迟极高

**掩码生成模型**：MAGNET（Ziv et al., 2024）利用离散 token 上的迭代掩码并行解码，推理速度远快于自回归模型（如 MusicGen、AudioGen）和扩散模型，但音频质量仍落后于扩散模型 SOTA

### 核心矛盾与动机
存在一个明显的质量-速度权衡困境：扩散模型质量高但慢，MAGNET 快但质量不足。一个直觉的改进方向是将 MAGNET 的离散 token 替换为连续表示（continuous representations），因为连续表示在文本到图像生成（Fan et al., 2024）、语音大语言模型（Yuan et al., 2024）、自动语音识别（Xu et al., 2024）等任务中已展现出优于离散 token 的性能。

然而，作者的初步实验表明，简单地在 MAGNET 中替换为连续表示会导致性能显著下降。这说明直接替换不可行，需要更合理的建模方式来处理连续表示。

**关键洞察**：潜在扩散模型（LDM）擅长建模连续表示，而 MGM 的迭代掩码并行解码可以替代 LDM 中重型的注意力骨干，用轻量 MLP 扩散头完成采样。两者结合可以同时获得连续表示的高保真度和并行解码的低延迟。

## 方法详解

### 整体框架
IMPACT 框架由三个核心组件组成：音频 VAE 编解码器、基于 Transformer 的潜在编码器（Latent Encoder）、轻量 MLP 扩散头（Diffusion Head）。训练分为两阶段，推理采用迭代掩码并行解码。

### 音频表示提取
给定音频输入，使用音频 VAE 将其编码为连续潜在表示序列 $\mathbf{z} = [z_1, z_2, \cdots, z_N]$，其中每个 $z_i$ 为连续向量。与 MAGNET 使用离散音频编解码器（如 EnCodec）不同，IMPACT 直接在 VAE 的连续潜在空间中操作，避免了离散化带来的信息损失。

### 阶段1：无条件预训练（Unconditional Pre-training）
- **目的**：在大规模无标注音频数据上学习音频生成的基本能力
- **方法**：不使用任何文本条件，仅对音频潜在序列进行掩码重建训练
- **掩码策略**：随机掩码部分潜在向量，训练模型从未掩码的上下文中重建被掩码的潜在向量
- **关键性**：作者实验证明此阶段对最终性能不可或缺（indispensable），直接跳过会导致性能大幅下降
- **优势**：可以利用大量无配对文本-音频数据，极大扩展可用训练数据规模

### 阶段2：文本条件训练（Text-conditional Training with MGM）
训练流程如下：

1. **文本编码**：将文本提示通过文本编码器转换为文本条件向量序列
2. **拼接输入**：将音频潜在序列 $\mathbf{z}$ 与文本条件向量序列拼接
3. **掩码应用**：对音频潜在序列的部分位置应用随机掩码
4. **潜在编码**：拼接后的序列通过 Transformer-based 潜在编码器（Latent Encoder），产生上下文感知的表示
5. **扩散头预测**：轻量 MLP 扩散头对被掩码位置的潜在向量进行扩散建模——预测用于腐蚀（corrupt）被掩码音频潜在的噪声

扩散头的工作机制：
- 对被掩码的潜在向量 $z_i$，在前向过程中添加高斯噪声 $\epsilon$ 得到 $z_i^t$
- MLP 扩散头以潜在编码器的输出为条件，学习预测噪声 $\epsilon$
- 由于 MLP 结构远轻于传统 LDM 使用的 U-Net 或 Transformer 骨干，每步扩散采样的计算开销极小

### 推理阶段：迭代掩码并行解码
推理过程从完全空的序列（所有位置被掩码）出发，逐步生成：

1. **初始化**：所有 $N$ 个位置均为掩码状态
2. **迭代生成**：在每个解码迭代中：
    - 潜在编码器根据当前已生成的上下文和文本条件编码序列
    - MLP 扩散头对仍被掩码的位置运行少量扩散采样步骤，生成候选潜在向量
    - 根据预测置信度选择一批位置揭示（unmask），将其从掩码转为已生成状态
3. **渐进揭示**：早期迭代揭示较少位置（上下文有限、置信度低），后期揭示更多（上下文丰富、置信度高）
4. **结束**：所有位置揭示后，将完整的潜在序列通过 VAE 解码器还原为音频波形

**速度优势来源**：
- 扩散采样仅在轻量 MLP 头上运行，而非重型骨干网络
- 每次迭代同时生成多个位置的潜在向量（并行解码）
- 总迭代次数远少于传统 LDM 的扩散步数

## 实验关键数据

### 实验设置
- **数据集**：AudioCaps（文本-音频对数据集）
- **评估指标**：
    - 客观指标：FD（Fréchet Distance）、FAD（Fréchet Audio Distance）、KL 散度、IS（Inception Score）
    - 主观指标：REL（文本相关性）、OVL（整体音频质量）
- **基线方法**：AudioLDM, AudioLDM2, Tango, Tango2, MAGNET-S, MAGNET-M, AudioGen, MusicGen

### Table 1: AudioCaps 主结果对比

| 方法 | 类型 | FD ↓ | FAD ↓ | KL ↓ | REL ↑ | OVL ↑ | 推理速度 |
|---|---|---|---|---|---|---|---|
| AudioLDM | 扩散 | 23.31 | 1.96 | 1.26 | — | — | 慢 |
| AudioLDM2 | 扩散 | 18.65 | 1.62 | 1.18 | — | — | 慢 |
| Tango | 扩散 | 17.52 | 1.59 | 1.15 | 3.65 | 3.72 | 慢 |
| Tango2 | 扩散 | 16.83 | 1.36 | 1.12 | 3.78 | 3.81 | 慢 |
| AudioGen | 自回归 | 20.87 | 2.15 | 1.35 | — | — | 中 |
| MAGNET-S | MGM | 22.15 | 2.08 | 1.38 | 3.32 | 3.25 | 快 |
| MAGNET-M | MGM | 19.63 | 1.85 | 1.28 | 3.45 | 3.42 | 中 |
| **IMPACT** | **MGM+LDM** | **15.92** | **1.28** | **1.09** | **3.85** | **3.88** | **快** |

IMPACT 在 FD 和 FAD 两个关键指标上取得 SOTA，分别较 Tango2 提升约 **5.4%** 和 **5.9%**，同时推理速度与 MAGNET-S 相当。

### Table 2: 消融实验

| 配置 | FD ↓ | FAD ↓ | KL ↓ |
|---|---|---|---|
| IMPACT（完整） | **15.92** | **1.28** | **1.09** |
| 去掉无条件预训练 | 21.37 | 1.95 | 1.31 |
| 离散 token（MAGNET 方式） | 22.15 | 2.08 | 1.38 |
| 连续表示 + 直接 MSE（无扩散头） | 25.84 | 2.46 | 1.52 |
| 重型注意力扩散骨干（替代 MLP 头） | 16.15 | 1.31 | 1.11 |

**核心发现**：
- **无条件预训练至关重要**：去掉后 FD 从 15.92 恶化到 21.37（+34.2%），这是因为仅靠有限的配对文本-音频数据不足以学习高质量的音频生成基础能力
- **连续表示 + 扩散头是关键组合**：仅替换为连续表示但不用扩散建模（直接 MSE 回归）效果最差，说明连续空间需要扩散建模来有效处理
- **MLP 头 vs 重型骨干**：用重型注意力骨干替代 MLP 头仅带来微小的质量提升（FD 16.15 vs 15.92），但推理速度大幅下降，验证了 MLP 头的高效性

## 亮点与洞察

- **精准定位设计空间的交叉点**：将 MGM 的并行解码效率与 LDM 的连续表示建模能力巧妙结合，避免了两类方法各自的弱点——MGM 的离散 token 保真度瓶颈和 LDM 的推理延迟瓶颈
- **轻量扩散头的设计思路**：既然掩码并行解码的潜在编码器已经建模了序列级上下文，扩散头只需在单个 token 级别运行去噪，因此可以用 MLP 替代重型骨干，实现近乎免费的扩散采样
- **无条件预训练的实证贡献**：明确展示了在 MGM 训练范式中，无条件预训练不仅是锦上添花而是不可或缺的，为后续 MGM 研究提供了重要参考
- **受 MAR 启发的跨模态迁移**：MAR 在图像生成中验证了连续表示 + 掩码并行解码 + 扩散头的范式，IMPACT 首次将此范式成功迁移到音频生成领域，展现了该方案的通用性
- **Pareto 最优**：在质量-速度 Pareto 前沿上取得新的最优解，既不像扩散模型那样牺牲速度，也不像 MAGNET 那样牺牲质量

## 局限与展望

- **评估局限**：主要在 AudioCaps 单一数据集上评估，未验证在更多样化的音频生成场景（如音乐生成、语音合成、环境音效）中的泛化能力
- **无条件预训练依赖大规模数据**：无条件预训练阶段需要大量无标注音频数据，对数据获取和存储有一定要求
- **VAE 的瓶颈**：模型最终生成质量仍受限于音频 VAE 的重建能力；如果 VAE 本身存在信息损失，IMPACT 无法弥补
- **固定的解码调度策略**：迭代掩码并行解码中的揭示调度（每步揭示多少位置）可能需要根据不同音频类型自适应调整，当前方案是否为最优尚不明确
- **文本编码器未深入探讨**：论文未详细讨论不同文本编码器（如 CLAP、FLAN-T5、T5）对性能的影响
- **长音频生成**：迭代解码的步数与序列长度相关，对于超长音频（数分钟级别）的生成效率和质量是否仍能保持未得到验证

## 相关工作与启发

### 掩码生成模型（MGM）
- **MaskGIT / MUSE / MAGE**：在图像生成领域验证了掩码并行解码的有效性，但均操作离散 token
- **MAGNET**：将 MGM 应用于音频/音乐生成，实现快速推理但质量不足
- **SoundStorm**：掩码并行解码用于语音合成
- **MAR**：在图像生成中首次将 MGM 与连续表示+扩散头结合，取得 SOTA → IMPACT 的直接灵感来源

### 潜在扩散模型（LDM）
- **AudioLDM / AudioLDM2**：用 VAE 编码音频到潜在空间，再用 U-Net/Transformer 做扩散去噪，质量高但慢
- **Tango / Tango2**：类似框架，引入 DPO 等对齐技术进一步提升质量
- **IMPACT 的定位**：保留 LDM 在连续潜在空间中的建模优势，但将重型扩散骨干替换为 MGM 的并行解码+轻量扩散头，实现质量与速度的双赢

### 离散 vs 连续表示
- 多项研究（Fan et al., 2024; Yuan et al., 2024; Xu et al., 2024）已证明连续表示在多种生成任务中优于离散 token
- IMPACT 进一步证实：在音频生成中，连续表示同样优于离散 token，但前提是需要合适的建模方式（扩散头而非简单回归）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 将 MAR 的连续掩码并行解码范式首次引入音频生成领域，无条件预训练阶段有独立贡献
- 实验充分度: ⭐⭐⭐⭐ — 客观+主观评估，消融实验充分验证了各组件贡献，但仅单一数据集略有不足
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法说明条理分明，相关工作定位准确
- 价值: ⭐⭐⭐⭐ — 为文本到音频生成提供了质量-速度 Pareto 最优的新方案，轻量扩散头设计可推广到其他模态

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](../../ACL2026/audio_speech/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)
- [\[ICML 2025\] FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)
- [\[ICLR 2026\] The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](../../ICLR2026/audio_speech/the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](../../AAAI2026/audio_speech/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)
- [\[ICML 2025\] ETTA: Elucidating the Design Space of Text-to-Audio Models](etta_elucidating_the_design_space_of_text-to-audio_models.md)

</div>

<!-- RELATED:END -->
