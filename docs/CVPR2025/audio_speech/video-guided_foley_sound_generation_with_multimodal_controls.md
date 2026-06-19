---
title: >-
  [论文解读] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls
description: >-
  [CVPR 2025][音频/语音][Foley音效生成] 提出 MultiFoley，基于 Diffusion Transformer 的视频引导 Foley 音效生成系统，支持文本语义控制和参考音频风格控制，通过联合训练视频-音频和文本-音频数据集实现 48kHz 高质量音频生成，在人类评估中以 90% 胜率碾压现有方法。
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "Foley音效生成"
  - "视频到音频"
  - "扩散模型"
  - "多模态控制"
  - "音视频同步"
---

# MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls

**会议**: CVPR 2025  
**arXiv**: [2411.17698](https://arxiv.org/abs/2411.17698)  
**代码**: [https://ificl.github.io/MultiFoley/](https://ificl.github.io/MultiFoley/)  
**领域**: 音频语音 / 生成模型  
**关键词**: Foley音效生成, 视频到音频, 扩散模型, 多模态控制, 音视频同步

## 一句话总结

提出 MultiFoley，基于 Diffusion Transformer 的视频引导 Foley 音效生成系统，支持文本语义控制和参考音频风格控制，通过联合训练视频-音频和文本-音频数据集实现 48kHz 高质量音频生成，在人类评估中以 90% 胜率碾压现有方法。

## 研究背景与动机

**领域现状**：视频到音频（V2A）生成旨在为无声视频自动配音。现有方法（如 FoleyCrafter、Frieren、VAB）大多在 16kHz 低采样率下工作，且只接受视觉输入——无法通过文本指定想要的声音语义或通过参考音频指定音色风格。

**现有痛点**：三个关键问题：（1）音频质量低——16kHz 采样率丢失了大量高频信息，专业 Foley 工作流需要 48kHz；（2）控制性差——用户无法指定"脚踩在木板上"vs"脚踩在碎石上"的区别；（3）数据不匹配——高质量音效库（如 HQ-SFX）只有文本标注无视频，视频数据集（如 VGGSound）音质较差。

**核心矛盾**：需要同时实现视觉同步、语义可控和高音质，但这三者分别需要不同类型的数据和模型能力。

**本文目标** 统一视频引导、文本控制和参考音频迁移三种生成模式，在 48kHz 下实现高质量、可控的 Foley 音效生成。

**切入角度**：联合训练——将 VGGSound（视频-音频对）和 HQ-SFX（文本-音频对）混合训练，用质量标签区分音质分布，通过随机 dropout 各条件实现灵活的多模态推理。

**核心 idea**：DiT + 混合数据联合训练 + 多条件随机 dropout = 同时具备视觉同步、文本可控和高音质的 Foley 生成。

## 方法详解

### 整体框架

输入可包含：（1）无声查询视频 $v_q$（CAVP 编码，8FPS，64帧）；（2）文本提示 $t_c$（T5-base 编码）；（3）参考音频-视频对 $(a_c, v_c)$。音频用 DAC-VAE 编码为 40Hz 隐变量（维度 64），12 层 DiT（332M 参数）在隐空间中进行扩散去噪，条件通过拼接（视频）和交叉注意力（文本）注入。

### 关键设计

1. **混合数据联合训练 + 质量标签**:

    - 功能：同时学习视频-音频时序对齐和文本-音频语义对应
    - 核心思路：VGGSound（168K 视频-音频样本）标记为"low quality"，HQ-SFX（400K 文本-音频对）标记为"high quality"。训练时 60:40 混合采样。推理时用"high quality"标签引导生成高质量音频分布
    - 设计动机：解决了数据矛盾——VGGSound 提供视觉同步监督但音质差，HQ-SFX 提供高质量音频但无视频。质量标签让模型学会区分两种分布

2. **多条件随机 Dropout**:

    - 功能：使模型能灵活处理任意条件组合
    - 核心思路：训练时以一定概率随机丢弃文本/视频/音频条件（0.25 概率）。对音频隐变量也随机 mask 0-2 秒区间。这使得模型可以在推理时接受任意条件子集：仅视频、视频+文本、视频+参考音频等
    - 设计动机：避免模型对某个特定条件过度依赖。消融显示去掉文本后语义指标骤降（CLAP 34.4%→19.4%），但同步性保持（0.80s→0.77s），说明条件解耦成功

3. **带负提示的分类器无关引导（CFG）**:

    - 功能：通过正负文本提示精确控制生成语义
    - 核心思路：推理时同时输入正面提示（想要的声音）和负面提示（不想要的声音），CFG 公式 $\hat\epsilon = (\gamma+1) \cdot \epsilon_\theta(z_t, v_q, t_{\text{pos}}) - \gamma \cdot \epsilon_\theta(z_t, \varnothing_v, t_{\text{neg}})$。负提示替代了简单的无条件噪声估计
    - 设计动机：文本控制 Foley 时，排除不需要的声音类型（如"不要背景音乐"）比只指定想要的更有效

### 损失函数 / 训练策略

标准 LDM 去噪损失 $\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\epsilon, t}[\|\epsilon - \hat\epsilon\|_2^2]$。对被 mask 的条件区域不计算损失。推理用 DDIM 100 步采样，CFG scale 3.0。先在混合数据上训练大模型，再在高音视频对应度的 VGGSound 子集上微调。

## 实验关键数据

### 主实验

VGGSound 测试集（8702 视频）上的 V2A 生成对比：

| 方法 | ImageBind↑ | CLAP↑ | AV-Sync↓ | 采样率 |
|------|-----------|-------|---------|--------|
| FoleyCrafter | 30.2% | 25.3% | 0.87s | 16kHz |
| Frieren | 26.1% | 27.6% | 0.87s | 16kHz |
| **MultiFoley** | 28.0% | **34.4%** | **0.80s** | **48kHz** |
| Oracle (DAC-VAE) | 35.4% | 28.2% | 0.62s | 48kHz |

人类评估（400 组 vs FoleyCrafter）：

| 维度 | MultiFoley 胜率 |
|------|----------------|
| 语义对齐 | **85.8%** |
| 同步性 | **94.5%** |
| 音频质量 | **86.5%** |
| 总体 | **90.2%** |

### 消融实验

| 配置 | ImageBind | CLAP | AV-Sync |
|------|-----------|------|---------|
| 完整模型 (VT2A) | 28.0% | 34.4% | 0.80s |
| 无文本 (V2A) | 22.4% | 19.4% | 0.77s |
| 无子集微调 | 27.3% | 33.8% | 0.81s |
| Low quality 标签 | - | 34.4% | - |
| High quality 标签 | - | 34.9% | - |

### 关键发现
- **文本条件对语义至关重要**：去掉文本后 CLAP 从 34.4% 降到 19.4%（-15 %），但 AV-Sync 几乎不变（0.77 vs 0.80s），证明视频条件主导时序同步，文本主导语义内容
- **质量标签有效分离分布**：High quality 标签提升 FAD@AUD（更接近专业音频分布）但降低 FAD@VGG（偏离网络视频分布），说明质量控制确实有效
- **CFG scale 3.0 是最优**：过低（1.0）语义不清晰，过高（7.0）开始过拟合出伪影
- **音频-视觉扩展任务中条件叠加有效**：Full conditions（视频+音频+视频条件+文本）CLAP 64.3% vs 视频 only 55.4%

## 亮点与洞察

- **混合数据训练的优雅解法**：质量标签是解决"好数据无视频、有视频数据不好"矛盾的极简方案。推理时切换标签就能控制音质风格，无需复杂的数据筛选或多阶段训练
- **48kHz 全带宽生成**：相比 16kHz 基线翻了 3 倍的频带宽度，对专业 Foley 制作有实际意义。DAC-VAE 的高效压缩（48kHz→40Hz 隐变量）是关键使能技术
- **条件解耦的自然实现**：随机 dropout 不同条件使模型自动学到每个条件的独立作用——视频管同步、文本管语义、参考音频管风格——无需显式解耦设计

## 局限与展望

- **ImageBind 分数低于部分基线和 Oracle**：28.0% vs FoleyCrafter 30.2% 和 Oracle 35.4%，说明跨模态语义对齐仍有提升空间
- **多事件处理困难**：场景中有多个声音事件叠加时，模型难以区分各事件的时序关系
- **训练数据规模有限**：仅用 VGGSound(168K) + HQ-SFX(400K)，更大规模高质量数据集可能显著提升性能
- **推理成本**：100 步 DDIM 扩散采样 + 48kHz 音频解码的计算开销较大
- **潜在滥用风险**：可以为任意视频生成极逼真的音效，存在 Deepfake 风险

## 相关工作与启发

- **vs FoleyCrafter**: FoleyCrafter 在 ImageBind 指标上更高但 CLAP 更低，说明其视频对齐好但语义理解差。MultiFoley 通过文本条件大幅提升语义控制
- **vs Frieren**: Frieren 有较好的 FAD 但同步性差。MultiFoley 的视频特征直接拼接到 DiT 输入保证了更好的时序对齐
- **vs Make-An-Audio**: 纯文本到音频方法无法保证视频同步，MultiFoley 统一了文本和视频两种控制

## 评分
- 新颖性: ⭐⭐⭐⭐ 多模态控制的统一框架和质量标签机制是关键创新
- 实验充分度: ⭐⭐⭐⭐⭐ 自动指标+人类评估+多种消融+扩展任务，评估非常全面
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但部分符号表示可简化
- 价值: ⭐⭐⭐⭐ 对专业 Foley 制作和视频后期制作有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/audio_speech/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)
- [\[CVPR 2025\] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation](vintage_joint_video_and_text_conditioning_for_holistic_audio_generation.md)
- [\[CVPR 2025\] HOP: Heterogeneous Topology-based Multimodal Entanglement for Co-Speech Gesture Generation](hop_heterogeneous_topology-based_multimodal_entanglement_for_co-speech_gesture_g.md)

</div>

<!-- RELATED:END -->
