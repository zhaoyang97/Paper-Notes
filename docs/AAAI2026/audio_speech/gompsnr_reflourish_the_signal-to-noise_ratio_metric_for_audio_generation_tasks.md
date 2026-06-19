---
title: >-
  [论文解读] GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks
description: >-
  [AAAI2026][音频/语音][signal-to-noise ratio] 通过引入全方位相位导数（omnidirectional phase derivatives）替换瞬时相位来重构 SNR 指标，提出 GOMPSNR 作为更可靠的音频质量评估指标，并衍生出一系列新的损失函数显著提升神经声码器性能。
tags:
  - "AAAI2026"
  - "音频/语音"
  - "signal-to-noise ratio"
  - "phase derivatives"
  - "audio quality metric"
  - "loss function"
  - "neural vocoder"
---

# GOMPSNR: Reflourish the Signal-to-Noise Ratio Metric for Audio Generation Tasks

**会议**: AAAI2026  
**arXiv**: [2601.13758](https://arxiv.org/abs/2601.13758)  
**代码**: [lingling-dai/GOMPSNR](https://github.com/lingling-dai/GOMPSNR)  
**领域**: 音频语音  
**关键词**: signal-to-noise ratio, phase derivatives, audio quality metric, loss function, neural vocoder  

## 一句话总结
通过引入全方位相位导数（omnidirectional phase derivatives）替换瞬时相位来重构 SNR 指标，提出 GOMPSNR 作为更可靠的音频质量评估指标，并衍生出一系列新的损失函数显著提升神经声码器性能。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：信噪比（SNR）长期以来是音频生成任务中评估质量的基础客观指标。然而越来越多研究表明 SNR 及其变体（segSNR、SI-SNR 等）与 PESQ、UTMOS 等感知指标的相关性很低，逐渐被边缘化。与此同时，具有类似数学形式的 MCD 和 M-STFT 仍然是主流指标。这一矛盾促使作者追问两个问题：

1. **SNR 为何失效？** 作者将 SNR 从时域展开到时频域后发现，SNR 隐式地耦合了幅度和相位的度量。幅度谱的残差具有清晰结构，但相位谱的残差呈现无信息的噪声图案，说明传统的瞬时相位（IP）距离度量本身就不可靠。
2. **如何修复 SNR？** 相位导数（瞬时频率 IF 和群延迟 GD）比瞬时相位具有更清晰的结构，可用于替代 IP 来计算相位距离。

### 解决思路

**本文目标**：- SNR 中相位距离的不准确度量是导致其与人耳感知不一致的关键因素
- 相位谱因 wrapping 特性（值域限制在 $[-\pi, \pi)$）和对波形平移的高敏感性，直接计算距离不可靠
- SNR 公式中相关项 $C$ 的符号在 $\theta - \hat{\theta}$ 约 $\pm\pi/2$ 附近翻转，造成数值振荡，使 SNR 对相位误差过度敏感

## 方法详解

### 1. 全方位相位导数（Omnidirectional Phase Derivatives）
使用 9 个固定参数的 $3 \times 3$ 卷积核 $\mathcal{K} \in \mathbb{R}^{9 \times 3 \times 3}$，从时频图上 8 个相邻方向加上瞬时相位本身提取全方位相位导数：

$$\nabla\theta = \theta \circledast \mathcal{K}$$

配合反缠绕函数 $f_{AW}(x) = |x - 2\pi \cdot \text{round}(x / 2\pi)|$ 解决相位 wrapping 问题。

### 2. GOMPSNR 指标
SNR 在时频域展开为：

$$SNR = 10\log_{10} \frac{\sum_{k,l} |Y|^2}{\sum_{k,l}(|Y|^2 + |\hat{Y}|^2 + C)}$$

其中相关项 $C = -2|Y||\hat{Y}|\cos(\theta - \hat{\theta})$。改进经历两步：

- **OMPSNR**：将 IP 替换为全方位相位导数，$C = -\frac{2}{9}|Y||\hat{Y}|\sum_i \cos(\nabla_i\theta - \nabla_i\hat{\theta})$
- **GOMPSNR**：进一步将 $\cos$ 替换为线性映射的反缠绕函数，使 $C$ 始终非正，消除符号翻转引起的数值振荡：$C = \frac{2}{9}|Y||\hat{Y}|\sum_i(\frac{1}{\pi}f_{AW}(\nabla_i\theta - \nabla_i\hat{\theta}) - 1)$

### 3. 新损失函数族
基于相同的相位导数思想，提出三类新损失函数：

- **WOP Loss（幅度加权的全方位相位损失）**：用幅度谱加权 OP loss，使高能量区域获得更多关注
- **OmniRI Loss**：将传统 RI loss 中的 IP 替换为全方位相位导数，解耦相位和幅度的联合优化
- **CORI Loss（Coupled OmniRI）**：将幅度距离与相位导数距离以乘积形式耦合，同时优化两者

### 4. 最优损失函数组合
在幅度损失（Log/Lin）、相位损失（WOP）和联合优化损失（CORI）三个维度搜索最优组合，最终推荐 Lin + WOP + CORI(L1) 的组合。

## 实验关键数据

**指标验证**：在 LibriTTS 上用官方预训练 Vocos 计算 PCC 和 SRCC：
- SNR 与感知指标的相关性不超过 0.1，几乎无效
- GOMPSNR 与 PESQ、UTMOS、VQScore、NISQA、DistillMOS 均表现出较强相关性

**Vocos 损失函数消融（LJSpeech）**：
- 原始配置：PESQ 3.749，UTMOS 4.128，GOMPSNR 4.299
- +WOP：PESQ 3.928（+0.18），GOMPSNR 5.232（+0.93）
- +WOP+CORI(L1)：PESQ 4.001，MCD 2.238，GOMPSNR 5.674

**跨声码器验证（LJSpeech，Lin+WOP+CORI 组合 vs 原始）**：
- Vocos：PESQ 3.749→4.035，GOMPSNR 4.299→5.749
- APNet2：PESQ 3.643→3.901，GOMPSNR 4.961→5.533
- RNDVoc：PESQ 4.033→4.121，GOMPSNR 5.655→5.822

**Neural Audio Codec**：WavTokenizer 和 Vocos codec 在各带宽下均获提升，低带宽（高压缩率）下提升更显著。

## 亮点与洞察
- **问题分析深入**：通过数学推导和可视化清晰定位 SNR 失效的根因在于相位距离度量，论证逻辑严密
- **指标和损失函数双线推进**：从同一核心洞见同时改进评估指标和训练损失，方法论统一且实用
- **广泛实验覆盖**：4 个声码器（Vocos、APNet、APNet2、RNDVoc）× 2 个数据集（LJSpeech、LibriTTS）+ Neural Audio Codec，验证充分
- **即插即用**：所提损失函数无需修改模型架构，可直接替换原始损失，工程友好

## 局限与展望
- 实验仅限于声码器和音频编解码器，未验证在语音增强、语音分离等上游任务中的效果
- GOMPSNR 依赖参考信号（侵入式指标），无法用于无参考场景
- 全方位相位导数使用固定的 $3 \times 3$ 卷积核，未探索更大感受野或可学习核的效果
- 损失函数组合搜索仍依赖人工枚举，缺乏自动化搜索策略
- 未与近年的非侵入式感知指标（如 DNSMOS、SpeechLMScore）做直接对比

## 相关工作与启发

| 方法 | 类型 | 相位处理 | 与感知指标相关性 |
|------|------|---------|----------------|
| SNR/SI-SNR | 指标 | 隐式（瞬时相位） | 极低（PCC/SRCC < 0.1） |
| OP Loss | 损失函数 | 全方位相位导数 | — |
| **GOMPSNR** | **指标+损失** | **全方位相位导数+反缠绕+线性映射** | **显著提升** |
| M-STFT | 指标 | 幅度谱距离（忽略相位） | 中等 |
| PESQ/UTMOS | 感知指标 | 基于听觉模型 | 作为参考标准 |

与同期工作 RNDVoc（IJCAI 2025）使用的 OP 表示高度相关，GOMPSNR 可视为其在指标层面的推广。

## 相关工作与启发
- 相位距离度量的不可靠性可能也影响其他依赖时频表示的任务（如音乐生成、声音事件检测），值得迁移验证
- WOP loss 中"用幅度加权相位损失"的思路可推广到其他多分量信号的联合优化场景
- GOMPSNR 的设计思路（找到传统指标失效的数学根因→针对性修正）可借鉴到其他领域的指标改进

## 评分
- 新颖性: 7/10 — 核心贡献在于将全方位相位导数引入 SNR 重构，思路朴素但有效
- 实验充分度: 9/10 — 多声码器、多数据集、多指标的系统性验证
- 写作质量: 8/10 — 数学推导清晰，问题动机阐述充分
- 价值: 8/10 — 提供了可直接替代 SNR 的新指标和即插即用的损失函数，对音频生成社区有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] MECAT: A Multi-Experts Constructed Benchmark for Fine-Grained Audio Understanding Tasks](../../ICML2026/audio_speech/mecat_a_multi-experts_constructed_benchmark_for_fine-grained_audio_understanding.md)
- [\[ICML 2026\] Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models](../../ICML2026/audio_speech/focus_then_listen_an_empirical_study_of_plug-and-play_audio_enhancer_for_noise-r.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](../../NeurIPS2025/audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)
- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](../../ACL2026/audio_speech/planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)

</div>

<!-- RELATED:END -->
