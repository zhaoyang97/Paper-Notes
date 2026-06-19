---
title: >-
  [论文解读] VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation
description: >-
  [CVPR 2025][音频/语音][视频文本联合音频生成] 提出 VinTAGe，首个联合视频+文本条件的音频生成模型，通过可学习层权重平衡视觉/文本引导，用教师-学生框架缓解模态偏置，在画内音和画外音生成上实现全面最优（FAD 3.05，MOS 3.36）。 领域现状：现有音频生成分为 V2A（视频到音频）和 T2A（文…
tags:
  - "CVPR 2025"
  - "音频/语音"
  - "视频文本联合音频生成"
  - "Transformer"
  - "模态偏置"
  - "教师引导"
  - "画外音"
---

# VinTAGe: Joint Video and Text Conditioning for Holistic Audio Generation

**会议**: CVPR 2025  
**arXiv**: [2412.10768](https://arxiv.org/abs/2412.10768)  
**代码**: 待公开  
**领域**: 音频语音 / 生成模型  
**关键词**: 视频文本联合音频生成, Flow Transformer, 模态偏置, 教师引导, 画外音

## 一句话总结

提出 VinTAGe，首个联合视频+文本条件的音频生成模型，通过可学习层权重平衡视觉/文本引导，用教师-学生框架缓解模态偏置，在画内音和画外音生成上实现全面最优（FAD 3.05，MOS 3.36）。

## 研究背景与动机

**领域现状**：现有音频生成分为 V2A（视频到音频）和 T2A（文本到音频）两条线，各自只用一种模态条件。真实视频的音频既包含与画面对应的画内音（如脚步声）也包含文本描述的画外音（如背景音乐）。

**现有痛点**：V2A 模型只能生成与画面对应的声音（忽略画外音），T2A 模型没有视觉时序对齐。简单串联两种条件会导致模态偏置——模型过度依赖某一模态而忽略另一模态（如 FoleyCrafter 画内64.9%/画外21.7%，严重偏向视觉）。

**核心矛盾**：视觉和文本条件对音频的贡献方式不同——视觉提供时序同步信号，文本提供语义分类信号——简单融合无法平衡两者。

**切入角度**：用预训练的 V2A 和 T2A 教师模型分别提供单模态指导，学习可调的层权重控制每层中各模态的贡献度。

**核心 idea**：Flow Transformer + 可学习层权重 + 教师引导去偏置 = 平衡画内外音的联合音频生成。

## 方法详解

### 关键设计

1. **VT-Encoder（视觉-文本联合编码器）**:

    - 功能：将视频和文本条件融合为统一的条件表示
    - 核心思路：CLIP 编码视频帧+光流/位置嵌入，FLAN-T5 编码文本。两者通过门控交叉注意力融合
    - 设计动机：光流嵌入捕捉运动信号，位置嵌入提供时间线索

2. **Joint VT-SiT（可学习层权重的 Flow Transformer）**:

    - 功能：逐层平衡视觉和文本条件的贡献
    - 核心思路：扩展 SiT（Scalable Interpolant Transformers），在每层引入可学习权重 $\omega_l$ 控制文本/视觉条件的混合比例
    - 设计动机：不同层可能需要不同的模态依赖——浅层可能更依赖视觉（时序），深层更依赖文本（语义）

3. **教师-学生去偏置框架**:

    - 功能：防止联合模型退化为只依赖单一模态
    - 核心思路：预训练独立的 V2A 教师和 T2A 教师。训练联合模型时额外对齐其预测与两个教师的预测：$\mathcal{L} = \mathcal{L}_{main} + \lambda_v \mathcal{L}_v + \lambda_t \mathcal{L}_t$
    - 设计动机：VinTAGe 画内57.7%/画外43.6% 远比 FoleyCrafter 64.9%/21.7% 更平衡

### 损失函数 / 训练策略

$\mathcal{L} = \|v_\theta - (\dot\alpha_t x - \dot\beta_t \epsilon)\|^2 + \lambda_v \mathcal{L}_v + \lambda_t \mathcal{L}_t$，classifier-free guidance $s_{vis} = s_{txt} = 2.5$。

## 实验关键数据

### 主实验

VinTAGe-Bench（636 对）：

| 指标 | VinTAGe | FoleyCrafter | Tango2 |
|------|---------|-------------|--------|
| FAD↓ | **3.05** | 3.81 | 8.01 |
| AV对齐 | **22.29** | 21.45 | - |
| MOS质量 | **3.36** | 3.02 | 2.71 |
| 画内/画外 | 57.7/43.6 | 64.9/21.7 | - |

### 关键发现
- **模态平衡是关键区分点**：VinTAGe 画内外比例 57.7/43.6 远比 FoleyCrafter 的 64.9/21.7 更平衡
- **教师引导有效**：无教师时模型倾向忽略文本条件

## 亮点与洞察
- **首次定义联合视频+文本音频生成任务**——配套了 VinTAGe-Bench 和画内/画外评估协议
- **教师去偏置的可推广性**——这种用单模态教师引导多模态学生的框架可以用在任何多条件生成任务中

## 局限与展望
- VinTAGe-Bench 较小（636 对），评估有限
- 画外音质量依赖 LLM 生成的文本描述
- 需要预训练两个教师模型，训练成本高

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次联合视频+文本音频生成
- 实验充分度: ⭐⭐⭐⭐ 新基准+主观评估+消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 为音频生成社区开辟了多模态条件新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] OmniSonic: Towards Universal and Holistic Audio Generation from Video and Text](../../CVPR2026/audio_speech/omnisonic_towards_universal_and_holistic_audio_generation_from_video_and_text.md)
- [\[CVPR 2025\] Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model](enhancing_dance-to-music_generation_via_negative_conditioning_latent_diffusion_m.md)
- [\[CVPR 2025\] Synchronized Video-to-Audio Generation via Mel Quantization-Continuum Decomposition](synchronized_video-to-audio_generation_via_mel_quantization-continuum_decomposit.md)
- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)
- [\[CVPR 2025\] MultiFoley: Video-Guided Foley Sound Generation with Multimodal Controls](video-guided_foley_sound_generation_with_multimodal_controls.md)

</div>

<!-- RELATED:END -->
