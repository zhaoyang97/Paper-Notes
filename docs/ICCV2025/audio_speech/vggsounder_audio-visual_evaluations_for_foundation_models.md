---
title: >-
  [论文解读] VGGSounder: Audio-Visual Evaluations for Foundation Models
description: >-
  [ICCV 2025][音频/语音][音视频分类] 针对 VGGSound 数据集在多标签缺失、类别重叠和模态错位方面的局限性，构建了 VGGSounder——一个带有模态标注的多标签音视频分类基准，并提出"模态混淆"度量来揭示基础模型在多模态融合上的不足。 VGGSound 是音视频分类领域最常用的基准数据集…
tags:
  - "ICCV 2025"
  - "音频/语音"
  - "音视频分类"
  - "多标签基准"
  - "模态标注"
  - "基础模型评估"
  - "VGGSound"
---

# VGGSounder: Audio-Visual Evaluations for Foundation Models

**会议**: ICCV 2025  
**arXiv**: [2508.08237](https://arxiv.org/abs/2508.08237)  
**代码**: [项目主页](https://vggsounder.github.io/)  
**领域**: 音频-视觉学习 / 基准评测  
**关键词**: 音视频分类, 多标签基准, 模态标注, 基础模型评估, VGGSound

## 一句话总结

针对 VGGSound 数据集在多标签缺失、类别重叠和模态错位方面的局限性，构建了 VGGSounder——一个带有模态标注的多标签音视频分类基准，并提出"模态混淆"度量来揭示基础模型在多模态融合上的不足。

## 研究背景与动机

VGGSound 是音视频分类领域最常用的基准数据集，拥有约 20 万个 10 秒视频片段和 309 个类别。然而，随着音视频基础模型的快速发展，VGGSound 的局限性日益凸显：

**标注不完整**：VGGSound 仅为每个样本标注一个类别，但绝大多数视频实际上同时包含多个类别（如乐队演奏中同时出现多种乐器）。这导致模型预测了正确的额外类别却被判为错误。

**类别重叠**：309 个自动生成的类别中存在同义类（如 "timpani" 和 "tympany"）、子类-超类关系（如 "male speech" 和 "people speaking"）以及频繁共现类（如 "playing drums" 和 "playing drum kit"）。

**模态错位**：尽管 VGGSound 声称进行了视觉和听觉验证，约 **48.43%** 的测试样本存在模态错位——标注的类别要么不可听（如静态图片配背景音乐），要么不可见（如旁白叙述中提到的内容）。这挑战了 VGGSound 具有强模态对齐的广泛假设。

**核心动机**：缺乏模态感知的多标签标注使得我们无法准确评估基础模型的音频和视觉能力——一个模型预测了正确的听觉类别，如果该类别不在原始单标签中就会被判错。这种系统性低估尤其影响零样本使用的基础模型。

## 方法详解

### 整体框架

VGGSounder 的构建流程包含：提案生成 → 人工标注 → 自动补全 → 标签合并。

### 关键设计

1. **金标准参考集**：4 位计算机视觉专家手动标注了 417 个随机选择的样本，确保每个类别至少覆盖一次。使用多数投票合并标注。这个参考集用于评估自动提案策略的召回率。

2. **标签提案生成**：组合多个 SOTA 模型（CAV-MAE、AV-Siam、Equi-AV、DeepAVFusion、Gemini 1.5 Flash/Pro）的 top-k 预测。不同模态（音频/视觉/音视频）和不同 top-k 的预测被组合使用。对高频类别（如 speech、bird sounds）强制提案。最终策略达到 **93% 召回率**，平均每个样本约 30 个提案。

3. **人工标注（Amazon Mechanical Turk）**：对每个样本，标注者回答：

    - 是否包含背景音乐 / 旁白 / 静态图片（元标签）
    - 每个提案类别是否**可听**和/或**可见**
    - 是否有遗漏的类别

   每批 20 个样本包含 2 个金标准质控样本，F1 低于 25% 的批次被拒绝并重新标注。

4. **自动补全**：当检测到子类存在时自动添加超类（如检测到 "eagle screaming" → 自动添加 "bird squawking"）；同义类互相补全。

5. **模态混淆度量（Modality Confusion, $\mu$）**：衡量模型在获得额外模态输入后反而预测错误的样本比例：

$$\mu_M = 100 \cdot \frac{\sum_{x \in M} \mathcal{I}[a(x)\text{-correct} \cap av(x)\text{-wrong}]}{N_{total}}$$

这个指标揭示了**多模态输入并非总是更好**——模型可能被额外模态干扰。

### 损失函数 / 训练策略

VGGSounder 本身是评测基准，不涉及模型训练。评估使用多标签分类指标（Subset Accuracy、F1、Hit）。对嵌入模型使用 top-k 预测，对基础模型使用 LLM 辅助评估（Qwen-3 评判模型输出与目标类别的匹配度）。

## 实验关键数据

### 主实验

11 个音视频模型的 VGGSounder 评测结果（F1 分数）：

| 模型 | a(A) | v(V) | av(AV) | $\mu_A$↓ | $\mu_V$↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| CAV-MAE | 34.46 | 34.91 | 42.62 | 3.96 | 6.01 |
| AV-Siam | 33.30 | 35.41 | 39.43 | 10.30 | 8.69 |
| Gemini 1.5 Pro | 19.26 | 49.73 | 53.74 | 3.07 | 4.23 |
| VideoLLaMA 2 | 38.87 | 47.82 | 52.35 | 14.34 | 5.43 |
| Ola | 47.70 | 24.85 | 46.48 | 17.07 | 6.32 |

- 嵌入模型在音频输入上通常优于视觉输入，但基础模型呈现**相反趋势** — 偏向视觉信息
- 所有模型都存在显著的模态混淆（$\mu$），4-17% 的样本在加入额外模态后反而预测错误

### 消融实验（元标签分析）

不同元标签对 F1 分数的影响（$\Delta$ F1）：

| 条件 | 嵌入模型（音频） | 基础模型（音频） | 嵌入模型（视觉） | 基础模型（视觉） |
|------|:---:|:---:|:---:|:---:|
| 有背景音乐 | -3.4~-4.1 | -0.5~-11.8 | -2.5~-4.9 | -0.9~-4.6 |
| 有旁白 | -7.1~-9.1 | -4.0~+18.2 | +4.3~+5.2 | -3.7~-8.2 |
| 有静态图片 | +15.9~+22.1 | +1.4~+19.0 | +10.4~+19.8 | +4.2~+22.1 |
| 无任何元标签 | — | — | — | — |

- 背景音乐对所有模型都是挑战
- 基础模型对旁白的鲁棒性更好（部分模型性能反而提升），但嵌入模型受干扰严重
- 静态图片降低了视觉分类性能，但提升了音频分类（缺少视觉干扰时音频分类更准确）

### 关键发现

- VGGSounder 上的性能显著高于 VGGSound（由于多标签消除了假阴性），均差达 15-29%
- 人工标注贡献远大于自动补全（Hit 提升 8-28% vs 0.2-2.1%）
- Gemini 系列在音频输入上表现极差，说明其**几乎完全依赖视觉模态**
- 专用嵌入模型和通用基础模型的性能已经趋近

## 亮点与洞察

- **模态混淆度量是核心创新**：首次量化了"多模态输入可能比单模态更差"的现象，揭示了当前模型在模态融合上的根本缺陷
- **模态标注的引入至关重要**：使得首次能够分别评估模型在"仅可听但不可见"和"仅可见但不可听"样本上的表现
- **方法论上有借鉴意义**：标签提案 + 人工验证 + 自动补全的流水线可以推广到其他多标签基准的构建
- 发现了基础模型偏向视觉、嵌入模型偏向音频的系统性差异

## 局限与展望

- 多标签标注的完整性仍依赖于提案生成的召回率（93%），可能遗漏罕见类别
- 仅重新标注了测试集（15,446 样本），训练集可能存在相同问题
- 元标签仅覆盖三种情况（背景音乐/旁白/静态图片），其他干扰源未考虑
- LLM 辅助评估可能引入额外偏差
- 类别数固定为 VGGSound 的 309 个，无法评估开放词汇能力

## 相关工作与启发

- 受 ImageNet 重新标注（Beyer et al., 2020）和 MMLU 修正（Gema et al., 2024）启发
- 与 AudioSet 的多标签设计互补，但增加了模态级标注
- 模态混淆度量的思想可扩展到任何多模态融合系统的评估
- 为未来音视频基础模型的训练策略提供了重要诊断工具

## 评分

- **新颖性**: 7/10 — 基准重构本身不是新方法，但模态混淆度量是新贡献
- **技术质量**: 8/10 — 标注流程严谨，评测全面，覆盖 11 个模型
- **实用性**: 9/10 — 直接可用的基准，揭示了基础模型的真实能力
- **写作质量**: 8/10 — 结构清晰，可视化丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Mind the Gap! Static and Interactive Evaluations of Large Audio Models](../../ACL2025/audio_speech/mind_the_gap_static_and_interactive_evaluations_of_large_audio_models.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](../../NeurIPS2025/audio_speech/textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](../../NeurIPS2025/audio_speech/data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)

</div>

<!-- RELATED:END -->
