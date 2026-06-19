---
title: >-
  [论文解读] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
description: >-
  [ACL 2026][视频生成][文生视频] 提出 OSCBench——首个专门评估文生视频模型中物体状态变化（OSC）能力的基准，基于烹饪场景构建 1,120 条提示覆盖常规/新颖/组合三类场景，揭示即使最强 T2V 模型在 OSC 准确率上也仅达 0.786。 领域现状：T2V 模型在视觉质量和时间一致性上取得显著进展…
tags:
  - "ACL 2026"
  - "视频生成"
  - "文生视频"
  - "物体状态变化"
  - "评估基准"
  - "烹饪场景"
  - "多模态评估"
---

# OSCBench: Benchmarking Object State Change in Text-to-Video Generation

**会议**: ACL 2026  
**arXiv**: [2603.11698](https://arxiv.org/abs/2603.11698)  
**代码**: [项目页面](https://hanxjing.github.io/OSCBench)  
**领域**: 视频生成  
**关键词**: 文生视频, 物体状态变化, 评估基准, 烹饪场景, 多模态评估

## 一句话总结
提出 OSCBench——首个专门评估文生视频模型中物体状态变化（OSC）能力的基准，基于烹饪场景构建 1,120 条提示覆盖常规/新颖/组合三类场景，揭示即使最强 T2V 模型在 OSC 准确率上也仅达 0.786。

## 研究背景与动机

**领域现状**：T2V 模型在视觉质量和时间一致性上取得显著进展，现有基准主要评估感知质量、文本-视频对齐或物理合理性。

**现有痛点**：现有基准忽略了动作理解的关键维度——由文本提示显式指定的物体状态变化（如削土豆、切柠檬）。T2V 模型可能在高层语义上对齐良好，但生成的物体状态变化不正确、不完整或不一致。

**核心矛盾**：高质量的视觉外观掩盖了动作后果建模的缺陷——视频看起来逼真但物体并未正确变化状态。

**本文目标**：构建系统化的 OSC 评估基准，诊断 T2V 模型在状态变化建模上的具体不足。

**切入角度**：选择烹饪场景作为评估领域（状态变化频繁、多样、定义明确），并设计常规/新颖/组合三类场景测试不同层次的能力。

**核心 idea**：将 OSC 评估分为状态变化准确率和状态变化一致性两个子维度，配合 CoT 引导的 MLLM 自动评估。

## 方法详解

### 整体框架
OSCBench 从 HowToChange 数据集出发，先经人机协作把 20 种动作、134 种物体抽象为 9 类动作和 8 类物体（28 子类），据此构建常规/新颖/组合三类 OSC 场景（108/20/12），每个场景含 8 个动作-物体组合，共 1,120 条提示；评测则沿语义遵循、OSC 表现、场景对齐、感知质量四个维度展开。

### 关键设计

**1. 三类 OSC 场景：把"记忆"和"推理"分开考**

如果只用常见动作-物体组合测试，模型可能靠见过的素材蒙混过关，看不出它是否真懂状态变化。本文据此设计三档场景：常规场景用常见组合（如切柠檬）测基础能力；新颖场景用不常见但物理合理的组合（如捣碎葡萄柚），这类组合无法靠记忆解决、必须从动作语义推断状态变化，因而测的是泛化；组合场景串起连续多动作（如先削皮再切片）测时间一致性。三档由易到难，正好把"记忆能解决的"和"必须推理的"区分开。

**2. CoT 引导的 MLLM 评估：让打分器先讲理由再给分**

判断物体是否变到正确状态、变化过程是否平滑，本身需要多步推理，把 MLLM 当黑盒直接打分并不可靠。本文用思维链引导 MLLM 走一条"标准接地 → 证据提取 → 分数论证"的推理链——先对齐评判标准，再从视频里提取证据，最后据证据给出分数，使状态变化的判断更有据可循，也更容易扩展到大规模自动评测。

**3. 多维度评估体系：分维度定位 OSC 到底错在哪**

OSC 失败可能出在多个环节，单看一个总分看不出病根。本文把评测拆成四块：语义遵循（主体/物体/动作三项对齐）、OSC 表现、场景对齐、感知质量（真实感 + 美学），每项用 1-5 Likert 量表、人类评估取三人均值。其中 OSC 表现由两个核心指标刻画——状态变化准确率衡量物体是否达到正确的目标状态，状态变化一致性衡量变化过程是否平滑连贯，二者合起来回答"动作有没有真正改变物体"，正好把模型"画面好看但状态没变"的缺陷暴露出来。

### 损失函数 / 训练策略
本文是评估工作，不涉及模型训练。

## 实验关键数据

### 主实验（人类评估，归一化 0-1）

| 模型 | 主体对齐 | 物体对齐 | 动作对齐 | OSC准确率 | OSC一致性 | 真实感 |
|------|---------|---------|---------|----------|----------|--------|
| Veo-3.1-Fast | 0.936 | 0.916 | **0.908** | **0.786** | **0.748** | 最高 |
| Kling-2.5-Turbo | **0.938** | 0.900 | 0.826 | 0.726 | 0.726 | 0.732 |
| Wan-2.2 | 0.904 | 0.842 | 0.616 | 0.560 | 0.668 | 0.702 |
| HunyuanVideo-1.5 | 0.914 | 0.902 | 0.656 | 0.524 | 0.608 | 0.618 |
| Open-Sora-2.0 | 0.860 | 0.734 | 0.518 | 0.380 | 0.428 | 0.416 |

### 关键发现
- 所有模型在主体/物体对齐上表现良好（>0.73），但 OSC 准确率和一致性显著偏低
- 最强模型 Veo-3.1 的 OSC 准确率仅 0.786，说明状态变化建模是 T2V 的关键瓶颈
- 新颖和组合场景比常规场景表现更差，揭示了泛化能力的不足
- 闭源模型（Veo/Kling）明显优于开源模型，差距在 OSC 维度尤为显著

## 亮点与洞察
- OSC 视角填补了 T2V 评估的重要空白——动作不仅是运动，更应该产生正确的物体状态变化
- 三类场景设计巧妙区分了记忆能力和推理能力
- CoT 引导的 MLLM 评估与人类评估高度相关，为大规模自动化 OSC 评估提供了可行路径

## 局限与展望
- 仅聚焦烹饪领域，其他领域（工艺制作、化学实验）的 OSC 评估有待扩展
- 当前仅评估单次动作或两步组合，更长序列的组合动作挑战更大
- MLLM 评估虽与人类相关但并非完美替代，极端失败情况下可能误判

## 相关工作与启发
- **vs VBench**: VBench 聚焦整体视频质量，缺乏对物体状态变化的专项评估
- **vs PhyWorldBench**: 关注物理合理性（重力、碰撞），OSCBench 关注动作后果建模
- **vs T2V-CompBench**: 评估组合生成能力但不涉及状态变化的准确性和时间一致性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个 OSC 专项基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 6个模型、人类+自动双评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，评估体系完整
- 价值: ⭐⭐⭐⭐ 为 T2V 研究指明了关键改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Ego-InBetween: Generating Object State Transitions in Ego-Centric Videos](../../CVPR2026/video_generation/ego-inbetween_generating_object_state_transitions_in_ego-centric_videos.md)
- [\[ICML 2026\] LocoT2V-Bench: Benchmarking Long-form and Complex Text-to-Video Generation](../../ICML2026/video_generation/locot2v-bench_benchmarking_long-form_and_complex_text-to-video_generation.md)
- [\[CVPR 2026\] TiViBench: Benchmarking Think-in-Video Reasoning for Video Generation](../../CVPR2026/video_generation/tivibench_benchmarking_think-in-video_reasoning_for_video_generation.md)
- [\[CVPR 2026\] GenHOI: Towards Object-Consistent Hand-Object Interaction with Temporally Balanced and Spatially Selective Object Injection](../../CVPR2026/video_generation/genhoi_towards_object-consistent_hand-object_interaction_with_temporally_balance.md)
- [\[CVPR 2026\] M4V: Multimodal Mamba for Efficient Text-to-Video Generation](../../CVPR2026/video_generation/m4v_multimodal_mamba_for_efficient_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
