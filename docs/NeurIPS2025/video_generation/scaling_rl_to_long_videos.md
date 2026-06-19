---
title: >-
  [论文解读] Scaling RL to Long Videos
description: >-
  [NeurIPS 2025][视频生成][长视频推理] 提出 LongVILA-R1 全栈框架，通过 104K 长视频推理数据集、两阶段 CoT-SFT + RL 训练管线、以及 MR-SP 多模态强化序列并行系统，将 VLM 的推理能力扩展到长视频（最高 8192 帧），在 VideoMME 上达到 65.1%/71.1%。
tags:
  - "NeurIPS 2025"
  - "视频生成"
  - "长视频推理"
  - "强化学习"
  - "视觉语言模型"
  - "序列并行"
  - "链式思维"
---

# Scaling RL to Long Videos

**会议**: NeurIPS 2025  
**arXiv**: [2507.07966](https://arxiv.org/abs/2507.07966)  
**代码**: [GitHub](https://github.com/NVlabs/Long-RL)  
**领域**: 视频生成  
**关键词**: 长视频推理, 强化学习, 视觉语言模型, 序列并行, 链式思维

## 一句话总结

提出 LongVILA-R1 全栈框架，通过 104K 长视频推理数据集、两阶段 CoT-SFT + RL 训练管线、以及 MR-SP 多模态强化序列并行系统，将 VLM 的推理能力扩展到长视频（最高 8192 帧），在 VideoMME 上达到 65.1%/71.1%。

## 研究背景与动机

1. **领域现状**: 长视频理解需要时间、空间、目标导向和叙事推理能力。GPT-4o、Gemini-1.5-Pro 等闭源模型展现了强大能力，开源 VLM 在短视频上也取得进展。

2. **现有痛点**: (1) 缺乏高质量长视频推理数据集——不同于数学/代码推理有结构化标注，长视频推理需标注复杂时间动态和叙事元素。(2) 长视频 RL 训练框架困难——处理成百上千帧导致巨大内存和超长 rollout 时间。

3. **核心矛盾**: 现有 RL 框架（如 R1-V、EasyR1）不是为长视频设计的，GRPO 的组采样在长上下文下计算成本极高，且视觉编码需要重复计算。

4. **本文目标**: 全栈解决长视频推理的数据、训练方法和训练系统三大挑战。

5. **切入角度**: 数据端通过 NVILA-8B + DeepSeek-R1-671B 自动生成长视频 CoT 标注；系统端通过序列并行和视频嵌入缓存加速 RL 训练。

6. **核心 idea**: 缓存视频嵌入 + 序列并行使长视频 RL 训练成为可能，而高质量 CoT 数据和难度过滤是推理能力涌现的关键。

## 方法详解

### 整体框架

三大组件：
1. **LongVideo-Reason 数据集**: 104K QA 对，18K 长视频
2. **两阶段训练**: Stage-1 CoT-SFT (36K) → Stage-2 GRPO RL (68K+102K)
3. **MR-SP 训练系统**: 序列并行 + 嵌入缓存

### 关键设计

**1. LongVideo-Reason 数据构建**

- **功能**: 提供大规模高质量长视频推理标注
- **核心思路**: 将视频切为 ~10s 片段 → NVILA-8B 生成描述 → 汇总所有片段描述 → DeepSeek-R1-671B 基于全视频描述生成 Question-Reasoning-Answer 对。四种推理类型：时间推理、目标/目的推理、空间推理、叙事推理。数据过滤：对每个问题推理 10 次，太简单（全对）和太难（全错）的过滤掉，只保留中等难度（GRPO 需要 rollout 多样性）。
- **设计动机**: GRPO 对批采样敏感——若所有 rollout 全对或全错则梯度消失，需要适当难度的数据。

**2. MR-SP 多模态强化序列并行**

- **功能**: 使长视频 RL 训练可行且高效
- **核心思路**: 
    - **Stage 1 - Rollout 并行编码**: 视频帧均匀分配到多个 GPU，各自独立编码，all-gather 汇聚嵌入。关键优化：缓存并复用视频嵌入，避免 8-16 次 rollout 中重复编码。
    - **Stage 2 - Prefilling 序列并行**: 对 policy/reference model 的 prefilling 操作做序列并行——将全局嵌入 padding 到统一长度后按 GPU 分片，各 GPU 只计算部分 token 的 logits。
- **设计动机**: 长视频产生海量 visual token（$10^4$-$10^5$），单 GPU 无法容纳。嵌入缓存避免 $G$ 次（rollout 数）的重复编码。

**3. 两阶段训练策略**

- **功能**: 先建立推理能力基础，再通过 RL 扩展
- **核心思路**: Stage-1 用 36K 高质量 CoT 数据做 SFT（格式为 `<think></think><answer></answer>`），使模型具备基本推理和指令跟随能力。Stage-2 用 68K 数据 + 102K 开源数据做 GRPO（准确率+格式的规则奖励），扩展推理能力。
- **设计动机**: 直接做 RL（跳过 SFT）性能下降，SFT 提供必要的推理 warm-up。

### 损失函数 / 训练策略

- Stage-1: 标准 SFT 交叉熵损失
- Stage-2: GRPO 目标 $\mathcal{J}(\theta)$，包含 clip 操作和 KL 正则。组 $G=8$，优势 $A_i$ 通过组内标准化计算
- 奖励: 规则基础（格式正确性 + 答案准确性）

## 实验关键数据

### 主实验

**VideoMME 基准**

| 模型 | w/o sub | Short | Medium | Long | w/ sub |
|------|---------|-------|--------|------|--------|
| LongVILA-7B | 60.1 | 69.0 | 58.3 | 53.0 | 65.1 |
| **LongVILA-R1-7B** | **65.1** | **76.8** | **63.2** | **55.2** | **71.1** |
| Video-R1-7B | 61.4 | - | - | - | - |
| Gemini-1.5-Pro | 75.0 | - | - | - | 81.3 |
| LLaVA-Video-7B | 63.3 | - | - | - | 69.7 |

**LongVideo-Reason-eval**

| 模型 | Temporal | Goal | Plot | Spatial | Overall |
|------|----------|------|------|---------|---------|
| Video-R1-7B | 61.4 | 85.0 | 62.0 | 58.5 | 68.1 |
| Gemini-1.5-Pro | 65.4 | 81.9 | 67.8 | 53.3 | 69.3 |
| LongVILA-7B | 58.0 | 80.2 | 57.1 | 46.7 | 62.7 |
| **LongVILA-R1-7B** | **68.1** | **85.7** | **70.6** | **53.3** | **72.0** |

### 消融实验

| 设置 | CoT-SFT | RL 数据 | LongVideo-Reason-eval |
|------|---------|--------|---------------------|
| 仅 Base | ✗ | ✗ | 62.7 |
| 仅 SFT | ✓ 本文数据 | ✗ | 提升显著 |
| 跳过 SFT 直接 RL | ✗ | ✓ | 性能下降 |
| **完整管线** | ✓ | ✓ | **72.0** |

**MR-SP 训练效率（8×A100, LongVILA-7B）**

| 帧数 | 无 MR-SP | MR-SP Stage 1 | 完整 MR-SP |
|------|----------|---------------|-----------|
| 256 | 正常 | 加速 | 加速 |
| 512 | 慢 | 加速但 OOM | **2.1× 加速** |
| 1024 | OOM | OOM | **可运行** |

### 关键发现

- RL 带来的推理能力随帧数增加持续提升（LongVILA-R1 在 16→512 帧持续增长，而 LongVILA 在 256 帧后饱和甚至下降）
- MR-SP 在 512 帧时实现 2.1× 加速，且是 1024 帧能跑的唯一方案
- CoT-SFT 是 RL 的必要前置——跳过则性能下降
- 在单个 A100 节点即可支持小时级视频（3600 帧）的 RL 训练

## 亮点与洞察

- 全栈方案：数据→训练→系统全部自洽
- MR-SP 使长视频 RL 从不可能变为实际可行，嵌入缓存避免 $G$ 倍重复编码是关键优化
- 数据过滤策略（去掉太简单/太难的）对 GRPO 至关重要——理论上梯度消失条件的实践解答
- 推理能力随帧数的持续提升验证了长视频 RL 的价值

## 局限与展望

- 数据生成消耗约 80,000 H100 GPU 小时，成本极高
- 推理依赖分段描述→LLM 生成推理，可能引入 caption 噪声
- 目前仅验证了 7B 规模模型，更大规模的效果未知
- 空间推理（Spatial）指标提升有限（53.3%），是已知弱项

## 相关工作与启发

- 建立在 LongVILA 的 MM-SP 基础上，前端数据利用 NVILA 和 DeepSeek-R1
- Video-R1 仅处理 16 帧短视频，本文将 RL for VLM 扩展到长视频
- 启发：RL 训练的精髓在于数据难度适配和系统工程优化

## 评分

- **新颖性**: ⭐⭐⭐⭐ 全栈整合而非单点突破，MR-SP 系统贡献显著
- **实验充分度**: ⭐⭐⭐⭐⭐ 多基准对比、消融充分、训练效率量化完整
- **写作质量**: ⭐⭐⭐⭐ 内容密集但组织清晰
- **价值**: ⭐⭐⭐⭐⭐ 为长视频 VLM 推理提供了可复现的完整方案，开源系统价值大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](../../ICML2026/video_generation/explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)
- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](../../ICML2026/video_generation/enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[NeurIPS 2025\] Radial Attention: O(n log n) Sparse Attention with Energy Decay for Long Video Generation](radial_attention_onlog_n_sparse_attention_with_energy_decay_for_long_video_gener.md)
- [\[CVPR 2025\] Towards Precise Scaling Laws for Video Diffusion Transformers](../../CVPR2025/video_generation/towards_precise_scaling_laws_for_video_diffusion_transformers.md)
- [\[ICCV 2025\] OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models](../../ICCV2025/video_generation/omnihuman-1_rethinking_the_scaling-up_of_one-stage_conditioned_human_animation_m.md)

</div>

<!-- RELATED:END -->
