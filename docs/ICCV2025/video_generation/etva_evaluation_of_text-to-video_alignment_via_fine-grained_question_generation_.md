---
title: >-
  [论文解读] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering
description: >-
  [ICCV 2025][视频生成][文本-视频对齐评估] 提出ETVA，一种基于细粒度问题生成与回答的文本-视频对齐评估方法，通过多智能体场景图遍历生成原子问题、知识增强多阶段推理回答问题，在与人类判断的相关性上大幅超越现有指标（Spearman's ρ 58.47 vs 31.0），并构建了包含2k prompts和12k问题的评估基准。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "文本-视频对齐评估"
  - "问答框架"
  - "场景图"
  - "多智能体"
  - "知识增强推理"
---

# ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering

**会议**: ICCV 2025  
**arXiv**: [2503.16867](https://arxiv.org/abs/2503.16867)  
**代码**: [eftv-eval.github.io/etva-eval](https://eftv-eval.github.io/etva-eval)  
**领域**: 视频生成  
**关键词**: 文本-视频对齐评估, 问答框架, 场景图, 多智能体, 知识增强推理

## 一句话总结

提出ETVA，一种基于细粒度问题生成与回答的文本-视频对齐评估方法，通过多智能体场景图遍历生成原子问题、知识增强多阶段推理回答问题，在与人类判断的相关性上大幅超越现有指标（Spearman's ρ 58.47 vs 31.0），并构建了包含2k prompts和12k问题的评估基准。

## 研究背景与动机

### 领域现状

文本到视频(T2V)生成模型（如Sora、Kling、HunyuanVideo）发展迅速，但缺乏可靠的自动评估指标来衡量生成视频与文本描述的语义对齐程度。

### 现有痛点

**粗粒度评分**：现有指标（CLIPScore、VideoScore等）只输出单一分数，无法描述具体哪些语义元素对齐/不对齐

**与人类判断偏差大**：如图示例中，人类认为Video 2更好地描绘了太空站微重力效果，但现有指标系统性偏好Video 1

**问题生成质量差**：简单的ICL方法生成的问题过于复杂（如同时包含动作+对象+环境），Video LLM难以准确回答

**Video LLM幻觉严重**：直接用Video LLM回答问题时，缺乏常识知识（如微重力下水的行为），且缺少深层推理过程

### 核心矛盾

如何生成"恰到好处"的问题（原子性+完整覆盖）并让Video LLM可靠地回答这些问题？

### 切入角度

模拟人类标注过程：先将文本解析为场景图并遍历生成原子问题（解决C1），再通过知识检索+多阶段推理让Video LLM像人类一样先回忆知识再观察分析（解决C2）。

## 方法详解

### 整体框架

ETVA分两阶段工作：
1. **问题生成(QG)**：多智能体系统解析文本prompt → 构建语义场景图 → 遍历生成原子yes/no问题
2. **问题回答(QA)**：辅助LLM检索常识知识 → Video LLM通过多阶段推理回答问题

最终对齐分数：$S = \frac{1}{n}\sum_{i=1}^{n} S_i$，其中$S_i \in \{0, 1\}$。

### 关键设计

#### 1. **多智能体问题生成**

- **功能**：将文本prompt分解为原子性问题，确保完整覆盖且不冗余
- **核心思路**：三个协作Agent——
    - **元素提取器**：识别实体(如"杯子"、"太空站")、属性(如"玻璃材质"、"透明")、关系(如"从...倒出"、"在...中")
    - **图构建器**：将元素组织为层次化场景图，实体节点为中心锚点，所有关系和属性节点必须连接到至少一个实体。属性节点只有出边，关系节点维持双向连接
    - **图遍历器**：按顺序处理——先确认实体 → 再验证属性 → 最后检查关系（仅在两端实体都验证后）
- **设计动机**：基于场景图的遍历保证了问题的原子性（每次只问一件事）和逻辑依赖（先确认对象再问关系），避免了ICL方法的冗余和不可回答问题

#### 2. **知识增强多阶段推理**

- **功能**：减少Video LLM的幻觉，模拟人类的认知过程
- **核心思路**：
    - **知识增强(KA)**：用辅助LLM（Qwen2.5-72B-Instruct）根据prompt检索相关常识知识。如"太空站倒水"→微重力下液体形成漂浮球状而非流下
    - **多阶段推理**：
    1. **视频理解阶段**：Video LLM自主提取逐帧描述（不看文本）
    2. **通用反思阶段**：结合观察、问题和常识知识进行交叉分析
    3. **结论阶段**：给出Yes/No答案并提供显式的视觉-语言对齐检查
- **设计动机**：模拟人类标注过程——先回忆相关知识→仔细观察视频→深度推理→给出结论，避免Video LLM"不经思考直接回答"的幻觉问题

#### 3. **ETVABench 构建**

- **功能**：构建专门用于文本-视频对齐评估的全面基准
- **核心思路**：
    - 从VBench、EvalCrafter、T2V-ComBench等基准收集2k diverse prompts
    - 基于问题类型分类为10个类别：existence, action, material, spatial, number, shape, color, camera, physics, other
    - ETVABench-2k (2k prompts, 12k questions) + ETVABench-105 (105 prompts精简版)

## 实验关键数据

### 主实验（与人类判断的相关性比较）

| 指标 | Kendall's τ | Spearman's ρ |
|------|------------|-------------|
| BLIP-BLEU | 8.5 | 12.1 |
| CLIPScore | 10.3 | 13.8 |
| ViCLIPScore | 19.4 | 25.9 |
| VideoScore | 23.7 | 31.0 |
| **ETVA** | **47.2** | **58.5** |

**分维度对比**（Spearman's ρ）：

| 维度 | VideoScore | ETVA | 提升 |
|------|-----------|------|------|
| Existence | 30.6 | 57.4 | +87.6% |
| Material | 37.3 | 66.1 | +77.2% |
| Spatial | 31.7 | 66.8 | +110.7% |
| Shape | 35.7 | 75.1 | +110.4% |
| Physics | 23.9 | 60.4 | +152.7% |
| Camera | 26.3 | 44.2 | +68.1% |

### 消融实验

**问题生成部分消融**：

| 配置 | Kendall's τ | Spearman's ρ | 说明 |
|------|------------|-------------|------|
| 多智能体QG | **47.16** | **58.47** | 完整方法 |
| Vanilla ICL QG | 35.04 | 42.87 | 直接ICL生成问题 |
| 提升 | +12.12 | +15.60 | +34.6% |

**问题回答部分消融**：

| 配置 | 准确率 | Kendall's τ | Spearman's ρ |
|------|--------|------------|-------------|
| ETVA (完整) | **89.27** | **47.16** | **58.47** |
| w/o 知识增强(KA) | 67.34 | 27.34 | 35.54 |
| w/o 视频理解(VU) | 82.73 | 37.56 | 44.81 |
| w/o 批判反思(CR) | 68.74 | 28.73 | 38.21 |
| 仅KA | 65.48 | 24.72 | 33.12 |
| 直接回答 | 63.07 | 18.18 | 23.84 |

### T2V模型评估（ETVABench-105）

| 模型 | Existence | Action | Spatial | Physics | Camera | Avg |
|------|-----------|--------|---------|---------|--------|-----|
| Latte | 0.519 | 0.504 | 0.444 | 0.350 | 0.105 | 0.474 |
| CogVideoX-5B | 0.644 | 0.664 | 0.630 | 0.500 | 0.474 | 0.620 |
| HunyuanVideo | 0.727 | 0.693 | 0.704 | 0.300 | 0.421 | 0.686 |
| Kling-1.5 | 0.754 | 0.675 | 0.754 | 0.500 | 0.383 | 0.707 |
| Pika-1.5 | 0.801 | 0.752 | 0.778 | 0.450 | 0.421 | 0.738 |
| Sora | 0.815 | 0.759 | **0.870** | 0.550 | 0.316 | 0.757 |
| **Vidu-1.5** | 0.792 | **0.766** | 0.862 | **0.600** | 0.421 | **0.761** |

### 关键发现

- **ETVA与人类判断相关性提升88%**：Spearman's ρ从31.0飙升至58.5
- **知识增强是最关键模块**：去除KA导致准确率下降21.93%（最大降幅），证明常识知识对抗幻觉至关重要
- **多智能体QG贡献14.67%**：结构化场景图生成比vanilla ICL质量高得多
- **所有T2V模型在Physics和Camera维度表现最差**：最高分仅0.600和0.474，暴露了物理模拟和相机控制的根本不足
- **闭源模型整体领先但非全面碾压**：开源HunyuanVideo在shape(0.824)上匹敌甚至超越Sora(0.765)
- **仅用KA不够**：需要结合多阶段推理才能有效利用知识（仅KA准确率65.48 vs 完整89.27）

## 亮点与洞察

1. **场景图驱动的原子问题生成**：解决了"什么样的问题Video LLM能准确回答"这一核心难题
2. **认知架构模拟**：知识检索→视频观察→反思推理→结论的多阶段过程，与人类标注流程高度一致
3. **细粒度诊断能力**：不仅给出总分，还能精确指出每个语义元素是否对齐，对模型改进有直接指导价值
4. **10维度分类体系**：基于问题的分类方法比基于prompt的分类更精确，同一prompt可涵盖多个类别
5. **15个T2V模型的系统性评测**：揭示了物理和相机控制是当前T2V模型的最大短板

## 局限与展望

1. **依赖大型LLM**：QG（Qwen2.5-72B）和QA（Qwen2-VL-72B）都需要大规模模型，评估成本高
2. **仅支持Yes/No问题**：无法评估"程度"型的语义对齐（如动作的准确度）
3. **KA可能引入偏差**：如果辅助LLM的常识知识不准确，可能误导评估
4. **闭源模型评估受限**：ETVABench-2k仅评估了开源模型，因为大规模调用闭源API成本过高
5. **Color维度表现最差**（39.7 Spearman's ρ），说明颜色评估可能需要更专业的方法
6. **未考虑时间动态的深层评估**：如动作的流畅性、因果关系的准确性

## 相关工作与启发

- **TIFA** 是QG&QA评估文本-图像对齐的先驱，但直接迁移到视频面临复杂度和幻觉挑战
- **VBench** 是当前最全面的T2V评估基准，但其文本对齐维度使用ViCLIPScore，粒度较粗
- **VideoScore** 通过微调MLLM实现评分，但本文证明其与人类判断的相关性仅31.0
- 本文的多智能体QG框架和知识增强QA框架有潜力扩展到其他多模态评估任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 场景图驱动QG + 知识增强多阶段QA的组合新颖，但单个技术（场景图、CoT）已有先例
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7种baseline指标对比、详细消融、15个模型评测、人类标注验证，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，C1/C2挑战框架有说服力，案例分析直观
- **价值**: ⭐⭐⭐⭐⭐ — 为T2V评估提供了远超现有指标的方法，构建的benchmark对社区有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[ICCV 2025\] WorldScore: A Unified Evaluation Benchmark for World Generation](worldscore_a_unified_evaluation_benchmark_for_world_generation.md)
- [\[NeurIPS 2025\] DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](../../NeurIPS2025/video_generation/densedpo_finegrained_temporal_preference_optimization_for_vi.md)
- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[ICML 2026\] DFSAttn: Dynamic Fine-Grained Sparse Attention for Efficient Video Generation](../../ICML2026/video_generation/dfsattn_dynamic_fine-grained_sparse_attention_for_efficient_video_generation.md)

</div>

<!-- RELATED:END -->
