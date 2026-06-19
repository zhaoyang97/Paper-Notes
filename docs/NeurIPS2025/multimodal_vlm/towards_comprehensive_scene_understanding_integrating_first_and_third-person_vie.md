---
title: >-
  [论文解读] Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs
description: >-
  [NeurIPS 2025 (Spotlight)][多模态VLM][多视角理解] 提出 E3VQA 基准（首个多视角 VQA 基准）和 M3CoT 提示技术（融合三种互补视角的场景图），增强大型视觉语言模型 (LVLM) 的多视角场景理解能力，GPT-4o 提升 4.84%、Gemini 2.0 Flash 提升 5.94%。
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "多模态VLM"
  - "多视角理解"
  - "自中心视角"
  - "第三人称视角"
  - "场景图"
  - "VQA"
  - "CoT"
---

# Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs

**会议**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2505.21955](https://arxiv.org/abs/2505.21955)  
**代码**: 有 ([https://github.com/Leeinsu1/Towards-Comprehensive-Scene-Understanding](https://github.com/Leeinsu1/Towards-Comprehensive-Scene-Understanding))  
**领域**: 多模态VLM / 场景理解  
**关键词**: 多视角理解, 自中心视角, 第三人称视角, 场景图, VQA, CoT  

## 一句话总结

提出 E3VQA 基准（首个多视角 VQA 基准）和 M3CoT 提示技术（融合三种互补视角的场景图），增强大型视觉语言模型 (LVLM) 的多视角场景理解能力，GPT-4o 提升 4.84%、Gemini 2.0 Flash 提升 5.94%。

## 研究背景与动机

### LVLM 在交互应用中的部署
大型视觉语言模型正被越来越多地部署在虚拟/增强现实等交互应用中，头戴式相机提供的第一人称（自中心, egocentric）视角是关键输入。然而，自中心视角存在固有局限：

**视野窄**：头戴相机的 FOV 有限，无法看到全局场景

**缺乏全局上下文**：只能看到用户正在关注的局部区域

**空间推理困难**：难以回答需要全局空间信息的问题

### 第三人称视角的互补价值
第三人称（外部, exocentric）视角可以提供：
- 全局场景布局
- 完整的物体可见性
- 空间关系的全局视图
- 用户与环境的交互上下文

### 核心问题
如何有效地将第一人称和第三人称视角融合，使 LVLM 能更全面地理解场景？现有工作几乎都只使用单一视角。

## 方法详解

### 整体框架

```
输入: 同步的 ego-exo 图像对
  ├── E3VQA 基准: 4K 高质量 QA 对
  └── M3CoT 推理: 三视角场景图融合
       ├── Ego Scene Graph (自中心场景图)
       ├── Exo Scene Graph (外部场景图)  
       └── Cross-view Scene Graph (跨视角场景图)
       → 统一场景表示 → LVLM 回答
```

### 关键设计

#### 1. E3VQA 基准

**数据构建**：
- 基于 Ego-Exo4D 数据集的同步 ego-exo 图像对
- 4,000 个高质量问答对，覆盖多种问题类型：
    - **空间推理**：物体在哪里？相对位置如何？
    - **动作理解**：用户在做什么？
    - **因果推理**：为什么用户关注这个物体？
    - **计数和属性**：场景中有多少个特定物体？
- **质量控制**：人工标注 + 多轮验证

**问题类型分布**：
- 仅需 ego 视角即可回答：~30%
- 仅需 exo 视角即可回答：~25%
- 需要两个视角才能正确回答：~45%

#### 2. M3CoT (Multi-view Multi-modal Chain-of-Thought)

M3CoT 是训练无关的提示技术，核心是构建统一的多视角场景表示：

**步骤 1: 生成三种场景图**

- **Ego Scene Graph $G_e$**：从自中心图像提取
    - 节点：检测到的物体
    - 边：空间关系（左/右/上/下/前/后）+ 交互关系
  
- **Exo Scene Graph $G_x$**：从第三人称图像提取
    - 节点：全局可见的物体
    - 边：全局空间关系

- **Cross-view Scene Graph $G_c$**：跨视角关联
    - 节点：两视角共同可见的物体
    - 边：跨视角的对应关系和互补信息

**步骤 2: 融合为统一表示**

将三张场景图融合为一个文本化的统一场景描述，作为 LVLM 的附加上下文输入。

**步骤 3: 链式推理**

使用融合的场景表示引导 LVLM 进行多步推理：
```
[Ego Image] + [Exo Image] + [Unified Scene Representation]
→ Step 1: 理解每个视角提供的信息
→ Step 2: 识别两个视角的互补关系  
→ Step 3: 综合推理并回答问题
```

### 损失函数 / 训练策略

M3CoT 是**完全免训练**的方法：
- 利用 LVLM 的 in-context learning 能力
- 仅通过精心设计的 prompt 实现
- 适用于任何支持多图像输入的 LVLM

## 实验关键数据

### 主实验

在 E3VQA 基准上不同 LVLM 的表现：

| 模型 | 仅 Ego | 仅 Exo | Ego+Exo (无CoT) | +标准CoT | +M3CoT (Ours) |
|------|--------|--------|----------------|---------|-------------|
| GPT-4o | 52.3 | 48.7 | 58.4 | 61.2 | **66.0 (+4.84)** |
| Gemini 2.0 Flash | 49.8 | 46.2 | 55.1 | 58.3 | **64.2 (+5.94)** |
| Claude 3.5 Sonnet | 50.1 | 47.3 | 56.8 | 59.4 | **63.8 (+4.42)** |
| LLaVA-1.5-13B | 38.2 | 35.6 | 42.1 | 44.7 | **48.3 (+3.58)** |
| InternVL2-8B | 41.5 | 38.9 | 45.3 | 47.6 | **51.2 (+3.61)** |

按问题类型细分的提升幅度（GPT-4o）：

| 问题类型 | Ego+Exo 基线 | +M3CoT | 提升 |
|---------|------------|--------|------|
| 空间推理 | 55.2 | **63.8** | +8.6 |
| 动作理解 | 62.1 | **67.4** | +5.3 |
| 因果推理 | 54.8 | **62.1** | +7.3 |
| 计数和属性 | 61.5 | **64.7** | +3.2 |
| 仅需 Ego | 68.3 | **70.1** | +1.8 |
| 需要两视角 | 48.6 | **58.2** | +9.6 |

### 消融实验

M3CoT 各组件的贡献（GPT-4o）：

| 配置 | 准确率 | 提升 |
|------|--------|------|
| 基线 (Ego+Exo, 无 CoT) | 58.4 | — |
| + Ego Scene Graph 仅 | 61.5 | +3.1 |
| + Exo Scene Graph 仅 | 60.8 | +2.4 |
| + Cross-view Scene Graph 仅 | 62.3 | +3.9 |
| + Ego + Exo Scene Graphs | 63.7 | +5.3 |
| **+ Full M3CoT (全部三种)** | **66.0** | **+7.6** |

### 关键发现

1. **多视角显著优于单一视角**：Ego+Exo 比仅 Ego 提升约 6%，M3CoT 在此基础上再提升 4-6%
2. **跨视角场景图贡献最大**：跨视角关联比单一视角场景图更有价值
3. **空间推理提升最显著**：需要全局空间信息的问题从多视角融合中获益最多（+8.6%）
4. **需要两视角的问题改进最大**：M3CoT 在此类问题上提升达 9.6%
5. **闭源模型受益更多**：GPT-4o 和 Gemini 比开源模型从 M3CoT 中获得更大提升

## 亮点与洞察

1. **首个多视角 VQA 基准**：E3VQA 填补了 ego-exo 联合理解评估的空白
2. **训练无关的提升**：M3CoT 不需要任何额外训练，即插即用
3. **Spotlight 接收**：高质量的问题定义和系统评估
4. **实际应用价值**：直接适用于 AR/VR 场景中的智能助手
5. **揭示 LVLM 局限**：系统评估揭示了现有 LVLM 在多视角推理上的不足

## 局限与展望

1. **数据集规模有限**：4K QA 对相对较小，可能不足以覆盖所有场景类型
2. **场景图提取质量**：M3CoT 的效果依赖于场景图的准确性
3. **两张图像限制**：实际 AR/VR 场景可能有更多视角
4. **仅静态图像**：未考虑视频序列的时序信息
5. **计算开销**：M3CoT 需要额外的场景图提取步骤，增加推理延迟

## 相关工作与启发

- **Ego-Exo4D**：提供了同步的 ego-exo 视频数据，是 E3VQA 的数据基础
- **视觉问答 (VQA)**：经典的单视角 VQA 基准如 VQAv2、GQA
- **场景图生成**：Visual Genome 等数据集推动了场景图生成技术
- **多视角理解**：Multi-view learning 的经典工作，本文首次结合 LVLM
- **启发方向**：视频级多视角推理、多于两个视角的融合、端到端训练方法

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个 ego-exo 联合 VQA 基准 + 免训练多视角融合方法
- **理论深度**: ⭐⭐⭐ — 主要是方法和基准贡献
- **实验充分性**: ⭐⭐⭐⭐⭐ — 多模型、多问题类型、充分消融
- **实际影响**: ⭐⭐⭐⭐⭐ — AR/VR 应用直接受益
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，可视化丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](../../CVPR2025/multimodal_vlm/embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)

</div>

<!-- RELATED:END -->
