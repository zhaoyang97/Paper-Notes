---
title: >-
  [论文解读] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting
description: >-
  [多模态VLM] 提出 EgoGazeVQA 基准和三种注视引导提示策略（文本/视觉/显著图），首次系统验证了眼动注视信号对提升 MLLM 第一人称视频意图理解的关键价值，Qwen2.5-VL-72B + GazeS 策略在平均准确率上提升 5.8 个百分点。 领域现状：多模态大语言模型（MLLM）在视频理解任务上取得了显著…
tags:
  - "多模态VLM"
---

# In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting

## 一句话总结

提出 EgoGazeVQA 基准和三种注视引导提示策略（文本/视觉/显著图），首次系统验证了眼动注视信号对提升 MLLM 第一人称视频意图理解的关键价值，Qwen2.5-VL-72B + GazeS 策略在平均准确率上提升 5.8 个百分点。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在视频理解任务上取得了显著进展，第一人称（egocentric）视频因其与用户视角天然对齐，被视为实现主动式、个性化 AI 助手的理想载体。

**现有痛点**：现有的第一人称视频 QA 基准（如 QaEgo4D、EgoSchema、EgoTextVQA）仅利用全局视觉帧信息，完全忽略了眼动注视（gaze）这一反映用户注意力和意图的核心线索；MLLM 在推理用户空间和时间意图时准确率远低于人类。

**核心矛盾**：MLLM 使用全局帧构建视觉 token，能获取广泛上下文但无法捕捉佩戴者对具体物体/区域的显式关注信号，导致在需要理解"用户正在看什么"的问题上频繁失败。

**本文目标**：(1) 缺少一个集成注视信号的第一人称视频意图理解基准；(2) 缺少有效利用注视信号来增强 MLLM 意图理解的方法。

**切入角度**：从 Ego4D、EgoExo4D、EGTEA Gaze+ 三大数据集中提取带注视数据的视频片段，利用 MLLM 生成+人工审核构造高质量 QA 对，并设计多种注视引导提示策略。

**核心 idea**：眼动注视信号作为用户意图的直接指示器，可通过文本/视觉/显著图三种提示形式有效注入 MLLM，显著提升第一人称视频意图理解。

## 方法详解

### 整体框架

EgoGazeVQA 包含两大部分：**基准构建流水线** + **注视引导提示策略**。基准构建流水线从三大第一人称数据集提取视频片段、帧描述和注视坐标，利用 Qwen2.5-VL 生成空间/时间/因果意图 QA 对（每题5个选项），再经人工审核（6个维度）确保质量。推理阶段在标准 MLLM 输入基础上叠加三种注视引导策略来增强意图理解。

### 关键设计

**1. Gaze as Textual Prompt (GazeT)**

- **功能**：将逐帧注视坐标编码为归一化的 (x, y) 文本坐标，直接拼接到文本提示中
- **核心思路**：利用 MLLM 底层 LLM 的强文本理解能力，将空间注视信息转化为语言描述，避免视觉与注视尺度不匹配
- **设计动机**：所有注视数据标准化到 0-1 范围，抽象掉不同视频的尺寸差异；在时间意图理解上效果最优，因为 MLLM 更擅长从文本推理时间关系

**2. Gaze as Visual Prompt (GazeV)**

- **功能**：在每帧注视坐标位置绘制 25 像素红色圆圈标记，配合文本提示告知"红圈代表高关注区域"
- **核心思路**：将注视信号嵌入视觉模态，模拟人类视觉注意模式，引导模型关注帧内高显著性区域
- **设计动机**：通过视觉方式更直观地传达空间注意信息，让模型像人类一样被"引导"到关注区域

**3. Sequential Gaze Salience Map (GazeS)**

- **功能**：沿视频时间轴逐步累积注视显著图，将多帧注视轨迹合并为单张热力图，重复注视区域获得更高强度
- **核心思路**：建模注视的时间相关性，通过渐进增强注视显著度同时编码空间和时间意图
- **设计动机**：整体效果最优（尤其空间意图），因为它同时捕捉了"在哪看"和"看了多久/多少次"两个维度

### 损失函数 / 训练策略

- **微调策略**：采用 LoRA 对 Qwen2.5-VL-7B 进行轻量微调（基于 LLaMA-Factory），仅需约 500 条注视引导 QA 对即可带来显著提升
- **跨数据集迁移**：在 EGTEA 上微调 → 测试 Ego4D+EgoExo（平均准确率从 54.0% 提升至 69.5%），反向迁移同样有效（52.0% → 65.4%），说明注视引导能力具有跨域泛化性
- **评估指标**：5 选 1 多选准确率（随机基线 20%），分空间/时间/因果三个维度评估

## 实验关键数据

### 主实验

各 MLLM 在 EgoGazeVQA 上的表现（使用最优 GazeS 策略 vs. 无注视信号）：

| 模型 | 策略 | Spatial | Temporal | Causal | Avg |
|------|------|---------|----------|--------|-----|
| Human | — | 80.7 | 75.6 | 95.2 | 83.8 |
| Qwen2.5-VL-72B | w/o | 57.1 | 45.2 | 79.3 | 60.5 |
| Qwen2.5-VL-72B | GazeS | **64.3** | **50.3** | **84.3** | **66.3** |
| InternVL2.5-8B | w/o | 50.4 | 51.1 | 73.3 | 58.3 |
| InternVL2.5-8B | GazeV | **55.0** | 50.6 | **76.2** | **60.6** |
| MiniCPM-o 2.6 | w/o | 40.8 | 34.5 | 32.5 | 35.9 |
| MiniCPM-o 2.6 | GazeS | **43.2** | **43.5** | **74.4** | **53.7** |

### 消融实验

LoRA 微调跨数据集迁移效果（Qwen2.5-VL-7B）：

| 训练集 | 测试集 | Spatial | Temporal | Causal | Avg |
|--------|--------|---------|----------|--------|-----|
| — (Zero-shot) | Ego4D+EgoExo | 40.4 | 43.3 | 78.2 | 54.0 |
| EGTEA (~500对) | Ego4D+EgoExo | **67.7** | **56.5** | **84.4** | **69.5** |
| — (Zero-shot) | EGTEA | 45.7 | 32.9 | 77.3 | 52.0 |
| Ego4D+EgoExo | EGTEA | **66.5** | **47.0** | **82.8** | **65.4** |

### 关键发现

1. **GazeS 策略整体最优**：在空间意图上提升最显著（Qwen2.5-VL-72B: +7.2pp），但在时间意图上 GazeT 文本提示反而更好，说明 MLLM 从文本推理时间关系的能力优于从叠加注视标记的视觉帧中推理
2. **模型规模决定注视利用效率**：大模型（72B）从注视信号中获益远多于小模型（7B），小模型受限于模型容量难以充分解释注视相关指令
3. **MLLM 自估注视的局限**：让 Qwen2.5-VL-72B 自己预测注视点再用于提示，MSE 越低提升越大（EgoExo 上 MSE=0.038 带来 +2.8pp），但估计不准时甚至有损
4. **场景复杂度影响大**：厨房（目标多且杂乱）准确率最低，医疗/车库（目标少）相对较高；社交游戏场景中多人互动显著拉低性能
5. **少量注视标注数据即可大幅提升**：仅 500 对 LoRA 微调数据就让空间推理提升约 27pp，且具有跨数据集泛化性

## 亮点与洞察

- **首个注视引导的第一人称视频 QA 基准**：填补了 egocentric VQA 领域中注视信号的空白，913 个视频、1757 个 QA 对覆盖多场景多活动
- **三种提示策略的互补性**：GazeS 空间最优、GazeT 时间最优，揭示了 MLLM 对不同模态信息的偏好差异
- **干扰选项设计精巧**：包含反因果选项、空间邻近陷阱、社交影响干扰、高显著性无关物体等多种干扰策略
- **验证了「少量注视数据 + LoRA」的高效路径**：对落地部署友好，不需要大规模注视标注

## 局限与展望

1. **视频时长受限**：平均 30-60 秒的短片段，无法验证在更长视频中注视引导的效果
2. **注视获取依赖硬件**：实际部署需要眼动追踪设备，限制了应用范围；MLLM 自估注视精度不足时甚至有负面效果
3. **场景覆盖有限**：厨房和烹饪活动占比过高（约 60%），可能引入分布偏差
4. **未探索注视与其他传感器融合**：EgoExo4D 提供 IMU、音频等多通道数据，可进一步融合
5. **仅测试多选格式**：开放式问答中注视信号的效果未被验证

## 相关工作与启发

- **GazeGPT (Konrad et al. 2024)**：展示注视数据可改善 MLLM UI 设计，但缺少标准化基准评估 → 本文填补此空白
- **IntentQA (Li et al. 2023)**：针对意图推理但使用第三人称视频 → 转向第一人称+注视提供更自然的意图信号
- **EgoSchema / EgoMemoria**：第一人称长视频推理基准但无注视信号 → 注视可作为强先验显著降低推理难度
- **启发**：注视信号可类比为 visual grounding 中的 referring expression，未来可探索将注视信号作为 visual prompt tuning 的一种形式

## 评分

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首个将注视信号引入 MLLM 视频理解基准的工作，切入角度独特且有实际价值 |
| 技术深度 | ⭐⭐⭐ | 三种提示策略设计合理但技术门槛不高，LoRA 微调实验较为标准 |
| 实验完整度 | ⭐⭐⭐⭐ | 7个模型 × 3种策略 × 多场景/活动维度的全面评测，含微调、自估注视等深入分析 |
| 实用价值 | ⭐⭐⭐⭐ | 对 AR 眼镜、第一人称 AI 助手等场景有直接指导意义，"少量数据+LoRA"路径具有可操作性 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICCV 2025\] OrderChain: Towards General Instruct-Tuning for Stimulating the Ordinal Understanding Ability of MLLM](../../ICCV2025/multimodal_vlm/orderchain_towards_general_instruct-tuning_for_stimulating_the_ordinal_understan.md)
- [\[ICCV 2025\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning](../../ICCV2025/multimodal_vlm/metamorph_multimodal_understanding_and_generation_via_instruction_tuning.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

</div>

<!-- RELATED:END -->
