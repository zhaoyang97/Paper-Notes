---
title: >-
  [论文解读] Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis
description: >-
  [AAAI 2026][人体理解][面部情绪分析] 提出 Facial-R1，一个三阶段对齐训练框架（SFT → RL → 数据合成），通过将 AU 和情绪标签作为可验证奖励信号来对齐 VLM 的推理过程与情绪识别结果，在 8 个基准上达到 SOTA，并构建了 FEA-20K 数据集。 面部情绪分析（Facial Emoti…
tags:
  - "AAAI 2026"
  - "人体理解"
  - "面部情绪分析"
  - "强化学习"
  - "Action Unit"
  - "视觉语言模型"
  - "GRPO"
---

# Facial-R1: Aligning Reasoning and Recognition for Facial Emotion Analysis

**会议**: AAAI 2026  
**arXiv**: [2511.10254](https://arxiv.org/abs/2511.10254)  
**代码**: [https://github.com/RobitsG/Facial-R1](https://github.com/RobitsG/Facial-R1)  
**领域**: Human Understanding  
**关键词**: 面部情绪分析, 强化学习, Action Unit, 视觉语言模型, GRPO

## 一句话总结
提出 Facial-R1，一个三阶段对齐训练框架（SFT → RL → 数据合成），通过将 AU 和情绪标签作为可验证奖励信号来对齐 VLM 的推理过程与情绪识别结果，在 8 个基准上达到 SOTA，并构建了 FEA-20K 数据集。

## 研究背景与动机

面部情绪分析（Facial Emotion Analysis, FEA）是传统面部情绪识别（FER）的扩展，它不仅要给出情绪标签，还需要识别面部动作单元（AU），并基于 AU 生成可解释的情绪推理过程。近年来，视觉语言模型（VLM）如 LLaVA、InternVL 等被引入 FEA 任务，取得了不错的效果。

然而，现有方法存在两个核心痛点：

**推理幻觉**：VLM 缺乏情绪领域先验知识，容易生成看似合理但实际错误的情绪解释，比如遗漏关键面部特征或对 AU 的误判。

**推理与识别错位**：即使模型在推理过程中识别了正确的情绪线索，最终的情绪标签仍可能与推理结论矛盾，因为推理路径与标签之间缺乏内在的因果关联。

已有方法（如 ExpLLM、FABA）尝试通过构造细粒度指令微调数据来缓解这些问题，但高质量情绪推理数据难以大规模采集，且过度严格的指令微调限制了 VLM 的灵活思考能力。

本文的核心 idea：用**可验证的情绪因子（AU 和情绪标签）作为强化学习的奖励信号**，而非硬性规定推理路径，让模型在训练中自然涌现灵活的推理模式，从而同时解决幻觉和推理-识别错位问题。

## 方法详解

### 整体框架
Facial-R1 采用三阶段渐进式训练：
- **Stage 1: SFT**（监督微调）— 用少量高质量样本建立基础情绪推理能力
- **Stage 2: RL**（强化学习）— 用情绪因子作为奖励信号对齐推理与识别
- **Stage 3: Data Synthesis**（数据合成）— 迭代扩充训练数据实现自我提升

### 关键设计

1. **最小化监督微调（SFT）**:

    - 功能：用 GPT-4o-mini 生成仅 300 个高质量情绪分析样本进行微调
    - 核心思路：在指令中嵌入 AU 定义等情绪领域知识，让 VLM 建立面部表情与情绪之间的基本推理能力
    - 设计动机：以极低的初始化成本消除推理幻觉，避免大规模标注的瓶颈

2. **基于可验证奖励的强化学习（RL with GRPO）**:

    - 功能：使用 GRPO 算法，设计三种奖励成分引导模型训练
    - 核心思路：复合奖励 $R = R_{AU} + R_{acc} + R_{format}$
        - **AU 奖励 $R_{AU}$**：采用 F1 score 衡量模型预测 AU 与 ground truth 的匹配度，鼓励模型基于可观察的面部特征进行推理，缓解奖励稀疏问题
        - **准确率奖励 $R_{acc}$**：情绪标签预测正确为 1，否则为 0，直接对齐推理与识别
        - **格式奖励 $R_{format}$**：要求输出使用 `<think>` 和 `<answer>` 标签，规范化推理结构
    - 设计动机：与 SFT 相比，RL 阶段不限制具体推理路径，只要求模型关注两个情绪事实（AU 和情绪标签），从而增强灵活性和鲁棒性

3. **迭代数据合成**:

    - 功能：利用前两个阶段训练好的模型自动生成大规模情绪推理数据
    - 核心思路：从 FABA-Instruct 等数据集中取问题和 GT 标签构造指令，用训练好的 VLM 生成推理，再通过自动过滤（AU/情绪/格式三重检查）和人工审核确保质量
    - 设计动机：绕过人工标注瓶颈，通过多轮迭代训练持续扩充数据，最终构建出包含 17,737 训练样本和 1,688 测试样本的 FEA-20K 数据集

### 损失函数 / 训练策略
- SFT 阶段使用标准的交叉熵损失
- RL 阶段采用 GRPO 算法，通过组内相对优势 $A_i = (R^i - \text{mean})/\text{std}$ 进行策略优化
- 数据合成阶段引入反思机制：如果模型初次推理有误，会引导其自我修正后重新生成

## 实验关键数据

### 主实验

**AU 识别（F1↑）**:

| 数据集 | 指标 | Facial-R1 | 之前 SOTA | 提升 |
|--------|------|-----------|-----------|------|
| DISFA | F1 | **73.1** | 72.9 (Face-LLaVA) | +0.2 |
| BP4D | F1 | 67.4 | **69.3** (Norface) | -1.9 |
| RAF-AU | F1 | **70.2** | 69.5 (Exp-BLIP) | +0.7 |
| FABA-Instruct | F1 | **68.3** | 61.9 (FMAE) | +6.4 |

**情绪识别**:
- RAF-DB: Facial-R1 在全部 7 个情绪类别上排名第一，远超 GPT-4o（62.7% Acc）
- AffectNet: Facial-R1 达到 **65.2%** 准确率（8 类），在 happiness、sadness、anger、surprise、fear 上均最优

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Qwen2.5-VL (zero-shot) | 22.1 F1 (DISFA) | 基线，缺乏情绪先验 |
| + SFT only (300 samples) | 显著提升 | 消除幻觉，建立基础能力 |
| + SFT + RL | 进一步提升 | 对齐推理与识别 |
| + SFT + RL + Data Synthesis | **73.1** F1 (DISFA) | 完整框架，全面 SOTA |
| 仅用 SFT 大量数据 | 低于完整方案 | 过度限制推理灵活性 |

### 关键发现
- 仅 300 条 SFT 数据即可有效消除推理幻觉，建立基础情绪推理能力
- RL 阶段的 AU 奖励对减少虚假推理效果最显著，情绪准确率奖励对消除推理-识别错位效果最好
- 数据合成实现了无需大规模人工标注的"数据飞轮"效应
- 在 FABA-Instruct 上 F1 提升 6.4 个点，说明方法在复杂推理场景优势明显

## 亮点与洞察
- **极低初始化成本**：只需 300 条 GPT-4o-mini 生成的样本 + AU/情绪弱标签即可启动整个训练流程
- **将 DeepSeek-R1 的 GRPO 思路迁移到表情分析**：从数学推理的可验证奖励，拓展到情绪分析的可验证情绪因子
- **推理灵活性 vs 路径限制的权衡**：RL 比 SFT 更适合情绪推理，因为情绪表达是高度个人化的，不应强制统一路径
- 可视化显示 Facial-R1 能准确检测多个 AU 并合理推理出情绪，而基线 VLM 常出现 AU 误判

## 局限与展望
- 当前数据合成依赖 FABA-Instruct 的图像来源，多样性受限
- BP4D 上未超过 Norface（69.3 vs 67.4），说明对实验室控制场景的泛化仍有提升空间
- 仅支持离散情绪分类，未考虑连续维度（valence-arousal）或复合情绪
- 推理速度受限于 VLM 的自回归生成，实时应用有挑战

## 相关工作与启发
- DeepSeek-R1/GRPO 的可验证奖励 RL 范式在特定领域（情绪分析）也有效
- "最小监督 + RL + 数据合成"三阶段范式可推广到其他需要可解释推理的视觉任务
- AU 作为情绪的中间表示，为其他情感计算任务提供了可借鉴的结构化推理路径

## 评分
- 新颖性: ⭐⭐⭐⭐ — GRPO 应用到表情分析有创新，但三阶段框架并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ — 8 个基准、多任务评估、消融实验完整
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，动机和方法衔接自然
- 价值: ⭐⭐⭐⭐ — FEA-20K 数据集和低成本训练范式有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](../../ICLR2026/human_understanding/neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)
- [\[ECCV 2024\] Facial Affective Behavior Analysis with Instruction Tuning](../../ECCV2024/human_understanding/facial_affective_behavior_analysis_with_instruction_tuning.md)
- [\[ECCV 2024\] Generalizable Facial Expression Recognition](../../ECCV2024/human_understanding/generalizable_facial_expression_recognition.md)
- [\[CVPR 2026\] CLEX: Complementary Label Exchange Learning for Noisy Facial Expression Recognition](../../CVPR2026/human_understanding/clex_complementary_label_exchange_learning_for_noisy_facial_expression_recogniti.md)
- [\[CVPR 2026\] Dynamic Label Noise Suppression with Optimal Teacher Pool for Facial Expression Recognition](../../CVPR2026/human_understanding/dynamic_label_noise_suppression_with_optimal_teacher_pool_for_facial_expression_.md)

</div>

<!-- RELATED:END -->
