---
title: >-
  [论文解读] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA
description: >-
  [ACL 2025][多模态VLM][持续指令微调] 针对多模态持续指令微调(MCIT)中MoELoRA的参数低效和灾难性遗忘问题，提出BranchLoRA——一种非对称架构，共享矩阵A捕获跨任务通用模式、多路矩阵B编码任务特有知识，配合灵活调参-冻结机制和任务特定路由器，在CoIN benchmark上以更少参数大幅超越前SOTA MoELoRA（ACC: 44.20 vs 37.13, BWT: -20.98 vs -25.91）。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "持续指令微调"
  - "BranchLoRA"
  - "MoE"
  - "灾难性遗忘"
  - "多模态大语言模型"
---

# Enhancing Multimodal Continual Instruction Tuning with BranchLoRA

**会议**: ACL 2025  
**arXiv**: [2506.02041](https://arxiv.org/abs/2506.02041)  
**代码**: [GitHub](https://github.com/BladeDancer957/BranchLoRA)  
**领域**: 多模态VLM  
**关键词**: 持续指令微调, BranchLoRA, MoE, 灾难性遗忘, 多模态大语言模型

## 一句话总结

针对多模态持续指令微调(MCIT)中MoELoRA的参数低效和灾难性遗忘问题，提出BranchLoRA——一种非对称架构，共享矩阵A捕获跨任务通用模式、多路矩阵B编码任务特有知识，配合灵活调参-冻结机制和任务特定路由器，在CoIN benchmark上以更少参数大幅超越前SOTA MoELoRA（ACC: 44.20 vs 37.13, BWT: -20.98 vs -25.91）。

## 研究背景与动机

**领域现状**：多模态大语言模型(MLLM)通过指令微调实现与人类意图的对齐。实际应用中，模型需要不断适应新任务和新指令，但从头重训成本过高，因此出现了多模态持续指令微调(MCIT)范式。

**现有痛点**：
- MCIT面临灾难性遗忘(CF)——学新任务时旧任务性能急剧下降
- 现有MoELoRA方法聚合所有LoRA expert的输出，容易覆写旧知识
- MoELoRA的共享router持续更新，导致expert分配偏向最新任务
- 所有expert的矩阵A和B都独立维护，存在参数冗余

**核心矛盾**：MoELoRA中多个expert的矩阵A参数在持续训练中趋于收敛（捕获共性），而矩阵B保持可区分（捕获特性），说明为A维护多份独立副本是浪费参数的。

**本文目标** 在MCIT场景下，设计一种更高效的LoRA架构，同时解决MoELoRA的参数冗余和灾难性遗忘两个核心问题。

**切入角度**：通过实验分析发现MoELoRA的A矩阵收敛现象，据此设计非对称架构——共享A（树干），多路B（树枝），辅以冻结和路由机制防遗忘。

**核心 idea**：MoELoRA中A矩阵趋同B矩阵分化→共享A+多路B的非对称BranchLoRA + 灵活冻结机制 + 任务特定路由器 = 更少参数 + 更少遗忘。

## 方法详解

### 整体框架

BranchLoRA集成到MLLM每层的Feed-Forward模块中，pipeline：
1. 输入经multi-head attention得到中间表示x
2. x通过共享矩阵A投影到低维空间
3. 任务特定router对x的首token计算expert权重，稀疏选择top-k个B矩阵
4. 各B矩阵独立投影回高维空间，按router权重加权聚合
5. 推理时task selector自动路由到正确router（无需任务标识）

### 关键设计

1. **非对称架构（共享A + 多路B）**:
    - 功能：消除参数冗余，同时保留任务特有知识的编码能力
    - 核心思路：所有expert共享一个矩阵A（捕获跨任务共性），每个expert维护独立矩阵B（捕获任务特性），形成"树干-树枝"结构
    - 设计动机：实验观察到MoELoRA的A矩阵在持续训练中收敛（t-SNE可视化高度重叠），而B矩阵保持可区分→无需为A维护多份

2. **灵活调参-冻结机制（Flexible Tuning-Freezing）**:
    - 功能：保护旧任务知识同时允许跨任务知识迁移
    - 核心思路：训练完当前任务后，分析router输出分布，将最活跃的top-k个B矩阵冻结；新任务训练时，router可选择(a)仅可调expert、(b)可调+冻结expert混合、(c)仅冻结expert
    - 设计动机：冻结防遗忘（旧知识不被覆写），但允许router访问冻结expert实现跨任务知识迁移（类比大脑巩固记忆同时整合新信息）

3. **任务特定路由器 + 自动任务选择器**:
    - 功能：防止router偏向最新任务，且推理时无需任务标识
    - 核心思路：每训练一个新任务就增量引入一个新router（带独立W_r参数），并训练对应的task key（图像key + 文本key），通过cosine similarity alignment loss将key与任务样本的embedding对齐
    - 设计动机：共享router的持续更新导致旧任务的最优expert分配被遗忘；推理时通过计算测试样本与各task key的相似度自动选择router（准确率95.8%）

### 损失函数 / 训练策略

- 总损失：L_total = L_task + λ · L_align
- L_task：标准自回归生成损失
- L_align = Σ(1-cos(e_img, k_img)) + Σ(1-cos(e_txt, k_txt))，将task key与样本embedding对齐
- 参数设置：rank=128, α=256, N=8 experts, top-k=2, λ=1.0
- 冻结vision encoder和LLM，仅微调projector和LoRA
- 使用8×NVIDIA H800 GPU训练

## 实验关键数据

### 主实验（LLaVA-1.5-7B, CoIN benchmark, 8个sequential tasks）

| 方法 | ACC↑ | MAA↑ | BWT↑ | 可训练参数 |
|---|---|---|---|---|
| LoRA | 28.74 | 32.97 | -32.62 | - |
| LwF | 30.41 | 34.95 | -27.03 | - |
| EWC | 32.90 | 36.93 | -27.46 | - |
| MoELoRA | 37.13 | 42.76 | -25.91 | 350M |
| **BranchLoRA** | **44.20** | **49.94** | **-20.98** | **222M** |
| Multi-task (上界) | - | 57.18 | - | - |

### 模型规模扩展（LLaVA-1.5-13B）

| 方法 | ACC↑ | MAA↑ | BWT↑ |
|---|---|---|---|
| MoELoRA | 42.51 | 49.14 | -23.62 |
| **BranchLoRA** | **49.27** | **55.73** | **-19.29** |

### 消融实验（LLaVA-1.5-7B）

| 变体 | ACC↑ | MAA↑ | BWT↑ |
|---|---|---|---|
| MoELoRA baseline | 37.13 | 42.76 | -25.91 |
| + 共享矩阵A | 38.19 | 43.95 | -25.32 |
| + 动态稀疏选择 | 39.96 | 45.53 | -23.77 |
| + 灵活冻结机制 | 42.22 | 47.76 | -22.41 |
| + 任务特定router (完整BranchLoRA) | **44.20** | **49.94** | **-20.98** |

### 效率对比

| 方法 | 可训练参数 | 训练时间(ms/batch) |
|---|---|---|
| MoELoRA | 350M | 62 |
| BranchLoRA | **222M** | **51** |

### 关键发现

- 共享A不仅减少37%参数，还略微提升性能——验证了A矩阵收敛的observation
- 每个设计组件都带来增量改进：共享A → 稀疏选择 → 冻结机制 → 任务路由器
- 在7B和13B上都一致超越MoELoRA，表明方法具有可扩展性
- 更大模型(13B)遗忘更少（BWT: -19.29 vs -20.98），但遗忘仍然存在
- 增加指令多样性（10Type）可进一步提升BranchLoRA性能（ACC: 44.20→46.47）
- 任务选择器准确率95.8%——偶尔的误分类并未影响整体优势

## 亮点与洞察

- **数据驱动的架构设计**：不是凭直觉设计架构，而是先做参数分析实验发现A矩阵收敛现象，再据此设计非对称结构——方法论值得学习
- **"树干-树枝"比喻精准**：共享A如树干（稳定的共性基础），多路B如树枝（灵活的任务适配），形象且准确
- **冻结机制模拟人脑记忆巩固**：已学知识冻结保护 + 新知识通过router访问旧expert迁移——biologically inspired
- **效率与效果双赢**：比MoELoRA少37%参数、快18%训练速度，但ACC高7个点——稀有的帕累托改进
- **实用的推理方案**：task selector消除了对任务标识的依赖，使方法更贴近真实应用

## 局限与展望

- 实验仅在CoIN benchmark上验证，任务多样性有限（8个多模态数据集）
- 任务序列顺序是否影响结果未充分探讨
- top-k的选择（k=2）是否在不同场景下需要调整未讨论
- 当任务数量非常大时，冻结expert可能导致可调expert不足
- 未与model merging等方法进行比较或结合
- 非多模态任务上的效果未验证

## 相关工作与启发

- **vs MoELoRA (CoIN)**：直接基线——BranchLoRA通过非对称架构+冻结+任务路由三重改进大幅超越
- **vs HydraLoRA (Tian et al. 2024)**：HydraLoRA在多任务场景中也观察到类似的A矩阵收敛，但本文在持续学习场景中进一步发展了这一发现
- **vs EWC/LwF**：传统持续学习方法在MCIT场景下效果有限，远不如基于MoE的方案
- **vs 标准LoRA**：LoRA无任何遗忘缓解措施，BWT=-32.62远差于BranchLoRA的-20.98

## 评分

- 新颖性: ⭐⭐⭐⭐ 从参数分析observation出发设计非对称架构，灵活冻结+任务路由组合新颖
- 实验充分度: ⭐⭐⭐⭐ 消融完整，双尺度模型验证，效率分析充分；但benchmark单一
- 写作质量: ⭐⭐⭐⭐ 动机清晰（参数分析→架构设计），图表直观，逻辑连贯
- 价值: ⭐⭐⭐⭐ 为MLLM的持续学习提供了比MoELoRA更优的方案，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[CVPR 2026\] Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance](../../CVPR2026/multimodal_vlm/multimodal_continual_instruction_tuning_with_dynamic_gradient_guidance.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](../../ICML2025/multimodal_vlm/dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](../../NeurIPS2025/multimodal_vlm/visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->
