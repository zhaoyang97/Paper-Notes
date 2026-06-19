---
title: >-
  [论文解读] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning
description: >-
  [NeurIPS 2025][多模态VLM][VLM decision-making] 发现VLM的决策推理能力可与视觉感知解耦——用文本描述替代图像时决策准确率不降反升；据此提出Praxis-VLM，在纯文本场景上通过多阶段GRPO与自适应reward训练决策推理能力，推理时零样本迁移到视觉输入，在三大决策benchmark上全面超越SFT基线，尤其在OOD场景泛化优势显著。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLM decision-making"
  - "text-driven RL"
  - "GRPO"
  - "跨模态"
  - "embodied reasoning"
---

# Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2503.16965](https://arxiv.org/abs/2503.16965)  
**代码**: [https://github.com/Derekkk/Praxis-VLM](https://github.com/Derekkk/Praxis-VLM)  
**领域**: 多模态VLM / Agent决策 / 强化学习  
**关键词**: VLM decision-making, text-driven RL, GRPO, cross-modal transfer, embodied reasoning

## 一句话总结

发现VLM的决策推理能力可与视觉感知解耦——用文本描述替代图像时决策准确率不降反升；据此提出Praxis-VLM，在纯文本场景上通过多阶段GRPO与自适应reward训练决策推理能力，推理时零样本迁移到视觉输入，在三大决策benchmark上全面超越SFT基线，尤其在OOD场景泛化优势显著。

## 研究背景与动机

**领域现状**：VLM在视觉理解领域表现出色，但在需要"先思考再决策"的复杂情境决策任务（如看到交通事故该怎么做、机器人应该选择哪个动作）中，缺乏显式的多步推理能力。近期DeepSeek-R1、OpenAI o1等工作证明了RL可以显著增强LLM的推理能力，也有工作（R1-OneVision、Vision-R1、OpenVLThinker）尝试将这种推理增强迁移到VLM。

**现有痛点**：现有的VLM推理增强方法严重依赖大规模图文配对数据——需要图像+问题+推理链的三元组来做SFT或RL训练。在决策场景中，这类配对数据极度稀缺且标注成本极高，涵盖多样化的真实世界决策情境更是困难重重。

**核心矛盾**：增强VLM决策推理需要大量图文配对数据 vs 决策场景中图文配对数据极度稀缺——数据需求与数据可获得性之间存在根本冲突。

**本文目标**：如何在不依赖图文配对数据的前提下，赋予VLM强大的情境决策推理能力，并使其能泛化到多种视觉决策场景？

**切入角度**：作者做了一个关键的预实验——在VIVA和PCA-Bench两个视觉决策benchmark上，将图像输入替换为文本描述（GPT-4o caption或数据集标注文本），发现Qwen2.5-VL的决策准确率竟然与直接用图像输入持平甚至更高。这揭示了一个深刻洞察：决策推理的核心能力存在于语言域，可以与视觉感知解耦学习。这与认知科学中的"心理模型理论"一致——人类也是通过构建内部语言表征来推理和决策，然后将这些内部模型应用于感知体验。

**核心 idea**：用纯文本数据训练VLM的决策推理能力（通过多阶段GRPO+自适应reward），训练好的推理能力可以在推理时自动迁移到处理视觉输入的场景，实现数据高效的VLM决策增强。

## 方法详解

### 整体框架

Praxis-VLM包含三个关键阶段：

1. **文本决策数据构造**：用GPT-4o批量合成纯文本决策场景数据（10K训练+1K验证），每个样本包含文本情境描述+多选决策问题+正确答案，不需要任何图像数据
2. **多阶段GRPO训练**：Stage 1用geometry3k数学数据做冷启动建立基础推理能力→ Stage 2在文本决策数据上做RL训练激发复杂决策推理能力，每阶段配合不同的自适应reward
3. **视觉推理迁移**：推理时将文本输入替换为真实视觉输入（图像/视频帧），完整VLM架构（含vision encoder）处理多模态输入，文本训练获得的推理能力自动迁移

核心设计要点：训练时只更新LLM参数，不碰vision encoder；推理时用完整VLM架构处理图像输入，实现文本→视觉的零样本能力迁移。

### 关键设计

1. **决策推理与视觉感知解耦的验证与利用**
    - 功能：作为整个方法论的实证基础，证明VLM的决策瓶颈在推理而非视觉感知
    - 核心思路：在VIVA和PCA-Bench上对比两种设置——(1) 用原始图像作为输入情境 (2) 用文本描述（GPT-4o caption或数据集标注文本）替代图像。实验发现文本情境设置的决策准确率与图像输入持平甚至更优，说明决策推理可以从文本表征中充分学习
    - 设计动机：如果核心决策能力不绑定于视觉感知，就可以绕开昂贵的图文配对数据，用纯文本数据来训练推理能力——这是整个Praxis-VLM框架的理论根基

2. **多阶段GRPO训练与自适应R1 Reward**
    - 功能：分阶段逐步建立从格式遵从→逻辑推理→复杂决策的能力链
    - 核心思路：
        - **Stage 1（冷启动）**：用geometry3k几何数据做GRPO训练。Reward = R_accuracy + R_format + 0.5·R_tag。R_tag确保模型学会`<think></think><answer></answer>`格式（检查每个特殊token恰好出现一次），一旦格式稳定就移除R_tag，转向以R_accuracy为主
        - **Stage 2（决策RL）**：在合成文本决策数据上训练。Reward = R_accuracy + 0.8·R_format + 0.5·R_len。R_len = min(word_count/250, 1.0)鼓励更长更充分的推理链
        - 关键发现：无需先做SFT冷启动再RL，只要配合自适应reward策略就可以直接在instruction-tuned VLM上做GRPO，简化了训练pipeline
    - 设计动机：直接在决策数据上做RL效果不佳（模型还不会格式化推理输出），数学数据的逻辑推理性质天然适合做冷启动；分阶段adaptive reward让模型在每个阶段专注学习不同技能，避免多目标冲突

3. **GPT-4o驱动的文本决策数据合成**
    - 功能：提供高质量、多样化的纯文本决策训练数据，支撑RL训练
    - 核心思路：先手工构造10个种子问题作为in-context examples，然后prompt GPT-4o批量生成（每批10个样本+去重），最终得到10K训练+1K验证样本。每个样本为文本情境描述+多选决策问题+答案的三元组，设计为需要多步推理才能正确决策，且可用简单规则评估（选择题格式）
    - 设计动机：需要挑战性足够高（迫使模型学习推理而非模式匹配）且可用规则评估（避免复杂reward modeling和reward hacking风险）的训练数据；批量生成+去重策略保证了数据多样性；不需要任何图像数据和人工过滤，实现域无关的快速数据构造

### 损失函数 / 训练策略

- **优化算法**：GRPO（Group Relative Policy Optimization）——对每个query采样G=5条response，计算组内归一化优势函数A_hat，用clip-PPO目标+KL散度惩罚更新策略
- **KL散度系数**：β=0.01，平衡策略更新幅度与参考策略的距离
- **学习率**：1e-6
- **训练范围**：全参微调（LLM部分），vision encoder在训练期间冻结
- **推理配置**：vLLM + greedy decoding，最大序列长度1024 tokens
- **硬件**：4×A100/H100 GPU

## 实验关键数据

### 主实验

| 模型 | VIVA (%) | PCA-Bench (%) | EgoNormia (OOD, %) |
|------|----------|---------------|---------------------|
| Qwen2.5-VL-3B | 76.61 | 48.58 | 51.92 |
| + SFT | 77.42 | 46.37 | 35.06 |
| + Reason SFT | 75.81 | 49.53 | 28.34 |
| **Praxis-VLM-3B** | **79.03** | **50.79** | **54.27** |
| Qwen2.5-VL-7B | 80.97 | 46.37 | 46.19 |
| + SFT | 81.13 | 45.74 | 34.83 |
| + Reason SFT | 78.79 | 53.00 | 34.08 |
| **Praxis-VLM-7B** | **84.03** | **60.25** | **54.33** |

核心发现：SFT和Reason SFT在OOD的EgoNormia上严重退化（从46.19降到34.83/34.08），而Praxis-VLM反而提升（54.33 > 46.19），RL学到的推理能力是真正可迁移的。

### 消融实验

| 消融维度 | VIVA (%) | PCA-Bench (%) | EgoNormia (%) |
|----------|----------|---------------|---------------|
| Praxis-VLM-7B (完整) | 84.03 | 60.25 | 54.33 |
| w/o math cold start (one-stage) | 83.87 | 58.99 | 49.57 |
| Praxis-VLM-3B (完整) | 79.03 | 50.79 | 54.27 |
| w/o math cold start (one-stage) | 79.52 | 50.79 | 53.13 |

**Diverse Sampling (7B, 8 samples, T=0.2)**:

| 方法 | VIVA Orig→Major→Pass@1 | PCA-Bench Orig→Major→Pass@1 | EgoNormia Orig→Major→Pass@1 |
|------|------------------------|-----------------------------|-----------------------------|
| Qwen2.5-VL-7B | 80.97→80.73→80.81 | 46.37→48.27→56.47 | 46.19→46.36→54.50 |
| Reason SFT | 78.79→80.64→89.03 | 53.00→58.36→82.33 | 34.08→35.69→66.04 |
| Praxis-VLM-7B | 83.87→84.36→89.27 | 58.99→61.83→77.92 | 49.57→55.08→72.23 |

### 关键发现

- **Math cold start的核心价值在OOD泛化**：域内任务（VIVA/PCA-Bench）差异不大，但在OOD的EgoNormia上7B模型从49.57提升到54.33（+4.76），说明数学冷启动强化了底层逻辑推理架构
- **推理长度与难度正相关**：将样本按Praxis-VLM生成的推理链长度分为5个quintile bin，发现更长推理对应更难样本；但在同等难度下Praxis-VLM始终优于无推理的baseline
- **Majority Vote优势**：Praxis-VLM在Majority Vote上全面优于Reason SFT，说明GRPO学到的推理分布中心更可靠，不仅能找到正确路径（高Pass@1），还能更稳定地收敛到正确答案
- **超长推理的overthinking风险**：最长20%的样本出现准确率下降，部分原因是超过1024 token被截断，部分原因是过长推理引入噪声干扰最终决策

## 亮点与洞察

- **"决策推理可与视觉感知解耦"是极具认知科学启发的发现**——呼应心理模型理论，人类也是通过语言构建内部表征来推理决策，再将其应用于感知体验。这一发现为VLM训练开辟了全新的数据高效路径
- **纯文本训练→视觉推理迁移**的范式非常优雅：训练时完全不需要图文配对数据，推理时直接处理视觉输入，实现了训练数据需求与推理能力之间的解耦
- **跳过SFT冷启动直接GRPO**的发现简化了训练pipeline——只要设计好自适应reward（先格式后准确率），instruction-tuned模型就能直接through RL学习推理
- **R_len reward的有效性**打破了"更长推理不一定更好"的刻板印象——在决策任务中，更充分的情境分析确实带来更好的决策质量
- **4维推理分析**（情境分析、行动结果评估、安全风险管理、规则规范遵从）揭示了模型学到了结构化的决策思维模式，不是黑箱式的端到端映射
- **错误分析**的三大失败模式（情境误解、安全优先级错误、规范对齐缺失）为后续改进指明了具体方向

## 局限与展望

- 仅在3B/7B模型上验证，更大模型（如72B）上的效果尚不清楚，特别是文本→视觉迁移是否随模型规模线性提升
- 文本决策数据由GPT-4o合成，可能存在域偏差——合成数据的情境分布能否覆盖真实世界的长尾场景存疑
- EgoNormia用视频帧拼接为单图输入，对VLM视频理解能力的评估不够原生，需要在真正的视频输入上验证
- 推理链最大长度限制为1024 tokens，部分复杂场景的推理被截断——需要探索更长context下的推理性能
- 未与VLA模型等其他决策增强方法进行直接对比，也未探索与其他RL方法（如DPO、PPO）的组合效果
- stage 1冷启动用固定的geometry3k数据集，是否存在更优的冷启动数据选择策略值得探索

## 相关工作与启发

- **vs R1-OneVision / Vision-R1**：这些方法在图文配对数据上做RL增强VLM推理，Praxis-VLM证明对于决策任务，纯文本训练就足够——两者可以互补，文本训练处理高层推理，视觉训练处理感知细节
- **vs NoisyRollout**：NoisyRollout通过视觉扰动增强RL exploration，Praxis-VLM完全绕开视觉域用纯文本训练——两种数据效率策略从不同维度解决训练数据瓶颈
- **vs DeepSeek-R1**：Praxis-VLM将R1的RL推理增强范式从纯LLM扩展到跨模态VLM决策，验证了RL推理能力的跨模态可迁移性
- **心理模型理论的AI实践**：该工作为认知科学假说（人类通过语言表征做决策推理）提供了计算验证——可以反过来启发认知科学研究
- **数据效率启示**：对于需要"理解+推理"的VLM任务，可以拆解为视觉感知（用已有VLM能力）和高层推理（用文本数据训练）两个独立模块，大幅降低训练数据需求

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — "决策推理与视觉感知可解耦"这一发现既有实验支撑又有认知理论呼应，纯文本训练VLM决策推理的范式具有开创性意义
- **实验充分度**: ⭐⭐⭐⭐ — 3个benchmark+多种baseline+diverse sampling+推理维度分析+错误分析，非常全面；扣分点是模型规模仅到7B，且缺少与VLA等其他决策方法的对比
- **写作质量**: ⭐⭐⭐⭐⭐ — 从preliminary finding到方法设计再到实验验证的叙事逻辑极为流畅，"Language is the dress of thought"引用画龙点睛，每个设计选择都有清晰动机
- **对我的价值**: ⭐⭐⭐⭐⭐ — "文本训练→视觉迁移"范式可推广到更多视觉推理任务，自适应reward设计策略可直接复用，4维决策推理框架可用于设计更细粒度的reward
---
title: >-
  [论文解读] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning
description: >-
  [NeurIPS 2025][多模态][VLM decision-making] 发现VLM的决策推理能力可以与视觉感知解耦——用文本描述替代图像时决策性能不降反升，据此提出Praxis-VLM：在纯文本场景上用GRPO训练决策推理能力，然后零样本迁移到视觉输入推理，在VIVA/PCA-Bench/EgoNormia三个决策benchmark上超越SFT基线且泛化性更强。
tags:
  - NeurIPS 2025
  - 多模态
  - VLM decision-making
  - text-driven RL
  - GRPO
  - 跨模态
  - embodied reasoning
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[ICML 2025\] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning](../../ICML2025/multimodal_vlm/towards_efficient_online_tuning_of_vlm_agents_via_counterfactual_soft_reinforcem.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)

</div>

<!-- RELATED:END -->
