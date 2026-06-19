---
title: >-
  [论文解读] Unified Reinforcement and Imitation Learning for Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][VLM蒸馏] 提出 RIL（Unified Reinforcement and Imitation Learning）训练框架，结合 GRPO 强化学习和 GAIL 对抗模仿学习，让小型 VLM（7B）通过学习大型 VLM（72B）的文本生成风格来大幅提升性能，无需增加推理延迟或"思考"过程。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLM蒸馏"
  - "强化学习"
  - "模仿学习"
  - "GRPO"
  - "GAIL"
---

# Unified Reinforcement and Imitation Learning for Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.19307](https://arxiv.org/abs/2510.19307)  
**代码**: 无（NVIDIA内部）  
**领域**: 多模态VLM  
**关键词**: VLM蒸馏, 强化学习, 模仿学习, GRPO, GAIL

## 一句话总结

提出 RIL（Unified Reinforcement and Imitation Learning）训练框架，结合 GRPO 强化学习和 GAIL 对抗模仿学习，让小型 VLM（7B）通过学习大型 VLM（72B）的文本生成风格来大幅提升性能，无需增加推理延迟或"思考"过程。

## 研究背景与动机

VLM 通过整合视觉和语言模态实现了强大的多模态理解能力，但大模型（72B+）的部署受限于计算资源。现有提升 VLM 性能的路径各有局限：

**现有策略及痛点**：

**模型规模扩大**：GPT-4o、Qwen2.5-VL-72B 等效果好但无法在手机、AR设备上部署

**Think-Answer 范式**：DeepSeek-R1 等通过长思维链显著提升推理能力，但推理延迟和计算开销大幅增加

**架构修改**：增加多视觉编码器等，改变推理流程，部署灵活性降低

**传统知识蒸馏**：基于特征距离的蒸馏在高维空间中效果有限

**核心切入点**：能否让小模型学习大模型的"说话方式"（文本生成风格），而不需要改变架构或增加推理成本？RIL 结合强化学习（优化答案质量）和模仿学习（学习表达风格），用对抗框架统一两者。

## 方法详解

### 整体框架

RIL 包含三个核心组件：学生 VLM（Generator）、判别器（Discriminator）、LLM-as-a-Judge。训练流程交替进行判别器预训练和 RIL 联合训练，后者在每步结合强化奖励和模仿奖励更新学生模型。

### 关键设计

1. **判别器架构与预训练**

    - 判别器与学生 VLM 共享相同架构和初始参数，仅将语言头（$\mathbb{R}^{d \times v}$ → $\mathbb{R}^{d \times 1}$）替换为线性判别头 + sigmoid
    - 预训练目标：区分学生生成的回答与教师生成的回答
    - 架构一致性设计防止生成器-判别器的"天平问题"，避免一方过强导致训练崩溃
    - 判别器输出的连续分数被**二值化**为 0/1，提供更清晰的学习信号

2. **RIL 奖励设计（双重奖励）**

    - **相似性奖励 $r_s$**：来自判别器，评估学生回答与教师风格的相似度（二值化后）
    - **正确性奖励 $r_a$**：来自 LLM-as-a-Judge（Qwen2.5-32B），评估回答是否与 ground truth 语义一致（同样二值化）
    - 两类奖励互补：判别器关注"说得像不像"，Judge 关注"说得对不对"

3. **GRPO 阶段的教师引导**

    - 在 GRPO 更新中，同时使用学生和教师 VLM 的回答作为候选
    - 当学生对某个问题所有采样回答都错误时，教师的正确回答提供了"逃脱零奖励困境"的路径
    - 这不仅稳定训练，还使学生有机会超越教师

4. **多教师策略**

    - 使用多个大教师 VLM（如 Qwen2.5-VL-72B + InternVL3-78B）提供更多样的回答风格
    - 多教师使判别器更鲁棒，不会过拟合某一种回答模式
    - 学生接触到更丰富的correct回答分布，学习效率更高

### 训练策略

- 学生模型：Qwen2.5-VL-7B/3B, InternVL3-8B/2B/1B
- 教师模型：Qwen2.5-VL-72B, InternVL3-78B
- Judge: Qwen2.5-32B
- 使用 Dr.GRPO（GRPO 的改进版本）作为 RL 基础
- 推理时无需判别器和 Judge，保持原始推理速度

## 实验关键数据

### 主实验（14 个 Benchmark 平均分）

| 模型 | AI2D | ChartQA | MathVista | MMB | MMMU | BLINK | 14 Bench 均分 |
|------|------|---------|-----------|-----|------|-------|--------------|
| Qwen2.5-VL-7B 原始 | 83.9 | 87.3 | 67.8 | 83.5 | 55.0 | 56.4 | ~70.8 |
| + RL (Dr.GRPO) | 84.5 | 90.0 | 69.5 | 84.3 | 57.2 | 60.7 | ~73.3 |
| **+ RIL (单教师)** | 86.7 | 95.4 | 74.5 | 86.8 | 61.8 | 68.5 | **~78.0** |
| **+ RIL (双教师)** | 86.1 | **95.6** | **79.7** | 86.3 | **65.7** | **70.0** | **~79.7** |
| InternVL3-8B + RIL | 87.4 | 95.5 | 74.1 | 88.7 | 66.8 | 60.1 | ~75.5 |

### 消融实验（多教师 vs 单教师）

| 配置 | MMMU | MathVista | BLINK | 均分趋势 |
|------|------|-----------|-------|---------|
| 单教师(同族72B) | 61.8 | 74.5 | 68.5 | 高 |
| 单教师(跨族78B) | 60.9 | 74.6 | 68.1 | 高 |
| **双教师(Both)** | **65.7** | **79.7** | **70.0** | **最高** |

### 关键发现

- **RIL 远超纯 RL**：在 Qwen2.5-VL-7B 上，RIL 比 Dr.GRPO 平均提升约 5-7 个百分点，ChartQA 从 90.0 跃升至 95.4
- **双教师显著优于单教师**：MathVista 上从 74.5/74.6（单教师）跃升至 79.7（双教师），MMMU 从 61.8/60.9 提至 65.7
- **小模型也能大幅提升**：InternVL3-1B 通过 RIL 在多个 benchmark 上提升 3-8 个百分点
- **蒸馏模型的协同效应**：已经经过feature蒸馏的VLM在RIL下表现更好，内在特征对齐与RIL目标互补

## 亮点与洞察

- **范式创新**：将 GAN 式对抗学习引入 VLM 训练，判别器区分"学生写的"和"老师写的"文本风格，思路非常巧妙
- **二值化奖励的稳定性**：连续判别器分数会引入歧义，二值化后训练大幅稳定，这与 GRPO 的 binary reward 设计哲学一致
- **不需要 Think-Answer**：与 Vision-R1 等方法不同，RIL 训练后推理无需思考链，保持原始推理速度
- **架构/tokenizer无关**：RIL 纯粹基于文本回答进行学习，学生和教师可以使用完全不同的图像嵌入和分词器
- **LLM-as-a-Judge 的通用性**：传统 RL 奖励依赖 answer parsing（只适用于数学等有标准答案的任务），Judge 可评估开放式问答

## 局限与展望

- 需要访问大教师 VLM 生成回答，训练成本包含大模型推理
- 判别器训练的稳定性仍需仔细调参（初始化、预训练步数等）
- Judge 模型本身可能有偏差，对某些领域的正确性评估不够准确
- 没有在实际部署场景（手机/边缘设备）上验证推理效率的实际收益
- 双教师来自不同 VLM 族，数据准备工作量较大

## 相关工作与启发

- **vs DeepSeek-R1/Vision-R1**：这些方法依赖 think-answer 过程增加推理延迟，RIL 保持原始推理速度
- **vs 传统知识蒸馏**：传统蒸馏在高维特征空间做对齐，RIL 在自然语言回答层面做对齐（"verbalization effect"），效果更好
- **vs GAIL**：经典 GAIL 用于机器人控制，RIL 首次将其有效适配到 VLM 训练，引入4项关键修改

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将对抗模仿学习引入VLM训练是全新的切入点，结合GRPO的统一框架设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 14个benchmark、多种模型尺寸(1B-8B)、多教师组合、完整消融
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但表格密集度较高，关键创新点需要反复阅读才能准确理解
- 价值: ⭐⭐⭐⭐⭐ 提供了一条不增加推理成本就能大幅提升小VLM性能的实用路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[CVPR 2026\] TTRV: Test-Time Reinforcement Learning for Vision Language Models](../../CVPR2026/multimodal_vlm/ttrv_test-time_reinforcement_learning_for_vision_language_models.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](../../ICCV2025/multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] GoalLadder: Incremental Goal Discovery with Vision-Language Models](goalladder_incremental_goal_discovery_with_vision-language_models.md)

</div>

<!-- RELATED:END -->
