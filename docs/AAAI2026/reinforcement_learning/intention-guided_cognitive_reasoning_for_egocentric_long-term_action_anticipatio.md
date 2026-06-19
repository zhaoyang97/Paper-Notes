---
title: >-
  [论文解读] Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation
description: >-
  [AAAI 2026][强化学习][长期动作预测] 提出INSIGHT框架，一个面向第一人称长期动作预测的两阶段统一框架：第一阶段通过手-物交互区域特征提取和动词-名词共现矩阵增强动作表示；第二阶段引入基于GRPO的强化学习认知推理模块，模拟"感知→推理→回答"的结构化认知过程进行意图推断和动作预测。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "长期动作预测"
  - "第一人称视频"
  - "手-物交互"
  - "认知推理"
  - "GRPO"
---

# Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation

**会议**: AAAI 2026  
**arXiv**: [2508.01742](https://arxiv.org/abs/2508.01742)  
**代码**: [github.com/CorrineQiu/INSIGHT](https://github.com/CorrineQiu/INSIGHT)  
**领域**: 强化学习  
**关键词**: 长期动作预测, 第一人称视频, 手-物交互, 认知推理, GRPO

## 一句话总结

提出INSIGHT框架，一个面向第一人称长期动作预测的两阶段统一框架：第一阶段通过手-物交互区域特征提取和动词-名词共现矩阵增强动作表示；第二阶段引入基于GRPO的强化学习认知推理模块，模拟"感知→推理→回答"的结构化认知过程进行意图推断和动作预测。

## 研究背景与动机

长期动作预测（LTA）旨在根据观测到的第一人称视频片段预测未来动作序列，是人机交互、增强现实和辅助系统中的关键能力。准确预测用户未来动作使AI系统能够主动适应行为、提供及时帮助。

**现有方法的三大局限**：

**忽视细粒度视觉线索**：现有方法对手-物交互（HOI）区域的细粒度信息利用不足。HOI区域密集包含与动作高度相关的线索，对于区分微妙的上下文相关行为至关重要。通用视觉编码器直接处理整帧图像，丢失了这些关键的第一人称感知细节。

**忽略动词-名词语义关联**：独立预测动词和名词可能产生不合理的组合（如"drink + guitar"），降低预测可靠性。现有方法缺少对动词-名词共现统计的显式建模。

**缺乏显式认知推理**：大多数方法将LTA视为被动的序列预测任务，缺少主动的决策推理过程。基于LLM的方法虽引入文本推理，但仅依赖静态先验，缺乏动态意图推断能力，在复杂的扩展时间场景中表现脆弱。

## 方法详解

### 整体框架

INSIGHT由两个阶段组成：
- **第一阶段：手-物语义动作识别** — 提取判别性视觉特征，增强语义一致性
- **第二阶段：显式认知推理预测** — 模拟 "think → reason → answer" 的认知过程

### 关键设计

#### 1. **HOI增强特征提取**

传统方法直接对整帧图像应用视觉编码器。INSIGHT引入HOI聚焦的特征提取策略：

- 对每个视频段 $S_k$ 均匀采样4帧 $F_{k,T}$
- 使用预训练的100DOH检测器对每帧进行HOI区域检测，再用SAM2细化高分辨率掩码，得到精确的HOI区域掩码 $R_{k,T}$
- 采用双流EgoVideo-V架构同时编码全帧和HOI区域：

$$(\mathbf{X}_{k,T}^{ori}, \mathbf{X}_{k,T}^{mask}) = \text{EgoVideo-V}(F_{k,T}, R_{k,T})$$

- 两路嵌入拼接后通过线性MLP融合，Transformer模块捕获时空关系

这种设计将全局场景上下文与局部HOI细节有机结合，显著提升动词-名词预测的语义准确性。

#### 2. **动词-名词共现语义矫正**

Transformer模型输出经过双分类器（动词分类器+名词分类器），但独立预测的动词-名词对可能不合理。INSIGHT构建共现矩阵进行语义矫正：

从训练数据统计共现矩阵 $\mathbf{C} \in \mathbb{N}^{|\mathcal{V}| \times |\mathcal{N}|}$：

$$\mathbf{C}_{v,n} = \sum_{k=1}^{K} \mathbf{1}_{\{v_k = v \wedge n_k = n\}}$$

行/列归一化得到条件概率 $\mathbf{P}^{(n|v)}$ 和 $\mathbf{P}^{(v|n)}$，矫正后的联合概率为：

$$\tilde{p}(v_k, n_k) = p(v_k) \cdot p(n_k) \cdot \frac{1}{2}(\mathbf{P}^{(n|v)}_{v,n} + \mathbf{P}^{(v|n)}_{v,n})$$

最终通过MAP估计选择最佳动词-名词对。这有效过滤了语义上不合理的组合，增强了预测的可靠性。

#### 3. **基于GRPO的认知推理模块**

第二阶段用Qwen2.5-VL-7B作为骨干，引入结构化推理流程 "think → reason → answer"：

- **think（视觉感知）**：`<think>...</think>` 感知当前场景
- **reason（意图推断）**：`<intention>...</intention>` 推断用户的高层任务意图
- **answer（动作预测）**：`<answer>...</answer>` 输出预测的动作序列

**格式奖励**（确保结构化输出）：
- 长度奖励 $S_{len}$：预测动作对数量是否达标
- 标签顺序奖励 $S_{fmt}$：是否遵循think→intention→answer结构
- 语言一致性奖励 $S_{lang}$：输出是否全为英文
- 软超长惩罚 $R_{Soft}$：线性递减惩罚过长输出

**内容奖励**：
- **准确性奖励** $S_{acc}$：基于编辑距离（ED）归一化到[0,1]

$$S_{acc} = 1 - \frac{d_{ED}^Z}{|\mathbf{s}_{true}|}$$

- **意图奖励** $S_{int}$：用Sentence-BERT计算生成意图与GPT-4.1生成的伪ground-truth意图的余弦相似度，经缩放sigmoid归一化

$$S_{int} = \min\left(\frac{1}{1+\exp[-\gamma(sim-\beta)]} \Big/ \frac{1}{1+\exp[-\gamma(1-\beta)]}, 1\right)$$

**总奖励整合**：

$$R = \omega_1 S_{len} R_{task} + \omega_2 R_{Soft}$$

其中 $R_{task} = \omega_3 S_{acc} + \omega_4 S_{int} + \omega_5 S_{lang} + \omega_6 S_{fmt}$

### 损失函数 / 训练策略

- 视觉编码器：冻结的EgoVideo-V，Transformer 4层8头
- 认知推理：Qwen2.5-VL-Instruct-7B骨干，基于Swift框架的GRPO训练
- 6×NVIDIA H20-SXM5-96GB GPU
- batch size 24，学习率 3e-6，温度 0.9，KL系数 0.08
- 奖励权重：$\omega_1=0.90, \omega_2=0.10, \omega_3=0.85, \omega_4=0.05, \omega_5=0.05, \omega_6=0.05$
- 意图奖励参数：$\beta=0.8, \gamma=40$
- 总训练500步，约90 GPU小时

## 实验关键数据

### 主实验

**Ego4D-v2验证集（编辑距离ED，越低越好）**：

| 方法 | LLM | Verb↓ | Noun↓ | Action↓ |
|------|-----|-------|-------|---------|
| AntGPT | LLaMA2-7B | 0.6728 | 0.6755 | 0.8931 |
| PALM | LLaMA2-7B | 0.7111 | 0.6465 | 0.8819 |
| EgoVideo | Vicuna-7B | 0.6576 | 0.6264 | 0.8619 |
| ICVL | LLaMA3-8B | 0.6516 | 0.6194 | 0.8570 |
| **INSIGHT** | **Qwen2.5-VL-7B** | **0.6643** | **0.6092** | **0.8463** |

**EPIC-Kitchens-55 / EGTEA Gaze+（mAP，越高越好）**：

| 方法 | EK-55 ALL↑ | EK-55 FREQ↑ | EK-55 RARE↑ | EGTEA ALL↑ | EGTEA FREQ↑ | EGTEA RARE↑ |
|------|-----------|------------|------------|-----------|------------|------------|
| AntGPT | 40.1 | 58.8 | 31.9 | 80.2 | 84.8 | 72.9 |
| ICVL | 43.3 | 61.6 | 33.8 | 81.0 | 85.2 | 73.7 |
| **INSIGHT** | **45.2** | **62.4** | **36.0** | **81.7** | **85.9** | **74.4** |

### 消融实验

| 配置 | Verb ED↓ | Noun ED↓ | Action ED↓ | 说明 |
|------|----------|----------|------------|------|
| w/o HOI feature | 0.6719 | 0.6158 | 0.8595 | 去除HOI特征，性能下降 |
| w/o Semantic correction | 0.6716 | 0.6108 | 0.8587 | 去除共现矫正 |
| w/o Cognitive reasoning | 0.6750 | 0.6176 | 0.8612 | 影响最大，直接预测 |
| w/o Intention | 0.6685 | 0.6104 | 0.8571 | 保留推理但去除意图监督 |
| **INSIGHT (full)** | **0.6643** | **0.6092** | **0.8463** | 所有模块协同最优 |

### 关键发现

1. **认知推理是最关键组件**：去除结构化推理（w/o Cognitive reasoning）导致最大性能下降，Action ED从0.8463升至0.8612，说明"think→reason→answer"的显式推理对长期预测至关重要

2. **HOI特征对名词预测贡献最大**：在Ego4D-v2上，INSIGHT在名词预测上比最强基线ICVL好1.02%，归因于HOI聚焦的特征提取捕获了关键的物体操作信息

3. **稀有类动作提升显著**：在EK-55的RARE类别上，INSIGHT比ICVL提升6.5%（33.8→36.0），说明认知推理和意图对齐有效减少了长尾类别混淆

4. **冻结编码器的优势**：INSIGHT使用冻结的视觉编码器即超越了使用微调编码器的EgoVideo，表明微调后的语言模型和认知推理模块能有效补偿视觉模糊性

5. **训练收敛稳定**：GRPO训练500步内收敛，意图奖励曲线与总奖励高度吻合，验证了意图监督与任务目标的对齐

## 亮点与洞察

- **两阶段设计的互补性**：第一阶段强化视觉表示质量（HOI+共现），第二阶段引入认知推理能力（GRPO+意图），各消融实验表明两者缺一不可
- **认知推理的仿生设计**：think→reason→answer模拟人类决策过程，使模型从被动序列预测转为主动意图推断，是视频理解领域的重要范式转变
- **意图奖励的巧妙设计**：利用GPT-4.1生成伪意图标签作为监督，避免了昂贵的人工标注，同时sigmoid归一化确保了奖励信号的梯度友好性
- **共现矩阵的简洁有效**：简单的统计先验即可显著减少语义不合理的预测，实现成本极低

## 局限与展望

- 意图的伪ground-truth依赖GPT-4.1生成，引入了外部模型偏差，且生成质量难以保证
- 视觉编码器冻结限制了模型对特定场景的适应性，端到端微调可能进一步提升
- HOI检测依赖预训练的100DOH检测器，在非厨房场景中检测质量可能下降
- GRPO训练仅500步，训练时间虽短但可能限制了模型的推理深度
- 动词-名词共现矩阵来自训练集统计，可能无法覆盖测试集中的新颖组合
- 未探讨更长时间跨度（如Z>20）的预测能力

## 相关工作与启发

- 与AntGPT、PALM等基于LLM的方法相比，INSIGHT的关键创新在于用RL replace SFT来训练推理过程
- 结构化推理（think→reason→answer）的设计灵感来自DeepSeek-R1的成功，但进行了任务特定的适配
- HOI检测+SAM2细化的pipeline可以作为其他第一人称视频任务的通用特征增强方案
- 意图奖励的设计模式（LLM伪标签+嵌入相似度+sigmoid归一化）可迁移到其他需要中间推理监督的RL任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将认知推理+GRPO应用到LTA是新颖的组合，但各组件有prior work支撑
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个主流基准+详细消融+训练动态+定性对比
- **写作质量**: ⭐⭐⭐⭐ — 框架图清晰，方法描述详尽
- **价值**: ⭐⭐⭐⭐ — 在LTA领域建立新SOTA，认知推理范式有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Unveiling the Cognitive Compass: Theory-of-Mind-Guided Multimodal Emotion Reasoning](../../ICLR2026/reinforcement_learning/unveiling_the_cognitive_compass_theory-of-mind-guided_multimodal_emotion_reasoni.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ICLR 2026\] LoongRL: Reinforcement Learning for Advanced Reasoning over Long Contexts](../../ICLR2026/reinforcement_learning/loongrl_rl_for_reasoning_long_contexts.md)
- [\[AAAI 2026\] Efficient Multiagent Planning via Shared Action Suggestions](efficient_multiagent_planning_via_shared_action_suggestions.md)
- [\[AAAI 2026\] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach](vision-language_reasoning_for_geolocalization_a_reinforcement_learning_approach.md)

</div>

<!-- RELATED:END -->
