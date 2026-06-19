---
title: >-
  [论文解读] Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions
description: >-
  [ACL 2025][合成用户] 提出一个端到端框架，基于真实人口学、健康/生活方式和行为/心理特征数据生成有健康状况的合成用户（涵盖睡眠和糖尿病管理），用于评估健康教练Agent的交互质量，并通过人类专家评估验证其显著优于通用合成用户。 交互式健康教练Agent需要通过与用户的交互来评估效果，但收集和评估多样化、长期的人类…
tags:
  - "ACL 2025"
  - "合成用户"
  - "健康状况建模"
  - "教练式对话Agent"
  - "用户模拟"
  - "LLM评估"
---

# Sleepless Nights, Sugary Days: Creating Synthetic Users with Health Conditions for Realistic Coaching Agent Interactions

**会议**: ACL 2025  
**arXiv**: [2502.13135](https://arxiv.org/abs/2502.13135)  
**代码**: [有](https://anonymous.4open.science/r/sleepless_nights)  
**领域**: 其他  
**关键词**: 合成用户, 健康状况建模, 教练式对话Agent, 用户模拟, LLM评估

## 一句话总结

提出一个端到端框架，基于真实人口学、健康/生活方式和行为/心理特征数据生成有健康状况的合成用户（涵盖睡眠和糖尿病管理），用于评估健康教练Agent的交互质量，并通过人类专家评估验证其显著优于通用合成用户。

## 研究背景与动机

交互式健康教练Agent需要通过与用户的交互来评估效果，但收集和评估多样化、长期的人类交互既昂贵又耗时。LLM 生成的合成用户为此提供了自动化评估的可能，但现有方法存在关键局限：

**缺乏真实健康状况锚定**：通用合成用户不能准确反映特定健康条件下用户的需求和挑战

**人口统计偏差**：LLM 训练数据偏向英语文化和高在线参与度人群，不能代表真实患者分布

**缺乏情境化知识**：LLM 可以引用睡眠困难等现象，但这些不代表扎根于生活经历的情境化知识

**因果推断风险**：向 LLM 提供特定建议可能无意中改变合成用户的其他隐含特征

核心理念：合成用户应该基于真实数据生成——从真实的人口统计、健康指标和行为心理特征出发构建，而非完全依赖 LLM 自由生成。

## 方法详解

### 整体框架

两阶段构建合成用户：

1. **结构化数据生成**：基于真实人口学、健康/生活方式和行为心理数据生成结构化属性
2. **完整画像生成**：基于结构化数据，用 LLM 生成完整的用户"小传"(vignette)

然后让合成用户与教练Agent进行模拟交互，通过 Concordia 系统或直接 LLM 调用。

### 关键设计

1. **基于真实数据的属性锚定**

    - 睡眠场景：使用 LifeSnaps 公开数据集（68 人，含人口统计、睡眠数据、大五人格等）
    - 糖尿病场景：使用 PBHS 纵向队列（345 名 2 型糖尿病患者，含详细人口学、社会经济、临床数据）
    - 设计动机：直接采样真实数据的分布，避免 LLM 的分布偏差

2. **多层次用户建模**

   对于睡眠场景：
    - 基础属性：年龄、性别、BMI、睡眠时长和效率、大五人格
    - LLM 生成的睡眠档案：主要睡眠关切、睡眠目标、目标原因、障碍
    - 可选扩展：COM-B 行为模型框架的挑战、丰富的背景故事

   对于糖尿病场景：
    - 从 246 个真实挑战中按 COM-B 模型分布采样障碍
    - 基于患者的人口学、社会经济和临床数据构建小传
    - 生成沟通风格（语气、冗长度、信心水平）

3. **交互模拟**

    - 使用 Concordia 生成式agent框架实例化合成用户
    - Concordia 提供关联记忆、思维链推理、模块化架构
    - 睡眠Agent采用"Talker-Reasoner"双Agent架构（System 1 + System 2）
    - 使用 Gemini 1.5 Pro 作为底层 LLM

4. **多维评估策略**

    - 自动评估：比较教练Agent的内部用户模型与真实用户档案
    - 专家评估：训练有素的人类评估者盲评交互质量
    - 对比评估：完整合成用户 vs 仅人口统计的基线用户

### 损失函数 / 训练策略

本文为框架性工作，不涉及模型训练。核心是合成数据生成和评估流程设计。

## 实验关键数据

### 睡眠教练实验（68 个合成用户，10 轮交互）

| 评估维度 | 指标 |
|----------|------|
| 主要睡眠关切识别准确率 | **89.7%** |
| 障碍召回率 | 71.4% |
| 障碍精确率 | 72.5% |
| 睡眠目标召回率 | 66.4% |
| 睡眠目标精确率 | 84.2% |

### 人类专家评估（睡眠场景）

| 评估项 | 完整用户 vs 基线偏好 | 评分者一致性 |
|--------|---------------------|-------------|
| 整体偏好 | 完整用户显著获胜 | Fleiss' κ = 0.67 |
| p-value | 3.7 × 10⁻¹² | |
| 5/5 完全一致率 | 64% | |
| ≥4/5 一致率 | 91% | |

### 糖尿病教练实验（200 个合成用户）

| 评估维度 | 专家评分 |
|----------|---------|
| 用户一致性 | **92%** |
| 障碍展示忠实度 | **100%** |

### 关键发现

1. 教练Agent能以 89.7% 的准确率识别合成用户的主要睡眠关切，表明合成用户确实在交互中有效传达了分配的健康属性
2. 基于完整健康/行为属性的合成用户显著优于仅基于人口统计的基线用户（p < 10⁻¹²）
3. 评分者间一致性高（κ = 0.67），说明质量差异明显且容易判断
4. 框架在两个独立开发的Agent和场景中均通过验证，具有通用性

## 亮点与洞察

1. **端到端框架设计**：从真实数据采样 → 属性生成 → 小传构建 → 交互模拟 → 多维评估的完整流程
2. **真实数据锚定的关键性**：实验有力证明了仅有人口统计信息远不够，健康条件和行为特征是生成逼真合成用户的关键
3. **两场景独立验证**：睡眠和糖尿病场景由不同团队独立开发，增强了结论的可信度
4. **对 LLM 偏差的系统性思考**：明确识别并缓解了 LLM 作为合成用户的多种偏差来源
5. **支持隐私保护**：合成用户可以生成新个体，不依赖真实患者的直接数据

## 局限与展望

1. 合成用户仍可能缺乏真实生活经验的深度和细微差别
2. 仅评估了目标和障碍的获取，未评估后续行为改变过程
3. 依赖 Gemini 系列模型，不同 LLM 可能产生不同质量的合成用户
4. 使用开源模型（如 Gemma 2-27B）替代时性能有所下降
5. 长期交互（超过 10 轮）的效果未验证

## 相关工作与启发

- AMIE (Tu et al., 2024)：医疗诊断对话Agent，但合成患者设计有人口统计偏差等局限
- Yu et al. (2024)：基于知识图谱的患者 LLM，适合临床但不适合健康教练场景
- Castricato et al. (2024)：基于美国人口普查的合成用户，但仅考虑人口统计未考虑健康状况
- Concordia (Vezhnevets et al., 2023)：本文使用的生成式agent框架

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次系统性地将真实健康数据整合到合成用户生成中用于教练Agent评估
- **实验充分度**: ⭐⭐⭐⭐ 双场景验证、自动+专家评估、对比实验设计完善
- **写作质量**: ⭐⭐⭐⭐ 框架描述清晰，背景综述全面
- **价值**: ⭐⭐⭐⭐ 为健康AI领域的Agent评估提供了实用方法论

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[ACL 2025\] Using Shapley Interactions to Understand How Models Use Structure](using_shapley_interactions_to_understand_how_models_use_structure.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)

</div>

<!-- RELATED:END -->
