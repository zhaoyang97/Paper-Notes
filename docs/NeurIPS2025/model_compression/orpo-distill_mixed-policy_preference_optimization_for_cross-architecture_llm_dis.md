---
title: >-
  [论文解读] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation
description: >-
  [NeurIPS 2025][模型压缩][知识蒸馏] 提出 ORPO-Distill，将跨架构 LLM 知识蒸馏重新定义为偏好优化问题：使用教师模型生成正样本推理链、学生模型生成负样本推理链，通过 ORPO 对比损失训练，并引入混合策略（mixed-policy）更新学生负样本，在 5 个 QA 基准上一致超越黑盒 KD 基线。
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "知识蒸馏"
  - "偏好优化"
  - "ORPO"
  - "混合策略"
  - "跨架构蒸馏"
---

# ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2509.25100](https://arxiv.org/abs/2509.25100)  
**代码**: 未公开  
**领域**: LLM对齐  
**关键词**: 知识蒸馏, 偏好优化, ORPO, 混合策略, 跨架构蒸馏

## 一句话总结

提出 ORPO-Distill，将跨架构 LLM 知识蒸馏重新定义为偏好优化问题：使用教师模型生成正样本推理链、学生模型生成负样本推理链，通过 ORPO 对比损失训练，并引入混合策略（mixed-policy）更新学生负样本，在 5 个 QA 基准上一致超越黑盒 KD 基线。

## 研究背景与动机

LLM 知识蒸馏的两大范式：

**白盒 KD**：依赖教师 logits，要求师生共享词表/架构，限制灵活性

**黑盒 KD**：仅需教师生成序列，允许跨架构，但通常只用 CoT 蒸馏

**现有黑盒 KD 的三个问题**：

**单一 CoT 蒸馏**监督信号有限：只用一条教师推理链

**缺少对比学习**：未区分"好的推理路径"和"差的推理路径"

**分布不匹配**：训练时用教师序列，推理时学生自回归生成，分布存在 gap

**本文的三个创新点**：

1. 用**多样化推理链**（diverse reasoning traces）替代单一 CoT
2. 用 **ORPO 偏好优化**做对比蒸馏（教师正 + 学生负）
3. 用**混合策略更新**（mixed-policy）解决分布不匹配问题

## 方法详解

### 整体框架

输入 prompt → 教师生成 K 条正推理链 → 学生生成 K 条负推理链 → 构建偏好数据集 $\langle$Prompt, Chosen, Rejected$\rangle$ → ORPO 对比训练 → 混合策略更新负样本

### ORPO 损失函数

ORPO（Odds Ratio Preference Optimization）将 SFT 和偏好对齐合并为单一目标：

$$L_{SFT} = -\log q_\theta(y_P \mid x)$$

$$L_{OR} = -\log \sigma\left(\log \frac{\text{odds}\;q_\theta(y_P|x)}{\text{odds}\;q_\theta(y_N|x)}\right)$$

其中 odds 函数定义为：

$$\text{odds}\;q_\theta(y|x) = \frac{q_\theta(y|x)}{1 - q_\theta(y|x)}$$

最终损失：

$$L_{ORPO} = L_{SFT} + \lambda \cdot L_{OR}$$

- $\lambda = 0.1$：轻微偏好倾斜（如人类偏好对齐）
- $\lambda = 1$：强适应（本文使用，裁剪学生的错误生成路径）

### 偏好数据集构建

**关键发现**：负样本来自学生模型优于来自教师模型

| 实验设置 | MedQA | ARC-C |
|:---|:---:|:---:|
| (正-教师, 负-教师) | 41.72 | 45.87 |
| (正-教师, 负-学生) | **49.33** | **56.48** |

**多样性采样**：
- 温度采样 $\tau = 0.8$，生成 $K$ 条推理链
- 教师和学生均使用 Reason-then-Answer 格式
- 拒绝采样：丢弃 ROUGE-L 重叠 > 0.80 的冗余链
- $K \in \{4, 8, 12\}$ 实验表明 $K=8$ 之后收益递减

### 混合策略更新

三种策略定义：

| 策略 | 参数 $\phi$ | 负样本来源 |
|:---|:---:|:---|
| Off-policy | $\phi = 0$ | 固定为初始学生模型生成 |
| On-policy | $\phi = 1$ | 每 epoch 用最新 checkpoint 重采样 |
| Mixed-policy | $\phi = 0.5$ | 以概率 $\phi$ 用最新 checkpoint，否则用初始模型 |

**为什么 on-policy 反而更差？** 频繁更新的负样本质量更高但多样性下降，缩小了对比学习的区分空间。Mixed-policy 通过锚定初始学生模型保持负样本多样性。

### 训练策略

- 教师模型：InternLM 2.5 7B-Chat
- 学生模型：InternLM 2.5 1.8B-Chat、TinyLlama 1.1B-Instruct
- 全参数微调，5 个 epoch
- $K = 8$，$\lambda = 1$，$\phi = 0.5$

## 实验关键数据

### 主实验：不同学生模型在 5 个数据集上的准确率（%）

| 实验 | MedQA | ARC-C | StrategyQA | OBQA | GSM8K | 平均 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TinyLlama 1.1B** | | | | | | |
| Zero-shot CoT | 29.78 | 29.95 | 43.52 | 26.60 | 11.97 | 28.36 |
| Single CoT FT | 32.10 | 32.63 | 46.25 | 29.05 | 31.56 | 34.32 |
| Diverse CoT FT | 34.85 | 35.40 | 47.84 | 33.60 | 36.22 | 37.58 |
| Off-Policy ORPO | 38.95 | 41.20 | 49.77 | 37.45 | 39.45 | 41.36 |
| On-Policy ORPO | 35.11 | 38.01 | 49.24 | 35.60 | 36.88 | 38.97 |
| **Mixed-Policy ORPO** | **40.25** | **43.55** | **51.25** | **40.10** | **40.72** | **43.17** |
| **InternLM 1.8B** | | | | | | |
| Zero-shot CoT | 35.82 | 37.12 | 54.15 | 27.40 | 41.02 | 39.10 |
| Single CoT FT | 37.94 | 40.45 | 57.50 | 41.35 | 44.38 | 44.32 |
| Diverse CoT FT | 40.56 | 42.15 | 58.66 | 54.50 | 47.50 | 48.67 |
| Off-Policy ORPO | 49.33 | 56.48 | 59.39 | 53.20 | 51.25 | 53.93 |
| On-Policy ORPO | 43.25 | 49.80 | 58.50 | 52.79 | 47.94 | 50.46 |
| **Mixed-Policy ORPO** | **50.43** | **59.32** | **61.75** | **55.22** | **52.47** | **55.84** |

教师模型（InternLM 2.5 7B-Chat）Zero-shot CoT 平均：59.58%

### 消融实验：各组件贡献

| 组件 | TinyLlama 平均提升 | InternLM 1.8B 平均提升 |
|:---|:---:|:---:|
| 单CoT → 多样CoT | +3.26 | +4.35 |
| 多样CoT → Off-Policy ORPO | +3.78 | +5.26 |
| Off-Policy → Mixed-Policy | +1.81 | +1.91 |
| **总提升（单CoT → Mixed-Policy）** | **+8.85** | **+11.52** |

### 关键发现

1. **每个组件都有贡献**：多样化推理链（+3~5%）、ORPO 对比（+4~5%）、mixed-policy（+1~2%），呈递增关系
2. **On-policy 反而不如 Off-policy**：验证了负样本多样性比质量更重要的假设
3. **混合策略统一最优**：在所有数据集和学生模型上一致最佳
4. **跨架构蒸馏有效**：TinyLlama（Llama 架构）和 InternLM 1.8B（InternLM 架构）都从 InternLM 7B 教师获益
5. **小模型收益更大**：1.1B 模型平均绝对提升 +14.81%（vs Zero-shot），1.8B 模型 +16.74%

## 亮点与洞察

1. **将蒸馏问题重新定义为偏好优化**是关键洞察：黑盒 KD 本质上就是让学生学会"偏好"教师的推理方式，ORPO 提供了自然的框架
2. **学生负样本 > 教师负样本**的发现很有启发：因为学生负样本直接暴露了学生的弱点，对比训练更具针对性
3. **Mixed-policy 的设计巧妙平衡了质量与多样性**：类似于 RL 中 ε-greedy 的思想
4. **方法极其简洁**：不需要 reward model、不需要 PPO 训练、不需要白盒访问，实现成本低
5. **K=8 的经验发现**对后续工作有实用参考价值

## 局限与展望

1. **仅验证了多选 QA 任务**：依赖 ground truth label 判断正负样本，开放式生成任务需要额外的验证器
2. **教师模型仅 7B**：未测试更大教师（如 70B）的效果上限
3. **$\phi$ 的设置较粗糙**：仅测试了 $\phi \in \{0, 0.5, 1\}$，未做细粒度搜索或自适应调节
4. **全参数微调**：在资源受限场景下，LoRA 等参数高效方法的兼容性未验证
5. **缺少与 DPO、PPO 等其他偏好优化方法的对比**
6. **不应将 on-policy 和 off-policy 的差异简单归因于多样性**——可能还有优化景观等因素

## 相关工作与启发

- **ORPO**（Hong et al., 2024）：原始论文将 SFT 和偏好对齐合并，本文将其应用于蒸馏场景
- **On-policy Distillation**（Agarwal et al., 2024）：白盒方法中 on-policy 更好，但本文在黑盒场景中发现 mixed-policy 更优
- **MAGDI**（Chen et al., 2024）：使用多教师对比蒸馏，但不利用学生生成输出
- **DistillM**（Ko et al., 2024）：白盒方法中 SGO 的重要性，本文将其推广到黑盒

## 评分

- ⭐⭐⭐⭐ (4/5)
- **创新性** ⭐⭐⭐⭐：将蒸馏→偏好优化的重定义很优雅，mixed-policy 是有效的新贡献
- **实验充分性** ⭐⭐⭐⭐：5 个数据集、2 个学生模型、逐组件消融，但缺少大模型教师和更多偏好优化方法对比
- **写作质量** ⭐⭐⭐⭐：简洁清晰，算法伪代码规范
- **实用价值** ⭐⭐⭐⭐⭐：方法简单有效，易于在实际小模型部署中使用
- **理论深度** ⭐⭐⭐：虽然有直觉解释（多样性 vs 质量），但缺少理论分析

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](../../ICML2025/model_compression/confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](../../ICCV2025/model_compression/cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[AAAI 2026\] CTPD: Cross Tokenizer Preference Distillation](../../AAAI2026/model_compression/ctpd_cross_tokenizer_preference_distillation.md)
- [\[NeurIPS 2025\] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation](ppg-distill_efficient_photoplethysmography_signals_analysis_via_foundation_model.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)

</div>

<!-- RELATED:END -->
