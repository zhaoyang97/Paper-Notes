---
title: >-
  [论文解读] Collaborative Performance Prediction for Large Language Models
description: >-
  [ACL 2025][LLM 其他][性能预测] 本文提出协同性能预测框架CPP，利用多个LLM在多个任务上的历史性能数据及模型/任务的设计因素进行协同过滤式预测，突破了传统Scaling Law仅限单模型族预测的限制，能跨模型族准确预测LLM的下游性能。 领域现状：理解和预测大语言模型在各种下游任务上的性能是NLP研究中的…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "性能预测"
  - "Scaling Law"
  - "协同过滤"
  - "LLM评估"
  - "模型选择"
---

# Collaborative Performance Prediction for Large Language Models

**会议**: ACL 2025  
**arXiv**: [2407.01300](https://arxiv.org/abs/2407.01300)  
**代码**: [https://github.com/Don-Joey/CPP_LLM](https://github.com/Don-Joey/CPP_LLM)  
**领域**: LLM/NLP  
**关键词**: 性能预测, Scaling Law, 协同过滤, LLM评估, 模型选择

## 一句话总结
本文提出协同性能预测框架CPP，利用多个LLM在多个任务上的历史性能数据及模型/任务的设计因素进行协同过滤式预测，突破了传统Scaling Law仅限单模型族预测的限制，能跨模型族准确预测LLM的下游性能。

## 研究背景与动机

**领域现状**：理解和预测大语言模型在各种下游任务上的性能是NLP研究中的关键挑战。Scaling Law（如Kaplan et al., Chinchilla）揭示了模型性能与计算量之间的幂律关系，已被用于指导模型设计和训练资源分配。近期工作（如Hu et al., 2024; Isik et al., 2024）将Scaling Law扩展到预测下游任务性能。

**现有痛点**：现有的下游性能预测方法存在三个关键限制：（1）需要透明的设计因素（如FLOPs、训练数据量）——很多闭源模型不提供这些信息；（2）仅限于同一模型族内预测（如只能用LLaMA-7B预测LLaMA-70B），忽略了不同模型族之间的相似性；（3）评估成本高——测试一个LLM在某些基准上需要$10K+和4K+ GPU hours，对新任务和新模型的评估成本不可接受。

**核心矛盾**：Scaling Law的"单模型族"假设过于局限——不同模型族之间实际存在强相似性（如擅长推理的模型在多个推理任务上都表现好），但现有方法无法利用这种跨模型族的相似性。

**本文目标**：设计一个能利用"模型×任务"的协同信息来预测性能的框架，类似推荐系统中的协同过滤——已知部分模型在部分任务上的成绩，预测未知的成绩。

**切入角度**：作者将LLM性能预测类比为推荐系统中的评分预测问题——"模型"对应"用户"，"任务"对应"物品"，"性能分数"对应"评分"。如果模型A在任务1-3上与模型B表现相似，那么模型A在任务4上的表现可以参考模型B在任务4上的成绩。

**核心 idea**：构建协同数据矩阵（模型×任务的性能分数矩阵+额外设计因素），使用矩阵分解和辅助因素增强的协同过滤方法预测缺失的性能分数。

## 方法详解

### 整体框架
CPP框架包含两个核心组件：（1）协同数据——从在线平台（如Open LLM Leaderboard）收集模型×任务的性能分数矩阵，并附带模型设计因素（参数量、训练方法、架构类型）和任务设计因素（任务类型、难度、评估指标）；（2）协同预测方法——基于矩阵分解+辅助因素的预测模型，对给定的模型ID和任务ID，利用两者的协同信息预测性能分数。

### 关键设计

1. **协同数据构建（Collaborative Data Construction）**:

    - 功能：构建用于协同预测的模型×任务性能数据集
    - 核心思路：从Hugging Face Open LLM Leaderboard等平台爬取大量模型在多个基准任务上的评测结果，形成稀疏的分数矩阵 $R \in \mathbb{R}^{M \times T}$。同时收集每个模型的元信息（参数量 $N$、架构类型 $a$、训练数据量 $D$、微调方法 $f$ 等）和每个任务的属性（类型 $c$、few-shot设置 $k$、评估指标 $m$ 等）。这些设计因素作为辅助信息增强预测
    - 设计动机：纯矩阵分解在分数矩阵稀疏时效果不佳，辅助因素提供了额外的归纳偏置

2. **增强型矩阵分解预测器（Factor-Enhanced Matrix Factorization）**:

    - 功能：融合协同信号和设计因素进行性能预测
    - 核心思路：将模型 $i$ 和任务 $j$ 分别用潜在向量 $u_i$ 和 $v_j$ 表示（类似推荐系统中的用户/物品embedding）。预测分数为 $\hat{R}_{ij} = u_i^T v_j + f(x_i^{model}, x_j^{task})$，其中 $f$ 是一个小型MLP网络，输入为模型和任务的设计因素特征向量。矩阵分解部分捕捉协同信号（相似模型在相似任务上的表现相近），MLP部分利用设计因素进行预测，两者互补
    - 设计动机：传统Scaling Law只使用设计因素（如 $Loss \propto N^{-\alpha}$），忽略协同信号；纯协同过滤不使用先验知识。本方法将两者统一

3. **因素重要性分析模块（Factor Importance Analysis）**:

    - 功能：量化各个设计因素对性能预测的贡献
    - 核心思路：利用MLP分支中各输入因素的梯度幅度和SHAP值来估计每个设计因素的重要性。通过排列不同因素的输入并观察预测分数的变化来量化每个因素的贡献。例如分析"参数量"、"训练数据类型"、"微调方法"等因素各贡献了多少预测accuracy
    - 设计动机：以往Scaling Law只考虑少数预定义因素（如N、D、C），本方法可以数据驱动地发现更多重要因素

### 损失函数 / 训练策略
使用均方误差（MSE）损失拟合已知的性能分数，加上L2正则化防止过拟合。对稀疏数据使用负采样策略。

## 实验关键数据

### 主实验

| 方法 | RMSE↓ | MAE↓ | R²↑ | 跨模型族预测RMSE↓ |
|------|-------|------|-----|------------------|
| Chinchilla Scaling Law | 5.82 | 4.31 | 0.62 | 8.47 |
| Observational Scaling (Ruan et al.) | 4.15 | 3.02 | 0.74 | 6.23 |
| FPE (Isik et al.) | 3.87 | 2.78 | 0.78 | 5.81 |
| 纯矩阵分解 | 3.21 | 2.35 | 0.83 | 3.45 |
| **CPP（本文）** | **2.43** | **1.76** | **0.91** | **2.68** |

### 消融实验

| 配置 | RMSE↓ | 说明 |
|------|-------|------|
| CPP完整 | 2.43 | 协同+因素 |
| w/o 设计因素（纯协同） | 3.21 | 去掉因素后退化为纯MF |
| w/o 协同信号（纯因素） | 3.95 | 去掉协同后退化为Scaling Law变体 |
| w/o 模型因素 | 2.87 | 模型元信息贡献+0.44 |
| w/o 任务因素 | 2.71 | 任务属性贡献+0.28 |
| 稀疏度20%已知 | 3.12 | 少量已知数据时仍有效 |
| 稀疏度50%已知 | 2.43 | 较充分数据时最优 |

### 关键发现
- 协同信号是最大贡献项：仅用协同过滤就比最好的Scaling Law方法低0.66 RMSE，证明了模型间相似性的价值
- CPP在跨模型族预测上优势最大（RMSE 2.68 vs 5.81），因为协同过滤不需要模型族内的训练资源曲线
- 因素重要性分析发现：微调方法（RLHF vs SFT）对下游任务性能的影响甚至超过参数量，这是传统Scaling Law忽略的
- 即使只有20%的成绩矩阵可用，CPP仍然优于需要完整设计因素的传统方法

## 亮点与洞察
- 将LLM性能预测建模为推荐系统问题是一个精彩的类比——"用户对电影的评分"对应"模型在任务上的得分"，成熟的推荐系统技术可以直接迁移
- 因素重要性分析揭示了非显而易见的洞察（如微调方法比参数量更重要），这对LLM开发决策具有实际指导意义
- 协同数据的自动收集机制可以持续更新，随着更多模型和评测结果的发布，预测精度会越来越高

## 局限与展望
- 协同数据的质量依赖于在线平台的评测标准一致性，不同平台的评测设置差异可能引入噪声
- 对全新的、与现有模型族完全不同的架构（如SSM），冷启动问题可能影响预测
- 当前只预测单一指标的性能分数，未考虑多指标间的相关性
- 可以扩展为动态预测——根据训练过程中已有的checkpoint性能，预测最终收敛性能

## 相关工作与启发
- **vs Scaling Laws (Kaplan et al., 2020)**: 传统Scaling Law基于计算量的幂律，本文引入协同过滤突破了单模型族限制
- **vs FPE (Isik et al., 2024)**: FPE利用模型族内相似性，本文进一步利用跨模型族的相似性
- **vs Observational Scaling (Ruan et al., 2024)**: 通过观察性数据拟合Scaling曲线，本文的协同方法不需要设计因素即可预测

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 推荐系统思路引入LLM性能预测是很巧妙的创新
- 实验充分度: ⭐⭐⭐⭐⭐ 对比全面、消融详细、分析深入
- 写作质量: ⭐⭐⭐⭐ 框架表述清晰，类比恰当
- 价值: ⭐⭐⭐⭐⭐ 对LLM评估和模型选择有很高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Open-Set Living Need Prediction with Large Language Models](open-set_living_need_prediction_with_large_language_models.md)
- [\[ACL 2025\] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)
- [\[ACL 2025\] Prediction Hubs are Context-Informed Frequent Tokens in LLMs](prediction_hubs_are_context-informed_frequent_tokens_in_llms.md)
- [\[ACL 2025\] DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services](disco_device-server_collaborative_llm-based_text_streaming_services.md)
- [\[ACL 2025\] Comparing Large Language Models in Extracting Subjective Information from Political News](comparing_large_language_models_in_extracting_subjective_information_from_politi.md)

</div>

<!-- RELATED:END -->
