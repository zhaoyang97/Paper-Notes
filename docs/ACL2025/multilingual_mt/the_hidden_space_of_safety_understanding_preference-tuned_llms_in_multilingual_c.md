---
title: >-
  [论文解读] The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts
description: >-
  [ACL2025][多语言/翻译][multilingual alignment] 本文系统分析了偏好调优（RLHF/DPO 等）对 LLM 内部表示空间在多语言场景下的影响，发现对齐机制在英语上能有效分离有害/无害内容的隐空间表示，但在印地语、中文、德语等非英语语言上效果显著退化，揭示了当前对齐方法存在严重的单语偏差问题。
tags:
  - "ACL2025"
  - "多语言/翻译"
  - "multilingual alignment"
  - "LLM safety"
  - "preference tuning"
  - "representation analysis"
  - "cross-lingual"
---

# The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual Contexts

**会议**: ACL2025  
**arXiv**: [2504.02708](https://arxiv.org/abs/2504.02708)  
**代码**: 待确认  
**领域**: 多语言翻译  
**关键词**: multilingual alignment, LLM safety, preference tuning, representation analysis, cross-lingual

## 一句话总结

本文系统分析了偏好调优（RLHF/DPO 等）对 LLM 内部表示空间在多语言场景下的影响，发现对齐机制在英语上能有效分离有害/无害内容的隐空间表示，但在印地语、中文、德语等非英语语言上效果显著退化，揭示了当前对齐方法存在严重的单语偏差问题。

## 背景与动机

1. **对齐调优是 LLM 安全的核心**：RLHF、DPO 等偏好优化方法已成为确保 LLM 安全、可靠和符合人类价值观的标准后训练阶段。
2. **偏好数据以英语为主导**：当前对齐数据集（如 HH-RLHF、Anthropic 数据等）绝大部分为英语，低资源语言的偏好数据极度匮乏。
3. **多语言安全表现不一致**：尽管 LLM 在英语上能正确拒绝有害请求（如"如何制造炸弹"），但相同请求以印地语提问时模型可能直接输出详细步骤。
4. **隐空间机制尚不明确**：对齐如何重塑模型内部表示空间、安全约束是否跨语言泛化，缺乏系统性量化研究。
5. **越狱攻击在多语言场景更易成功**：已有研究表明非英语语言可以绕过安全过滤器，本质上反映了多语言对齐不足。
6. **影响公平性和全球部署**：对齐效果的语言差异意味着非英语用户面临更高的有害内容暴露风险，这是模型大规模部署的伦理障碍。

## 方法详解

### 核心分析框架

本文提出一套基于隐空间分析的多语言对齐评估方法，核心思路是：**对齐应该在模型的表示空间中将有害内容和无害内容分离开来，分离程度反映对齐强度**。

### 数据准备

- **平衡毒性数据集**：每种语言 5,000 个样本（2,500 有毒 + 2,500 无毒），覆盖 9 种语言，重点分析英语(en)、印地语(hi)、中文(zh)、德语(de)
- **多语言平行文本去毒数据集**：包含毒性-去毒版本的平行句对，语义一致但毒性表达不同，用于更严格的控制实验

### 表示提取与可视化

- 对参考模型 $\pi_{ref}$（SFT 后、对齐前）和对齐模型 $\pi_\theta$（对齐后）分别进行前向传播
- 提取最终层嵌入表示，**仅在输入处理阶段探测**，避免解码阶段的记忆和偏差污染
- 使用 PCA 降维到 2D 进行可视化，观察有害/无害聚类的分离情况

### 量化分离度的指标体系

1. **Between-class variance ratio**：衡量有害/无害两类表示之间的方差比例，值越大说明分离越好
2. **Bhattacharyya Distance**：度量两个分布的重叠程度，对数尺度（范围 $10^{-3}$ 到 $10^{+1}$），值越大分离越明确
3. **Silhouette Score**：评估聚类质量，衡量样本归属正确聚类的紧密度和分离度

### 评估的模型

覆盖 7 个开源模型，4 个模型家族：Llama-2 (7B)、Llama-3.1 (8B)、Llama-Guard-3 (8B)、Qwen-2.5 (7B)、Gemma-2 (9B)、Gemma-3 (12B)、Phi-4 (14B)。每个模型比较参考版本和对齐版本。

## 实验关键数据

### 表1：Llama-2 对齐前后 Between-class Variance 提升

| 语言 | 对齐前 Between-class Var | 对齐后 Between-class Var | 提升幅度 |
|:---|:---:|:---:|:---:|
| English | 0.81% | 61.20% | **+60.39%** |
| Hindi | - | - | +19.98% |
| Chinese | - | - | +10.09% |
| German | - | - | +26.85% |

- 英语的类间方差提升幅度是中文的 6 倍、印地语的 3 倍
- PCA 解释方差比为 49.61%，后续使用前 10 个主成分进行更充分的度量

### 表2：Bhattacharyya Distance 跨模型跨语言对比（对数尺度）

| 模型 | en (Δ方向) | hi (Δ方向) | zh (Δ方向) | de (Δ方向) |
|:---|:---:|:---:|:---:|:---:|
| Llama-2 | ↑ 显著增大 | ↑/↓ 不稳定 | ↑ 轻微增大 | ↑ 中等增大 |
| Llama-3.1 | ↑ 显著增大 | ↓ 反向退化 | ↑ 轻微增大 | ↑ 中等增大 |
| Gemma-2 | ↑ 显著增大 | ↑ 中等增大 | ↑ 中等增大 | ↑ 中等增大 |
| Qwen-2.5 | ↑ 显著增大 | ↑ 中等增大 | ↑ 中等增大 | ↑ 中等增大 |

- 英语在所有模型上都表现出 consistent 的分离增强
- 印地语在部分模型上出现"反向效应"：对齐后的模型反而比未对齐模型的聚类分离更差
- Silhouette Score 呈现一致模式：英语的聚类质量提升远高于其他语言

### 平行去毒实验（更困难设定）

- 英语：即使有害句和无害句仅差 1-2 个词，对齐后仍能维持有意义的表示分离
- 印地语：无论对齐前后，模型在低维空间中都无法捕捉到清晰的分布偏移

## 亮点

- **分析视角独特**：不通过生成内容评估安全性，而是直接探测隐空间的表示分布变化，提供了理解对齐机制的新维度
- **量化指标体系完整**：组合使用 PCA 可视化、Bhattacharyya Distance、Silhouette Score 三种互补指标，结论互相验证
- **揭示了关键安全隐患**：印地语的"反向效应"（对齐后分离反而下降）为多语言越狱攻击提供了表示层面的解释
- **模型覆盖全面**：横跨 4 个模型家族、7 个模型，结论具有较好的泛化性
- **实验设计严谨**：使用平行去毒语料进行控制实验，在语义几乎相同的条件下测试对齐的敏感度

## 局限与展望

- **仅关注安全维度**：对齐方法还影响推理、指令遵循、规划等多种能力，本文仅分析毒性/安全一个维度
- **语言覆盖偏少**：仅分析 3 种非英语语言且都是中等资源语言，未涉及真正的低资源语言（如斯瓦希里语、泰语等）
- **分析而非解决方案**：本文以诊断问题为主，未提出具体的改进对齐方法或多语言微调策略
- **数据集规模有限**：每种语言 5,000 个样本虽已超过先前工作，但可能不足以代表多语言场景的全部复杂性
- **Phi-4 缺少参考模型**：Phi-4 没有公开的参考（未对齐）模型，无法进行前后对比

## 与相关工作的对比

### vs Dang et al. (2024) "RLHF Can Speak Many Languages"

Dang et al. 从训练角度研究如何让 RLHF 适应多语言设定，提出跨语言偏好传递方法。本文与之互补：不关注如何改进训练，而是从表示空间角度诊断现有对齐模型在多语言上究竟失败在哪里、失败到什么程度。两者结合可以形成"诊断→治疗"的完整流程。

### vs Lin et al. (2024) 多语言对齐分析

Lin et al. 也分析了对齐在多语言场景的效果，但样本量仅约 200 个。本文使用 5,000 个样本且引入了 Bhattacharyya Distance 和 Silhouette Score 等更严格的量化指标，在方法论和统计可靠性上有显著提升。

### vs Son et al. (2024) 多语言越狱研究

Son et al. 从对抗攻击角度发现非英语语言更容易越狱 LLM。本文从表示空间角度为这一现象提供了机理解释：安全约束在非英语语言的隐空间中分离度不足，使得模型无法在内部有效区分有害/无害输入。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 从隐空间表示角度分析多语言对齐的视角新颖且有深度
- 实验充分度: ⭐⭐⭐ — 模型覆盖全面但语言覆盖偏少，缺低资源语言验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，可视化丰富，伦理讨论到位
- 价值: ⭐⭐⭐⭐ — 为多语言 LLM 安全提供了量化诊断工具和重要的实证发现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] Implicit Cross-Lingual Rewarding for Efficient Multilingual Preference Alignment](implicit_cross-lingual_rewarding_for_efficient_multilingual_preference_alignment.md)

</div>

<!-- RELATED:END -->
