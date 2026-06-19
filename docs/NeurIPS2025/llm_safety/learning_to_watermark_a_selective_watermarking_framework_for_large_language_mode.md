---
title: >-
  [论文解读] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization
description: >-
  [NeurIPS 2025][LLM安全][LLM水印] 提出LTW（Learning to Watermark）框架，使用一个轻量级选择器网络基于句子嵌入、token熵和当前水印比例来自适应决定何时施加水印，通过多目标优化（MGDA）在可检测性和文本质量之间达到Pareto最优，在不降低检测性能的前提下显著提升水印文本质量。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "LLM水印"
  - "选择性水印"
  - "多目标优化"
  - "文本质量"
  - "水印可检测性"
---

# Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2510.15976](https://arxiv.org/abs/2510.15976)  
**代码**: [https://github.com/fattyray/learning-to-watermark](https://github.com/fattyray/learning-to-watermark)  
**领域**: AI安全  
**关键词**: LLM水印, 选择性水印, 多目标优化, 文本质量, 水印可检测性

## 一句话总结
提出LTW（Learning to Watermark）框架，使用一个轻量级选择器网络基于句子嵌入、token熵和当前水印比例来自适应决定何时施加水印，通过多目标优化（MGDA）在可检测性和文本质量之间达到Pareto最优，在不降低检测性能的前提下显著提升水印文本质量。

## 研究背景与动机

**领域现状**：LLM快速发展带来版权和滥用风险，水印技术（如KGW、Unigram）成为检测AI生成文本的重要工具。KGW通过将词表分为"绿名单"和"红名单"并偏向绿名单token来嵌入可检测信号。

**现有痛点**：现有水印方法面临检测性与文本质量的权衡——系统性地偏移token选择会破坏语义连贯性。TS-watermark限制用户调参灵活性、NS-watermark严重拖慢生成速度且检测性脆弱、EXP-edit检测极慢。

**核心矛盾**：选择性水印（只对部分token施加水印）是解决质量-检测权衡的有前途方向，但现有方法SWEET仅用熵阈值作为选择标准，需要大量网格搜索手动调参，且忽略了语义上下文等重要信息。

**本文目标**：如何自动学习最优的水印施加决策——何时该给token加水印、何时不该？

**切入角度**：将选择性水印决策建模为一个可学习的网络输出，利用多目标优化同时优化检测性和文本质量两个冲突目标。

**核心idea**：训练一个轻量MLP作为"选择器"，综合利用语义嵌入、token熵和水印比例三种信息，通过MGDA在检测性和质量之间寻找Pareto最优解。

## 方法详解

### 整体框架
在LLM的每个token生成步，选择器网络根据当前上下文决定是否对该token施加水印。训练时使用连续输出实现可微优化，推理时通过自适应阈值离散化为二值决策。检测时重建相同的选择决策，只检测被选中的token。

### 关键设计

1. **选择器网络 (Selector Network)**:

    - 功能：在每个生成步骤决定当前token是否应被水印
    - 核心思路：输入为三部分的拼接——(a) 前 $k$ 个token的句子嵌入 $\mathcal{E}_{\text{sem}}(s_{n-k+1:n})$（SimCSE提供）；(b) 当前token概率分布的Shannon熵 $e$；(c) 当前已水印token的比例 $r$。MLP网络输出 $\mathbf{m}_{\text{wm}} \in [0,1]$：
    $\mathbf{m}_{\text{wm}} = \mathcal{M}_\theta(\mathcal{E}_{\text{sem}}(s_{n-k+1:n}),\; e,\; r)$
    - 设计动机：熵单独不够——语义上下文可以区分哪些词类（如介词、连词）更不宜被水印；水印比例提供全局调控信号

2. **可微水印框架**:

    - 功能：将标准KGW水印过程松弛为可微版本以支持梯度训练
    - 核心思路：原始logits修正为 $\tilde{\mathbf{l}}_{t+1} = \mathbf{l}_{t+1} + \delta \cdot \mathbf{m}_{\text{wm}}^{[t+1]} \mathbf{m}_{\text{green}}$，其中 $\delta$ 是水印强度。训练时 $\mathbf{m}_{\text{wm}}$ 保持连续值；推理时通过阈值 $\tau$ 离散化：$m_{\text{wm}} = \mathbb{1}[\mathbf{m}_{\text{wm}} > \tau]$
    - z-score检测公式松弛为可微版本：
    $z = \frac{\sum_{t=1}^T p_{gr}^{[t]} \cdot m_{\text{wm}}^{[t]} - \gamma \sum_{t=1}^T m_{\text{wm}}^{[t]}}{\sqrt{\sum_{t=1}^T m_{\text{wm}}^{[t]} \gamma(1-\gamma)}}$

3. **自适应阈值 (Adaptive Threshold)**:

    - 功能：根据当前水印比例动态调整二值化阈值
    - 核心思路：水印比例低时降低阈值→更多token被水印以保证检测性；比例高时提高阈值→减少水印以保持质量
    - 设计动机：静态阈值无法平衡不同生成阶段的需求，动态调整可以在全局层面维持检测性和质量的平衡

### 损失函数与训练策略

两个优化目标通过MGDA（Multiple Gradient Descent Algorithm）联合优化：

**质量导向损失**：
$$\mathcal{L}_Q = \lambda_{\text{sim}} \mathcal{L}_S + \lambda_{\text{entropy}} \mathcal{L}_{\text{entropy}} + \lambda_{\text{fix}} \mathcal{L}_{\text{output\_fix}}$$

- $\mathcal{L}_S = -\cos_{\text{sim}}(E_w, E_s)$：最大化水印文本与非水印文本的语义相似度
- $\mathcal{L}_{\text{entropy}} = \text{BCE}(m_{\text{wm}}, \sigma(\lambda_e(e - \mu_e)))$：鼓励在高熵时水印、低熵时不水印
- $\mathcal{L}_{\text{output\_fix}} = -\frac{1}{T}\sum_{t=1}^T (m_{\text{wm}}^{[t]} - 0.5)^2$：正则化输出趋向0/1的mask行为

**检测导向损失**：
$$\mathcal{L}_D = -\lambda_z z + \lambda_{\text{wm}} \mathcal{L}_{\text{wm\_ratio}} + \lambda_{\text{fix}} \mathcal{L}_{\text{output\_fix}}$$

- $-\lambda_z z$：最大化z-score
- $\mathcal{L}_{\text{wm\_ratio}} = \text{MSE}(m_{\text{wm}}^{[t]}, f(r_t))$：水印比例低时鼓励多水印，高时鼓励少水印

训练使用OPT-1.3b在C4 RealNewsLike子集上完成（10k训练，500测试）。

## 实验关键数据

### 主实验

在OPT-6.7B上对比各水印方法：

| 方法 | TPR@2% | AUROC | 最佳F1 | 困惑度↓ | 语义相似度↑ |
|------|--------|-------|--------|---------|-----------|
| KGW | 0.998 | 0.9999 | 0.999 | 20.52 | 0.547 |
| Unigram | 1.000 | 1.0000 | 1.000 | 17.68 | 0.531 |
| EXP-edit | 0.972 | 0.9874 | 0.980 | 23.93 | 0.505 |
| SWEET | 1.000 | 1.0000 | 1.000 | 17.93 | 0.555 |
| TS-watermark | 0.996 | 0.9997 | 0.996 | 14.08 | 0.565 |
| **LTW-1 (Ours)** | **1.000** | **1.0000** | **1.000** | **13.62** | **0.570** |
| **LTW-0 (Ours)** | **1.000** | **1.0000** | **1.000** | **13.62** | **0.574** |

核心发现：LTW在保持完美检测性（AUROC=1.0）的同时，困惑度仅13.62，远低于KGW(20.52)和SWEET(17.93)，也低于TS-watermark(14.08)。

### 消融实验——自适应阈值模块

| 配置 | $\delta$=1.5 | $\delta$=2 | $\delta$=2.5 | $\delta$=3 | $\delta$=3.5 | $\delta$=4 |
|------|-------------|-----------|-------------|-----------|-------------|-----------|
| 有自适应 - PPL | 11.69 | 12.35 | 13.00 | 13.41 | 13.98 | 14.03 |
| 有自适应 - z-score | 6.84 | 9.29 | 11.24 | 12.67 | 13.78 | 14.50 |
| 无自适应 - PPL | 11.80 | 12.52 | 13.22 | 13.40 | 13.92 | 14.31 |
| 无自适应 - z-score | 6.83 | 9.03 | 11.12 | 12.44 | 13.57 | 14.37 |

### 关键发现
- LTW在所有水印强度设置下都实现了比KGW更优的Pareto frontier（更高检测率+更低困惑度）
- 自适应阈值在6种水印强度中的全部6种都提升了z-score，并在4/6种中降低了困惑度
- 网络输出与词性高度相关：倾向于不水印介词、连词、标点和符号（这些影响语义连贯性），而倾向于水印副词和形容词（这些有更多同义替换）
- LTW-0（基于Unigram）在释义攻击下表现更优，因为固定的绿/红名单不受前一token变化影响
- 在GPT-J-6B上的结果一致，困惑度从KGW的16.55降至LTW-0的9.56

## 亮点与洞察
- **可学习的选择策略**：首次用训练神经网络替代手工规则来决定水印施加时机。网络自动学到了"不水印功能词、多水印描述词"的策略，这与语言学直觉一致但无需人工编码
- **多目标优化的优雅应用**：用MGDA处理检测性和质量的冲突目标，避免了手动权衡超参数。这种multi-objective框架可迁移到其他存在conflicting objectives的AI安全场景
- **训练-推理分离的巧妙设计**：训练时用连续mask确保可微性，推理时用硬binary mask增强鲁棒性，二者通过输出正则化（推向0/1）无缝衔接

## 局限与展望
- 选择器网络在OPT-1.3b上训练，迁移到更大模型时是否仍然最优未验证
- sentence embedding窗口大小的选择（文中讨论了大小的权衡但未系统搜索最优值）
- 释义攻击下LTW-1的鲁棒性略逊于Unigram基线，说明KGW基础上的选择性水印在鲁棒性方面仍有提升空间
- 仅在文本续写(text completion)任务上评估，对话/代码生成等其他场景的表现待验证

## 相关工作与启发
- **vs KGW**: LTW-1在KGW基础上添加选择性，困惑度从20.5降到13.6（33%降幅），检测性不变
- **vs SWEET**: 同为选择性水印，但SWEET依赖手动熵阈值网格搜索，LTW用网络自动学习；且LTW利用了语义嵌入和水印比例等额外信息
- **vs TS-watermark**: TS-watermark训练 $\gamma$ 和 $\delta$ 生成器来自适应调整水印超参数；LTW从选择"哪些token"的角度切入，更直接地保护关键token
- **vs NS-watermark**: NS-watermark最小化水印token数但z-score刚过阈值，极易受攻击；LTW保持高z-score的同时通过选择性降低质量损失

## 评分
- 新颖性: ⭐⭐⭐⭐ 用可学习网络替代规则驱动的选择策略是清晰的创新点，多目标优化的应用自然合理
- 实验充分度: ⭐⭐⭐⭐⭐ 两个LLM、多种基线、释义攻击、不同水印强度的Pareto曲线对比、消融实验和词性分析均很全面
- 写作质量: ⭐⭐⭐⭐ 方法描述详细清晰，公式推导完整，但部分符号稍显冗余
- 价值: ⭐⭐⭐⭐ 在LLM水印这一热门话题上提供了实用且有效的改进方案，代码已开源

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Machine Unlearning via Adaptive Gradient Reweighting and Multi-stage Objective Optimization](../../CVPR2026/llm_safety/machine_unlearning_via_adaptive_gradient_reweighting_and_multi-stage_objective_o.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](../../ICML2025/llm_safety/de-mark_watermark_removal_in_large_language_models.md)
- [\[ACL 2025\] From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](../../ACL2025/llm_safety/from_tradeoff_to_synergy_a_versatile.md)
- [\[ACL 2025\] Improved Unbiased Watermark for Large Language Models](../../ACL2025/llm_safety/improved_unbiased_watermark_for_large_language.md)
- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](../../ICML2025/llm_safety/improving_llm_safety_alignment_with_dual-objective_optimization.md)

</div>

<!-- RELATED:END -->
