---
title: >-
  [论文解读] Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM
description: >-
  [NeurIPS 2025][LLM安全][数据提取攻击] 揭示了即使精确遗忘（从头重训练去除数据影响）也存在隐私泄露风险：攻击者利用遗忘前后两个模型检查点的差异，通过逆向模型引导和 token 过滤策略，可显著提升已删除数据的提取成功率，在某些场景下提取率翻倍。 大语言模型（LLM）训练数据中可能包含敏感个人信息（如医疗记…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "数据提取攻击"
  - "机器遗忘"
  - "LLM隐私"
  - "逆向模型引导"
  - "精确遗忘"
---

# Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM

**会议**: NeurIPS 2025  
**arXiv**: [2505.24379](https://arxiv.org/abs/2505.24379)  
**代码**: [GitHub](https://github.com/Nicholas0228/unlearned_data_extraction_llm)  
**领域**: 医学图像  
**关键词**: 数据提取攻击, 机器遗忘, LLM隐私, 逆向模型引导, 精确遗忘

## 一句话总结

揭示了即使精确遗忘（从头重训练去除数据影响）也存在隐私泄露风险：攻击者利用遗忘前后两个模型检查点的差异，通过逆向模型引导和 token 过滤策略，可显著提升已删除数据的提取成功率，在某些场景下提取率翻倍。

## 研究背景与动机

大语言模型（LLM）训练数据中可能包含敏感个人信息（如医疗记录），GDPR 和 CCPA 等法规赋予用户"被遗忘权"，机器遗忘成为刚需：

**近似遗忘的脆弱性**：梯度上升、NPO 等近似方法虽计算高效，但已被证明可被攻击恢复遗忘数据（Lucki et al., Hu et al.），且缺乏形式化保证

**精确遗忘被视为"金标准"**：通过在去除目标数据后从头重训练，精确遗忘被普遍认为能完全消除数据影响，对提取/反演攻击免疫

**被忽视的威胁模型**：在实际部署中，遗忘前的模型检查点/logits API 通常是可获取的——开源模型的快照可被保存，API 用户可能已保存之前的查询结果。这意味着攻击者可同时访问遗忘前后两个版本

本文的核心发现具有反直觉的矛盾性：**遗忘操作本身反而增加了信息泄露风险**——因为它为攻击者提供了额外的信号（两个模型行为的差异精确编码了被删除数据的分布信息）。

## 方法详解

### 整体框架

攻击者拥有遗忘前模型 $\theta$ 和遗忘后模型 $\theta'$ 的检查点或 logits API，以及每条待提取数据的前缀 $x_{\leq i}$（如患者姓名、出生日期等已知信息）。攻击目标是利用两个模型的差异重建被删除的数据 $X_0$。方法核心是将遗忘视为微调的逆过程，通过逆向模型引导构建近似被删除数据分布的"伪预测器"。

### 关键设计

1. **逆向模型引导（Reversed Model Guidance）**：将遗忘过程视为微调的逆过程——$\theta$ 可看作在 $\theta'$ 基础上用 $X_0$ 微调的结果。假设从 $\theta'$ 重新学习 $X_0$ 分布的参数化近似为 $p_\theta(x_{i+1}|x_{\leq i}) \propto p_{\theta'}^{1-\lambda}(x_{i+1}|x_{\leq i}) \cdot q^\lambda(x_{i+1}|x_{\leq i})$，通过对数化和重参数化推导出：

    $\log q(x_{i+1}|x_{\leq i}) = \log p_{\theta'}(x_{i+1}|x_{\leq i}) + w(\log p_\theta(x_{i+1}|x_{\leq i}) - \log p_{\theta'}(x_{i+1}|x_{\leq i}))$

   其中 $w = 1/\lambda$ 是引导尺度。设计动机：类比扩散模型中的 classifier guidance，利用遗忘前后模型的 logits 差作为引导信号，差异越大的 token 越可能属于被删除数据的分布。

2. **Token 过滤策略**：直接使用 logits 差异可能产生不连贯的生成。借鉴对比解码（Contrastive Decoding），限制候选 token 必须在遗忘前模型 $\theta$ 中有足够高的概率：

    $V' = \{v \in V \mid p_\theta(v|x_{\leq i}) \geq \gamma \max_{v \in V} p_\theta(v|x_{\leq i})\}$

   然后在 $V'$ 中选择引导分布 $\log q$ 最高的 token。设计动机：遗忘前模型仍保留对 $X_0$ 的残留知识，限制在其高概率 token 中选择可消除低频噪声 token，保持生成文本的质量和连贯性。

3. **贪心解码集成**：最终的下一 token 选择为：$x_{\text{next}} = \arg\max_{v \in V'} \log q(v|x_{\leq i})$，结合了引导信号的方向性和 token 过滤的质量控制。

### 损失函数 / 训练策略

- 攻击方法本身**无需训练**，仅需模型推理
- 精确遗忘的实现：在完整数据集上微调得到 $\theta$，去除 $X_0$ 后重新微调预训练模型得到 $\theta'$
- 默认配置：遗忘集占10%，引导尺度 $w=2.0$（Phi-1.5）/$w=1.4$（Llama2-7B），过滤阈值 $\gamma=10^{-5}$
- 评估指标：Rouge-L(R) 和 A-ESR（平均提取成功率，阈值 $\tau=1.0$/$\tau=0.9$）

## 实验关键数据

### 主实验

三个标准基准上的提取攻击结果（10% 遗忘集）：

| 数据集 | 模型 | 方法 | Rouge-L(R) | A-ESR$_{0.9}$ | A-ESR$_{1.0}$ |
|---|---|---|---|---|---|
| MUSE | Phi-1.5 | 仅预遗忘生成 | 0.473 | 0.114 | 0.101 |
| MUSE | Phi-1.5 | **本文攻击** | **0.606** | **0.249** (+118%) | **0.224** (+121%) |
| MUSE | Llama2-7b | 仅预遗忘生成 | 0.675 | 0.424 | 0.384 |
| MUSE | Llama2-7b | **本文攻击** | **0.744** | **0.496** (+17%) | **0.438** (+14%) |
| TOFU | Phi-1.5 | 仅预遗忘生成 | 0.566 | 0.100 | 0.070 |
| TOFU | Phi-1.5 | **本文攻击** | **0.643** | **0.202** (+102%) | **0.120** (+71%) |
| WMDP | Phi-1.5 | 仅预遗忘生成 | 0.429 | 0.079 | 0.069 |
| WMDP | Phi-1.5 | **本文攻击** | **0.567** | **0.218** (+175%) | **0.192** (+178%) |

### 消融实验

| 配置 | Rouge-L(R) | A-ESR$_{1.0}$ | 说明 |
|---|---|---|---|
| 仅遗忘后模型生成 | 0.296 | 0.004 | 遗忘后模型几乎无法提取 |
| 仅遗忘前模型生成 | 0.473 | 0.101 | 基线提取能力 |
| 引导 $w=1.0$ | ~0.52 | ~0.14 | 引导不足 |
| 引导 $w=2.0$ (最优) | **0.606** | **0.224** | 最佳引导强度 |
| 引导 $w=4.0$ | ~0.55 | ~0.18 | 过度引导导致退化 |
| $\gamma=10^{-1}$ (过严过滤) | ~0.50 | ~0.13 | 过滤过严影响引导效果 |
| $\gamma=10^{-5}$ (最优) | **0.606** | **0.224** | 适度过滤最优 |
| $\gamma=0$ (无过滤) | ~0.56 | ~0.19 | 不过滤轻微降低 |

### 关键发现

- **医疗场景模拟**：在合成的 SOAP 格式患者诊断记录上，攻击方法将精确匹配提取率从 14% 提升到 **21%**（+50%），证明了在真实医疗隐私场景中的严重威胁
- **训练轮数效应**：模型记忆程度越高（更多训练轮次），最优引导尺度 $w$ 越小——这与理论推导一致（$w = 1/\lambda$，$\lambda$ 随训练增加）
- **遗忘集大小影响较小**：提取效果更多取决于实例级记忆，而非遗忘数据的整体规模
- **近似遗忘也受影响**：方法在 GA、NPO 等近似遗忘上也有效，但改进幅度随模型效用下降而减小

## 亮点与洞察

- **深刻的矛盾性发现**：遗忘操作本身成为了隐私泄露的信号来源，这对整个机器遗忘领域的安全假设提出了根本性挑战
- **类比扩散模型引导的巧妙迁移**：将 classifier-free guidance 从生成模型迁移到隐私攻击，视遗忘为微调的逆过程
- **实用威胁模型**：攻击者仅需 API 级别的 logits 访问（不需要模型权重），大幅降低了攻击门槛
- **对遗忘评估标准的警示**：现有评估仅考虑遗忘后模型，本文表明必须考虑攻击者可获取历史检查点的威胁模型

## 局限与展望

- 假设攻击者拥有数据前缀（如患者姓名），这在某些场景中可能不成立
- 当前仅验证了 Phi-1.5 和 Llama2-7B 等中小模型，对更大规模模型效果待验证（附录有初步结果）
- 医疗数据集为合成数据，真实医疗 LLM 的记忆和遗忘行为可能不同
- 引导尺度 $w$ 需要手动调节，自适应选择策略有待开发
- 未深入讨论有效的防御措施（仅简单实验了加噪数据等策略）

## 相关工作与启发

- 与 Carlini et al. (2021) 数据提取攻击的关系：本文将提取攻击扩展到遗忘场景，利用两个模型差异作为额外信号
- 与 Contrastive Decoding (Li et al., 2023) 的联系：借鉴了受限 token 选择的思想，但目的从提升生成质量变为提升攻击效率
- 对 LLM 隐私治理的启发：模型版本管理需更严格的安全策略，旧版本检查点应视为敏感资产
- 对医疗 AI 的警示：在处理患者数据的 LLM 中，简单的精确遗忘不足以保证隐私安全

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 揭示精确遗忘的隐私悖论，逆向引导攻击的思路极具创新性
- **实验充分度**: ⭐⭐⭐⭐ 三个标准基准+医疗模拟+消融研究+泛化分析，较完整，但防御实验略少
- **写作质量**: ⭐⭐⭐⭐ 问题阐述清晰，威胁模型定义严谨，图示有效（尤其是 Fig.1 的患者数据提取示例）
- **价值**: ⭐⭐⭐⭐⭐ 对 LLM 安全和隐私治理有重大影响，特别是医疗 AI 领域应高度重视此类风险

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On Optimal Steering to Achieve Exact Fairness](on_optimal_steering_to_achieve_exact_fairness.md)
- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](../../ICLR2026/llm_safety/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[CVPR 2025\] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](../../CVPR2025/llm_safety/forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)
- [\[NeurIPS 2025\] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)

</div>

<!-- RELATED:END -->
