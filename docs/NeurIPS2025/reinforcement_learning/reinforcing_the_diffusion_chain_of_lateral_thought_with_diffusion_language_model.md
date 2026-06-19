---
title: >-
  [论文解读] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models
description: >-
  [NeurIPS 2025][强化学习][扩散语言模型] 提出扩散横向思维链（DCoLT），将扩散语言模型逆向过程中的每个中间步视为潜在"思考"动作，通过基于最终结果的强化学习优化整条推理轨迹，在SEDD和LLaDA两种扩散语言模型上实现了数学和代码生成的SOTA表现。 现有大语言模型的推理能力主要依赖链式思维（CoT）…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "扩散语言模型"
  - "横向思维"
  - "GRPO"
  - "Plackett-Luce"
---

# Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.10446](https://arxiv.org/abs/2505.10446)  
**代码**: [GitHub](https://github.com/maple-research-lab/LLaDOU)  
**领域**: 图像生成  
**关键词**: 扩散语言模型, 横向思维, 强化学习, GRPO, Plackett-Luce

## 一句话总结

提出扩散横向思维链（DCoLT），将扩散语言模型逆向过程中的每个中间步视为潜在"思考"动作，通过基于最终结果的强化学习优化整条推理轨迹，在SEDD和LLaDA两种扩散语言模型上实现了数学和代码生成的SOTA表现。

## 研究背景与动机

现有大语言模型的推理能力主要依赖链式思维（CoT），即逐步分解问题生成中间推理步骤。但CoT受限于自回归模型的因果注意力机制，推理只能沿单一方向线性展开。这与人类认知不符——人在思考初期并不要求完整的语言结构，概念和想法先是零散涌现，再逐步组织成连贯表达。这种非线性创造性推理被称为"横向思维"（lateral thinking）。

扩散语言模型（DLM）的逆向过程天然适合模拟横向思维：

**双向推理**：每个token可自由关注所有其他token（双向注意力），不受因果掩码限制

**格式自由**：中间步不需要遵循语法规则，允许不完整或无序的内容

**非线性生成**：可在多个位置同时生成token，不必从左到右线性生成

然而现有DLM训练方法（如DoT使用带标注的CoT数据做SFT）仍鼓励顺序推理，未充分利用横向思维能力。关键问题在于：**如何强化整条扩散逆向过程使其作为完整的横向思维链被优化？**

## 方法详解

### 整体框架

DCoLT将扩散逆向过程 $x_{0:N}$ 定义为完整的横向思维链。每一步 $n$ 从 $x_{n-1}$ 生成 $x_n$ 视为一个"动作"，整条链用基于最终结果的RL联合优化。不对中间步施加任何显式监督，仅根据最终答案 $x_N$ 的正确性给予奖励（reward=1正确/0错误）。使用GRPO计算优势函数：

$$A^g = \frac{r^g - \text{mean}(r^{1:G})}{\text{std}(r^{1:G})}$$

每步的loss为：

$$\mathcal{L}_{\theta,n} = -\frac{1}{G}\sum_{g=1}^{G}\frac{\pi_{\theta,n}(x_n^g|x_{n-1}^g)}{\pi_{\text{old},n}(x_n^g|x_{n-1}^g)}A^g$$

沿所有步累积梯度后更新参数。

### 关键设计

1. **DCoLT-SEDD（连续时间DLM）**

   SEDD学习concrete score $s_\theta(x,t)_y \approx p_t(y)/p_t(x)$，用于估计逆向转移率。通过τ-leaping将序列级策略定义为所有token转移概率的乘积：

    $\pi_{\theta,n}(x_n|x_{n-1}) = \prod_{i=1}^{|x_n|}p_{\theta,t_n}(x_n^i|x_{n-1})$

   其中每个token的转移概率可从concrete score和转移率矩阵 $Q_t$ 闭式计算。这使得整条轨迹的策略概率可微，可直接用GRPO训练。

2. **DCoLT-LLaDA + 排序去掩码策略模块（UPM）**

   LLaDA是离散时间掩码扩散模型，每步选择部分掩码token揭露为文本。作者发现**去掩码顺序对推理至关重要**——应该先揭露确信度高的token。为此引入Unmask Policy Module（UPM）：

    - UPM为每个token预测排名分数 $h_{\theta,n}^i$，使用Plackett-Luce模型定义排序抽样策略：

    $\pi_{\theta,n}^{\text{unmask}}(\mathcal{U}_n|x_{n-1}) = \prod_{k=1}^{K}\frac{\exp(h_{\theta,n}^{u_n(k)})}{\sum_{j=k}^{K}\exp(h_{\theta,n}^{u_n(j)}) + \sum_{j \in \mathcal{M}_n}\exp(h_{\theta,n}^{u_n(j)})}$

    - token生成策略：$\pi_{\theta,n}^{\text{token}}(x_n|x_{n-1},\mathcal{U}_n) = \prod_{i \in \mathcal{U}_n}p_{\theta,n}(x_n^i|x_{n-1})$

    - 完整策略为两者之积：$\pi_{\theta,n}(x_n|x_{n-1}) = \pi_{\theta,n}^{\text{unmask}} \cdot \pi_{\theta,n}^{\text{token}}$

   UPM仅含一个Transformer block，通过自适应层归一化嵌入扩散步 $n$ 和掩码指示信息，计算开销很小。该模型命名为LLaDOU。

3. **分步梯度累积策略**

   逆向过程包含多步生成，完整展开计算图会导致显存溢出。采用每步反向传播梯度后累积，最后统一更新参数的策略。

### 损失函数 / 训练策略

- 奖励函数：规则验证（数学判断答案正确性，代码判断unit test通过）
- RL算法：GRPO（组相对策略优化）
- 训练数据：GSM8K和MATH共15K公开问题（无需推理过程标注）
- 硬件：16块H800 GPU

## 实验关键数据

### SEDD 400M实验

| 模型 | 后训练 | Sudoku 4×4 | GSM8K-Aug |
|------|--------|-----------|-----------|
| GPT2 + CoT | SFT | 71.5% | 43.9% |
| GPT2 + CoT | RL | 74.6% | - |
| SEDD + DoT | SFT | 79.4% | 53.5% |
| **SEDD + DCoLT** | **RL** | **96.2%** | **57.0%** |

### LLaDOU 8B实验

| 模型 | 后训练 | GSM8K | MATH | HumanEval | MBPP |
|------|--------|-------|------|-----------|------|
| LLaDA 8B | baseline | 78.3% | 38.9% | 39.6% | 40.2% |
| d1-LLaDA | SFT+RL | 82.1% | 40.2%‡ | - | - |
| **LLaDOU** | **RL** | **88.1%** | **44.6%** | **59.1%** | **51.6%** |
| DeepseekMath-RL 7B | SFT†+RL† | 88.2% | 51.7% | - | - |

### 消融实验

| 训练组件 | UPM | LLaDA | GSM8K |
|----------|-----|-------|-------|
| 无训练 | × | × | 47.27% |
| 仅UPM(含AdaLN) | ✓(AdaLN) | × | 69.24% |
| UPM(无AdaLN)+LLaDA | ✓(无AdaLN) | ✓ | 80.53% |
| UPM(含AdaLN)+LLaDA | ✓(AdaLN) | ✓ | **81.06%** |

### 关键发现

1. **UPM的关键作用**：仅训练UPM（冻结LLaDA）就能从47.27%提升到69.24%，说明去掩码顺序对推理极为关键
2. **数据高效**：LLaDOU仅用15K公开问题训练就接近DeepseekMath（776K问题SFT + 144K问题RL）
3. **长度外推**：训练时固定生成长度256，推理时增加到384可在MATH上额外提升1.6%，难题提升更明显
4. **横向思维优于CoT**：同为400M模型，DCoLT在Sudoku上超过CoT/DoT约17-25个百分点

## 亮点与洞察

1. **范式创新**：首次提出将扩散模型的逆向过程视为完整推理链并用RL端到端优化，而非单步独立训练
2. **去掩码顺序可学习**：Plackett-Luce模型提供了优雅的数学框架将离散排序问题纳入可微策略优化
3. **无监督的推理涌现**：不需要任何推理过程标注，仅靠最终答案的正确性信号就能让模型学会有效的横向推理模式
4. **渐进式生成涌现**：训练后自然出现从易到难的渐进生成模式（先填入确定性高的token），与人类思维习惯一致

## 局限与展望

- 仅在可验证任务（数学/代码）上验证，泛化到开放域需要reward model
- 训练数据量和计算资源有限，性能仍有提升空间
- 序列长度限制在256-512 token，长推理任务受限
- 多步生成的计算图展开导致显存开销较大

## 相关工作与启发

- **扩散语言模型**: SEDD（连续时间离散扩散）、LLaDA（掩码扩散）、Dream
- **推理强化学习**: DeepSeek-R1（自回归RL推理）、GRPO
- **扩散推理**: DoT（用SFT训练扩散模型做CoT推理）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — DCoLT开辟了扩散语言模型推理的新范式
- 实验充分度: ⭐⭐⭐⭐☆ — 多任务多消融对比充分，但缺少更大模型规模验证
- 写作质量: ⭐⭐⭐⭐☆ — 结构清晰、公式推导完整
- 价值: ⭐⭐⭐⭐⭐ — 证明了非自回归模型也能通过横向思维实现强推理能力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](../../ICML2026/reinforcement_learning/learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ACL 2026\] d-TreeRPO: Towards More Reliable Policy Optimization for Diffusion Language Models](../../ACL2026/reinforcement_learning/d-treerpo_towards_more_reliable_policy_optimization_for_diffusion_language_model.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)

</div>

<!-- RELATED:END -->
