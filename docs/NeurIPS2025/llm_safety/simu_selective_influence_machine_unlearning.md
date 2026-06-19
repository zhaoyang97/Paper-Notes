---
title: >-
  [论文解读] SIMU: Selective Influence Machine Unlearning
description: >-
  [NeurIPS 2025][LLM安全][机器遗忘] 提出 SIMU 两阶段框架：先通过梯度聚合识别编码遗忘集信息的关键 MLP 神经元，再仅对这些神经元进行二阶（Sophia）优化遗忘，在保持遗忘效果的同时大幅提升模型原有能力的保留。 大语言模型（LLM）会记忆训练数据中的敏感信息，这在数据隐私和 AI 安全方面引发严重…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "机器遗忘"
  - "二阶优化"
  - "神经元定位"
  - "影响函数"
---

# SIMU: Selective Influence Machine Unlearning

**会议**: NeurIPS 2025  
**arXiv**: [2510.07822](https://arxiv.org/abs/2510.07822)  
**代码**: 未公开  
**领域**: LLM安全  
**关键词**: 机器遗忘, LLM安全, 二阶优化, 神经元定位, 影响函数  

## 一句话总结

提出 SIMU 两阶段框架：先通过梯度聚合识别编码遗忘集信息的关键 MLP 神经元，再仅对这些神经元进行二阶（Sophia）优化遗忘，在保持遗忘效果的同时大幅提升模型原有能力的保留。

## 研究背景与动机

大语言模型（LLM）会记忆训练数据中的敏感信息，这在数据隐私和 AI 安全方面引发严重关注。机器遗忘（Machine Unlearning）旨在让模型精确"遗忘"特定数据的影响，而无需从头重训。

现有遗忘方法面临核心矛盾：

**梯度上升类方法**（如 Gradient Ascent）容易过度遗忘，导致模型整体能力严重退化

**仅在保留集微调**则容易遗忘不充分，目标数据的影响仍残留

**正则化方法**（如 GradDiff、NPO）虽然尝试平衡两者，但在激进遗忘场景下仍会损害模型效用

**二阶优化方法**（如 SOUL）利用 Hessian 信息进行更精准的参数更新，表现较好，但 Hessian 近似误差会累积

关键洞察：**没有先前工作将定位感知技术与二阶影响函数遗忘结合起来**，即同时解决"在哪更新"和"如何更新"两个问题。SIMU 正是填补这一空白。

## 方法详解

### 整体框架

SIMU 是一个两阶段框架：

- **阶段一**：关键神经元识别（Critical Neuron Identification）—— 找到哪些 MLP 神经元主要编码了遗忘集信息
- **阶段二**：选择性影响函数遗忘（Selective Influence Unlearning）—— 仅对这些关键神经元执行二阶优化更新

### 关键设计一：关键神经元识别

基于 Meng et al. 的发现：Transformer 的 MLP 层充当键值记忆，存储模型的事实知识。SIMU 扩展了 Privacy Neuron Detector，为自回归语言模型设计梯度聚合方案：

**步骤 1**：将遗忘集的 QA 对转换为多个 next-token prediction 样本

**步骤 2**：对第 $l$ 层第 $k$ 个 MLP down-sample 神经元 $w_l^k$，获取其在样本 $i$ 上的原始激活值 $\beta_{l,i}^k$

**步骤 3**：将激活从 0 到 $\beta_{l,i}^k$ 分 $m$ 步均匀缩放，在每步计算损失变化

**步骤 4**：计算归因分数：

$$\text{Att}(w_l^k) = \frac{1}{m} \sum_{j=1}^{m} \sum_{i=1}^{|D|} \beta_{l,i}^k \frac{\partial L_i(a_{j,l,i}^k)}{\partial a_{j,l,i}^k}$$

**步骤 5**：按层阈值化生成二值掩码。令 $M_l = \max_k \text{Att}(w_l^k)$，若 $\text{Att}(w_l^k) > t \cdot M_l$ 则标记为关键神经元，$t \in (0,1]$ 控制每层选中的神经元比例。

### 关键设计二：选择性影响函数遗忘

在第二阶段使用 Sophia 优化器在二阶迭代框架中微调：

- **冻结**：除 attention 投影层和 MLP down-sample 层外的所有参数
- **稀疏更新**：MLP 内仅更新关键神经元（由掩码 $\mathbf{M}$ 控制）
- **全量更新**：attention 层保持完整更新（保留序列建模能力）

Sophia 的每步更新为裁剪的拟牛顿步：

$$\theta_{t+1} = \theta_t - \eta_t \cdot \text{clip}\left(\frac{m_t}{\max\{\gamma H_t, \epsilon\}}, 1\right)$$

其中 $m_t$ 是一阶动量的 EMA，$H_t$ 是 Gauss-Newton 对角 Hessian 的 EMA。

### 损失函数 / 训练策略

掩码在三个关键位置施加，确保非关键神经元完全不受影响：

1. **一阶动量 EMA 之后**：$m_t = \mathbf{M} \odot m_t' + \bar{\mathbf{M}} \odot m_{t-1}$
2. **二阶曲率 EMA 之后**：$H_t = \mathbf{M} \odot H_t' + \bar{\mathbf{M}} \odot H_{t-1}$
3. **参数更新之后**：$\theta_t = \mathbf{M} \odot \theta_t' + \bar{\mathbf{M}} \odot \theta_{t-1}$

这种三重掩码机制确保遗忘更新严格限制在关键神经元上，最大限度减少对保留知识的附带损害。遗忘目标使用 GradDiff：在遗忘集上做梯度上升（增大损失），在保留集上做梯度下降（维持损失），两者加权组合。

## 实验关键数据

### 主实验

在 TOFU 和 LUME 两个基准上，使用 LLaMA2-7B 和 OLMo-1B 进行评估。

**TOFU 基准结果**：

| 模型 | 方法 | 聚合分数 ↑ | Forget EM ↓ | Retain EM ↑ | World Facts EM ↑ |
|------|------|:---------:|:----------:|:----------:|:---------------:|
| LLaMA2-7B | FO-GradDiff | 0.4738 | 72.75% | 76.50% | 79.49% |
| LLaMA2-7B | SO-GradDiff | 0.7957 | 10.25% | 72.25% | 82.05% |
| LLaMA2-7B | **SIMU-GradDiff** | **0.7963** | 20% | **78.00%** | **82.90%** |
| OLMo-1B | FO-GradDiff | 0.7059 | 26.50% | 63.00% | 0.85% |
| OLMo-1B | SO-GradDiff | 0.8235 | 22.75% | 78.00% | 38.46% |
| OLMo-1B | **SIMU-GradDiff** | **0.8438** | **10.25%** | 75.50% | **42.74%** |

**LUME 基准结果**：

| 模型 | 方法 | 聚合分数 ↑ | Forget Overall ↓ | Utility Overall ↑ |
|------|------|:---------:|:----------------:|:-----------------:|
| LLaMA2-7B | SO-GradDiff | 0.607 | 0.0187/0.00 | 0.7714/0.6212 |
| LLaMA2-7B | **SIMU** | **0.659** | **0.0025/0.00** | **0.8295/0.7149** |
| OLMo-1B | SO-GradDiff | 0.728 | 0.0055/0.0 | 0.9244/0.8499 |
| OLMo-1B | **SIMU** | **0.740** | **0.0015/0.0** | **0.9365/0.8540** |

### 消融实验

- 对 LLaMA2-7B，SIMU 相对 SO-GradDiff 的效用提升约 **5-6%**
- 对 OLMo-1B，效用提升约 **1-2%**
- 差异归因于 LLaMA2-7B 的遗忘集信号更集中在少数关键神经元中

### 关键发现

1. **大模型获益更大**：LLaMA2-7B 比 OLMo-1B 从 SIMU 中获益更多，因其遗忘信号更集中
2. **稀疏 MLP + 全量 Attention 是最优组合**：在遗忘和效用保持之间取得理想平衡
3. **关键神经元掩码有效减少 Hessian 近似误差的传播**

## 亮点与洞察

1. **理论与实践的精妙结合**：将"MLP 是事实记忆"的理论洞察转化为实用遗忘策略
2. **三重掩码设计精巧**：在一阶动量、二阶曲率和参数更新三个位置同时施加掩码，确保更新严格受限
3. **渐进式激活缩放**的归因方法比简单梯度归因更精确
4. **注意力层全量更新 + MLP 稀疏更新**的设计直觉清晰：保留上下文建模能力，仅修正事实存储
5. 首次将定位感知与二阶影响函数遗忘结合，填补了研究空白

## 局限与展望

1. **计算开销**：关键神经元识别阶段需逐神经元多步激活缩放和梯度计算，大模型上成本较高
2. **阈值 $t$ 的敏感性**：阈值选择影响较大，论文未详细讨论自动调参策略
3. **仅适用于固定遗忘集**：持续遗忘请求需每次重算关键神经元掩码
4. **评估局限**：仅在两个基准上测试，缺乏更大模型（70B）的验证
5. 可探索将神经元归因扩展到 Attention 层，实现更精细选择性遗忘

## 相关工作与启发

- **SOUL**（Jia et al., 2024）：二阶影响函数遗忘的基础工作，SIMU 在此基础上加入定位
- **ROME/MEMIT**（Meng et al., 2022/2023）：MLP 作为事实记忆的理论基础
- **Privacy Neuron Detector**（Wu et al., 2023）：神经元级隐私检测，SIMU 扩展到自回归模型
- **GradDiff**（Liu et al., 2022）：梯度差分遗忘的经典方法

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：定位感知与二阶优化结合是自然但有效的创新
- **实验充分度** ⭐⭐⭐⭐：两个基准、两个模型的全面评估，改进一致
- **理论深度** ⭐⭐⭐：有直觉解释但缺乏严格理论分析
- **实用性** ⭐⭐⭐⭐：方法相对简单，可直接应用于现有 LLM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[CVPR 2026\] pH-Strips for Selective Forgetting: A Blunt but Fast Diagnostic Baseline for Machine Unlearning](../../CVPR2026/llm_safety/ph-strips_for_selective_forgetting_a_blunt_but_fast_diagnostic_baseline_for_mach.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/llm_safety/munba_machine_unlearning_via_nash_bargaining.md)
- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](../../ACL2025/llm_safety/lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [\[ICML 2025\] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](../../ICML2025/llm_safety/negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)

</div>

<!-- RELATED:END -->
