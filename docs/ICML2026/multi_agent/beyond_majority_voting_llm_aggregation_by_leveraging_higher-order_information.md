---
title: >-
  [论文解读] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information
description: >-
  [ICML 2026][多智能体][LLM聚合] 本文提出两种利用高阶信息的 LLM 回答聚合算法——基于一阶准确率信息的 Optimal Weight (OW) 和基于二阶相关性信息的 Inverse Surprising Popularity (ISP)，在不需要标签的条件下证明性优于多数投票，并在 UltraFeedback、MMLU 和医疗健康数据集上验证了一致的提升。
tags:
  - "ICML 2026"
  - "多智能体"
  - "LLM聚合"
  - "多智能体推理"
  - "信息聚合"
  - "贝叶斯最优"
  - "无监督标注"
---

# Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information

**会议**: ICML 2026  
**arXiv**: [2510.01499](https://arxiv.org/abs/2510.01499)  
**代码**: 无  
**领域**: 多智能体  
**关键词**: LLM聚合, 多智能体推理, 信息聚合, 贝叶斯最优, 无监督标注  

## 一句话总结

本文提出两种利用高阶信息的 LLM 回答聚合算法——基于一阶准确率信息的 Optimal Weight (OW) 和基于二阶相关性信息的 Inverse Surprising Popularity (ISP)，在不需要标签的条件下证明性优于多数投票，并在 UltraFeedback、MMLU 和医疗健康数据集上验证了一致的提升。

## 研究背景与动机

**领域现状**：多智能体 LLM 推理（如 LLM debate、LLM council）已广泛使用，在聚合多个模型回答时，绝大多数工作直接采用多数投票 (Majority Voting, MV) 作为标准聚合策略。

**现有痛点**：MV 是一种"零阶"方法，仅依赖原始回答的频次，完全忽略了不同 LLM 之间的能力异质性（有的模型准确率 90%、有的只有 60%）和回答之间的相关性。这意味着一个弱模型和一个强模型在投票中拥有相同权重，且当多个弱模型犯相同错误时容易误导最终结果。

**核心矛盾**：要利用模型准确率（一阶信息）来加权需要大量带标签数据来估计准确率，但在自动标注、预测市场等无监督场景中根本没有标签；而经典的 Surprisingly Popular (SP) 方法虽然利用了二阶相关性信息且不需要标签，但在 LLM 场景下反而不如 MV——因为 LLM 不像人类群体那样存在系统性偏差，SP 所利用的信号反而起了反效果。

**本文目标**：设计不依赖标签、利用高阶信息的聚合算法，在理论上可证地优于 MV。

**切入角度**：作者观察到随机打乱选项顺序后，联合分布具备对称结构（所有错误选项等概率），这种结构允许推导出闭式最优解。同时 SP 失败的原因在于其"放大偏差"的方向恰好与 LLM 场景相反，反转 SP 的计算方向即可纠正。

**核心 idea**：用准确率的逆 sigmoid 函数作为最优权重实现贝叶斯最优聚合；在没有标签时，将 SP 的预测方向反转（从"反事实"视角计算分数）来利用二阶信息超越 MV。

## 方法详解

### 整体框架

这篇论文要解决的是：当 $N$ 个 LLM 对一道 $K$ 选项的选择题各自给出回答 $a_1, \ldots, a_N$ 后，怎么把它们聚合成一个比多数投票更准的答案。整套方法先做一步预处理——对每个问题随机打乱选项顺序，让正确答案之外的所有错误选项在统计上等概率出现，从而让回答的联合分布具备一种对称结构，正是这个结构让后面的最优解能写成闭式。聚合时按手头有什么信息分三档走：知道每个模型准确率就用 OW 做贝叶斯最优加权；一个标签都没有就用只依赖回答间相关性的 ISP，或者先用 ISP/拟合估出准确率再回到 OW（即 OW-I / OW-L）。最后把选出的答案按逆置换映射回原始选项顺序输出。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：N 个 LLM 对 K 选项题的回答 a_1 … a_N"]
    A --> B["预处理：随机打乱选项<br/>使错误选项等概率、联合分布对称"]
    B -->|已知各模型准确率| OW["Optimal Weight（OW）<br/>逆 sigmoid 加权 ω_i = σ_K⁻¹(x_i)<br/>贝叶斯最优"]
    B -->|无标签·二阶相关性| ISP["Inverse Surprising Popularity（ISP）<br/>反转 SP 方向，用反事实条件概率打分"]
    B -->|无标签·先估准确率| EST["OW-L / OW-I<br/>OW-L 拟合二阶概率反解 x_i<br/>OW-I 拿 ISP 结果当伪标签估 x_i"]
    EST -->|代回 σ_K⁻¹(x_i)| OW
    OW --> OUT["逆置换映射回原始选项<br/>输出聚合答案"]
    ISP --> OUT
```

### 关键设计

**1. Optimal Weight（OW）：在已知准确率时把加权投票做到贝叶斯最优**

多数投票最大的毛病是给所有模型同样的话语权，一个 90% 准确率的强模型和一个 60% 的弱模型投票等权，几个弱模型一起犯错就能带偏结果。OW 的做法是给第 $i$ 个 agent 赋予权重 $\omega_i = \sigma_K^{-1}(x_i)$——其中 $x_i$ 是它的准确率，$\sigma_K(x) = e^x / (K-1+e^x)$ 是把 logistic 推广到 $K$ 类的广义 sigmoid——然后按 $f_{OW} = \arg\max_s \sum_i \sigma_K^{-1}(x_i) \cdot \mathbb{1}\{a_i = s\}$ 选票数最高的选项。关键不在"准确率越高权重越大"这个直觉，而在于：在随机打乱引出的对称信息结构下，这组逆 sigmoid 权重恰好是后验概率最大化的充要解，是所有可能聚合器（不限于线性加权）里的贝叶斯最优，而非众多加权方案中碰巧不错的一个。两个推论也随之而来：$K=2$ 时它退化成 logistic 的逆，与 Bradley-Terry 模型对上号；当所有 agent 同质（准确率相同）时它又退化回 MV——这说明 MV 只有在 self-consistency 这种"同一模型多次采样"的特例下才是最优的。

**2. Inverse Surprising Popularity（ISP）：没有标签时靠回答间的相关性反超多数投票**

OW 虽好但需要准确率，而自动标注、预测市场这类场景根本没有标签。经典的 Surprisingly Popular（SP）方法本来能在无标签下利用二阶相关性信息 $\mathbb{P}(A_i|A_j)$，可在 LLM 上它反而不如 MV——因为 SP 赖以工作的前提是人群有"系统性低估正确答案"的偏差，而 LLM 群体没有这种偏差，SP 放大的信号方向就反了。ISP 的修法是把 SP 的预测方向整个反转：SP 算的是每个 agent "预测别人会怎么回答"，ISP 算的是"假如别的 agent 给了一个反事实的回答，我会怎么预测"，分数写成 $S_{ISP}(s,i) = \frac{1}{N-1}\sum_{j \neq i} \frac{1}{K-1}\sum_{a \neq a_j} \mathbb{P}(A_i=s|A_j=a)$，再用优势函数 $Adv_{ISP}(s) = \sum_i \mathbb{1}\{a_i=s\} - \sum_i S_{ISP}(s,i)$ 取最大的选项。用反事实条件概率代替真实条件概率，等于让预测分数偏向错误选项，反过来就把正确答案的优势放大了。论文给出了链式不等式 $\mathbb{E}[Adv_{ISP}(s^*)] \geq \mathbb{E}[Adv_{MV}(s^*)] \geq \mathbb{E}[Adv_{SP}(s^*)]$，正好解释了为什么 ISP 强于 MV、而 MV 又强于原版 SP。

**3. OW-L / OW-I：用二阶信息把准确率估出来，桥接回一阶最优框架**

ISP 能无标签工作但只用到了二阶信息，而真正的贝叶斯最优 OW 还是需要准确率。这两个变体就是在没有标签时把缺的准确率补出来，再回到 OW：OW-L 通过最小化经验条件概率与理论条件概率之间的均方误差，反解出各 agent 的准确率 $\hat{x}_1, \ldots, \hat{x}_N$；OW-I 则更直接，拿 ISP 的聚合结果当伪标签，统计每个 agent 与伪标签的一致率作为准确率估计。两者估出 $\hat{x}_i$ 后都代回权重公式 $\sigma_K^{-1}(\hat{x}_i)$ 做聚合。这样就把"无标签可用的二阶信息"转化成了"OW 需要的一阶最优权重"，实验里两个变体在真实数据上表现几乎一致，且都比直接用 ISP 更好。

## 实验关键数据

### 主实验（模拟数据）

| 方法 | $K=2$ | $K=4$ | $K=6$ | $K=8$ | $K=10$ |
|------|-------|-------|-------|-------|--------|
| MV | 85.13% | 92.64% | 94.22% | 94.85% | 95.54% |
| SP | 79.94% | 90.52% | 92.68% | 93.66% | 94.40% |
| Single Best | 90.34% | 89.94% | 90.31% | 89.95% | 90.05% |
| **ISP (本文)** | **90.48%** | **94.45%** | **95.78%** | **96.23%** | **96.49%** |
| OPT (clairvoyant) | 91.37% | 94.94% | 96.05% | 96.46% | 96.81% |

### 真实数据集实验（4 个强模型）

| 方法 | UltraFeedback | MMLU | ARMMAN |
|------|--------------|------|--------|
| MV | 72.21% | 89.32% | 85.24% |
| ISP | 73.26% | 90.01% | 85.78% |
| **OW-L** | **73.66%** | **90.37%** | **85.78%** |
| **OW-I** | **73.66%** | **90.37%** | **85.78%** |
| Single Best (oracle) | 73.14% | 91.02% | 85.32% |

### 关键发现

- ISP 在所有 $K$ 值下均优于 MV，且两者差距随 $K$ 增大而缩小（$\Theta(1/K)$），与理论预测一致
- 在 16 种模型组合中，OW-L 在 97.92% 的情况下优于 MV，绝对提升最高达 14.20%；MV 在所有组合中从未取得最佳
- 假设检验 t-statistic 分别为 12.53（UltraFeedback）、23.39（MMLU）和 3.22（ARMMAN），p-value 均 < 0.001，提升在统计上显著
- 在"强干扰项"子集（MMLU-hard，至少两个模型选了相同错误选项）上，OW-L/OW-I 比 MV 提升超过 7%（17.23% → 24.79%），说明高阶信息在困难场景下更有价值

## 亮点与洞察

- **逆 sigmoid 权重的贝叶斯最优性**：看似简单的加权方案 $\omega_i = \sigma_K^{-1}(x_i)$ 实际上是所有可能聚合器中的最优解（不限于线性），这为 Bradley-Terry 模型在 RLHF 中的使用提供了理论背书。这个结论非常优雅且可直接应用
- **反转 SP 的反直觉设计**：经典 SP 在人类群体中有效但在 LLM 中失败，作者深入分析了原因（LLM 缺少人类的系统性偏差），然后将方向反转得到 ISP，这种"诊断失败原因→针对性修改"的思路值得借鉴
- **二阶信息桥接一阶最优**：OW-L/OW-I 将"无标签可用的二阶信息"转化为"需要标签的一阶最优权重"，这种间接利用信息层级的思路可以迁移到其他无监督聚合场景

## 局限与展望

- 理论分析依赖条件独立假设（给定正确答案后各 LLM 独立），虽然实验表明在违反假设时仍有效，但缺乏正式的鲁棒性界
- 所有模型对同一问题使用相同的全局权重，未考虑不同问题类型上模型能力的差异（如某模型擅长数学但不擅长语言理解），prompt-specific 权重是明确的改进方向
- 仅处理封闭选择题（$K$ 个选项），对开放式生成任务的扩展尚不明确
- 位置偏差的理论假设（LLM 不受选项顺序影响）在弱模型上不完全成立，虽然实验中不需要去偏就有效，但对弱模型的理论保证有待加强

## 相关工作与启发

- **Surprising Popularity (Prelec et al., 2017)**：经典的基于二阶信息的聚合方法，但作者证明在 LLM 场景下 SP 劣于 MV，ISP 是对 SP 的针对性改进
- **Bradley-Terry 模型**：Corollary 3.2 建立了 OW 与 BT 模型的联系，为 RLHF 中 BT 模型的有效性提供了理论支持
- **Self-Consistency (Wang et al., 2022)**：Corollary 3.3 证明了当 agent 同质时 MV 即最优，即 self-consistency 场景下无需更复杂的聚合
- **启发**：该框架可直接作为多 LLM 系统中 MV 的 drop-in 替代品，且计算开销仅为 CPU 级别的几秒钟，适用于 API 调用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs](systematic_failures_in_collective_reasoning_under_distributed_information_in_mul.md)
- [\[NeurIPS 2025\] Adaptive Coopetition: Leveraging Coarse Verifier Signals for Resilient Multi-Agent LLM Reasoning](../../NeurIPS2025/multi_agent/adaptive_coopetition_leveraging_coarse_verifier_signals_for_resilient_multi-agen.md)
- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ICML 2026\] Voting Protocols as Coordination Mechanisms for Role-Constrained Multi-Agent Tutoring Systems](voting_protocols_as_coordination_mechanisms_for_role-constrained_multi-agent_tut.md)
- [\[AAAI 2026\] Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion](../../AAAI2026/multi_agent/beyond_detection_exploring_evidence-based_multi-agent_debate_for_misinformation_.md)

</div>

<!-- RELATED:END -->
