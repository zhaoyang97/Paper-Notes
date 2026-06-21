---
title: >-
  [论文解读] Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection
description: >-
  [ICLR 2026][LLM对齐][LLM 可信度] 针对 SFT 导致 LLM 可信度下降的问题，提出两阶段修复框架：先用 DPP 正则化子集选择定位"有害训练样本"，再用 PBRF 梯度上升修复模型，以 ≤1% 困惑度代价换取最高 21% 的可信度提升。 领域现状：大语言模型在下游任务上做监督微调（SFT）已成标准流程…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "LLM 可信度"
  - "SFT 修复"
  - "数据归因"
  - "DPP 子集选择"
  - "Proximal Bregman Response Function"
---

# Enhancing Trustworthiness of Fine-Tuned LLMs via Regularized Subset Selection

**会议**: ICLR 2026  
**代码**: [kyrs/tracing-llm-trust](https://github.com/kyrs/tracing-llm-trust)  
**领域**: LLM 对齐 / 模型修复  
**关键词**: LLM 可信度、SFT 修复、数据归因、DPP 子集选择、Proximal Bregman Response Function

## 一句话总结

针对 SFT 导致 LLM 可信度下降的问题，提出两阶段修复框架：先用 DPP 正则化子集选择定位"有害训练样本"，再用 PBRF 梯度上升修复模型，以 ≤1% 困惑度代价换取最高 21% 的可信度提升。

## 研究背景与动机

**领域现状**：大语言模型在下游任务上做监督微调（SFT）已成标准流程，但多项研究发现，即使在良性数据集上微调，模型的可信度指标（真实性、刻板偏见、机器伦理）也会随之下降。  
**现有痛点**：现有修复手段主要依赖 RLHF/DPO（需要大量标注偏好数据、成本高）或过滤器（可被绕过）；而直接重新训练耗时数小时且无法保证不进一步损害性能。  
**核心矛盾**：SFT 带来的困惑度提升（下游收益）与可信度退化之间存在内在张力——如何在修复可信度的同时不牺牲模型在原有任务上的性能？  
**本文目标**：在固定计算预算下，对已完成 SFT 的模型进行后验修复，同时提升三项可信度维度并保持困惑度几乎不变。  
**核心 idea**：把可信度退化问题转化为"数据归因 + 有针对性梯度上升"的优化问题：先找到导致退化的那少数训练样本，再对这些样本施加 Bregman 散度约束下的梯度上升，从而"取消"它们的影响。

## 方法详解

### 整体框架

方法分两个阶段：（1）**样本定位**——用影响函数（EK-FAC 近似 IHVP）在训练集上打分，再用 DPP 选出一个小而多样的有害子集 $S$；（2）**模型修复**——在 PBRF 目标下对子集 $S$ 执行梯度上升，同时通过 Gauss-Newton Hessian 约束保持原有下游性能。

```mermaid
flowchart LR
    A[Post-SFT 模型 θ_post] --> B[影响函数打分\nEK-FAC 近似 IHVP]
    B --> C[DPP 正则化子集选择\n最大化多样性+归因得分]
    C --> D[有害子集 S]
    D --> E[PBRF 梯度上升\n保持困惑度≤ε]
    E --> F[修复后模型 θ*]
```

### 关键设计

**1. 基于对数概率的可信度度量**  
为了让可信度指标可微分，论文将每项度量 $F_j$ 定义为"对立回应"与"正面回应"的条件对数似然之差：

$$F_j(\theta) = \mathbb{E}_{(m,p,o)\sim P^j_\text{trust}}\left[\log P_\theta(o \mid m) - \log P_\theta(p \mid m)\right]$$

其中 $p$ 为正向（可信）回应，$o$ 为反向（不可信）回应。该定义与 Bradley-Terry 模型等价：$F_j<0$ 说明模型对正向回应赋予更高概率，即更可信。这一设计将三个异质性指标（真实性、偏见、伦理）统一到同一可微框架，使后续梯度运算成为可能。

**2. PBRF 约束下的梯度上升修复**  
直接在训练样本上做梯度上升（SGA/GA）会破坏原有下游性能。PBRF（Proximal Bregman Response Function）通过在**函数空间**和**参数空间**双重约束来稳定更新：

$$\theta(β; S) = \arg\min_\theta \frac{1}{|N|}\sum_{i}\Psi(M(x_i,\theta), M(x_i,\theta_\text{post}); y_i) - \beta\sum_{(x,y)\in S} L(M(x,\theta),y) + \frac{\lambda}{2}\|\theta-\theta_\text{post}\|^2$$

其中 $\Psi$ 为预测空间的 Bregman 散度，约束更新后模型的函数行为不偏离 $\theta_\text{post}$。在小 $\beta$ 近似下，更新规则退化为一次带 Gauss-Newton 预处理的梯度步，可用 EK-FAC 高效近似逆 Hessian 向量积（IHVP）。

**3. DPP 正则化子集选择**  
Proposition 1 证明：对某训练样本施加梯度上升后，邻近的"有用样本"也会受到牵连（loss 传播）。因此子集 $S$ 应**小且多样**。论文引入 DPP 来最大化子集多样性：

$$S_j = \arg\max_{S, |S|\le\rho}\;\log\det(K_S + I) + \eta\cdot\log\sum_{v\in D^j_\text{trust}}\gamma_j(v, S)$$

第一项 $\log\det$ 惩罚冗余（相关样本的行列式趋近于零），第二项最大化对可信度指标的总归因贡献 $\gamma_j$。该目标是两个次模函数之和，可用贪心算法在多项式时间内求近优解。

**4. 跨指标统一子集**  
上述每个指标 $j$ 各选一个子集。论文还在附录中提出选一个**公共子集**同时修复所有可信度维度的变体，并发现公共子集在某些模型上（如 Qwen2.5-7B 真实性 +21.73%）表现更好，但对于指标间存在冲突的样本会互相干扰。

## 实验关键数据

### 主实验

以刻板偏见（Stereotypical Bias）为例，代表性结果：

| 模型 | 指标 | Post-SFT | Ours | Δ (相对) |
|------|------|----------|------|---------|
| Pythia-1.4B | Log-Odds↓ | −0.484 | −0.549 | +13.4% |
| Pythia-6.9B | Log-Odds↓ | −0.380 | −0.449 | +18.2% |
| Qwen2.5-7B | Log-Odds↓ | −0.691 | −0.780 | +12.9% |
| Pythia-1.4B | PPL↓ | 6.016 | 6.065 | −0.8% |
| Qwen2.5-7B | PPL↓ | 5.401 | 5.408 | −0.1% |

真实性最优改善：Qwen2.5-7B +9.6%；机器伦理最优：Qwen2.5-7B +8.7%；困惑度降幅全部 ≤2.4%。

### 消融实验

| 配置 | Log-Odds 改善 | PPL 变化 | 说明 |
|------|--------------|---------|------|
| SGA（随机梯度上升） | 低/发散 | 显著增大 | 无约束，不稳定 |
| GA（批梯度上升） | 低/发散 | 显著增大 | 同上 |
| GA+KL | 边际提升 | 稳定 | 仅机器伦理接近本文 |
| **Ours (PBRF+DPP)** | **最优** | **≤2%** | 一致优于所有 baseline |

### 关键发现

- PBRF 约束显著优于 SGA/GA，说明函数空间的近邻约束对保持下游性能至关重要
- DPP 正则化在高学习率场景下尤其能稳定优化，防止大子集导致的灾难性干扰
- 计算效率：修复 10 个样本仅需 10.96 秒，而重训练需 6+ 小时（对 Pythia-1.4B）
- 与 DPO 相比，本方法在小模型上困惑度保留更好，且无需构造偏好数据集

## 亮点与洞察

- **数据归因 + 模型修复的统一框架**：把"哪些训练样本有害"和"如何消除其影响"两个问题在 PBRF 框架下优雅地结合，理论推导扎实（Proposition 1 给出 loss 传播的量化上界）
- **DPP 作为修复稳定器**：将 DPP 用于选"要反训练的样本"而非通常的"要保留的样本"，视角新颖，解释了为何多样性对梯度上升至关重要
- **无需额外标注数据**：不依赖人类偏好标注，只需已有的可信度评估数据集（TruthfulQA、DecodingTrust 等），部署门槛低
- **可扩展性**：EK-FAC 近似使方法可扩展到 7B 参数量级，且修复开销远小于全量重训练

## 局限与展望

- 方法依赖可信度评估数据集的质量；若评估集存在分布偏移，归因可能失准
- 仅验证到 7B 量级，对 70B+ 模型的 EK-FAC 近似质量尚不明确
- 公共子集修复在部分指标间存在"此消彼长"现象，跨维度协同优化仍有空间
- PBRF 的 $\beta$ 超参数选取目前依赖经验，缺乏自动化调优机制

## 相关工作与启发

- **vs 影响函数（Koh & Liang 2017）**：本文扩展影响函数到 LLM 的可信度修复，而非传统的数据清洗；EK-FAC 解决了 LLM 规模下 IHVP 的可扩展性瓶颈
- **vs RLHF/DPO**：RLHF 需要偏好数据集且修复成本高；本文方法只需识别有害样本并执行少量梯度步，无需重新收集数据
- **vs 机器遗忘（Machine Unlearning）**：方法与遗忘同属"消除特定样本影响"的范式，但专注于可信度修复而非隐私保护，PBRF 约束是其区别于标准遗忘的核心
- **启发**：DPP + 数据归因的组合思路可迁移到其他需要"少量样本 targeted intervention"的场景（如安全微调、幻觉修复）

## 评分

- 新颖性: ⭐⭐⭐⭐ DPP 正则化子集选择与 PBRF 修复的组合在可信度修复场景下是新颖的，但各组件本身已有先例
- 实验充分度: ⭐⭐⭐⭐ 覆盖 6 个模型×3 个指标，消融实验完整，计算效率有量化对比，附录内容丰富
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，Proposition 1 给出严格证明，图表简洁
- 价值: ⭐⭐⭐⭐ 在"无需重训练、无需偏好数据"的约束下大幅提升可信度，工程实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Shelf Life of Fine-Tuned LLM-Judges: Future-Proofing, Backward-Compatibility, and Question Generalization](on_the_shelf_life_of_fine-tuned_llm-judges_future-proofing_backward-compatibilit.md)
- [\[ICLR 2026\] Data Selection for LLM Alignment Using Fine-Grained Preferences](data_selection_for_llm_alignment_using_fine-grained_preferences.md)
- [\[AAAI 2026\] Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal States](../../AAAI2026/llm_alignment/enhancing_uncertainty_estimation_in_llms_with_expectation_of_aggregated_internal.md)
- [\[ICLR 2026\] Anchored Supervised Fine-Tuning](anchored_supervised_fine-tuning.md)
- [\[ICML 2026\] SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](../../ICML2026/llm_alignment/spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)

</div>

<!-- RELATED:END -->
