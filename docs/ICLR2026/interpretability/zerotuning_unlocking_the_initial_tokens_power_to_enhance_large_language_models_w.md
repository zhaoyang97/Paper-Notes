---
title: >-
  [论文解读] ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training
description: >-
  [可解释性] 提出 ZeroTuning，仅需对初始 token（如 <BOS）的注意力分数进行头部特异性缩放，即可在无训练情况下提升 LLM 在 15 个数据集上的表现，仅需修改 4 行代码。 核心问题 Token 级注意力调优（如 PASTA、ACT）虽有效，但依赖外部启发式：识别任务特定的"重要" token…
tags:
  - "可解释性"
---

# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

## 论文信息
- **会议**: ICLR 2026
- **arXiv**: [2505.11739](https://arxiv.org/abs/2505.11739)
- **代码**: [https://anonymous.4open.science/r/ZeroTuning](https://anonymous.4open.science/r/ZeroTuning)
- **领域**: 可解释性
- **关键词**: 注意力调优, 初始 token, attention sink, 零训练增强, 头部特异性

## 一句话总结
提出 ZeroTuning，仅需对初始 token（如 `<BOS>`）的注意力分数进行头部特异性缩放，即可在无训练情况下提升 LLM 在 15 个数据集上的表现，仅需修改 4 行代码。

## 研究背景与动机

### 核心问题
Token 级注意力调优（如 PASTA、ACT）虽有效，但依赖**外部启发式**识别任务特定的"重要" token，带来偏差和适用性限制。能否找到一个**通用、任务无关的控制点**？

### Attention Sink 现象
初始 token 倾向于成为 attention sink（注意力汇聚点），但其增强性能的潜力尚未被开发。

### 关键发现
- 调整初始 token 的注意力一致产生最大且最稳定的增益
- 增益方向取决于任务：分类任务需上调（$\gamma > 1$），QA 任务需下调（$\gamma < 1$）
- 不同注意力头对初始 token 缩放的响应异质

## 方法详解

### 整体框架

ZeroTuning 把初始 token（如 `<BOS>`）当作一个统一的、任务无关的注意力杠杆：先用一个缩放因子 $\gamma$ 对该 token 的注意力分数做头部特异性缩放，再 softmax 重归一化，整个过程不改任何权重、只动注意力分布，落地仅需 4 行代码。其底层逻辑是——初始 token 本就是 attention sink（注意力汇聚点），调它的权重能成比例地重塑其余 token 间的相对注意力差异，从而在不引入任何外部"重要 token"启发式的前提下校正模型偏置。

### 关键设计

论文把整个方法归纳为三步：头部行为分析（head behavior profiling）→ 选择性缩放（selective rescaling）→ 重归一化（renormalization），校准环节再分监督 / 无监督两种变体。下面四个设计点分别讲清这套机制怎么算、为什么动初始 token、在哪儿动、以及 $\gamma$ 怎么定。

**1. 初始 token 缩放算子：用一个因子撬动整张注意力分布**

给定一层一头的注意力权重，引入缩放因子 $\gamma > 0$ 只作用在初始 token 上：$a_0' = \frac{\gamma a_0}{D}$、$a_i' = \frac{a_i}{D}$，其中归一化项 $D = (\gamma-1)a_0 + 1$ 保证缩放后仍是合法的概率分布。这个算子的妙处在于它**只动初始 token、却保持其余 token 之间的相对比例不变**：$\gamma > 1$ 时放大初始 token、把剩余分布压平（更均匀），$\gamma < 1$ 时压缩初始 token、把剩余分布锐化（更聚焦）。于是一个标量就能在"平坦化"与"锐化"两个方向上连续调节注意力，无需逐 token 设计启发式。

**2. 杠杆效力分析：为什么动初始 token 增益最大**

它要回答的是设计 1 之外的一个根问题——为什么偏偏选初始 token 当那个杠杆。缩放带来的任意两 token 注意力差异的变化量可写成 $E_{\text{diff},i,j} = |a_i - a_j| \cdot \frac{|\gamma-1| a_0}{(\gamma-1) a_0 + 1}$。对初始 token 原始权重 $a_0$ 求导得 $\frac{\partial E_{\text{diff},i,j}}{\partial a_0} = |a_i - a_j| |\gamma-1| \cdot \frac{1}{((\gamma-1)a_0+1)^2} \geq 0$，恒为非负。这意味着初始 token 的注意力越大，它作为杠杆的调控效力越强——而 attention sink 恰恰让初始 token 天然占据极高权重，这就从理论上解释了"为什么单调初始 token 就能稳定拿到最大增益"，也回答了选它而非其他 token 作为通用控制点的动机。

**3. 层级与头部的选择性施加：只在该动的地方动**

缩放并非全模型一刀切，而是分层、分头有选择地施加，这正是论文三步里的"头部行为分析"环节。层级上，浅层（1–10）和中间层（11–21）的调整通常比深层（22–31）更有效，因为前者主要承担表示学习与知识整合、调控空间大，深层则已聚焦任务特定推理。头部上，不同注意力头对初始 token 缩放的响应是异质的：有的头放大初始 token 才提升性能（up-effective），有的头反而要缩小才提升（down-effective），这种分化源自预训练中各头的功能特化。ZeroTuning 因此先做一次头部行为分析评估每个头的敏感性，只对占主导的头类型施加 $\gamma$（消融显示调一个适中子集、约 40%–70% 的头最优），避免相反方向的头互相抵消。

**4. 两种校准模式：有标注和零标注都能定 $\gamma$**

确定缩放因子 $\gamma$ 有两条路。监督模式在标注验证集上直接搜索最大化准确率的 $\gamma$；无监督模式则改为**最小化输出熵**——论文观察到使熵最小的 $\gamma$ 与使准确率最大的 $\gamma$ 强相关，因此在完全无标注时也能选出近优的缩放强度，且无标注输入既可取自同域离线语料、也可直接用测试批做 test-time 搜索。两种模式定好 $\gamma$ 后，最后都要用 softmax 把缩放过的分数重归一化以保持分布有效（即三步里的"重归一化"）；在无法直接改注意力分数的高效内核里，ZeroTuning 改为等价地缩放 query/key states，从而兼容 SDPA 与 FlashAttention。

## 实验

### 分类任务

| 模型 | Vanilla | ACT | Auto-PASTA | **ZeroTuning** |
|------|---------|-----|------------|---------------|
| Llama-3.1-8B Avg | 59.59 | 60.11 | 63.73 | **71.44** |
| Qwen-2-7B Avg | 55.10 | - | 65.57 | **68.19** |
| Deepseek-R1-14B Avg | 67.67 | - | 69.04 | **71.87** |

最大单数据集提升：SST-2 上 73.20 → 91.60（+18.4%），SUBJ 上 44.60 → 66.60（+22.0%）。

### 多选 QA 任务

| 模型 | Vanilla | Auto-PASTA | **ZeroTuning** |
|------|---------|------------|---------------|
| Llama-3.1-8B Avg | 58.84 | 60.18 | **61.48** |
| Qwen-2-7B Avg | 63.10 | 64.01 | **64.84** |
| Deepseek-R1-14B Avg | 60.05 | 60.31 | **62.20** |

LogiQA 上 Deepseek-R1-14B: 27.80 → 35.60（+7.80%）。

### MT-Bench 对话

| 模型 | Vanilla | ZeroTuning |
|------|---------|-----------|
| Llama-3.1-8B | 7.804 | **7.966** |
| Llama-2-13B | 6.650 | **6.916** |

### 关键发现

1. **仅调整单个 token 即一致超越多 token 调优方法**
2. **准确率与输出熵的强逆相关性**：验证无监督模式的可行性
3. **头部特异性调优远优于均匀调优**
4. **对量化推理、长上下文、few-shot 设置均保持稳健**
5. **仅需 4 行代码修改**

## 亮点

1. **极简主义设计**：一个 token、一个缩放因子、4 行代码
2. **理论分析深入**：从注意力重塑到偏差校正的完整推导
3. **无监督模式**：基于熵最小化，无需标注数据
4. **内核无关**：兼容 SDPA 和 FlashAttention
5. **跨模型跨任务一致有效**

## 局限性

1. 最优缩放方向依赖于任务（分类 vs QA），需要初步实验或启发式判断
2. 头部行为分析仍需一定计算成本
3. 提升幅度在已经很强的大模型上（如 Deepseek-R1-14B）相对有限
4. Up+Down 混合策略未超越单一策略，联合优化尚需探索
5. 对生成任务（开放式对话）的提升有限

## 相关工作
- **注意力调优**: PASTA, Auto-PASTA, ACT — 需要识别重要 token
- **Attention Sink 研究**: StreamingLLM, Barbero et al. — 解释现象但未利用
- **推理时优化**: 自一致性, CoT — 提示工程方向

## 评分
- **创新性**: ⭐⭐⭐⭐ — 极简但有效的创意，将 attention sink 从被动观察转为主动利用
- **实验充分性**: ⭐⭐⭐⭐⭐ — 15 个数据集、4 个模型、多维度分析
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑递进，由理论到方法到实验的完整故事
- **实用性**: ⭐⭐⭐⭐⭐ — 4 行代码即可部署，无训练无额外内存

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICML 2025\] DeltaSHAP: Explaining Prediction Evolutions in Online Patient Monitoring with Shapley Values](../../ICML2025/interpretability/deltashap_explaining_prediction_evolutions_in_online_patient_monitoring_with_sha.md)
- [\[ACL 2025\] Enhancing Automated Interpretability with Output-Centric Feature Descriptions](../../ACL2025/interpretability/output_centric_interpretability.md)
- [\[ACL 2025\] A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](../../ACL2025/interpretability/a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)
- [\[ICLR 2026\] On the Predictive Power of Representation Dispersion in Language Models](on_the_predictive_power_of_representation_dispersion_in_language_models.md)

</div>

<!-- RELATED:END -->
