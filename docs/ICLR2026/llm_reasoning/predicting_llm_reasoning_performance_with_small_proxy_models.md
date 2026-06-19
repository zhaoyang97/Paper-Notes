---
title: >-
  [论文解读] Predicting LLM Reasoning Performance with Small Proxy Models
description: >-
  [ICLR 2026][Reasoning][小模型代理] 提出 rBridge 方法，通过结合前沿模型推理轨迹 (reasoning trace) 的 NLL 评估与 token 级任务对齐权重，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，数据排序计算成本降低 100 倍以上。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "小模型代理"
  - "推理能力预测"
  - "预训练数据"
  - "scaling law"
  - "NLL"
---

# Predicting LLM Reasoning Performance with Small Proxy Models

**会议**: ICLR 2026  
**arXiv**: [2509.21013](https://arxiv.org/abs/2509.21013)  
**代码**: 未开源  
**领域**: LLM/NLP  
**关键词**: 小模型代理, 推理能力预测, 预训练数据, scaling law, NLL

## 一句话总结

提出 rBridge 方法，通过结合前沿模型推理轨迹 (reasoning trace) 的 NLL 评估与 token 级任务对齐权重，使 ≤1B 的小模型能有效预测 13B-32B 大模型的推理性能，数据排序计算成本降低 100 倍以上。

## 研究背景与动机

### 预训练中的代理模型需求

预训练大语言模型的成本极高（7B 模型 500B tokens 训练成本超 5 万美元），因此用小模型作为代理 (proxy) 来优化数据集和预测大模型表现是关键研究方向。现有方法（scaling law、数据排名不变性）在非推理任务上有效，但**推理能力的涌现特性**使小模型代理面临根本挑战。

### 核心挑战：推理的涌现性

推理能力展现出明显的涌现行为——仅在 7B+ 模型中可靠出现：

- 1B 模型在 MATH500 上噪声大且斜率方向错误（$R^2$ 极低）
- 相比之下，TriviaQA、HellaSwag 等非推理任务在小规模即有平滑改善
- 这使得从业者被迫使用 7-15B 的 "代理模型"，成本高昂

### 问题定义

设 $\pi^{\text{p}}, \pi^{\text{t}}$ 分别为小代理和大目标模型。目标是设计代理评估指标 $\text{metric}^{\text{p}}$ 使得：

$$\text{metric}^{\text{p}} := \max_{\text{metric}} \text{corr}(\text{metric}(\pi^{\text{p}}), \text{metric}^{\text{t}}(\pi^{\text{t}}))$$

即小模型指标的改善应可靠地预测大模型指标的改善。

## 方法详解

### 整体框架

rBridge 把"小模型预测大模型推理"的失败归因到两条对齐轴上：评估目标没和预训练目标对齐，目标任务没和被评 token 对齐。它不训练任何新模型，只是把标准的 token 级 NLL 改造成一个新评估指标——用前沿模型的推理轨迹当 gold label，再用前沿模型对每个 token 的置信度作权重，使小代理模型的得分曲线能跟上大模型推理性能的走势。

### 关键设计

**1. 用前沿模型推理轨迹 $R^\phi$ 作 gold label：让 NLL 信号落回小模型的分布内**

第一处错位是评估目标不对齐：Accuracy / Pass@K 与预训练的 next-token 目标不连续，1B 模型上噪声极大；而 NLL 看似连续，却并非所有 NLL 都等价——若 gold label $Y^*$ 对小模型是 OOD（分布外）的，NLL 信号同样会被噪声淹没。问题恰恰出在标签上：ScalingBench 之类的 $Y^*$ 把推理轨迹和最终答案一起当标签，夹带 "\n"、"Final Answer:"、"I hope it is correct." 等格式 token，对小预训练模型完全是分布外的文本。rBridge 的做法是丢掉最终答案的格式部分，只取前沿模型 $\pi^\phi$ 生成的推理轨迹 $R^\phi$ 当 gold label。

这样改的依据是：$R^\phi$ 本身是一段流畅的连续自然语言，更贴近以长文本为主的预训练分布（ID），论文用 $-\log p(Y^*)$ 度量 ID 程度，发现换成 $R^\phi$ 后五个推理基准上平均 NLL 下降 74.7%，信号噪声被压下来；同时 $R^\phi$ 又是一条通往正确答案的推理链，天然与推理任务对齐。一个标签替换就同时堵住「预训练目标对齐」和「任务对齐」两条错位轴的源头。

**2. token 级任务对齐权重：把"哪些 token 重要"交给前沿模型自动判断**

光替换标签还不够——标准 NLL 对 $R^\phi$ 里所有 token 一视同仁，分不清像 "sum modulo 9" 这种决定推理对错的任务关键 token 和换行、编号这类格式 token，后者会稀释信号。rBridge 进一步用前沿模型对每个 token 的置信度 $p^\phi(\text{token}_i)$ 作为这个 token 任务重要性的自动代理，把它乘进 NLL：

$$\text{rBridge NLL}(\text{token}_i) := \underbrace{-\log\big(p^{\text{p}}(\text{token}_i)\big)}_{\text{标准 NLL}} \cdot \underbrace{\frac{1}{|\text{token}_i|} \sum_{\text{letter} \in \text{token}_i} p^\phi(\text{letter})}_{\text{tokenizer 无关的任务对齐权重}}$$

前沿模型越确信的 token 往往越是推理骨架，权重也越高。为了让代理模型和前沿模型用不同 tokenizer 时权重仍可比，权重不在 token 级直接算，而是先在字母（letter）级取出 $p^\phi$、再在每个 token 内取平均，最后对整条权重因子做 MinMax 归一化以放大区分度（对应 Figure 1 中展开的完整形式）。这样小代理模型在任务关键 token 上的 NLL 改善被放大、格式 token 的贡献被压低，加权求和得到的 rBridge score 自然更贴合大模型的真实推理表现。

### 损失函数 / 训练策略

rBridge 不引入任何训练，整套流程就是一次前向打分：先用前沿模型对每道题生成推理轨迹 $R^\phi$，再让待评的小代理模型在 $R^\phi$ 上逐 token 算 NLL，同时取前沿模型的 token 概率算出任务对齐权重，最后按上式加权求和得到 rBridge score。因为只需小模型跑一遍前向，数据排名的计算开销相比直接训练大代理模型可省下两到三个数量级。

## 实验关键数据

### 主实验 1：数据集排名（<100M → 1.2B）

在 DataDecide 协议下，用代理模型排列 25 个预训练数据集：

| 方法 | 最佳 DAcc. | 计算节省 |
|------|-----------|---------|
| CF Accuracy（最大代理） | ~基线 | 1× |
| Norm Correct Prob | ~50%（随机水平） | - |
| **rBridge (3.7M 模型)** | 比基线高 27% | **100.2×-733.4×** |

rBridge 在最小代理规模（3.7M，训练 87.3M tokens）即达到远超基线的排名准确率。

### 主实验 2：代理-目标关系（1B → 13B/32B）

6 个推理基准上的 5-fold 交叉验证结果：

| 方法 | 1B→13B Avg Train $R^2$ ↑ | 1B→13B Avg Test MAE ↓ | 1B→32B Avg $R^2$ ↑ |
|------|--------------------------|----------------------|---------------------|
| Acc./p@1 | 0.304 | 3.709 | 0.312 |
| iSFT | 0.290 | 5.123 | 0.349 |
| TED | 0.375 | 3.377 | 0.352 |
| MPCA | 0.194 | 302.642 | 0.205 |
| NLL | 0.485 | 5.173 | 0.488 |
| $R^\phi$ | 0.867 | 1.455 | 0.820 |
| **rBridge** | **0.874** | **1.384** | **0.826** |

rBridge 在 10/12 个指标上取得最优，平均 $R^2$ 从基线 0.304 提升到 0.874。

### 消融实验

| 设置 | 1B→13B $R^2$ | 1B→13B+SFT $R^2$ | 1B→32B $R^2$ |
|------|-------------|-------------------|-------------|
| 标准 NLL | 0.485 | 0.413 | 0.488 |
| NLL on $R^\phi$ | 0.867 | 0.820 | 0.820 |
| **rBridge (+ 权重)** | **0.874** | **0.846** | **0.826** |

每个组件（推理轨迹、权重、归一化）都带来一致提升。

### 零样本迁移（1B → 7B 跨数据集）

在一个数据集上拟合的 rBridge-目标函数可零样本迁移到另一数据集：

| 方法 | 数据集排名正确率 | 平均 MAE |
|------|----------------|---------|
| $R^\phi$ | 4/5 | 3.425 |
| **rBridge** | **5/5** | **2.490** |

### 关键发现

1. **rBridge 性能超越 7-13 倍大的代理模型**：1B rBridge 优于 7B/13B 模型直接用 Acc. 评估
2. **计算节省巨大**：数据排名任务中节省 100-733 倍 FLOPs
3. **跨数据集零样本迁移有效**：一次拟合可复用
4. **非连续指标（Acc./p@k、iSFT）表现最差**：验证了评估目标对齐的重要性

## 亮点与洞察

1. **深刻的方法论洞察**：问题不在于小模型 "不够大"，而在于评估方式与预训练目标和任务不对齐
2. **优雅的解决方案**：不训练新模型，仅改变评估指标即可达到数量级的改善
3. **前沿模型概率的巧妙使用**：将前沿模型的 token 概率作为任务重要性的自动度量
4. **处理 tokenizer 不匹配**：字母级概率平均的设计简洁实用
5. **理论动机清晰**：从预训练目标对齐和 ID/OOD 分析出发，每一步设计都有理论支撑

## 局限性

1. 需要前沿模型生成推理轨迹，虽然成本低（每基准 <$10）但引入了对前沿模型的依赖
2. 实验中最大目标规模为 32B，更大模型（70B+）是否成立未验证
3. 零样本迁移仅在 2 个数据集间验证，泛化性有待进一步确认
4. 方法专注推理任务，对非推理任务（知识、理解）的应用未探索

## 相关工作与启发

- **Scaling Laws (Kaplan et al. 2020)**：rBridge 可视为推理领域的改进版 scaling law
- **ScalingBench**：rBridge 是 ScalingBench 的改进版，关键改进是去掉答案格式只用推理轨迹
- **DataDecide (Magnusson et al. 2025)**：rBridge 在其协议上验证了数据排名能力
- **启发**：评估指标的设计比模型规模更重要——对齐评估目标和任务是利用小模型的关键

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统解决小模型预测大模型推理性能的问题
- **理论深度**: ⭐⭐⭐⭐ — ID/OOD 分析和评估对齐框架完整
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三类实验（排名/关系/迁移）覆盖 6 个基准，对比 6+ 基线
- **实用价值**: ⭐⭐⭐⭐⭐ — 100× 计算节省的实际意义巨大
- **总评**: ⭐⭐⭐⭐☆ — 问题重要、方法优雅、实验全面，对预训练数据优化有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Model](predicting_llm_reasoning_performance_with_small_proxy_model.md)
- [\[ICLR 2026\] No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes](no_answer_needed_predicting_llm_answer_accuracy_from_question-only_linear_probes.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICLR 2026\] Best-of-∞: Asymptotic Performance of Test-Time LLM Ensembling](best-of-infinity_asymptotic_performance_of_test-time_llm_ensembling.md)
- [\[ICLR 2026\] Nudging the Boundaries of LLM Reasoning](nudging_the_boundaries_of_llm_reasoning.md)

</div>

<!-- RELATED:END -->
