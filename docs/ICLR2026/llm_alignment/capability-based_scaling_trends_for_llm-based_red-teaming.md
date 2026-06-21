---
title: >-
  [论文解读] Capability-Based Scaling Trends for LLM-Based Red-Teaming
description: >-
  [ICLR 2026][LLM对齐][红队测试] 在 600+ 对攻击者-目标 LLM 组合上系统评估了 4 种越狱方法，发现攻击成功率（ASR）与攻击者-目标的能力差距遵循 sigmoid 缩放定律（R^2=0.83），能力差距可用 MMLU-Pro 的 logit 变换量化。 领域现状：LLM 红队测试（red-team…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "红队测试"
  - "越狱攻击"
  - "能力缩放"
  - "安全评估"
  - "攻击成功率"
---

# Capability-Based Scaling Trends for LLM-Based Red-Teaming

**会议**: ICLR 2026  
**arXiv**: [2505.20162](https://arxiv.org/abs/2505.20162)  
**代码**: [https://github.com/kotekjedi/capability-based-scaling](https://github.com/kotekjedi/capability-based-scaling)  
**领域**: 人类理解 / AI安全 / LLM对齐  
**关键词**: 红队测试, 越狱攻击, 能力缩放, 安全评估, 攻击成功率

## 一句话总结
在 600+ 对攻击者-目标 LLM 组合上系统评估了 4 种越狱方法，发现攻击成功率（ASR）与攻击者-目标的能力差距遵循 sigmoid 缩放定律（R^2=0.83），能力差距可用 MMLU-Pro 的 logit 变换量化。

## 研究背景与动机

**领域现状**：LLM 红队测试（red-teaming）通过模拟攻击来评估模型安全性。现有研究通常只在少量模型对上评估，缺乏对攻击成功率如何随模型能力变化的系统性理解。

**现有痛点**：不同攻击方法的 ASR 差异巨大，且在不同模型对上表现不一致。缺乏统一的框架来预测新模型组合的攻击脆弱性。每次新模型发布都需要重新全量评估。

**核心矛盾**：安全评估是资源密集型的（每对模型都要测试），能否用缩放律来预测而非全量测试？同时，随着模型能力提升，人类红队测试是否终将失效？

**本文目标** 发现并量化 ASR 与模型能力差距之间的缩放关系，为安全评估提供预测性框架。

**切入角度**：将 MMLU-Pro 分数做 logit 变换作为"能力"的统一代理指标，计算攻击者与目标的能力差。

**核心 idea**：越狱成功率是攻击者-目标能力差距的 sigmoid 函数——攻击者越强且目标越弱，成功率越高；当目标超越攻击者时 ASR 急剧下降。

## 方法详解

### 整体框架
本文不提新攻击，而是把红队测试当成一个可测量的物理现象来刻画：要回答的问题是「攻击成功率到底由什么决定、能不能预测」。做法分三步——先把所有攻击者模型「解锁」掉，剥离安全对齐带来的拒绝，让它们愿意全力攻击；再用 4 种类人越狱攻击（PAIR、TAP、PAP、Crescendo）在 600+ 对攻击者-目标组合（29 个目标模型，含闭源前沿模型）上各跑一遍，按 HarmBench 前 50 条有害行为记录每对的攻击成功率（ASR，每条行为取 best-of-25 多次尝试中的最优）；最后把每对的 ASR 与攻击者-目标的「能力差距」对齐，拟合出二者的函数关系。能力统一用 MMLU-Pro 分数代理，于是整套分析的落点是一条把能力差映射到 ASR 的曲线。

> 这是一篇缩放趋势/测量分析论文，核心贡献是一条回归曲线而非多模块系统，pipeline 是线性的「解锁→批量攻击→记录 ASR→拟合曲线」，画框架图信息量低，故跳过 Mermaid 图；下面三个设计仍按测量流程的先后顺序排列，保持整体框架与关键设计一致。

### 关键设计

**1. 模型解锁：剥掉拒绝，只测攻击力**

测量的第一步要排除一个混淆因素：如果攻击者因为自身安全对齐而拒绝配合，那测到的就不是「它多会攻击」，而是「它愿不愿意攻击」，两者混在一起会污染缩放律。本文沿用「安全对齐很浅、可被轻易撤销」的观察，对全部攻击者做 LoRA 微调来去除安全护栏：用 BadLlama 与 Shadow Alignment 两个数据集混合、共约 1500 条有害样本训练，在移除拒绝行为的同时保留通用能力，并用直接 HarmBench 查询的 ASR 验证解锁是否到位。只有这样，后面 $\delta$ 与 ASR 的关系才反映纯粹的能力差距效应，而不被对齐强度搅浑。

**2. 能力差度量：把"谁更强"压成一个实数**

要把 ASR 回归到「能力差」上，先得给「谁更强、强多少」一个可回归的标量。直接比较两个模型的 MMLU-Pro 准确率不好用，因为准确率被压在 $[0,1]$ 区间、两端饱和、间距不均匀。本文先对分数做 logit 变换 $\text{logit}(p)=\log\!\big(p/(1-p)\big)$ 把它拉到整条实数轴，再定义能力差

$$\delta = \text{logit}(\text{攻击者 MMLU-Pro}) - \text{logit}(\text{目标 MMLU-Pro})$$

$\delta>0$ 表示攻击者更强（strong-to-weak），$\delta<0$ 表示目标反超（weak-to-strong）。这样攻击的脆弱性就由一个标量主导，跨模型家族、跨参数规模也能放到同一根坐标轴上横向比较。

**3. Sigmoid 缩放律：用一条曲线预测任意组合的 ASR**

有了 $\delta$，本文对每个目标模型在 logit 空间拟合 ASR 关于 $\delta$ 的线性回归，再映射回概率空间，自然得到一条 sigmoid 曲线——攻击者越强、目标越弱，ASR 越接近 1；目标反超后 ASR 急剧跌向 0。不同目标的斜率与偏移各异但形状一致，中位线参数为 $k=1.73,\ b=-0.79$，整体拟合达到 $R^2=0.83$。它的实用价值正在于此：只要知道两个模型的 MMLU-Pro 分数、算出 $\delta$，就能预测这对组合的红队 ASR，省去逐对全量评测的巨大开销，也让「人类红队随目标变强而失效」这类外推有了定量依据。

## 实验关键数据

### 主实验（600+ 攻击者-目标对）

| 攻击方法 | 模型对数 | avg ASR vs MMLU-Pro $\rho$ | R² (sigmoid 拟合) | 关键发现 |
|---------|---------|----------------------|------------------|---------|
| PAIR | 600+ | >0.88 | 0.83 | 最基础的 LLM 攻击 |
| TAP | 600+ | >0.85 | ~0.80 | 整体 ASR 最高 |
| PAP | 600+ | >0.82 | ~0.78 | 说服力导向攻击 |
| Crescendo | 600+ | >0.80 | ~0.75 | 多轮攻击 |

### 核心缩放趋势

| 趋势 | 量化 | 意义 |
|------|------|------|
| 更强模型=更好攻击者 | avg ASR 与 MMLU-Pro 线性相关（$\rho>0.88$） | 能力提升同时提升攻击力 |
| 更强模型=更难攻破 | Target ASR 随 MMLU-Pro 下降（R²=0.83） | 但下降趋势可预测 |
| 能力差主导 | ASR 主要取决于 $\delta$（差距），而非攻击者绝对能力 | 相对能力比绝对能力重要 |
| 社会科学>STEM | 心理学/健康/哲学分项相关性最强 | 说服力比知识量更关键 |

## 亮点与洞察
- **预测性工具**：知道两个模型的 MMLU-Pro 分数，就可以预测红队测试的大致 ASR，减少昂贵的全量评测——节省可能上万 GPU 小时的测试成本
- **安全投资的量化**：sigmoid 的右移量可以衡量安全训练的"等效能力提升"——Llama3 的右移比 Qwen2.5 更大，反映更强的安全投入
- **武器化风险的定量评估**：随着开源模型能力提升，攻击者的可用能力池增大——一个新的 70B 开源模型发布，所有现有部署系统的攻击暴露面都需要重新评估
- **Judge 不重要的实用启示**：昂贵的闭源 Judge 对最终 ASR 无显著影响——社区不需要在 Judge 上花费大量 API 成本

## 消融实验与深入分析

| 分析维度 | 发现 |
|----------|------|
| MMLU-Pro 分项相关性 | 社会科学分项（心理学、健康、哲学）与 ASR 相关性最强，超过 STEM |
| Judge 影响 | 强 Judge 提升 ASR@1（选择效果）但不影响 ASR@25（生成质量） |
| 攻击方法对比 | TAP 整体最强，Crescendo 的原始成功归因于 GPT-4 攻击者而非方法本身 |
| Llama2 异常 | 4 个早期 Llama 模型偏离趋势（过度拒绝+对抗训练），后续版本回归趋势 |
| 人类红队预测 | 假设人类 MMLU-Pro=0.898，ASR 随目标模型能力增长持续下降 |

## 局限与展望
- MMLU-Pro 作为能力代理可能在特定领域不够精确——Llama2 系列的偏离说明安全训练可以打破能力-鲁棒性的一般趋势
- 仅测试了 4 种自动攻击（PAIR/TAP/PAP/Crescendo），手工攻击或组合攻击可能不遵循同一缩放律
- 模型解锁（LoRA 微调去除安全对齐）可能不完全恢复攻击能力——解锁质量是潜在混淆因素。特别是某些模型（如 Claude）无法通过简单微调完全解锁
- 未考虑白盒攻击（GCG 等），仅限黑盒类人攻击——白盒攻击可能遵循不同的缩放规律
- Llama2 系列的偏离说明 MMLU-Pro 不是完美的防御能力代理——更好的代理可能是安全训练的 FLOPs 投入，但此数据不公开

## 相关工作与启发
- **vs Ren et al. (2024) 安全基准分析**：他们首次量化了 jailbreak 成功率与模型能力的负相关；本文进一步精确刻画为 sigmoid 函数
- **vs Howe et al. (2025) GCG 缩放**：研究 GCG 攻击在模型规模内的缩放，本文研究跨模型家族的能力差距缩放，维度互补
- **vs PAIR/TAP/PAP**：本文不是提出新攻击，而是用已有攻击揭示缩放规律——强调"规律"而非"方法"
- **启发——社会科学能力是盲点**：当前安全评估聚焦于技术能力（代码、化学），但说服力和社会工程能力才是攻击成功的关键预测因子

## 评分
- 新颖性: ⭐⭐⭐⭐ 红队测试的缩放律是新发现，sigmoid 拟合可直接用于预测
- 实验充分度: ⭐⭐⭐⭐⭐ 600+ 模型对的大规模评测，含闭源前沿模型
- 写作质量: ⭐⭐⭐⭐ 结果清晰，图表信息密度高
- 价值: ⭐⭐⭐⭐⭐ 对 AI 安全评估有直接的工程指导意义和政策启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [\[ICLR 2026\] Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment](inverse_reinforcement_learning_with_dynamic_reward_scaling_for_llm_alignment.md)
- [\[ACL 2026\] ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System](../../ACL2026/llm_alignment/ares_adaptive_red-teaming_and_end-to-end_repair_of_policy-reward_system.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ICLR 2026\] IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](ideal_data_equilibrium_adaptation_for_multi-capability_language_model_alignment.md)

</div>

<!-- RELATED:END -->
