---
title: >-
  [论文解读] Probabilistic Token Alignment for Large Language Model Fusion
description: >-
  [NeurIPS 2025][可解释性][最优传输] 将 LLM 融合中的 token 对齐问题重新建模为最优传输（Optimal Transport）问题，用动态 token 配对 + Sinkhorn 算法实现"软"概率对齐取代传统硬映射，在 6 大基准 78 个任务上相比 FuseLLM 平均提升 +1.72%，同时在困难任务上大幅缓解性能退化（从 -13.04% 降至 -4.07%）。
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "最优传输"
  - "Sinkhorn算法"
  - "概率Token对齐"
  - "Logit融合"
  - "跨架构模型融合"
  - "Knowledge Fusion"
---

# Probabilistic Token Alignment for Large Language Model Fusion

**会议**: NeurIPS 2025  
**arXiv**: [2509.17276](https://arxiv.org/abs/2509.17276)  
**代码**: [runjia.tech/neurips_pta-llm](https://runjia.tech/neurips_pta-llm)  
**领域**: 可解释性  
**关键词**: 最优传输, Sinkhorn算法, 概率Token对齐, Logit融合, 跨架构模型融合, Knowledge Fusion

## 一句话总结
将 LLM 融合中的 token 对齐问题重新建模为最优传输（Optimal Transport）问题，用动态 token 配对 + Sinkhorn 算法实现"软"概率对齐取代传统硬映射，在 6 大基准 78 个任务上相比 FuseLLM 平均提升 +1.72%，同时在困难任务上大幅缓解性能退化（从 -13.04% 降至 -4.07%）。

## 研究背景与动机

**领域现状**：从头训练 LLM 成本高昂，模型融合（Model Fusion）成为构建更强基线的高效替代方案。主流方法包括模型集成（Model Ensemble）、权重合并（Weight Merging）和知识融合（Knowledge Fusion）三大类。

**现有痛点**：FuseLLM 等知识融合方法依赖手工定义的硬 token 映射（基于最小编辑距离），存在两个关键缺陷——❶ 硬映射过于简化，无法捕捉不同上下文中 token 的多样对齐模式，引入偏差降低学习能力；❷ source 和 target 模型的 top-k token 集合独立对齐，未考虑概率值与整体分布，只能达到局部最优。

**核心矛盾**：模型集成需运行多个模型推理开销大；权重合并要求架构一致不具通用性；知识融合虽灵活但 token 对齐方法粗糙导致部分任务性能下降。

**切入角度**：不同 LLM 虽然词表和 token ID 不同，但其 logit 概率分布编码了相似的语义知识——通过在分布层面对齐（而非字符串层面硬匹配）可以实现更连贯的融合。

**核心 idea**：将 token 对齐重新建模为最优传输问题，用 Sinkhorn 算法求解全局传输计划，实现 logit 分布的软概率对齐。

## 方法详解

### 整体框架
PTA-LLM 遵循知识融合范式：将多个 source LLM 的概率分布矩阵通过 token 对齐融合到 target LLM（Llama-2 7B），采用 CLM loss + Fusion loss 的加权组合训练目标 $\mathcal{L} = \lambda \mathcal{L}_{\text{CLM}} + (1-\lambda) \mathcal{L}_{\text{Fusion}}$。核心创新集中在两阶段概率 token 对齐：动态 token 配对 → 最优传输概率对齐。

### 关键设计

1. **动态 Token 配对（Dynamic Token Pairing）**:

    - 功能：解决 source 和 target 模型因不同分词器产生的序列长度差异，找到最优 token 配对
    - 核心递归：$f(k,j) = \min\{f(k-1,j)+c, f(k,j-1)+c, f(k-1,j-1)+c\}$，其中 $c(\mathcal{B}_k, \mathcal{A}_j)$ 为预定义距离
    - 关键突破：放松传统一对一约束，允许一个 source token 对应多个 target token（反之亦然），适应不同分词方案的粒度差异
    - 计算效率：动态规划算法避免了 $L \times N$ 暴力搜索

2. **概率 Token 对齐（Probabilistic Token Alignment via OT）**:

    - 功能：在配对好的 token 之间进行 logit 级别的软对齐，解决 token ID 不匹配问题
    - 形式化：对每个 token 对 $(\mathcal{A}_j \in \mathbb{R}^{V_s}, \mathcal{B}_k \in \mathbb{R}^{V_t})$，求解最优传输计划 $\hat{\mathcal{T}} = \arg\min_{\mathcal{T} \geq 0} \sum_{x=1}^n \sum_{y=1}^m c_{xy} \mathcal{G}_{xy}$
    - 边际约束：$\sum_y \mathcal{G}_{xy} = \mathcal{A}_j[x]$，$\sum_x \mathcal{G}_{xy} = \mathcal{B}_k[y]$——保证概率质量守恒
    - 代价矩阵：$c_{xy}$ 定义为解码后文本的最小编辑距离，同时结合 logit 值进行"surface-level + logit-level"双层优化
    - 融合 logit 选取：对传输矩阵每行取最大值对应的 target 索引，多个 source 映射到同一 target 时概率累加
    - 窗口大小：默认 top-10 logits（$n=m=10$）

3. **Sinkhorn 算法求解**:

    - 初始化传输矩阵 $\mathcal{T} = \exp(-\lambda C)$
    - 交替缩放行列使边际分布匹配 source 和 target 的 token 概率分布
    - 收敛阈值 $10^{-5}$ 效果最佳（更严格的约束产生更连贯的融合）

4. **融合策略（Fusion Strategy）**:

    - 用交叉熵损失评估每个 source LLM 的预测质量
    - 选择交叉熵最低的 source 分布矩阵（MinCE 策略优于 AvgCE）
    - 组合权重 $\lambda = 0.8$——较低值意味着更依赖融合矩阵

### 训练策略
- 目标模型：Llama-2 7B
- Source 模型：OpenLLaMA 7B + MPT 7B
- 训练数据：MiniPile，batch size 256，max sequence length 2048
- 实现：PyTorch + HuggingFace Transformers + FlashAttention

## 实验关键数据

### 主实验（6 基准 78 任务）

| 基准 [任务数] | OpenLLaMA | MPT | Llama-2 | FuseLLM | PTA-LLM | 提升 |
|--------------|-----------|-----|---------|---------|---------|------|
| GSM [1] | 7.81 | 9.17 | 14.18 | 14.56 | **14.71** | +1.03% |
| BBH [27] | 33.87 | 33.38 | 39.70 | 41.01 | **41.08** | +0.17% |
| MultiPL-E [10] | 18.11 | 17.26 | 14.63 | 15.56 | **15.88** | +2.06% |
| MMLU [17] | 42.11 | 27.84 | 46.94 | 48.77 | **49.38** | +1.25% |
| ToxiGen [14] | 18.94 | 18.42 | 18.56 | 18.19 | **18.89** | +3.85% |
| TyDi QA [9] | 27.32 | 22.11 | 31.42 | 32.99 | **34.07** | +3.27% |
| **平均 [78]** | 24.69 | 21.36 | 27.57 | 28.51 | **29.00** | **+1.72%** |

### 困难任务稳定性（FuseLLM 退化的任务）

| 任务 | Llama-2 | FuseLLM | PTA-LLM | PTA vs Llama-2 |
|------|---------|---------|---------|----------------|
| Causal Judgement [BBH] | 50.80 | 46.52 (-8.43%) | **50.80** | +0.00% |
| Geometric Shapes [BBH] | 34.40 | 22.80 (-33.72%) | **26.80** | -22.09% |
| Tracking Shuffled (7) [BBH] | 11.20 | 10.40 (-7.14%) | **14.00** | **+25.00%** |
| Chemistry [MMLU] | 35.97 | 34.98 (-2.75%) | **36.96** | +2.75% |
| Arabic [TyDi QA] | 8.47 | 5.65 (-33.29%) | **7.49** | -11.57% |
| **平均 7 任务** | 30.22 | 26.28 (-13.04%) | **28.99** | **-4.07%** |

### 可解释性量化分析

| 指标 | FuseLLM | PTA-LLM | 说明 |
|------|---------|---------|------|
| 内部距离（Inner Distance） | 257.83 | **239.44** | 融合 token 更紧凑 |
| 中心距离（Center Distance） | 136.95 | **22.25** | 更接近 target token 分布 |

### 消融实验

| 参数 | 选择 | BBH | ME | MMLU |
|------|------|-----|-----|------|
| OT 收敛阈值 | 1e-4 → **1e-5** | +1.33% | -0.38% | +0.80% |
| 对齐窗口 | 5 → **10** | +0.97% | +1.70% | 持平 |
| 组合权重 | 0.9 → **0.8** | +1.71% | +1.04% | +0.92% |
| 融合函数 | AvgCE → **MinCE** | +1.38% | +1.23% | +1.00% |

### 关键发现
- **全面提升**：6 个基准全部超越 FuseLLM，78 任务平均 +1.72%
- **困难任务大幅缓解**：在 FuseLLM 退化的 7 个任务上，将平均退化从 -13.04% 降至 -4.07%（缓解 8.97%）
- **安全和多语言显著提升**：ToxiGen +3.85%，TyDi QA +3.27%——FuseLLM 在安全基准上甚至低于原 Llama-2，PTA-LLM 完全修复
- **更多模型融合效果更好**：从 1 → 2 → 3 个 source 模型，性能持续提升

## 亮点与洞察
- **优雅的问题重建模**：将 token 对齐从"字符串匹配"重新表述为"概率分布传输"，自然地引入最优传输理论框架，数学上优美且实践中有效
- **软对齐 vs 硬映射的直觉**：硬映射像"搬家"（改变位置不改大小），概率对齐像"物流配送"（将概率质量分配到最优位置）——论文的可视化分析很好地支撑了这个直觉
- **可解释性的量化验证**：通过 Isomap+PCA 降维可视化 token 分布，量化了内部距离和中心距离，提供了 token 对齐为何有效的分布层面解释
- **MinCE 优于 AvgCE**：选择当前预测最好的 source 模型（而非加权平均）效果更好，说明"精选"比"混合"更重要

## 局限与展望
- **仅限 7B 模型**：所有实验基于 Llama-2 7B / OpenLLaMA 7B / MPT 7B，未验证更大规模模型的效果
- **提升幅度有限**：平均 +1.72% 的相对增益在某些场景可能不够显著
- **BBH 上改进极小**：+0.17%，当 source 模型在某领域较弱时，更精确的对齐也可能传递"噪声知识"
- **窗口 top-10 的限制**：OT 计算仅在 top-10 logits 上进行，可能遗漏长尾 token 的信息
- **改进方向**：(1) 扩展到更大模型验证可扩展性；(2) 探索加入语义 embedding 距离的代价函数；(3) 动态调整 OT 窗口大小；(4) 结合 LoRA 等参数高效方法降低训练成本

## 相关工作与启发
- **vs FuseLLM**：PTA-LLM 是 FuseLLM 的直接改进，核心差异在于将硬映射 token 对齐替换为 OT 软概率对齐
- **vs Model Ensemble**：集成需要推理时维护多个模型（高内存+延迟），PTA-LLM 融合后只保留一个模型
- **vs Weight Merging (SWA等)**：权重合并要求架构相同，PTA-LLM 支持跨架构融合
- **vs 传统 Token Alignment**：传统方法仅基于字符串编辑距离做硬映射，PTA-LLM 额外利用 logit 概率分布信息做全局最优匹配

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 OT 引入 token 对齐是自然但此前未被探索的方向，问题建模优雅
- 实验充分度: ⭐⭐⭐⭐ 6 基准 78 任务+困难任务分析+消融+可视化，但仅限 7B 模型规模
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，数学推导完整，可视化分析有说服力
- 价值: ⭐⭐⭐ 对知识融合范式的实用改进，但绝对提升幅度有限且应用场景偏窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](../../ICML2026/interpretability/discovering_implicit_large_language_model_alignment_objectives.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[NeurIPS 2025\] Table as a Modality for Large Language Models](table_as_a_modality_for_large_language_models.md)
- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)

</div>

<!-- RELATED:END -->
