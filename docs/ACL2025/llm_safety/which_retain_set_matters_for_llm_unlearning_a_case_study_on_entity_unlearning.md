---
title: >-
  [论文解读] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning
description: >-
  [ACL 2025][LLM安全][machine unlearning] 系统研究实体遗忘中 retain set 的选择问题，提出 Syntactically Similar Neighbor Set，发现句法相似性（而非领域/实体相似性）才是遗忘过程中知识退化的主要驱动因素，用句法相似的 retain set 做正则化可同时最优保护所有类型的邻居知识。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "machine unlearning"
  - "retain set"
  - "entity unlearning"
  - "syntactic similarity"
  - "neighbor set"
  - "regularization"
---

# Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning

**会议**: ACL 2025  
**arXiv**: [2502.11441](https://arxiv.org/abs/2502.11441)  
**代码**: 未公开  
**领域**: LLM安全  
**关键词**: machine unlearning, retain set, entity unlearning, syntactic similarity, neighbor set, regularization

## 一句话总结

系统研究实体遗忘中 retain set 的选择问题，提出 Syntactically Similar Neighbor Set，发现句法相似性（而非领域/实体相似性）才是遗忘过程中知识退化的主要驱动因素，用句法相似的 retain set 做正则化可同时最优保护所有类型的邻居知识。

## 研究背景与动机

**领域现状**：LLM 遗忘研究集中在遗忘方法设计（GA、DPO、NPO 等），对 retain set 的构成和选择缺少深入分析。

**现有痛点**：现有邻居集（Domain Neighbor、Entity Neighbor）基于领域或实体关系构建，但这些方法是否真正捕捉了遗忘过程中最脆弱的知识区域尚未验证。

**核心矛盾**：遗忘过程会连带损害 retain set 中的知识，但具体哪些知识最易受损？现有假设（领域/实体相似性最关键）是否正确？

**本文目标** 回答两个研究问题：(RQ1) 遗忘对不同邻居集的性能影响有何差异？(RQ2) 哪种邻居集做正则化效果最优？

**切入角度**：从句法结构（syntactic structure）而非语义/领域角度重新审视遗忘的传播模式。

**核心 idea**：遗忘主要沿句法模式传播——结构相似的问题最易被连带遗忘，也最适合用于保留正则化。

## 方法详解

### 整体框架
1. 定义三种邻居集：Domain Neighbor（同领域）、Entity Neighbor（实体关联）、Syntactically Similar Neighbor（句法结构相似）
2. 在 Real-world 和 TOFU 两个场景上，用 4 种遗忘方法（GA/DPO/NPO/IDK）分别遗忘
3. 分析各邻居集的 Relative Utility Drop (RUD)
4. 用 3×3 实验矩阵测试不同 retain set 配置的正则化效果

### 关键设计

1. **Syntactically Similar Neighbor Set 构建**

    - 第一步：GPT-4o 实体掩码（mask 人名、日期、组织名），聚焦句法结构
    - 第二步：计算掩码后问题的 Levenshtein 相似度，聚类（阈值 $\theta_{high}$，簇大小 ≥ 3）
    - 第三步：从 retain set 中选不在其他邻居集的实体，按聚类的句法模式生成新 QA 对
    - 第四步：模型探测验证（只保留模型能正确回答的 QA 对）

2. **评估指标**

    - Model Utility (MU)：ROUGE + BERT Cosine Sim + Probability + Entailment 的算术平均
    - Forget Efficacy (FE)：同指标集在 forget set 上的聚合
    - Relative Utility Drop (RUD)：$(MU_{after} - MU_{before}) / MU_{before} \times 100$
    - 所有遗忘方法的 Forget Efficacy 统一调到 0.65–0.75 区间（公平比较）

3. **正则化实验设计**

    - 两种正则化 loss：GD（标准梯度下降保留）和 KL（KL 散度保留）
    - 3×3 矩阵：train retain set（Domain / Entity / Syntactically Similar）× test retain set
    - 每格取 4 种遗忘方法的 RUD 平均

## 实验关键数据

### 表1：遗忘过程对不同邻居集的 RUD（Real-world，Llama-3-8B-Instruct）

| 邻居集类型 | GA | NPO | IDK | DPO | 趋势 |
|-----------|-----|-----|-----|-----|------|
| Domain Neighbor | 小降 | 小降 | 小降 | 小降 | 影响最小 |
| Entity Neighbor | 小降 | 小降 | 小降 | 小降 | 与 Domain 相当 |
| **Syntactically Similar** | **大降** | **大降** | **大降** | **大降** | **影响最大** |
| Syn Sim & Domain (交叉) | 更大降 | 更大降 | 更大降 | 更大降 | 重叠加剧遗忘 |

### 表2：正则化效果热力图（GD 正则化，RUD 平均 across GA/DPO/NPO/IDK）

| Train Retain ↓ \ Test Retain → | Domain | Entity | Syn. Similar |
|-------------------------------|--------|--------|-------------|
| Domain | 中 | 中 | 低 |
| Entity | 中 | 中 | 低 |
| **Syntactically Similar** | **中-高** | **中-高** | **高 (+14.7pp)** |

关键发现：用 Syntactically Similar 做 train retain 时，对所有 test retain 类型都表现最优或接近最优。

## 亮点

- **挑战传统假设**：此前普遍认为领域/实体相似性决定遗忘传播，本文用系统实验证明句法相似性才是主导因素
- **改写实验验证鲁棒性**：paraphrase 后句法相似邻居仍然遗忘更多，说明不仅仅是表面句法匹配
- **梯度分析提供机制解释**：句法相似实例的梯度范数在遗忘过程中上升更快（Frobenius norm），定量解释了传播机制
- **实用指导**：Syntactically Similar Neighbor Set 做正则化是最优策略，GD 比 KL 差异更显著（14.7pp vs 7.35pp）
- **跨场景一致**：Real-world 和 TOFU 两个场景结论一致

## 局限与展望

- 仅关注实体遗忘，有害知识遗忘和版权内容遗忘可能有不同规律
- 实验限于 7B/8B 模型（LLaMA-2-7B-Chat、LLaMA-3-8B-Instruct），更大模型的行为可能不同
- 句法相似集的构建依赖 GPT-4o 做实体掩码，引入额外成本和不确定性
- 未探讨多种邻居集混合做 retain set 的效果
- unlearning 方法本身未改进，仅分析 retain set 的数据选择

## 与相关工作的对比

| 维度 | 本文 | Opt-Out (Choi et al.) | TOFU (Maini et al.) | RWKU (Jin et al.) |
|------|------|----------------------|---------------------|-------------------|
| 研究重点 | retain set 选择 | 遗忘方法 (OT 正则化) | 虚构遗忘 benchmark | 真实实体 benchmark |
| 邻居集类型 | **Domain + Entity + Syntactic** | Domain + Entity | Domain | Entity |
| 关键发现 | 句法 > 领域/实体 | Wasserstein > L2 | — | — |
| 方法贡献 | 数据视角（retain set 选择） | 算法视角（正则化） | 数据集 | 数据集 + 评估 |
| 正则化分析 | GD vs KL × 3 retain 类型 | Wasserstein vs 多种距离 | 无 | 无 |

## 评分

- 新颖性: ⭐⭐⭐⭐ (句法相似性视角新颖，挑战了领域和实体相似性的传统假设)
- 实验充分度: ⭐⭐⭐⭐⭐ (4 方法 × 3 邻居类型 × 2 场景 × 2 正则化 + paraphrase + 梯度分析)
- 写作质量: ⭐⭐⭐⭐ (RQ 驱动结构清晰，图表直观)
- 价值: ⭐⭐⭐⭐ (为遗忘方法的 retain set 选择提供实用指导，与 Opt-Out 互补)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)
- [\[ACL 2025\] Language Models Can Subtly Deceive Without Lying: A Case Study on Strategic Phrasing](language_models_can_subtly_deceive_without_lying_a_case_study_on_strategic_phras.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](../../ACL2026/llm_safety/maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](../../ICML2025/llm_safety/targeted_unlearning_with_single_layer_unlearning_gradient.md)

</div>

<!-- RELATED:END -->
