---
title: >-
  [论文解读] RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation
description: >-
  [AAAI 2026][信息检索/RAG][RAG 安全] 提出 RAGFort，首个系统性防御 RAG 知识库抽取攻击的双路径框架，通过对比重索引（inter-class）隔离主题间边界和约束级联生成（intra-class）抑制敏感内容输出，在安全性上将知识恢复率降低至无保护的 0.51×，同时保持回答质量。
tags:
  - "AAAI 2026"
  - "信息检索/RAG"
  - "RAG 安全"
  - "知识库保护"
  - "对比重索引"
  - "级联生成"
  - "双路径防御"
---

# RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation

**会议**: AAAI 2026  
**arXiv**: [2511.10128](https://arxiv.org/abs/2511.10128)  
**代码**: [https://github.com/happywinder/RAGFort](https://github.com/happywinder/RAGFort)  
**领域**: 信息检索  
**关键词**: RAG 安全, 知识库保护, 对比重索引, 级联生成, 双路径防御

## 一句话总结

提出 RAGFort，首个系统性防御 RAG 知识库抽取攻击的双路径框架，通过对比重索引（inter-class）隔离主题间边界和约束级联生成（intra-class）抑制敏感内容输出，在安全性上将知识恢复率降低至无保护的 0.51×，同时保持回答质量。

## 研究背景与动机

### 问题背景

RAG 系统正越来越多地部署于高价值领域（医疗、金融），其知识库是核心知识产权。然而，攻击者可以通过黑盒查询系统性地重建知识库：

**类内抽取（Intra-class extraction）**：在特定主题内迭代细化查询，逐步提取该主题的详细信息

**类间抽取（Inter-class extraction）**：利用已提取的知识递归生成针对语义相关主题的查询，逐步扩大覆盖范围

例如 RAG-Thief 攻击使用自动代理，从初始查询开始，存储提取的 chunk，分析后生成更有针对性的查询，既能深入同一主题也能扩展到新主题，最终几乎重建整个知识库。

### 现有防御的致命缺陷

现有方法只防御一条路径：
- **类内防御**（如改写/摘要）：减少单个 chunk 内的泄露，但无法阻止跨主题聚合
- **类间防御**（如设置检索距离阈值）：限制主题漂移，但允许单一主题内的深度提取

更严重的是，两类防御**天然冲突**：类间保护要求系统只检索少数高相关 chunk；而类内保护（如摘要化）需要聚合多个 chunk 的信息生成多样化内容。

### 关键实验发现

作者设计了掩码实验验证两条路径的互补性：
- 仅类内或仅类间保护时，代理知识库仍保留大量可用信息
- 联合保护时，代理知识库的实用性显著下降
- 随保护比例增大，联合保护的效果远优于单路径

**结论**：有效防御必须同时保护两条路径。

## 方法详解

### 整体框架

RAGFort 包含两个协同模块：
1. **类间保护**：结构感知的对比重索引（Contrastive Reindexing）—— 重组检索器索引以增强主题间语义分离
2. **类内保护**：约束级联生成（Constrained Cascade Generation）—— 轻量 draft 模型 + 强大 verifier 模型的级联框架，抑制敏感 token 生成

### 关键设计

#### 1. **类间保护：对比重索引**

**核心思路**：通过无监督聚类发现知识库的主题结构，然后用有监督对比学习训练编码器，使不同主题在嵌入空间中更好分离，让攻击者更难通过一个主题检索到另一个主题的内容。

三个阶段：

**阶段一：HDBSCAN 伪标签生成**
- 用预训练编码器（如 Sentence-BERT）获取每个 chunk 的稠密表示 $E(k_i)$
- HDBSCAN 聚类将 chunk 分为 $C$ 个簇 $\{\mathcal{C}_1, \ldots, \mathcal{C}_C\}$，自动确定簇数，处理离群值
- 为每个 chunk 分配伪标签 $y_i \in \{1, \ldots, C\}$

**阶段二：SupCon 对比学习**
- 使用监督对比损失训练新编码器：
$$\mathcal{L}_{\text{sup}} = \sum_{i=1}^{B} -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\text{sim}(z_i, z_p)/\tau)}{\sum_{a \in A(i)} \exp(\text{sim}(z_i, z_a)/\tau)}$$
- 拉近同簇样本、推远异簇样本，强化主题边界

**阶段三：索引替换**
- 用训练好的结构感知编码器 $f_{\text{sup}}(\cdot)$ 重新编码所有 chunk
- 构建新的索引用于检索，但生成器仍使用原始 chunk 内容
- 查询编码器和解码器不做修改，保持回答质量

**设计动机**：攻击者依赖语义相似性进行类间扩展，增强类间分离使得从一个主题"漂移"到相邻主题的查询难以命中目标。

#### 2. **类内保护：约束级联生成**

**核心思路**：组合轻量 draft 模型和强大 reference 模型，设计近最优拒绝规则来抑制敏感内容的生成概率。

**级联框架**：
- Draft 模型先提出 $\gamma$ 个候选 token，同时计算概率 $q_t$
- Verifier 模型评估每个 token，根据拒绝规则决定是否接受
- 若拒绝，用 verifier 生成替代 token

**优化目标**：设计拒绝规则 $r(x_{<t}, z)$，最小化 verifier 监督下的期望损失，同时约束拒绝预算 $B$ 和敏感内容生成概率阈值 $C$：

$$\min_r \mathbb{E}[(1-r) \cdot \ell(y, q_t) + r \cdot \ell(y, p_t)]$$
$$\text{s.t. } r \cdot D_{\text{TV}}(p_t, q_t) \leq B, \quad (1-r) \cdot \frac{q_t(z)}{p_t(z)} \leq C$$

**近似拒绝规则**（Lemma 1 推导）：
$$\hat{r}(x_{<t}, z) = 1 \iff \max_y q_t(y) < \max_y p_t(y) - \alpha \cdot D_{\text{TV}}(p_t, q_t) + \eta \cdot \frac{q_t(z)}{p_t(z)}$$

当 draft 模型的置信度明显低于 verifier 时触发拒绝。

**理论保证**（Lemma 2）：近似规则与最优规则的遗憾界为 $\max_y |\mathbb{P}_t(y) - q_t(y)| + \max_y |\mathbb{P}_t(y) - p_t(y)|$，当两个模型都接近真实分布时遗憾很小。

### 损失函数 / 训练策略

- 类间保护的 SupCon 训练使用温度参数 $\tau$ 的标准对比损失
- 类内保护是推理时方法，不涉及额外训练
- Draft/Verifier 配对：Qwen-14B 用 Qwen-7B 做 draft；Gemma-3-27B 用 Gemma-3-4B；DeepSeek-R1-8B 自身同时充当两个角色

## 实验关键数据

### 主实验

三个数据集：HealthCareMagic（医疗QA）、Enron Email（企业邮件）、Math QA（数学推理），三个生成模型，两种攻击（Worm-Attack, RAG-Thief）。

| 防御方法 | 相对平均 CRR (↓) |
|---------|-----------------|
| 无保护 | 1× |
| Re-ranking（类间） | 0.91× |
| Summarization（类内） | 0.87× |
| **RAGFort（双路径）** | **0.51×** |

具体数据（HealthCareMagic, Qwen-14B）：

| 防御 | Worm CRR(%) | RAG-Thief CRR(%) |
|------|------------|------------------|
| 无保护 | 17.60 | 57.16 |
| RAGFort | **8.84** | **27.96** |
| Re-ranking | 16.28 | 56.44 |
| Summarization | 15.44 | 47.24 |

### 消融实验

| 配置 | 相对平均 CRR | 说明 |
|------|-------------|------|
| 完整 RAGFort | 0.51× | 双路径联合最优 |
| RAGFort_InterOnly | 0.75× | 仅类间保护不足 |
| RAGFort_IntraOnly | 0.83× | 仅类内保护更弱 |
| 无保护 | 1.00× | 基线 |

系统性能影响：

| 模型/数据集 | 保护前 ACC | 保护后 ACC | ACC 下降 |
|------------|----------|----------|---------|
| Qwen-14B / HealthCare | 68.16% | 66.57% | -1.59% |
| DeepSeek-R1-8B / HealthCare | 61.36% | 61.12% | -0.24% |
| Gemma-3-27B / HealthCare | 69.64% | 68.88% | -0.76% |
| Qwen-14B / Enron | 99.10% | 98.85% | -0.25% |
| Qwen-14B / MathQA | 85.31% | 82.81% | -2.50% |

### 关键发现

1. **双路径远优于单路径**：RAGFort（0.51×）vs Re-ranking（0.91×）vs Summarization（0.87×），降低 49% 的恢复率
2. **类间和类内保护互补**：单独使用只降到 0.75×/0.83×，联合使用降到 0.51×，验证了双路径互补假设
3. **ACC 下降极小**：所有场景下不超过 2.5 个百分点，证明防御不影响用户体验
4. **FLOPs 不增反降**：级联生成将大模型推理部分替换为小模型，计算量反而降低（如 Qwen-14B/HealthCare 从 38T 降至 19.4T）
5. **Draft 模型差距影响 ACC**：DeepSeek（自身做 draft）ACC 下降最小（0.24%），Qwen（用 7B 做 draft）下降稍大（1.59%）

## 亮点与洞察

1. **系统性思维**：首次将 RAG 知识保护形式化为双路径问题，而非零散的防御手段堆叠
2. **理论+实践结合**：级联生成的拒绝规则有严格的理论推导和遗憾界保证，不是简单的工程 trick
3. **无需修改 LLM**：两个模块都不需要重新训练生成模型，部署友好
4. **攻击-防御完整评估**：同时测试了两种主流攻击方式，覆盖三个领域和三个模型
5. **兼顾效率**：级联生成在提供安全保护的同时还能降低 FLOPs

## 局限与展望

1. 仅评估了黑盒攻击，白盒/灰盒场景下的鲁棒性未知
2. HDBSCAN 聚类质量可能影响类间保护效果，对知识库结构有隐式假设
3. 论文提到的参数敏感性分析未呈现，实际部署时需要调优
4. 当攻击者已知防御机制时的适应性攻击未考虑
5. 仅测试了 chunk 级恢复率，对于更细粒度的信息泄露评估不足
6. 虽然分类为"人体理解"，实际是 AI 安全/NLP 领域的工作

## 相关工作与启发

- **RAG-Thief**（Jiang 2024）是最强攻击基线，展示了自动化迭代提取的威力
- **Worm-Attack**（Cohen 2024）模拟了更极端的自复制传播攻击
- **Speculative Decoding** 启发了级联生成的设计思路，但原始方法关注推理加速，本文将其改造为安全工具
- **SupCon**（监督对比学习）在这里的应用非常自然——将文档聚类标签作为监督信号增强检索空间的类间分离

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首个系统性双路径 RAG 防御框架）
- 实验充分度: ⭐⭐⭐⭐（多数据集多模型多攻击，但缺乏参数敏感性和适应性攻击测试）
- 写作质量: ⭐⭐⭐⭐⭐（论文结构清晰，理论推导严谨）
- 价值: ⭐⭐⭐⭐⭐（解决 RAG 商业部署的实际安全痛点）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)
- [\[ACL 2026\] Quantifying and Improving the Robustness of Retrieval-Augmented Language Models Against Spurious Features in Grounding Data](../../ACL2026/information_retrieval/quantifying_and_improving_the_robustness_of_retrieval-augmented_language_models_.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](../../ACL2025/information_retrieval/seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience](../../ACL2025/information_retrieval/toward_structured_knowledge_reasoning_contrastive_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
