---
title: >-
  [论文解读] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation
description: >-
  [NeurIPS 2025][信息检索/RAG][RAG] 提出 CalibRAG 框架，通过训练一个温度条件化的 forecasting function 来确保 RAG 辅助决策过程中的置信度校准，不仅改善校准质量还提升了准确率。 - LLM 越来越多地用于辅助人类决策，但 LLM 经常以高置信度提供错误信息（hallu…
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "RAG"
  - "calibration"
  - "decision-making"
  - "retrieval augmented generation"
  - "confidence estimation"
---

# Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation

**会议**: NeurIPS 2025  
**arXiv**: [2411.08891](https://arxiv.org/abs/2411.08891)  
**代码**: 待确认  
**领域**: 信息检索  
**关键词**: RAG, calibration, decision-making, retrieval augmented generation, confidence estimation

## 一句话总结

提出 CalibRAG 框架，通过训练一个温度条件化的 forecasting function 来确保 RAG 辅助决策过程中的置信度校准，不仅改善校准质量还提升了准确率。

## 研究背景与动机

- LLM 越来越多地用于辅助人类决策，但 LLM 经常以高置信度提供错误信息（hallucination），导致用户做出次优决策
- 研究表明用户对 LLM 输出存在过度依赖，且依赖程度与模型置信度成正比
- RAG 通过引入外部文档来缓解幻觉，但 **RAG 的检索器可能返回不相关文档**，且 LLM 对检索文档过度自信
- 现有 RAG 方法只关注检索相关性，**未考虑用户决策是否校准良好**
- 传统 temperature scaling 不适用于长文本生成的校准问题
- 先前的 decision calibration 方法需要微调 3 个 LLM + PPO 训练，代价高且不稳定

## 方法详解

### 整体框架

CalibRAG 的核心思想：训练一个 forecasting function $f(t, q, d)$ 来预测"给定温度 $t$、查询 $q$ 和检索文档 $d$，用户决策正确的概率"。推理时用该函数对检索文档进行重排序，选择最有可能导致正确决策的文档。

四阶段推理流程：
1. **Stage 1 - 初始检索**：给定 query $q^*$，检索 Top-K 候选文档
2. **Stage 2 - 评分与选择**：用 $f(t, q^*, d_i^*)$ 对每个文档评分并重排序
3. **Stage 3 - 查询重构（可选）**：如果最高置信度低于阈值 $\epsilon = 0.5$，重构 query 再检索
4. **Stage 4 - 最终决策**：生成 guidance 和置信度，用户据此做出决策

### 关键设计

**Forecasting Function 建模**：

以冻结的 LLM $\mathcal{M}$ 作为特征提取器 $f_{\text{feat}}$，用 Fourier 位置编码处理温度参数：

$$\text{PE}(t) = [\sin(\omega_1 t), \cos(\omega_1 t), \ldots, \sin(\omega_N t), \cos(\omega_N t)]$$

其中 $\omega_n = 2^n \cdot \frac{2\pi}{t_{\max} - t_{\min}}$。最终模型：

$$f(t, q, d) = \sigma\left(W_{\text{head}}^\top \left(f_{\text{feat}}(\text{concat}[q, d]; W_{\text{LoRA}}) + W_p \cdot \text{PE}(t)\right) + b_{\text{head}}\right)$$

仅训练 LoRA 适配器和轻量头部，保持 LLM 冻结。

**合成监督数据生成**：

- 从 TriviaQA、SQuAD2.0、WikiQA 提取 $(x, y)$ 对
- 为每个 query 检索 Top-20 文档（而非仅 Top-1），原因：(1) 低排名文档也可能帮助正确决策；(2) 避免训练数据偏向负样本
- 使用代理用户模型 $U$ 在不同温度 $t$ 下采样 $R=10$ 次响应
- 软标签 $b \in [0,1]$ = 正确响应的比例

### 损失函数 / 训练策略

使用对数似然损失（strictly proper scoring rule）：

$$\mathcal{L} = -\frac{1}{|\mathcal{S}|} \sum_{(t,q,d,b) \in \mathcal{S}} \left[b \log f(t,q,d) + (1-b)\log(1-f(t,q,d))\right]$$

该损失是 strictly proper scoring rule（对数分数），保证唯一最大化器就是真实概率 $p$，确保校准收敛。

也探索了多类别变体 CalibRAG-multi，将正确性分布离散化到直方图 bins (0-10)。

## 实验关键数据

### 主实验：通用领域（NQ, WebQA）

使用 Llama-3.1-8B 作为 RAG 和决策模型，BM25 和 Contriever 两种检索器。

| 方法 | 指标 | NQ (BM25) | WebQA (BM25) |
|------|------|-----------|-------------|
| CT-probe | ECE↓ | ~0.35 | ~0.38 |
| Number-LoRA | ECE↓ | ~0.30 | ~0.33 |
| CalibRAG | ECE↓ | **~0.15** | **~0.18** |
| CalibRAG-multi | ECE↓ | **~0.14** | **~0.17** |

CalibRAG 在所有指标（1-AUROC、1-ACC、ECE、BS）上全面优于基线。

### 医学领域（MedCPT 检索器）

| 指标 | BioASQ-Y/N | MMLU-Med | PubMedQA |
|------|-----------|---------|---------|
| CalibRAG ECE↓ | 最优 | 最优 | 最优 |
| CalibRAG ACC↑ | 最优 | 最优 | 最优 |

CalibRAG 使用通用域数据训练，但在医学领域（unseen 检索器 + OOD 数据集）仍然表现最优。

### 与重排序 / Robust RAG 对比

| 数据集 | 方法 | AUROC↑ | ACC↑ | ECE↓ | BS↓ |
|--------|------|--------|------|------|-----|
| HotpotQA | Cross-encoder | 60.74 | 34.98 | 0.477 | 0.477 |
| HotpotQA | LLM-rerank | 60.57 | 38.52 | 0.248 | 0.297 |
| HotpotQA | **CalibRAG** | **72.47** | **42.37** | **0.106** | **0.206** |
| NQ | SelfRAG | 48.4 | 36.2 | 0.522 | 0.545 |
| NQ | **CalibRAG** | **63.5** | **37.4** | **0.258** | **0.287** |

### 消融实验

- **温度条件化**：去除温度条件后 ECE 显著增加，尤其在高温采样时，验证了温度建模的必要性
- **检索文档数量**：K=20 时性能最优，增加到 40 后收益递减
- **查询重构**：Stage 3 在所有设置中一致提升性能，但增加计算开销

### 关键发现

1. Top-1 检索文档往往不是最优的——低排名文档有时能带来更好的决策
2. 添加检索文档后虽然提升准确率，但也增加了 ECE（过度自信），需要额外校准
3. CalibRAG 虽然主要设计用于校准，但因为选择了更有可能导致正确决策的文档，准确率也得以提升

## 亮点与洞察

1. **问题定义新颖**：将 RAG 的目标从"检索相关文档"扩展到"确保校准良好的决策"，视角独特
2. **温度条件化设计巧妙**：通过 Fourier 编码建模用户行为差异，使得同一个模型可以适应不同风险偏好的用户
3. **跨域泛化能力强**：在通用域训练的模型可以直接应用于未见过的医学检索器和数据集
4. **轻量级方案**：仅需训练 LoRA 适配器和分类头，避免了 PPO 等不稳定训练
5. **严格理论保证**：使用 strictly proper scoring rule 作为损失函数，保证校准收敛性

## 局限与展望

- 合成数据生成和 forecasting function 训练有额外开销
- 依赖 GPT-4o-mini 作为评估模型 $\mathcal{G}$，可能引入评估偏差
- 代理用户模型 $U$ 可能无法完全模拟真实人类决策行为
- 温度参数 $t$ 的解释与实际用户行为之间还有差距
- 未探索 forecasting function 在更多 LLM backbone 上的效果
- 查询重构阶段的触发机制可以更精细化

## 相关工作与启发

- 将置信度校准从分类任务扩展到 RAG 长文本生成场景，有启发意义
- 文档重排序不等于校准——重排序优化排序指标，CalibRAG 优化决策正确性
- 可以与其他 Robust RAG 方法（如 SelfRAG）互补使用
- 对高风险决策场景（医疗、法律）中 LLM 的可靠部署有直接参考价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 将校准概念引入 RAG 决策场景，视角新颖但技术路线相对直接
- 实验充分度: ⭐⭐⭐⭐ 覆盖多数据集、多检索器、多领域，消融实验完整
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，数学形式化规范，但部分符号较密
- 价值: ⭐⭐⭐⭐ 对 RAG 系统的可靠性提升有实际价值，适用于高风险决策场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)
- [\[ACL 2025\] Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets](../../ACL2025/information_retrieval/genie_worksheets_tod_agent.md)
- [\[NeurIPS 2025\] Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)
- [\[ICML 2025\] POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval](../../ICML2025/information_retrieval/poqd_performance-oriented_query_decomposer_for_multi-vector_retrieval.md)

</div>

<!-- RELATED:END -->
