---
title: >-
  [论文解读] Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration
description: >-
  [ICML 2025 Spotlight][LLM推理][嵌入扰动] 本文提出 Soft Reasoning，通过在首个生成 token 的 embedding 空间注入高斯扰动并用贝叶斯优化搜索最优扰动向量，以黑盒方式引导 LLM 在推理过程中探索更优的解空间，无需访问模型参数或额外验证器，在数学推理等任务上以极低计算开销超越 temperature scaling 和 Best-of-N 等基线。
tags:
  - "ICML 2025 Spotlight"
  - "LLM推理"
  - "嵌入扰动"
  - "贝叶斯优化"
  - "解码策略"
  - "推理多样性"
  - "测试时计算"
---

# Soft Reasoning: Navigating Solution Spaces in Large Language Models through Controlled Embedding Exploration

**会议**: ICML 2025 Spotlight  
**arXiv**: [2505.24688](https://arxiv.org/abs/2505.24688)  
**代码**: [alickzhu/Soft-Reasoning](https://github.com/alickzhu/Soft-Reasoning)  
**领域**: LLM推理  
**关键词**: 嵌入扰动, 贝叶斯优化, 解码策略, 推理多样性, 测试时计算

## 一句话总结

本文提出 Soft Reasoning，通过在首个生成 token 的 embedding 空间注入高斯扰动并用贝叶斯优化搜索最优扰动向量，以黑盒方式引导 LLM 在推理过程中探索更优的解空间，无需访问模型参数或额外验证器，在数学推理等任务上以极低计算开销超越 temperature scaling 和 Best-of-N 等基线。

## 研究背景与动机

### 现状

LLM 在简单推理任务上表现出色，但在复杂推理任务（如多步数学推理）中仍然面临重大挑战。现有提升推理质量的主流策略可分为两类：

**多样性采样方法**：通过 temperature scaling、top-k、nucleus sampling 等增加生成多样性，以期在多个候选答案中命中正确解。

**规划搜索方法**：如 Chain-of-Thought (CoT)、Tree of Thoughts (ToT)、MCTS 等，通过语言指令或树结构搜索来探索推理路径。

### 痛点

- **Temperature scaling 的局限**：提高温度参数会平坦化整个 token 分布，不加区分地提升所有低概率 token 的采样概率，引入大量噪声而非有意义的探索，导致生成质量下降且不一定覆盖正确答案。
- **启发式搜索的低效**：ToT、MCTS 等方法依赖 prompt 层面的启发式策略，不直接作用于模型内部表示，搜索效率低且高度依赖 prompt 变体，容易陷入"无头苍蝇"式的随机搜索。
- **计算开销过大**：Best-of-N 方法需要大量采样（如 N=64），计算成本与 N 线性增长；ToT 和 TSE 等方法涉及多轮树搜索和回溯，开销同样显著。

### 核心矛盾

如何在保持生成质量和连贯性的同时，高效地探索 LLM 的解空间？现有方法要么牺牲质量换取多样性（温度采样），要么牺牲效率换取覆盖率（大规模采样/搜索）。

### 本文切入角度

与其在 token 概率层面做无差别扰动（temperature），不如直接在 embedding 空间进行有控制的扰动，并用贝叶斯优化来引导搜索方向。关键洞察是：**首个 token 的 embedding 对后续整条推理链有决定性影响**，因此只需优化首个 token 就能控制整体生成方向。

## 方法详解

### 整体框架

Soft Reasoning 的核心思路是将 LLM 推理问题转化为 embedding 空间中的优化问题：

1. **嵌入扰动（Embedding Perturbation）**：在首个答案 token 的 embedding 上注入高斯噪声向量 $\mathbf{z}$，得到扰动后的 embedding $\mathbf{e}_1 + \mathbf{z}$
2. **确定性解码**：给定扰动后的 embedding，后续 token 全部使用 greedy decoding，确保每个扰动向量 $\mathbf{z}$ 唯一对应一个生成序列
3. **奖励评估**：用验证器（可以是 LLM 自身）评估生成序列的正确性和连贯性，得到奖励信号 $r(\mathbf{z})$
4. **贝叶斯优化**：基于已观测的 $(\mathbf{z}, r)$ 对，用 Bayesian Optimization 选择下一个最有前景的扰动向量

整体流程形成一个闭环：**扰动 → 生成 → 验证 → 优化 → 更好的扰动**。

### 关键设计

#### 1. 首 Token 嵌入扰动

传统 temperature scaling 修改的是 softmax 分布：

$$P(w^{(t)} \mid w^{(1:t-1)}; \theta, \tau) = \frac{\exp(\ell_{t,w^{(t)}} / \tau)}{\sum_w \exp(\ell_{t,w} / \tau)}$$

温度 $\tau$ 对所有 token 无差别缩放，低 $\tau$ 退化为 greedy，高 $\tau$ 趋近均匀分布。

Soft Reasoning 的扰动方式不同：在首个 token 的 embedding 层注入高斯向量 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$，直接改变该位置的表示，从而**有选择地**影响后续 token 的生成路径。这带来三个优势：

- **可控性**：每个 $\mathbf{z}$ 确定唯一的生成序列（greedy decoding），搜索可重复
- **灵活性**：embedding 空间的扰动可以产生比 temperature 更丰富的分布变化
- **局部性**：只扰动首个 token，后续 token 自然传播差异，计算开销极小

#### 2. 贝叶斯优化搜索

将推理问题建模为黑盒优化：

$$\mathbf{z}^* = \arg\max_{\mathbf{z}} r(\mathbf{z})$$

其中 $r(\mathbf{z})$ 是给定扰动向量 $\mathbf{z}$ 后生成序列的奖励值。贝叶斯优化的核心组件：

- **高斯过程（GP）代理模型**：基于已观测的 $\{(\mathbf{z}_i, r_i)\}$ 建立 $r(\mathbf{z})$ 的概率模型，预测未探索区域的期望奖励及不确定性
- **采集函数（Acquisition Function）**：平衡探索（exploration）与利用（exploitation），选择信息量最大的下一个 $\mathbf{z}$
- **迭代更新**：每次获得新的观测后更新 GP，逐步缩小搜索空间

与 MCTS / ToT 的区别在于：Soft Reasoning 在连续 embedding 空间中进行梯度友好的搜索，而非在离散 token/路径空间中做组合搜索。

#### 3. 自验证机制

Soft Reasoning 的一大亮点是**无需外部验证器**。LLM 自身同时扮演生成器与验证器的角色：

- 给定问题和候选答案，同一 LLM 通过 prompt 方式判断答案的正确性和连贯性
- 奖励函数综合考虑正确性（答案是否正确）和连贯性（推理过程是否逻辑自洽）
- 这使得方法完全模型无关（model-agnostic），可以即插即用到任何 LLM 上

### 训练/推理策略

Soft Reasoning 是一种**纯推理时方法（test-time method）**，不需要训练或微调：

- **无需训练**：不修改模型参数，不需要额外训练数据
- **无需模型访问**：只需要模型的 embedding 接口和生成接口（黑盒）
- **迭代预算可控**：贝叶斯优化的迭代次数可根据计算预算灵活设定
- **与 greedy decoding 结合**：扰动后使用 greedy decoding，避免采样随机性

## 实验关键数据

### 主要推理任务结果

| 方法 | 搜索空间类型 | GSM8K | MATH | 推理开销 |
|------|-------------|-------|------|----------|
| Greedy Decoding | — | Baseline | Baseline | 1× |
| Temperature Sampling (τ=0.7) | token 分布 | +有限提升 | +有限提升 | N× |
| Best-of-N (N=64) | 独立采样 | 显著提升 | 显著提升 | 64× |
| ToT | 离散路径 | 中等提升 | 中等提升 | 高 |
| **Soft Reasoning** | **连续 embedding** | **最优** | **最优** | **远低于 Best-of-64** |

> Soft Reasoning 在远少于 64 次生成的预算下，准确率超越或持平 Best-of-64。

### 方法特性对比

| 方法 | 搜索空间 | 是否需要外部验证器 | 是否需要模型参数 | 计算可控性 |
|------|----------|-------------------|-----------------|-----------|
| Temperature Scaling | token 分布 | 否 | 否 | 低（仅调温度） |
| Best-of-N | 独立采样 | 是（选最优） | 否 | 差（线性增长） |
| Tree of Thoughts | 离散路径 | 是 | 部分 | 差（指数级） |
| MCTS | 离散路径 | 是 | 部分 | 中等 |
| **Soft Reasoning** | **连续 embedding** | **否（自验证）** | **否** | **强（BO 迭代数可控）** |

### 跨模型泛化性

论文验证了 Soft Reasoning 在多种 LLM 上的有效性，包括不同规模和架构的模型。由于方法完全黑盒化，切换模型只需更换底座，无需修改搜索流程。实验表明方法在不同模型上均有稳定提升。

## 亮点与洞察

1. **首 Token 决定论的实践验证**：论文思路建立在"首个 token 的 embedding 对整条推理链有决定性影响"这一洞察上，这与 Wang & Zhou (2024) 的发现一致。通过扰动首个 token 而非整条序列，极大降低了搜索维度。

2. **连续空间 vs 离散空间搜索**：与 ToT/MCTS 在离散 token 空间搜索不同，Soft Reasoning 在连续 embedding 空间中搜索，天然适合贝叶斯优化等连续优化方法，搜索效率更高。

3. **自验证的巧妙设计**：利用 LLM 自身作为验证器，避免了对外部奖励模型的依赖。虽然自验证的准确性不如专门训练的 verifier，但在贝叶斯优化的框架下，即使噪声较大的奖励信号也能引导搜索方向。

4. **greedy decoding 的确定性保证**：扰动后使用 greedy decoding，保证了 $\mathbf{z} \to \text{sequence}$ 的一一映射，使得搜索空间结构良好，贝叶斯优化的代理模型更容易拟合。

5. **模型无关的即插即用架构**：只需 embedding 输入接口和文本输出接口，不依赖特定模型架构、参数或训练流程，适用范围极广。

## 局限与展望

1. **首 Token 扰动的局限**：仅扰动首个 token 的假设在需要中途修正推理方向的场景中可能不够充分，某些复杂问题可能需要在中间步骤进行干预。

2. **贝叶斯优化的维度限制**：embedding 空间通常是高维的（如 4096 维），标准贝叶斯优化在如此高维空间中效率有限，可能需要降维策略（如 random embedding 或 PCA）。

3. **自验证质量**：LLM 自身作为验证器的准确性依赖于模型能力，对于模型本身就难以判断正确性的领域（如高级数学），自验证可能提供误导性的奖励信号。

4. **缺乏理论保证**：嵌入扰动如何影响后续 token 分布缺乏严格的理论分析，高斯扰动的先验假设可能不是最优选择。

5. **计算开销仍非零**：虽然比 Best-of-64 高效，但每次迭代仍需完整的序列生成和验证，对于实时推理场景可能仍有延迟。

## 相关工作与启发

- **解码策略**：与 nucleus sampling (Holtzman et al., 2020)、min-p sampling (Minh et al., 2025)、D3 (Bao et al., 2024) 形成互补。Soft Reasoning 从 embedding 空间而非 logit 空间切入多样性问题。
- **测试时计算优化**：Snell et al. (2025) 指出优化测试时计算分配比扩大模型更有效，Soft Reasoning 是这一方向的具体实现。
- **树搜索方法**：ToT (Yao et al., 2023)、TSE (Zhang & Liu, 2024) 等在离散空间搜索，Soft Reasoning 将其推广到连续 embedding 空间。
- **互推理与自对弈**：Qi et al. (2025b)、Yan et al. (2024) 的 MCTS + 自对弈思路与 Soft Reasoning 的自验证有相似之处。
- **首 Token 重要性**：Wang & Zhou (2024) 的研究表明首 token 选择对推理结果有关键影响，为本文的核心设计提供了实证基础。
- **贝叶斯优化在 NLP 中的应用**：将 BO 用于超参数搜索是常见做法，但用于推理时的 embedding 搜索是新颖的尝试。

## 评分

- **新颖性**: ⭐⭐⭐⭐☆ — 将推理问题转化为 embedding 空间的贝叶斯优化问题是全新视角，首 token 扰动的设计简洁优雅
- **实验充分度**: ⭐⭐⭐☆☆ — 覆盖了多种 LLM 和推理任务，但缓存截断导致无法完整评估实验细节
- **写作质量**: ⭐⭐⭐⭐☆ — 动机清晰，方法描述系统，与已有工作的对比到位
- **价值**: ⭐⭐⭐⭐☆ — 提供了一种新颖且实用的推理增强策略，模型无关的特性使其具有广泛应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving Rationality in the Reasoning Process of Language Models through Self-playing Game](improving_rationality_in_the_reasoning_process_of_language_models_through_self-p.md)
- [\[NeurIPS 2025\] Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties](../../NeurIPS2025/llm_reasoning/topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)
- [\[ICML 2025\] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models](emergent_symbolic_mechanisms_support_abstract_reasoning_in_large_language_models.md)
- [\[AAAI 2026\] Efficient Thought Space Exploration Through Strategic Intervention](../../AAAI2026/llm_reasoning/efficient_thought_space_exploration_through_strategic_intervention.md)
- [\[CVPR 2026\] ReLaX: Reasoning with Latent Exploration for Large Reasoning Models](../../CVPR2026/llm_reasoning/relax_reasoning_with_latent_exploration_for_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
