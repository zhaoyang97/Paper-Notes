# A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.15444](https://arxiv.org/abs/2510.15444)  
**代码**: [https://wnjxyk.github.io/RPC](https://wnjxyk.github.io/RPC) (有)  
**领域**: LLM推理 / 测试时缩放  
**关键词**: Self-Consistency, Perplexity, Test-time Scaling, Confidence Estimation, LLM Reasoning  

## 一句话总结
提出首个针对基于采样的测试时缩放方法的理论框架，将推理误差分解为估计误差和模型误差，揭示了Self-Consistency收敛慢、Perplexity模型误差大的局限，并提出RPC方法融合两者优势，在7个基准上以50%的采样成本达到同等推理性能。

## 背景与动机
测试时缩放（Test-time Scaling）通过在推理阶段增加计算资源来提升LLM推理能力，其中基于采样的方法（生成多条推理路径后选择最优）已被广泛使用。现有的置信度估计方法主要分两类：(1) **一致性方法**，以Self-Consistency（SC）为代表，通过多数投票估计答案置信度；(2) **概率方法**，以Perplexity（PPL）为代表，直接利用LLM内部概率。然而，这些方法虽然实验效果不错，但缺乏严格的理论分析来理解它们的机制、固有局限和改进方向。这篇论文正是要填补这一理论空白。

## 核心问题
在基于采样的测试时缩放中，Self-Consistency和Perplexity分别存在什么本质局限？能否找到一种方法同时获得两者的优点——既有快速的估计误差收敛，又能保持低模型误差？

## 方法详解

### 整体框架
给定推理问题 $(x, y)$，LLM采样 $n$ 条推理路径 $\tilde{t}_1, \ldots, \tilde{t}_n$，通过置信度估计函数 $\hat{p}$ 评估每个答案的可信度，最终选择最高置信度的答案。论文建立了一个理论框架，将推理误差（squared error）分解为两个独立组分，然后分析现有方法的不足并提出改进。

### 关键设计

1. **推理误差分解（Proposition 1）**: 对任意答案 $\hat{y}$，推理误差可分解为：
   $$\mathcal{E}_{\hat{p}}(\hat{y}) = \underbrace{\mathbb{E}[(\hat{p}(\hat{y}|x) - p(\hat{y}|x))^2]}_{\text{Estimation Error}} + \underbrace{(p(\hat{y}|x) - \mathbb{I}[\hat{y}=y])^2}_{\text{Model Error}}$$
   - **估计误差**：取决于采样数量和置信度估计策略，可通过更好的估计方法降低
   - **模型误差**：由LLM本身的推理能力决定，是固定的

2. **SC和PPL的理论分析**:
   - **SC**（Proposition 2）：估计误差为 $\frac{1}{n} p(\hat{y}|x)(1-p(\hat{y}|x))$，仅以 $O(1/n)$ 线性速率收敛，采样有限时效果差；但模型误差较低
   - **PPL**（Proposition 3）：估计误差以 $(1-p(\hat{t}|x))^n$ 指数速率收敛，但模型误差比SC大（因为不使用一致性函数聚合等价答案）；且当概率极低时收敛优势退化

3. **Perplexity Consistency（PC）**: 核心创新——将LLM内部概率融入Self-Consistency框架。对每个答案 $\hat{y}$，估计置信度为所有对应推理路径的概率之和：
   $$\hat{p}^{(\text{PC})}(\hat{y}|x) = \sum_{\tilde{t} \in \mathcal{R}} \mathbb{I}[g(\tilde{t})=\hat{y}] \cdot p(\tilde{t}|x)$$
   理论保证（Theorem 4）：PC的估计误差以指数速率 $\alpha^n$ 收敛（$\alpha = 1 - \frac{1}{k}p(\hat{y}|x)$），同时保持与SC相同的低模型误差。

4. **Reasoning Pruning（RP）**: 解决PC在低概率答案上的退化问题。当 $p(\hat{y}|x) \to 0$ 时，指数收敛退化为线性。RP通过建模概率分布来自动过滤低概率推理路径：
   - 用两个Weibull分布的混合模型拟合所有采样路径的概率分布
   - 通过最大似然估计参数，计算每条路径属于高概率分布的后验概率
   - 移除 $P_{\text{High}} < 0.5$ 的路径，并用Truncated Mean保底
   - Theorem 7证明在最优阈值下，RP以高概率实现最优误差缩减

5. **RPC方法**: 先Reasoning Pruning过滤低概率路径，再用Perplexity Consistency计算置信度，是一个无超参数的即插即用方法。

### 损失函数 / 训练策略
RPC是一个后处理（post-hoc）方法，不需要修改LLM架构或训练过程。Weibull混合分布参数通过最大似然估计在每个问题的采样路径上即时拟合。

## 实验关键数据

| 数据集 | 指标 | RPC | SC | PPL | 提升(vs SC) |
|--------|------|-----|-----|-----|-------------|
| MATH | Acc | 51.95 | 50.57 | 46.99 | +1.38 |
| MathOdyssey | Acc | 31.62 | 28.25 | 27.35 | +3.37 |
| OlympiadBench | Acc | 11.14 | 11.07 | 7.27 | +0.07 |
| AIME | Acc | 9.74 | 9.40 | 5.96 | +0.34 |

效率对比（达到SC最佳性能所需最少采样数）:

| 数据集 | SC需要 | RPC需要 | 采样减少 |
|--------|--------|---------|----------|
| MATH | 64 | 32 | -50.0% |
| MathOdyssey | 112 | 32 | -71.4% |
| OlympiadBench | 128 | 64 | -50.0% |
| AIME | 128 | 48 | -62.5% |

### 消融实验要点
- **PC模块贡献**：PC单独已经提升了收敛速率，但在部分数据集（如MathOdyssey）上受到退化问题影响
- **RP模块贡献**：RP对MATH和MathOdyssey提升最显著，有效解决了低概率路径的退化问题
- **跨模型泛化**：在InternLM2-Math 1.8B、7B和DeepSeek-Math 7B上均有效
- **跨任务泛化**：在代码生成（HumanEval/MBPP/APPS）、逻辑推理（GPQA/LogiQA）上同样有效
- **与R1模型兼容**：在DeepSeek-R1-Distill-Qwen-7B上，RPC (61.11) vs SC (57.22) vs PPL (60.04)（MATH数据集，16次采样）
- **与高级方法兼容**：RPC+ESC和RPC+BoN(RM)均优于原始方法
- **高温采样**：T=1.3时SC性能下降但RPC持续提升（MATH上RPC 53.12 vs SC 50.65）
- **超参数鲁棒性**：不同初始化和参数范围下性能稳定
- **计算开销**：RPC额外开销可忽略（0.036s vs 0.006s/question），远小于LLM推理时间

## 亮点
- **首个理论框架**：将基于采样的测试时缩放方法的推理误差清晰分解为估计误差和模型误差，为方法设计提供了原则性指导
- **精准的诊断**：理论分析精确揭示SC的线性收敛瓶颈和PPL的大模型误差问题，而非泛泛而谈
- **巧妙的融合**：PC通过一个简单的概率加权一致性公式，同时获得指数收敛和低模型误差，思路优雅
- **自动化剪枝**：用Weibull混合分布自动建模概率分布，无需手工设置阈值，实用性强
- **50%采样成本削减**：在保持同等性能的前提下大幅减少LLM推理次数，直接节省计算成本

## 局限性 / 可改进方向
- **理论假设较强**：假设LLM采样服从伯努利分布，且采样路径互不相同，在实际中未必严格成立
- **后处理方法天花板有限**：作为post-hoc方法不修改模型训练，性能提升受限于采样路径质量
- **仅分析了两种典型方法**：理论框架有潜力分析更多方法（如MCTS、reward model scoring等），但本文未展开
- **采样策略未深入**：理论提示多样性采样很重要，但文章未探索如何设计更好的采样策略
- **Weibull混合分布的适用性**：对所有问题假设两个Weibull分布可能过于简化

## 与相关工作的对比
- **vs SC (Wang et al., 2022)**: 本文理论证明SC的估计误差仅线性收敛，RPC通过概率加权将其提升为指数收敛
- **vs CISC (Taubenfeld et al., 2025)**: CISC也探索了置信度与自一致性的结合，但本文提供了更完整的理论框架和误差分解
- **vs ESC (Li et al., 2024)**: ESC通过早停减少SC成本，RPC则从估计方法本身提升效率，两者可以组合使用
- **vs TTSC (Huang et al., 2025)**: TTSC做了self-calibration，与本文核心思想一致，但本文给出了理论解释

## 启发与关联
- 理论框架的误差分解思路可以迁移到多模态推理中的测试时缩放分析
- Perplexity Consistency的概率加权一致性思想可以扩展到reward model scoring的场景——用reward代替LLM内部概率
- Reasoning Pruning的Weibull混合分布建模思路可以用于其他需要区分高/低质量样本的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个为采样测试时缩放建立理论框架，方法设计由理论驱动
- 实验充分度: ⭐⭐⭐⭐⭐ 7个数据集、多个模型规模/架构、多种消融、与高级方法兼容性测试
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，命题-定理-备注结构严谨，实验展示逻辑好
- 价值: ⭐⭐⭐⭐ 理论分析为领域提供了有价值的认知框架，RPC方法简单有效可直接用
