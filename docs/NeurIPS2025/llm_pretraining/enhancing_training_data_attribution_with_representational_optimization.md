---
title: >-
  [论文解读] Enhancing Training Data Attribution with Representational Optimization
description: >-
  [NeurIPS 2025][预训练][training data attribution] 提出 AirRep（Attentive Influence Ranking Representation），一种基于表示学习的训练数据归因方法，通过可训练编码器和注意力池化机制，在推理效率比梯度方法快约 80 倍的同时，达到甚至超越 SOTA 梯度方法的归因精度。
tags:
  - "NeurIPS 2025"
  - "预训练"
  - "training data attribution"
  - "表示学习"
  - "影响函数"
  - "注意力机制"
  - "数据选择"
---

# Enhancing Training Data Attribution with Representational Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2505.18513](https://arxiv.org/abs/2505.18513)  
**代码**: [github.com/sunnweiwei/AirRep](https://github.com/sunnweiwei/AirRep)  
**领域**: LLM预训练  
**关键词**: training data attribution, 表示学习, 影响函数, attention pooling, 数据选择

## 一句话总结

提出 AirRep（Attentive Influence Ranking Representation），一种基于表示学习的训练数据归因方法，通过可训练编码器和注意力池化机制，在推理效率比梯度方法快约 80 倍的同时，达到甚至超越 SOTA 梯度方法的归因精度。

## 研究背景与动机

训练数据归因（Training Data Attribution, TDA）旨在衡量训练数据如何影响模型预测，对 AI 透明度和可问责性至关重要。现有方法分为两大类：

**梯度方法**（如影响函数）：
- 理论基础扎实，通过梯度和 Hessian 逆近似模型预测的变化
- 但计算代价极高（需要梯度计算 + Hessian 近似），且依赖损失凸性和模型最优假设（在现代神经网络中均不成立）

**表示方法**（如嵌入相似度）：
- 高效可扩展，适合大规模应用
- 但依赖启发式设计的表示空间，未针对归因任务优化，精度有限

此外，两类方法在估计**群组影响**时都采用简单求和的线性假设，无法捕获样本间的交互效应。

本文的核心问题：**能否设计一种方法，兼具梯度方法的精度和表示方法的效率？**

## 方法详解

### 整体框架

AirRep 由可训练编码器 $\text{Enc}$ 和注意力池化层 $\text{Agg}$ 组成，输入目标样本 $x$ 和训练集 $S$，输出影响分数：

$$f_{\text{AirRep}}(x, S) = \text{Enc}(x)^\top \cdot \text{Agg}(\text{Enc}(z_i) \mid z_i \in S)$$

### 关键设计

#### 1. 注意力池化（Attention-based Influence Pooling）

传统方法用简单求和估计群组影响，忽略样本间交互。AirRep 引入注意力机制加权聚合：

$$f_{\text{AirRep}}(x, S) = \text{Enc}(x)^\top \cdot \sum_{i=1}^{n} \alpha_i \, \text{Enc}(z_i)$$

注意力权重：
$$\alpha_i = \frac{\exp(|\text{Enc}(x)^\top \cdot \text{Enc}(z_i)|)}{\sum_{j \in [n]} \exp(|\text{Enc}(x)^\top \cdot \text{Enc}(z_j)|)}$$

核心直觉：影响分数通常是稀疏的，每个测试样本只依赖少数训练点，其余的增加噪声。注意力机制实现了选择性池化，聚焦最相关的训练样本。

数学联系：可以证明注意力池化与高阶群组影响函数中的样本权重相关（Basu et al. 的二阶项分析），提供了理论支撑。

#### 2. 可训练编码器

以 GTE-Small（30M 参数）为基础，加上随机初始化的投影矩阵。通过任务感知训练优化编码器，使嵌入空间适配归因任务（而非通用文本相似度）。

#### 3. 自动数据生成流水线

构造训练归因信号的流程：
1. 从语料库采样 $N_v = 10^4$ 验证样本和 $N_t = 10^5$ 训练样本
2. 从训练集随机采样 $M = 100$ 个子集，每个含 $n = 1000$ 样本
3. 在每个子集上微调 LLM（Qwen2.5-0.5B），评估在验证集上的损失
4. 计算归一化损失作为归因标签：

$$\hat{r}(x, S_i) = -\frac{\ell(x; \theta_i) - \text{Mean}(\{\ell(x; \theta_j)\})}{\text{Var}(\{\ell(x; \theta_j)\})}$$

构造 100 个交叉验证实例，总计 $10^4$ 个训练子集和 $10^7$ 个训练样本。

### 损失函数 / 训练策略

采用**加权成对排序损失**，优化归因分数排序而非精确值匹配：

$$\mathcal{L}(x, \mathcal{S}) = -\sum_{i,j \in M} \mathbb{1}_{r_i > r_j} \, w_{i,j} \, \log \sigma(f_i - f_j)$$

权重函数处理标签噪声：

$$w_{i,j} = \begin{cases} 0, & \text{if } |r_i - r_j| < T_{\min} \\ \min\{|r_i - r_j|, T_{\max}\}, & \text{if } T_{\min} \leq |r_i - r_j| \end{cases}$$

其中 $T_{\min} = 0.1$，$T_{\max} = 5.0$。低差异对被忽略（标签不可靠），高差异对被裁剪（避免异常值影响）。

训练细节：AdamW，lr=$10^{-4}$，最多 2000 步，分布式训练最大化 GPU 利用率。

## 实验关键数据

### 主实验

**LDS 评估**（Qwen2.5-0.5B，4 个数据集平均）：

| 方法 | 维度 | Avg | FLAN | Alpaca | Tulu | SafeRLHF |
|------|------|-----|------|--------|------|----------|
| LoGra | 18432 (48×) | 18.45 | 19.75 | 12.38 | 14.88 | 26.82 |
| Dsdm | 18432 (48×) | 18.02 | 19.67 | 12.15 | 14.31 | 25.94 |
| LESS | 8196 (21×) | 16.16 | 16.40 | 9.59 | 13.02 | 25.63 |
| TracIn | 18432 (48×) | 11.33 | 14.75 | 9.21 | 10.75 | 10.60 |
| TF-IDF | - | 9.98 | 2.52 | 7.24 | 5.24 | 24.94 |
| **AirRep** | **384 (1×)** | **26.23** | **21.11** | **22.58** | **15.14** | **46.08** |

AirRep 在仅用 1/48 存储的情况下，平均 LDS 超越所有梯度方法 **7.78 分**。

**跨模型泛化**（AirRep 仅在 Qwen2.5-0.5B 上训练）：
- 在 Qwen2.5-1.5B、3B、7B 上均保持领先，说明小模型训练的 AirRep 可迁移到大模型
- 在不同架构（Llama-1B、TinyLlama、GPT-2）上也表现稳健

**数据分类准确率**：

| 方法 | FLAN | Tulu | SafeRLHF |
|------|------|------|----------|
| LoGra (18432) | 85.44 | 86.00 | 83.20 |
| GTE-Small | 50.59 | 76.60 | **90.60** |
| **AirRep** | **86.41** | **88.20** | 87.20 |

### 消融实验

从基础 GTE (7.65) 开始逐步叠加：
1. **+编码器优化**（无注意力）→ 19.82（+12.17），证明编码器优化是核心
2. **+注意力池化** → 26.23（+6.41），注意力池化显著提升群组影响估计
3. 直接给 GTE/LoGra 加注意力仅有微小提升 → 说明优化权重分布比简单加权重要

### 关键发现

1. **效率优势巨大**：AirRep 推理速度快约 80×，存储效率高约 50×，每秒可编码数十万样本
2. **训练成本可摊销**：约 475K 样本的交叉点后，AirRep（含重训练）总成本低于 LoGra。24 GPU 小时可处理 1 亿+ 样本vs LoGra 的 200 万
3. **跨模型/跨任务泛化**：在 Qwen-0.5B 上训练，可直接用于 7B 模型和不同架构
4. **无监督学习任务信息**：AirRep 训练不使用任何数据标签，但能学到任务相关的表示（FLAN 分类准确率 86.41%）

## 亮点与洞察

1. **打破梯度与表示方法的壁垒**：通过任务感知训练将表示方法提升到梯度方法的精度水平，同时保持前者的效率
2. **注意力池化有理论支撑**：与高阶群组影响函数建立数学联系，不是纯启发式
3. **加权排序损失的设计巧妙**：通过对标签差异进行裁剪和忽略处理，优雅地应对 LLM 训练的随机性带来的标签噪声
4. **大规模可扩展性**：24 GPU 小时处理 1 亿样本的吞吐量，真正适用于 LLM 预训练数据归因
5. **训练成本摊销思路**：定量分析了训练 overhead 被推理效率摊销的交叉点，实用参考

## 局限与展望

1. **训练数据生成成本**：需要训练 100 个 LLM 子集模型获取归因标签，虽可摊销但初始成本不低
2. **仅评估了 LLM 微调阶段**：预训练阶段的数据归因更具挑战性，效果待验证
3. **SafeRLHF 上弱于 GTE**：因训练数据（UltraChat）不含有害内容，缺乏安全相关学习信号
4. **模态局限**：目前仅验证了文本任务，虽声称模态无关但视觉/多模态场景未实验
5. **GTE-Small 作为基础编码器**：30M 参数是否限制了表示能力的上限？更大编码器能否进一步提升？

## 相关工作与启发

- **影响函数系列**（Koh & Liang → LoGra → TRAK）：AirRep 的比较基线，揭示了梯度方法在 LLM 规模下的计算瓶颈
- **DCLM / FineWeb-Edu**：数据选择中使用了表示方法，但未做归因优化
- **Datamodels**（Ilyas et al.）：LDS 评估框架的来源，AirRep 沿用其实验设置
- **启发**：(1) 任务感知的表示学习是提升 TDA 的关键路径；(2) 群组影响估计需要超越线性假设；(3) "小模型训练，大模型应用"的迁移模式在数据归因中同样有效

## 评分

- 新颖性: ⭐⭐⭐⭐ — 注意力池化 + 可训练编码器的组合，加上排序优化范式，有显著创新
- 实验充分度: ⭐⭐⭐⭐⭐ — LDS、数据选择、数据分类、消融、成本分析、跨模型/跨架构泛化，极为全面
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，技术路线逻辑通顺，实验组织良好
- 价值: ⭐⭐⭐⭐⭐ — 在 LLM 规模下实现高效精确的数据归因，对数据策展和模型可解释性有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Final-Model-Only Data Attribution with a Unifying View of Gradient-Based Methods](final-model-only_data_attribution_with_a_unifying_view_of_gradient-based_methods.md)
- [\[NeurIPS 2025\] Quantifying Task-Relevant Representational Similarity Using Decision Variable Correlation](quantifying_task-relevant_representational_similarity_using_decision_variable_co.md)
- [\[NeurIPS 2025\] Conformal Risk Training: End-to-End Optimization of Conformal Risk Control](conformal_risk_training_end-to-end_optimization_of_conformal_risk_control.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](../../ICML2025/llm_pretraining/llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[CVPR 2025\] AMO Sampler: Enhancing Text Rendering with Overshooting](../../CVPR2025/llm_pretraining/amo_sampler_enhancing_text_rendering_with_overshooting.md)

</div>

<!-- RELATED:END -->
