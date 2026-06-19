---
title: >-
  [论文解读] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning
description: >-
  [AAAI 2026][模型压缩][视觉Token剪枝] SharpV 提出一个两阶段无训练视觉Token剪枝框架，在Pre-LLM阶段基于时空信息自适应调整每帧剪枝比例，在Intra-LLM阶段基于视觉信息退化假说进行KV Cache剪枝，首次实现与Flash Attention完全兼容，在多个视频理解基准上以约12%的Token保留率达到与稠密模型相当甚至更优的性能。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "视觉Token剪枝"
  - "VideoLLM"
  - "KV Cache"
  - "自适应剪枝"
  - "注意力机制"
---

# Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning

**会议**: AAAI 2026  
**arXiv**: [2511.08003](https://arxiv.org/abs/2511.08003)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 视觉Token剪枝, VideoLLM, KV Cache, 自适应剪枝, Flash Attention

## 一句话总结
SharpV 提出一个两阶段无训练视觉Token剪枝框架，在Pre-LLM阶段基于时空信息自适应调整每帧剪枝比例，在Intra-LLM阶段基于视觉信息退化假说进行KV Cache剪枝，首次实现与Flash Attention完全兼容，在多个视频理解基准上以约12%的Token保留率达到与稠密模型相当甚至更优的性能。

## 研究背景与动机

### 领域现状
视频大语言模型（VideoLLM）在视频理解和推理任务上展现了强大能力，但长视频中大量的时空信息导致LLM输入Token过多，带来二次方计算复杂度和KV Cache膨胀问题。

### 现有方法的痛点

**固定剪枝比例问题**：现有方法（FastV、VTW、DyCoke、FrameFusion等）统一应用固定剪枝率，本质上是针对特定数据集的局部最优，缺乏基于视频信息量的自适应调整，泛化性和鲁棒性不足。

**计算开销问题**：一些方法依赖计算密集的聚类算法（PruneVid）或复杂的规划技术（LLaVA-Scissor、DivPrune），剪枝本身引入的开销可能抵消节省。

**Flash Attention不兼容**：Intra-LLM阶段的方法大多依赖暴露的注意力分数来剪枝，而Flash Attention为了将复杂度从 $O(n^2 \cdot d)$ 降到 $O(n \cdot d)$ 而隐藏了注意力分数，导致这些方法无法使用。

### 核心矛盾
如何在保持低计算复杂度的同时实现信息感知的自适应剪枝，并且完全兼容Flash Attention等硬件加速技术？

### 本文切入角度
从信息论视角出发，不依赖注意力分数，而是利用视觉特征与原始特征的相似度来度量信息退化，实现两阶段层次化剪枝。

## 方法详解

### 整体框架
SharpV是一个两阶段即插即用框架：
- **Visual SharpV（Pre-LLM阶段）**：基于时空重要性评分选择重要视觉Token，通过L2范数和不相似度模块自适应确定每帧剪枝比例
- **Memory SharpV（Intra-LLM阶段）**：通过评估层级视觉信息退化程度，动态丢弃KV Cache

### 关键设计

#### 1. **不相似度计算模块（Dissimilarity Computation Module）**
- **核心思路**：直接使用 $1 - \cos$ 在高维空间中有问题（随机向量余弦相似度趋近于0），因此使用单位向量的欧氏距离作为不相似度度量
- **关键公式**：$\text{Dissim}(\mathbf{v}_1, \mathbf{v}_2) = \|\hat{\mathbf{v}}_1 - \hat{\mathbf{v}}_2\|_2 = \sqrt{2 - 2\cos(\theta)}$
- **设计动机**：该度量能放大几乎对齐向量之间的微小方向差异，更适合高维空间中的Token区分

#### 2. **时空Token重要性评估**
- **空间重要性 $\mathcal{S}$**：每个Token与帧级平均表示的不相似度，衡量Token在空间维度的独特性
    - $\mathcal{S} = \text{Dissim}(F_t, \overline{F_t})$
- **时间重要性 $\mathcal{T}$**：相邻帧对应位置Token的不相似度，衡量运动变化
    - $\mathcal{T} = \text{Dissim}(\mathbf{F}_t, \mathbf{F}_{t-1})$
- **综合得分**：$\mathcal{I} = \mathcal{T} + w \cdot \mathcal{S}$，其中 $w$ 控制空间信息的融入比例
- **复杂度**：整个评估过程仅为 $O(n \cdot d)$，远低于基于聚类/$O(n^2)$ 的方法

#### 3. **信息感知自适应阈值剪枝**
- **核心思路**：利用时间重要性的L2范数量化帧间视觉变化，自动确定每帧保留率
- **阈值计算**：
    - 对于第 $t$ 帧（$t \geq 2$）：$\text{threshold}_t = \frac{\|\mathcal{T}_t\|_2}{2\sqrt{f}}$
    - 对于第1帧：$\text{threshold}_1 = \frac{\|\mathcal{S}_1\|_2}{2\sqrt{f}}$
- **效果**：高运动序列自动增加Token保留率，静态帧激进剪枝

#### 4. **视觉信息退化假说与退化感知剪枝（Memory SharpV）**
- **关键观察**：视觉Token在浅层与原始特征保持高余弦相似度，深层急剧下降后趋于稳定（<0.2），类似人类记忆曲线；系统和指令Token则迅速收敛到近零
- **理论解释**：与信息瓶颈原理一致——网络在连续变换中丢弃不相关细节，保留任务相关特征
- **剪枝策略**：当某层视觉Token与原始视觉信息的余弦相似度低于阈值 $M$ 时，丢弃该层KV Cache
    - $\text{Discard}(l) = \text{True}, \text{if } \cos(\mathbf{V}_l, \mathbf{V}) < M$
- **优势**：完全不依赖注意力分数，与Flash Attention完全兼容

### 训练策略
- 无需训练，即插即用
- 超参数：$M=0.2$，$w=1$，手动模式 $K=1.6$

## 实验关键数据

### 主实验

| 模型/方法 | Token Budget | MVBench | VideoMME(wo/mc) | NextQA | ActNet-QA | Avg. | TTFT加速 |
|-----------|-------------|---------|-----------------|--------|-----------|------|---------|
| LLaVA-OV-7B (Dense) | 100% | 57.7 | 58.7/79.1 | 51.9 | 2.86 | 61.9 | 1.00× |
| + FastV | 30% | 56.3 | 56.4/76.5 | 50.7 | 2.80 | 60.0 | 1.30× |
| + DyCoke | 19% | 57.7 | 59.3/78.4 | 52.1 | 2.88 | 61.9 | 1.36× |
| + SharpV (Adaptive) | 12% | **58.2** | **60.0/78.8** | 51.9 | 2.86 | **62.2** | **1.64×** |
| LLaVA-OV-0.5B (Dense) | 100% | 46.6 | 45.9/57.5 | 47.9 | 2.66 | 49.5 | 1.00× |
| + SharpV (Adaptive) | 12% | 46.6 | 46.9/57.7 | 47.8 | 2.65 | 49.8 | 1.65× |

### 消融实验

| 配置 | MVBench | VideoMME | 说明 |
|------|---------|---------|------|
| Visual SharpV | 48.0/43.6 (PLLaVA) | 52.1/42.2 | 完整时空评分 |
| V-Random† (保留比例同SharpV) | 46.3/42.0 | 50.3/40.4 | 随机选择证明评分有效 |
| V-Random* (随机比例) | 45.0/41.3 | 49.4/40.2 | 无自适应比例显著下降 |
| LLaVA-OV-7B Visual SharpV | 58.2/59.5 | 70.8/57.4 | **超过Dense的57.7/58.7** |

### 关键发现
1. SharpV以约12%的Token保留率在多个基准上**偶尔超过稠密模型**1-2%，说明适当剪枝可以减少视频噪声
2. 在0.5B小模型上同样有效，证明可扩展性
3. Memory SharpV中约54%的视觉Token层相似度低于0.1，验证了视觉信息退化假说
4. 参数 $w=1$ 和 $M \leq 0.2$ 时性能最优且稳定

## 亮点与洞察
1. **首个完全兼容Flash Attention的两阶段剪枝框架**：不依赖注意力分数，从信息论角度提供全新的Intra-LLM剪枝范式
2. **自适应剪枝偶尔超越稠密模型**：说明"越多Token越好"的假设不成立，适当去噪反而有益
3. **视觉信息退化假说**：揭示了VideoLLM中跨模态信息流的规律——LLM主要在浅层处理视觉信息
4. **极低复杂度**：时空评分和相似度计算均为 $O(n \cdot d)$，剪枝过程本身几乎不引入额外开销

## 局限与展望
1. 对细微视觉细节（如微表情）可能存在信息损失，可探索更细粒度的Token剪枝
2. 视觉退化现象缺乏严格的理论框架，深层Token变换机制有待进一步研究
3. 超参数（$w$, $M$）虽然鲁棒，但缺乏自动搜索机制
4. 未验证在更长视频（>100帧）上的效果

## 相关工作与启发
- DivPrune/LLaVA-Scissor等高复杂度方法的教训：剪枝方法自身的计算开销不可忽视
- 信息瓶颈理论在多模态LLM中的应用前景广阔
- Flash Attention兼容性是未来VLLM效率方法的必要条件

## 评分
- 新颖性: ⭐⭐⭐⭐ — 视觉退化假说和自适应剪枝超越稠密模型是亮点
- 实验充分度: ⭐⭐⭐⭐⭐ — 多模型、多基准、详细消融
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机论证充分
- 价值: ⭐⭐⭐⭐ — Flash Attention兼容性有很强的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression](infocom_kilobyte-scale_communication-efficient_collaborative_perception_with_inf.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[AAAI 2026\] Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)

</div>

<!-- RELATED:END -->
