---
title: >-
  [论文解读] From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit
description: >-
  [NeurIPS 2025][可解释性][稀疏自编码器] 提出 MP-SAE，将经典 Matching Pursuit 算法展开为 SAE 的序列化编码器，通过残差引导的贪心特征选择实现条件正交性，能捕捉标准 SAE 无法发现的层次结构、非线性可及和跨模态特征，并天然支持推理时自适应稀疏度调节。 领域现状：SAE（稀疏自编码…
tags:
  - "NeurIPS 2025"
  - "可解释性"
  - "稀疏自编码器"
  - "Matching Pursuit"
  - "层次表示"
  - "条件正交"
---

# From Flat to Hierarchical: Extracting Sparse Representations with Matching Pursuit

**会议**: NeurIPS 2025  
**arXiv**: [2506.03093](https://arxiv.org/abs/2506.03093)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 稀疏自编码器, Matching Pursuit, 层次表示, 可解释性, 条件正交

## 一句话总结
提出 MP-SAE，将经典 Matching Pursuit 算法展开为 SAE 的序列化编码器，通过残差引导的贪心特征选择实现条件正交性，能捕捉标准 SAE 无法发现的层次结构、非线性可及和跨模态特征，并天然支持推理时自适应稀疏度调节。

## 研究背景与动机

**领域现状**：SAE（稀疏自编码器）基于线性表示假说（LRH），已成为神经网络可解释性研究的主流工具。LRH 认为表示可以分解为大量近似正交的方向，每个方向对应一个可解释概念，SAE 通过学习超完备稀疏字典来恢复这些方向。

**现有痛点**：近期研究发现 LRH 不能完全解释真实表示——(1) 层次概念（如动物→哺乳类→猫）的父子概念跨越正交子空间而非全局近似正交；(2) "洋葱状"非线性表示无法通过单次线性投射访问；(3) 多维概念（如星期几）不能用单一方向表示。标准 SAE（包括带层次目标的 Matryoshka SAE）在这些结构上表现不佳。

**核心矛盾**：SAE 的设计假设（全局准正交 + 线性可及）与神经网络中实际存在的层次、非线性表示结构不匹配。

**本文目标**：(1) 验证标准 SAE 是否能捕捉超出 LRH 范围的结构；(2) 设计一种归纳偏好与层次/非线性结构匹配的 SAE 架构。

**切入角度**：引入"条件正交"——要求不同层级间的概念正交（父-子正交），同一层级内允许干扰，这与标准 LRH 的全局准正交本质不同。经典 Matching Pursuit 算法天然具有逐步正交化的特性，每步选择与当前残差最相关的特征。

**核心 idea**：将 MP 的贪心残差分解展开为 SAE 编码器，使其归纳偏好与条件正交和非线性概念结构匹配。

## 方法详解

### 整体框架
MP-SAE 共享编码和解码字典 $\bm{D}$。编码过程展开 $T$ 步 Matching Pursuit：初始化残差 $\bm{r}^{(0)} = \bm{x} - \bm{b}_{\text{pre}}$，每步选择与残差最相关的特征方向，减去其贡献更新残差，重复 $T$ 步得到稀疏码 $\bm{z}$（$\|\bm{z}\|_0 \leq T$）和重构 $\hat{\bm{x}}$。字典通过反向传播联合学习。

### 关键设计

1. **MP 展开编码器**:

    - 功能：将 MP 贪心推理过程展开为可微分的编码器
    - 核心思路：每步 $t$：选择最大投影方向 $j^{(t)} = \arg\max_j (\bm{D}^\top \bm{r}^{(t)})_j$，计算系数 $z_{j^{(t)}}^{(t)} = \bm{D}_{j^{(t)}}^\top \bm{r}^{(t)}$，更新重构 $\hat{\bm{x}}^{(t+1)} = \hat{\bm{x}}^{(t)} + z_{j^{(t)}}^{(t)} \bm{D}_{j^{(t)}}$，更新残差 $\bm{r}^{(t+1)} = \bm{r}^{(t)} - z_{j^{(t)}}^{(t)} \bm{D}_{j^{(t)}}$
    - 设计动机：标准 SAE 用单次线性投射编码，无法感知特征间的依赖关系；MP 的序列化分解天然地让后续特征解释前面未解释的部分

2. **逐步正交性保证条件正交**:

    - 功能：确保每步选择的特征与上一步正交
    - 核心思路：由残差更新规则直接保证 $\bm{D}_{j^{(t-1)}}^\top \bm{r}^{(t)} = 0$——被选中的特征从残差子空间中移除，后续选择只在正交补空间中进行
    - 设计动机：这精确对应条件正交的定义——跨层级（跨步骤）正交，层级内（同步骤候选）允许干扰。虽然 MP 只保证与最近一步正交（不像 OMP 对所有已选正交化），但实验中残差经验上近似与所有已选方向正交

3. **访问非线性可及特征**:

    - 功能：通过残差迭代实现从原始输入出发的非线性特征提取
    - 核心思路：分解 $\bm{x} = \underbrace{\bm{\varphi}(\bm{x})}_{\text{线性可及}} + \underbrace{\sum_{t=1}^T \bm{\varphi}(\bm{r}^{(t)})}_{\text{非线性可及}} + \bm{r}^{(T+1)}$。虽然每步 $\bm{\varphi}(\cdot)$ 是线性投射，但 $\bm{r}^{(t)}$ 是 $\bm{x}$ 的非线性函数，因此组合 $\bm{\varphi}(\bm{r}^{(t)})$ 构成非线性特征
    - 设计动机：为"暗物质"现象（标准 SAE 无法解释的表示部分）提供构造性解释——这些特征不是不存在，而是需要非线性访问

### 损失函数 / 训练策略
训练目标：$\mathcal{L} = \|\bm{x} - \hat{\bm{x}}\|_2^2 + \lambda \mathcal{R}(\bm{z}) + \alpha \mathcal{L}_{\text{aux}}$。使用 Adam 优化器，学习率 $5 \times 10^{-4}$，cosine 衰减至 $10^{-6}$，50 epoch。在 ImageNet-1K 上用冻结骨干最后一层表示训练，扩展因子 $p = 25m$。

## 实验关键数据

### 主实验（表达能力 R² vs 稀疏度 Pareto 前沿）

| 模型/SAE | SigLIP R²@k=32 | DINOv2 R²@k=32 | CLIP R²@k=32 |
|----------|---------------|----------------|--------------|
| Vanilla (ReLU) | ~0.65 | ~0.55 | ~0.60 |
| BatchTopK | ~0.70 | ~0.60 | ~0.65 |
| MP-SAE | ~0.78 | ~0.70 | ~0.72 |

MP-SAE 在所有测试骨干（SigLIP, DINOv2, CLIP, ViT）上，在可比稀疏度下均达到更高 R²。

### 合成实验（条件正交恢复）

| SAE | Flat MSE↓ | Hierarchical MSE↓ | 说明 |
|-----|----------|-------------------|------|
| Vanilla | 低 | 高 | 保持层内结构但丢失层级分离 |
| BatchTopK | 低 | 高 | 同上，受特征吸收影响 |
| Matryoshka | 高 | 低 | 保持层级但引入层内负干扰 |
| MP-SAE | **低** | **低** | 同时保持层内+层级结构 |

### 关键发现
- **特征吸收问题**：Vanilla 和 BatchTopK 将子概念方向与父概念对齐，导致层级结构坍塌——这是标准 SAE 的根本缺陷
- **Matryoshka 的权衡**：保持了层级分离但引入了兄弟概念间的负干扰，说明显式层级目标也无法完美解决问题
- **有效秩持续增长**：随稀疏度 $k$ 增加，MP-SAE 的共激活矩阵有效秩持续增长，而标准 SAE 快速饱和——说明 MP-SAE 发现了更多样化的特征组合
- **推理时条件正交**：MP-SAE 的字典全局 Babel 分数更高（更多干扰），但推理时实际选择的特征子集 Babel 分数更低——推理时条件正交自然涌现
- **跨模态特征恢复**：在 CLIP 联合嵌入空间上，标准 SAE 学到的特征呈双峰模态分数分布（要么只响应图像要么只响应文本），MP-SAE 则能恢复真正的跨模态特征（模态分数在中间范围有大量质量）
- 推理时自适应稀疏度：MP-SAE 是唯一在改变 k 时重构误差单调递减的架构，TopK SAE 在 k 偏离训练值时可能退化

## 亮点与洞察
- **从现象学出发设计方法**：论文核心论点是"可解释性应从表示的现象学出发，方法应跟随假设"而非反过来——这是很有深度的方法论主张
- **条件正交的形式化**：从 Park et al. 的观察提炼出条件正交定义，将 LRH 的全局准正交放松为跨层级正交+层内可干扰，既有理论基础又有实际动机
- **MP 的妙用**：经典 MP 算法在稀疏编码中已有数十年历史，将其重新定位为 SAE 编码器是巧妙的旧酒新瓶，每步正交化天然匹配条件正交需求
- **暗物质的构造性解释**：将非线性可及特征分解为 $\bm{\varphi}(\bm{r}^{(t)})$ 的形式，为"标准 SAE 解释不了什么"提供了数学框架
- 跨模态特征发现的实用价值——可用于检验 VLM 中视觉和文本嵌入是否真正对齐

## 局限与展望
- MP 是贪心算法，缺乏全局最优性保证，在极端噪声下可能脆弱
- 条件正交假设可能不适用于扁平或纠缠的表示空间
- 计算成本随步数 $T$ 线性增长，$T$ 较大时推理速度可能成为瓶颈
- 实验主要在视觉模型（DINOv2, CLIP, SigLIP）上验证，语言模型上仅有初步结果
- 只验证了简单的二级层次结构（合成实验），更深层次/更复杂的语义层次有待探索

## 相关工作与启发
- **vs Vanilla/TopK SAE**: 标准 SAE 强制全局准正交，推理时一次线性投射完成编码；MP-SAE 允许字典有干扰但推理时选择条件正交的子集，更灵活
- **vs Matryoshka SAE**: Matryoshka 通过嵌套训练目标显式建模层级，但仍用线性编码器，无法避免层内负干扰；MP-SAE 通过残差机制更自然地实现层级分离
- **vs 正交匹配追踪 (OMP)**: OMP 对所有已选特征重正交化，理论保证更强但计算更贵；MP-SAE 只对上一步正交化但实证效果已足够好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 条件正交的形式化和 MP 展开为 SAE 编码器都是新颖且深刻的贡献
- 实验充分度: ⭐⭐⭐⭐ 合成+真实模型验证全面，跨模态分析尤其有说服力；但语言模型实验较少
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，动机线索清晰，图表精心设计
- 价值: ⭐⭐⭐⭐⭐ 对可解释性领域有根本性贡献——挑战了 LRH 的充分性并提供了构造性替代方案

## 与相关工作的对比
- **vs Vanilla/TopK SAE**: 标准 SAE 强制全局准正交 + 单次线性编码，无法区分层级内和层级间的结构差异，导致特征吸收（子概念被父概念吞并）；MP-SAE 通过序列化残差分解天然避免此问题
- **vs Matryoshka SAE**: Matryoshka 通过嵌套训练目标显式建模层级，但仍用线性编码器，引入兄弟概念间的负干扰；MP-SAE 的残差机制更自然地实现层级分离而不损害层内结构
- **vs OMP (正交匹配追踪)**: OMP 对所有已选特征重正交化，理论保证更强但计算成本 $O(Tk^2)$；MP-SAE 只对上一步正交化（$O(Tk)$），实证效果已足够好且更适合端到端训练
- **vs JumpReLU/Gated SAE**: 这些改进的激活函数仍在线性编码器框架内，无法访问非线性可及特征；MP-SAE 的残差迭代本质上构建了非线性编码路径

## 启发与关联
- 条件正交的形式化为理解 LLM 内部层次概念组织提供了分析框架——可用于研究 LLM 中语义层次（如"动物→哺乳类→猫"）的编码方式
- MP-SAE 的跨模态特征发现能力可用于检验 VLM 中视觉和文本嵌入是否真正对齐（而非表面上的余弦相似度）
- 自适应稀疏度特性使 MP-SAE 适合需要动态调节解释粒度的应用——用少量步骤获取粗粒度解释，增加步骤获取精细解释
- "暗物质"的构造性解释为 SAE 社区指出了明确的改进方向——标准 SAE 未解释的部分并非噪声，而是需要非线性访问的有意义特征

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Hierarchical Concept Embedding & Pursuit for Interpretable Image Classification](../../CVPR2026/interpretability/hierarchical_concept_embedding_pursuit_for_interpretable_image_classification.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] SHAP Values via Sparse Fourier Representation](shap_values_via_sparse_fourier_representation.md)
- [\[NeurIPS 2025\] How Intrinsic Motivation Shapes Learned Representations in Decision Transformers: A Cognitive Interpretability Analysis](toward_explainable_offline_rl_analyzing_representations_in_intrinsically_motivat.md)
- [\[ACL 2026\] AdaptiveK: Complexity-Driven Sparse Autoencoders for Interpretable Language Model Representations](../../ACL2026/interpretability/adaptivek_complexity-driven_sparse_autoencoders_for_interpretable_language_model.md)

</div>

<!-- RELATED:END -->
