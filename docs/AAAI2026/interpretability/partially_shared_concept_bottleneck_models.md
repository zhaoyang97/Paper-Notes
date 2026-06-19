---
title: >-
  [论文解读] Partially Shared Concept Bottleneck Models
description: >-
  [AAAI 2026][可解释性][概念瓶颈模型] 提出PS-CBM框架，通过多模态概念生成（结合LLM语义与示例图像视觉线索）、部分共享概念策略（基于激活模式合并概念）和Concept-Efficient Accuracy（CEA）评估指标，在11个数据集上以更少的概念实现了更高的分类精度和可解释性。
tags:
  - "AAAI 2026"
  - "可解释性"
  - "概念瓶颈模型"
  - "视觉-语言模型"
  - "概念效率"
  - "图像分类"
---

# Partially Shared Concept Bottleneck Models

**会议**: AAAI 2026  
**arXiv**: [2511.22170](https://arxiv.org/abs/2511.22170)  
**代码**: [github.com/7494zdl/PS-CBM](https://github.com/7494zdl/PS-CBM)  
**领域**: 可解释性  
**关键词**: 概念瓶颈模型, 可解释性, 视觉-语言模型, 概念效率, 图像分类

## 一句话总结

提出PS-CBM框架，通过多模态概念生成（结合LLM语义与示例图像视觉线索）、部分共享概念策略（基于激活模式合并概念）和Concept-Efficient Accuracy（CEA）评估指标，在11个数据集上以更少的概念实现了更高的分类精度和可解释性。

## 研究背景与动机

### 问题定义

概念瓶颈模型（CBM）在输入和预测之间插入一层人类可理解的概念，以增强模型可解释性。近年来自动化概念生成（使用LLM和VLM）减轻了人工标注负担，但仍面临三个根本挑战。

### 三大核心挑战

#### 1. 视觉锚定不足（Poor Visual Grounding）
- LLM生成的概念语义丰富但常与实际视觉内容不对齐
- VLM-based方法提升了视觉保真度但牺牲了类级语义一致性且计算代价高
- 存在持续的**语义-视觉鸿沟**，削弱了精度和可解释性

#### 2. 概念冗余（Concept Redundancy）
- **独立池策略**：为每个类独立生成概念→语义重复，相似概念冗余分配给多个类
- **全局共享策略**：统一去重→压缩了冗余但迫使无关类共享固定概念池，损害类别区分力
- 两种策略都阻碍模型清晰度和训练稳定性

#### 3. 评估指标不足（Inadequate Metrics）
- 大多数CBM仅按分类精度评估，忽略了大型冗余概念集的可解释性代价
- 没有原则性指标来捕捉精度和概念效率之间的权衡
- 性能提升可能以可用性为代价

### 本文动机
设计一个统一框架，同时解决这三个问题——提出部分共享概念策略（Partially Shared），在独立和全局共享之间找到最优平衡。

## 方法详解

### 整体框架

PS-CBM包含三个阶段：
1. **多模态概念生成**：结合LLM语义和示例图像生成概念集
2. **部分共享概念策略**：基于激活模式合并概念并跨类分配
3. **CBM训练**：通过概念监督学习透明的预测模型

### 关键设计

#### 1. **多模态概念生成**

弥合LLM语义与视觉锚定的鸿沟：

- **少样本图像选择**：对每个类 $i$，用CLIP嵌入构建多样的few-shot示例集 $\bm{X}_i \subset \mathcal{X}_i$
    - 从随机图像初始化，迭代选择与已选样本余弦距离最大的新样本
    - 对噪声数据集（如Food101）采用随机采样

- **概念生成**：对每个类，将文本描述与选定示例结合构造提示，查询GPT-4o两次以减少随机性
    - 去重后得到候选概念集 $\mathcal{S} = \bigcup_{i=1}^{l} \mathcal{S}_i$
    - 每个概念 $\bm{c}_j$ 关联一个类集 $\mathcal{C}_j$

#### 2. **部分共享概念策略（三步精炼）**

核心创新，通过三步逐步精炼概念集：

**Step 1: 概念过滤（Filtering）**
- 计算图像-概念亲和度矩阵 $\bm{A}_{i,j} = \cos(\Phi(\bm{x}_i), \Psi(\bm{c}_j))$（$\Phi$为图像编码器，$\Psi$为文本编码器）
- 如果概念 $\bm{c}_j$ 与类图像的top-4平均对齐度超过置信阈值 $\tau_{\text{conf}}$，则保留

**Step 2: 概念合并（Merging）**
- 计算过滤后概念的相关矩阵 $\bm{Q}$：$\bm{Q}_{i,j} = \frac{\bm{A}_{:,i}^\top \bm{A}_{:,j}}{\|\bm{A}_{:,i}\| \|\bm{A}_{:,j}\|}$
- 贪婪合并：选择可合并概念最多的概念作为代表，合并超过阈值 $\tau_{\text{merge}}$ 的概念
- 合并后的概念继承原始概念类集的并集
- 每个类最多保留top-$K$个排他性概念（仅关联一个类的概念）

**Step 3: 概念标注（Labeling）**
- 每个图像的概念标签向量为one-hot编码：$s_{i,j} = 1$ 当且仅当 $y_i \in C_j$ 且 $\bm{A}_{i,j} > \tau_{\text{conf}}$
- 生成带概念标注的数据集 $\mathcal{D}' = \{(\bm{x}_i, \bm{s}_i, y_i)\}_{i=1}^n$

#### 3. **CEA指标（Concept-Efficient Accuracy）**

基于信息论的原则性指标：

$$\text{CEA} = \frac{\text{ACC}}{(\log_k m)^\beta}$$

其中：
- $k = \lceil \log_2 l \rceil$：区分 $l$ 个类的理论最小信息量（bits）
- $m$：使用的概念数
- $\beta \geq 0$：温度参数（较小关注精度，较大关注紧凑性）

三个优良性质：
- **最优效率**：ACC→1且m→k时CEA→1
- **自适应缩放**：以k为底的对数缩放根据任务复杂度自适应调整惩罚
- **理论基础**：与Shannon信息论对齐

### 训练策略

**概念瓶颈层（CBL）训练**：
- 冻结骨干编码器 $\bm{\phi}$，训练投影层 $\bm{g}: \mathbb{R}^d \to \mathbb{R}^{\hat{m}}$
- 损失函数：Binary Cross-Entropy
  $$\min_{\bm{g}} \mathcal{L}_{\text{CBL}} = \frac{1}{n}\sum_{i=1}^n \text{BCE}(\bm{g}(\bm{\phi}(\bm{x}_i)), \bm{s}_i)$$

**最终分类层（FCL）训练**：
- 稀疏线性分类器 $\bm{f}: \mathbb{R}^{\hat{m}} \to \mathbb{R}^l$
- 损失函数：Cross-Entropy + elastic-net正则化
  $$\min_{\bm{f}} \mathcal{L}_{\text{FCL}} = \frac{1}{n}\sum_{i=1}^n \text{CE}(\bm{f}(\hat{\bm{g}}(\bm{x}_i)), y_i) + \lambda R_\alpha(\bm{W}_F)$$
- 使用GLM-SAGA优化器求解

## 实验关键数据

### 主实验（11个数据集，CLIP_RN50骨干）

| 方法 | 平均ACC(%)↑ | 平均概念数↓ | 平均CEA(%)↑ |
|------|-----------|----------|-----------|
| LaBo | 72.8 | 7,900 | 51.6 |
| LF-CBM | 72.9 | 718 | 55.2 |
| LM4CV | 73.4 | 873 | 56.4 |
| DN-CBM | 77.3 | 8,192 | 53.4 |
| Res-CBM | 71.8 | 291 | 56.7 |
| VLG-CBM | 75.2 | 732 | 57.0 |
| V2C-CBM | 72.8 | 7,500 | 51.2 |
| DCBM | 70.9 | 2,048 | 49.5 |
| **PS-CBM** | **78.3** | **545** | **59.0** |

PS-CBM平均ACC超SOTA 1.0%~7.4%，CEA超2.0%~9.5%，同时使用的概念数仅为545（比DN-CBM少7,647个）。

### 消融实验

| 概念策略 | ACC↑ | CEA↑ | 说明 |
|---------|------|------|------|
| Independent | 较低 | 较低 | 冗余概念多 |
| Globally Shared | 中等 | 中等 | 不相关类被迫共享概念 |
| **Partially Shared** | **最高** | **最高** | 选择性共享，平衡特异性和紧凑性 |

| 置信阈值 $\tau_{\text{conf}}$ | 平均ACC(%) | 概念数 | CEA(%) |
|------|------|------|------|
| 0.10 | 76.20 | 548 | 57.41 |
| 0.15 | 76.14 | 548 | 57.36 |
| **0.20** | **78.35** | **545** | **59.02** |
| 0.25 | 72.71 | 458 | 55.16 |
| 0.30 | 57.55 | 145 | 46.84 |

排他性概念数量 $K$ 的分析：ACC在 $K=0→1$ 时急剧提升，$K \geq 2$ 后趋于稳定；CEA在 $K=1$ 时达峰，之后下降。

### CLIP Score对比（领域特定数据集）

| 方法 | DTD | Resisc45 | UCF101 | 概念生成 | 概念池 |
|------|-----|----------|--------|---------|--------|
| LaBo | 0.227 | 0.222 | 0.230 | Language | Independent |
| DN-CBM | 0.192 | 0.187 | 0.187 | Vision | Globally Shared |
| V2C-CBM | 0.246 | 0.216 | 0.247 | Vision | Independent |
| **PS-CBM** | **0.249** | **0.255** | **0.265** | Language+Vision | Partially Shared |

### 关键发现

1. **精度与概念数的脱钩**：PS-CBM用最少的概念（545）达到最高精度（78.3%），证明概念质量比数量更重要
2. **部分共享最优**：在独立和全局共享之间找到最佳平衡——选择性共享语义相似概念既减少冗余又保持区分力
3. **K=1最高效**：每个类仅需1个排他性概念即可获得最佳CEA，过多的类专属概念反而降低效率
4. **多模态概念生成优势**：结合LLM语义和视觉示例在领域特定数据集上的CLIP Score显著优于单模态方法
5. **概念-类映射的语义一致性**：CIFAR10的可视化显示共享概念正确反映了语义关系（如deer和horse共享"hooves"、"long neck"）

## 亮点与洞察

1. **部分共享的优雅设计**：基于激活模式的贪婪合并算法简洁有效——选最大可合并集、继承类集并集、限制排他性概念数
2. **CEA指标的理论基础**：基于Shannon信息论推导出的评估指标，有明确的上界（→1）和下界，比现有指标（CUE、NEC）更具理论优雅性
3. **实验广度**：11个数据集横跨通用、细粒度和领域特定任务，结论的泛化性强
4. **代码开源**：完整复现代码和配置已公开

## 局限与展望

1. **依赖CLIP编码器**：概念过滤和标注的质量受限于CLIP的对齐能力
2. **GPT-4o成本**：概念生成需要查询GPT-4o，大规模应用时成本较高
3. **超参数敏感**：$\tau_{\text{conf}}$和$\tau_{\text{merge}}$的选择对性能影响较大，需要仔细调参
4. **ImageNet采样**：由于规模大，仅使用10%训练图像进行合并，可能影响概念质量
5. **概念的动态更新**：当前框架是静态的概念集，不支持根据反馈动态调整

## 相关工作与启发

- **与LF-CBM的对比**：LF-CBM使用全局共享池但概念冗余；PS-CBM的部分共享策略则在紧凑和区分力之间取得平衡
- **与DN-CBM的对比**：DN-CBM使用稀疏自编码器发现视觉概念，精度虽高但需8192个概念；PS-CBM用545个概念即超越
- **CEA vs CUE**：CUE缺乏明确上界且对文本格式敏感；CEA是后验的、任务自适应的、训练无关的

## 评分

- 新颖性: ⭐⭐⭐⭐ — 部分共享策略和CEA指标设计合理但非颠覆性创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 11个数据集、8个baseline、多维度消融，非常充分
- 写作质量: ⭐⭐⭐⭐⭐ — 结构清晰，表格丰富，可视化优秀，Table 1的对比矩阵很直观
- 价值: ⭐⭐⭐⭐ — 对CBM领域有实质推动，CEA指标可被广泛采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Flexible Concept Bottleneck Model](flexible_concept_bottleneck_model.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)

</div>

<!-- RELATED:END -->
