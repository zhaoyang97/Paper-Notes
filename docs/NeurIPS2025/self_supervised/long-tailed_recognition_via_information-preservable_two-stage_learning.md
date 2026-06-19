---
title: >-
  [论文解读] Long-Tailed Recognition via Information-Preservable Two-Stage Learning
description: >-
  [NeurIPS 2025 Spotlight][自监督学习][长尾识别] 提出信息保持的两阶段学习框架：第一阶段用 Balanced Negative Sampling (BNS) 基于互信息最大化学习有效且可分的特征空间，第二阶段用 Information-Preservable DPP (IP-DPP) 采样数学上信息量最大的样本来纠正多数类偏向的决策边界，在多个长尾数据集上取得 SOTA。
tags:
  - "NeurIPS 2025 Spotlight"
  - "自监督学习"
  - "长尾识别"
  - "对比学习"
  - "信息论"
  - "行列式点过程"
  - "两阶段学习"
  - "表征学习"
---

# Long-Tailed Recognition via Information-Preservable Two-Stage Learning

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.08836](https://arxiv.org/abs/2510.08836)  
**代码**: [github.com/fudong03/BNS_IPDPP](https://github.com/fudong03/BNS_IPDPP)  
**作者**: Fudong Lin, Xu Yuan (University of Delaware)
**领域**: 自监督  
**关键词**: 长尾识别, 对比学习, 信息论, 行列式点过程, 两阶段学习, 表征学习

## 一句话总结

提出信息保持的两阶段学习框架：第一阶段用 Balanced Negative Sampling (BNS) 基于互信息最大化学习有效且可分的特征空间，第二阶段用 Information-Preservable DPP (IP-DPP) 采样数学上信息量最大的样本来纠正多数类偏向的决策边界，在多个长尾数据集上取得 SOTA。

## 研究背景与动机

长尾分布是现实数据的常态，头部类别主导训练过程使得深度网络决策边界偏向多数类，尾部类别性能严重下降。现有方法分为两类：

**一阶段方法**（re-weighting、过采样/欠采样）：受限于表征学习能力，难以在头尾类之间取得平衡

**两阶段方法**（解耦表征学习与分类）：已有工作存在两个关键瓶颈：
   - **第一阶段**：现有对比学习方法（KCL、TSC、SBCL）难以学到 head/tail 类之间清晰可分的特征空间
   - **第二阶段**：过采样易 mode collapse、欠采样导致严重信息丢失

本文从信息论角度出发，分别针对这两个阶段提出新方案。

## 方法详解

### 整体框架

两阶段流程：Stage 1 用 BNS 训练特征提取器 → Stage 2 在 IP-DPP 采样的平衡子集上微调分类器。

### Stage 1: Balanced Negative Sampling (BNS)

**核心思想**：最大化同一数据的两个增强视图之间的互信息 $MI(\boldsymbol{Q}, \boldsymbol{V})$，并证明这等价于最小化类内距离。

- 给定图像 $\boldsymbol{x}$，通过数据增强生成两个视图 $\boldsymbol{x}_i \in \mathbb{X}_Q$ 和 $\boldsymbol{x}_j \in \mathbb{X}_V$
- 借鉴 Noise Contrastive Estimation (NCE)，构建正负样本对进行对比学习
- 基础负采样损失 $\mathcal{L}_{NS}$ 可以缓解"标签偏差"但无法学到分离良好的空间

**BNS 的创新**：对给定锚点图像 $\boldsymbol{x}_i$，额外采样 $m$ 个同类图像 $\{\boldsymbol{x}_k\}_{k=1}^m$，形成 $(m+1)$ 个正对和 $n(m+1)$ 个负对：

$$\mathcal{L}_{BNS} = -\frac{1}{m+1}\left[\sum_{q_* \in \{q_i\} \cup \boldsymbol{Q}_{i,m}^+} \log\sigma\left(\frac{\boldsymbol{q}_*^\top \boldsymbol{v}_{j,i}^+}{\tau}\right) + \sum_{q_* \in \{q_i\} \cup \boldsymbol{Q}_{i,m}^+} \sum_{j=1}^n \log\sigma\left(-\frac{\boldsymbol{q}_*^\top \boldsymbol{v}_j^-}{\tau}\right)\right]$$

**双层语义分解**：

- **实例级语义**：锚点 $\boldsymbol{q}_i$ 与其正对 $\boldsymbol{v}_{j,i}^+$ 来自同一实例，保证高质量特征
- **类级语义**：附加同类样本 $\boldsymbol{q}_k$ 与正对 $\boldsymbol{v}_{j,i}^+$ 来自同一类别，促进类间分离

**定理 4.1（类内距离-互信息定理）**：$\max MI(\boldsymbol{Q}^c, \boldsymbol{V}^c) \propto \min D(\boldsymbol{Q}^c, \boldsymbol{V}^c)$，即最大化互信息等价于最小化类内距离。

### Stage 2: Information-Preservable DPP (IP-DPP)

**目标**：从多数类中采样平衡子集来纠正决策边界，同时保持信息量。

- 基于行列式点过程 (DPP)，构造对称随机矩阵 $\boldsymbol{S}$，其中 $\boldsymbol{S}_{i,j} = \frac{p(i)p(j)}{N}$（$i \neq j$），$p(i) = p_\phi(y_i|\boldsymbol{x}_i)$ 为正确分类的概率
- 证明 $\boldsymbol{S}$ 是半正定的且特征值在 $[0,1]$ 区间（Lemma 4.4, 4.5），满足 DPP 核矩阵条件
- **信息保持性**（Remark 4.7）：$\mathcal{P}_S(\mathbb{Y} \cup \{\boldsymbol{x}\}) \propto I(\boldsymbol{x})$，即分类置信度低（信息内容高）的样本更可能被采样
- 标准 DPP 期望采样 $N(1-\ln 2)$ 个样本（约三分之一），不足以平衡类先验，因此引入固定基数 $k$-DPP，从每个多数类采样固定 $k$ 个实例
- 基于谱分解设计高效采样算法（Algorithm 1）

## 实验关键数据

### 表1: CIFAR-10/100-LT 结果（IF=100）

| 方法 | CIFAR-10-LT Overall | CIFAR-100-LT Overall |
|------|:---:|:---:|
| Focal Loss | 69.2 | 43.5 |
| LDAM Loss | 71.5 | 44.1 |
| RIDE | 73.4 | 47.2 |
| SBCL | 72.6 | 48.5 |
| OTmix | 73.8 | 48.1 |
| DisA | 73.6 | 49.2 |
| **Ours** | **76.4** | **52.4** |

- CIFAR-10-LT 上总体准确率 76.4%，超出次优至少 2.6%
- CIFAR-100-LT 上总体准确率 52.4%，超出 DisA 3.2%
- 虽然 many-shot 不如 OTmix，但 medium-shot 和 few-shot 大幅领先（few-shot 提高 19.9% / 12.8%）

### 表2: ImageNet-LT 与 iNaturalist 2018

| 方法 | ImageNet-LT Overall | iNaturalist 2018 Overall |
|------|:---:|:---:|
| DisA | 49.4 | 69.8 |
| SBCL | 47.1 | 70.4 |
| **Ours** | **51.7** | **74.0** |

- ImageNet-LT 上超过 DisA 2.3%，iNaturalist 上超过 SBCL 3.6%
- 大规模数据集上同样保持 SOTA，证明泛化能力

### 不同不平衡因子 (IF) 的鲁棒性

- IF 从 10 增加到 200 时，本方法在 CIFAR-10-LT 上仅下降 10.2%（83.7%→73.5%），而 DisA 下降 15.2%
- 在所有 IF 值（10/20/50/100/200）上均取得最高总体准确率

### 表征学习评估

- 线性探测准确率：CIFAR-10-LT 上 68.2%，分别超 KCL/TSC/SBCL 7.2%/4.4%/3.5%
- BNS 的 many-shot 与 few-shot 差距仅 1.8%（69.4% vs 67.6%），SBCL 差距为 43.5%
- t-SNE 可视化清晰显示 BNS 学到的特征空间类间分离度远优于 SBCL

## 亮点

1. **理论扎实**：从信息论出发，严格证明互信息最大化等价于类内距离最小化（Theorem 4.1），DPP 采样的信息保持性有理论保证
2. **同时解决两阶段的瓶颈**：BNS 通过双层语义（实例级+类级）解决特征空间不可分问题，IP-DPP 通过信息含量优先采样解决欠采样信息丢失问题
3. **尾部提升显著**：在不大幅牺牲头部性能的前提下，few-shot 准确率提升极为显著（CIFAR-10 上比 OTmix 高 19.9%）
4. **鲁棒性强**：在极端不平衡（IF=200）下性能下降最小，且在小/大规模数据集上均 SOTA

## 局限与展望

1. **Many-shot 准确率有所牺牲**：方法在头部类别上不如部分baseline，存在以头换尾的 trade-off
2. **计算开销**：IP-DPP 需要谱分解对称随机矩阵，大规模数据集上的计算成本值得关注
3. **超参数依赖**：BNS 中的 $m$（同类采样数）和 IP-DPP 中的 $k$（固定采样数）需要调整，尤其 $m$ 受尾部类别样本数限制
4. **仅评估分类任务**：未在检测、分割等下游任务上验证，泛化性有待进一步测试
5. **与最新 foundation model 的结合**：未探索基于预训练大模型（如 ViT-L/CLIP）的长尾微调场景

## 与相关工作的对比

| 类别 | 代表方法 | 本文优势 |
|------|---------|---------|
| 重加权 | Focal Loss, LDAM | 本文通过两阶段解耦避免了过度依赖损失函数调权 |
| 过采样 | SMOTE, OTmix | IP-DPP 欠采样但保持信息，避免过采样的 mode collapse |
| 欠采样 | Random, Informed | IP-DPP 基于信息论优先保留高信息量样本，减少信息丢失 |
| 对比学习 | KCL, TSC, SBCL | BNS 同时捕获实例级+类级语义，特征空间分离度更好 |
| 最新SOTA | DisA (ICML'24) | 整体准确率一致超越，尾部提升尤为明显 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — BNS 的互信息视角与 IP-DPP 的信息保持采样均有新意，理论证明充分
- 实验充分度: ⭐⭐⭐⭐ — 4 数据集 × 9 baseline，含消融、不同 IF、线性探测、t-SNE 可视化
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，结构完整，理论与实验对应良好
- 价值: ⭐⭐⭐⭐ — 长尾识别领域的一致性 SOTA，理论与实践结合紧密

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition](../../AAAI2026/self_supervised/bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)
- [\[CVPR 2026\] Trust-calibrated Collaborative Learning for Long-Tailed Visual Recognition](../../CVPR2026/self_supervised/trust-calibrated_collaborative_learning_for_long-tailed_visual_recognition.md)
- [\[CVPR 2026\] Decision Boundary-aware Generation for Long-tailed Learning](../../CVPR2026/self_supervised/decision_boundary-aware_generation_for_long-tailed_learning.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](../../ICLR2026/self_supervised/maximizing_incremental_information_entropy_for_contrastive_learning.md)

</div>

<!-- RELATED:END -->
