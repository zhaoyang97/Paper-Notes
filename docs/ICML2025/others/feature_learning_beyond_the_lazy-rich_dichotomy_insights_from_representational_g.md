---
title: >-
  [论文解读] Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry
description: >-
  [ICML2025 (Spotlight)][feature learning] 提出用**流形容量 (manifold capacity)** 及其关联的几何度量 (GLUE) 来刻画特征学习的丰富程度，超越传统的 lazy vs rich 二分法，揭示了不同学习阶段、学习策略以及在神经科学和 OOD 泛化问题中的新洞察。
tags:
  - "ICML2025 (Spotlight)"
  - "feature learning"
  - "lazy-rich regime"
  - "manifold capacity"
  - "representational geometry"
  - "GLUE"
---

# Feature Learning beyond the Lazy-Rich Dichotomy: Insights from Representational Geometry

**会议**: ICML2025 (Spotlight)  
**arXiv**: [2503.18114](https://arxiv.org/abs/2503.18114)  
**代码**: [GitHub](https://github.com/chung-neuroai-lab/feature-learning-geometry)  
**领域**: 特征学习理论  
**关键词**: feature learning, lazy-rich regime, manifold capacity, representational geometry, GLUE

## 一句话总结

提出用**流形容量 (manifold capacity)** 及其关联的几何度量 (GLUE) 来刻画特征学习的丰富程度，超越传统的 lazy vs rich 二分法，揭示了不同学习阶段、学习策略以及在神经科学和 OOD 泛化问题中的新洞察。

## 研究背景与动机

### Lazy vs Rich 二分法的局限

- 现有理论将神经网络学习分为 **lazy regime**（权重几乎不变，相当于随机特征模型）和 **rich regime**（主动学习任务相关特征）
- 这一二分法过于粗糙：rich regime 内部也存在巨大差异——不同架构、初始化、学习率会导致截然不同的特征学习机制，但都被笼统归为"rich"
- 传统度量方法（权重变化量、NTK-label alignment、representation-label alignment）各有缺陷：
    - 权重变化：仅衡量变化大小，无法量化学到的任务相关特征多少
    - NTK/representation-label alignment：在某些设定下会给出错误的排序
    - 均不是纯粹基于表征的方法，不适用于神经科学中无法精确追踪突触权重变化的场景

### 核心问题

1. 如何用一个**基于表征的度量**来量化特征学习的丰富程度？
2. Rich regime 内部是否存在**子类型** (subtypes)？
3. 该框架能否为神经科学和机器学习中的开放问题提供新见解？

## 方法详解

### 1. 任务相关流形 (Task-Relevant Manifolds)

- 对于分类任务：第 $i$ 类的流形 $\mathcal{M}_i = \text{conv}(\{\Phi(x) : x \in \mathcal{X}_i\})$，即该类所有输入在某层的神经表征的凸包
- 关键思路：**特征学习 = 流形解缠 (manifold untangling)**——学习使任务相关流形在表征空间中更容易分离

### 2. 流形容量 (Manifold Capacity) $\alpha_M$

- **直觉**：衡量在给定维度的表征空间中能"打包"多少个可线性分离的流形
- 模拟容量的定义：对随机二分法 $\mathbf{y} \in \{±1\}^P$ 和随机投影 $\Pi_n$，估计线性分离成功的概率 $p_n$，然后
$$\alpha_{\text{sim}} = \frac{P}{\sum_{n \in [N]}(1 - p_n)}$$
- 实际使用**均场版本** $\alpha_M$，可通过求解二次规划高效计算，且与模拟版本的误差为 $O(1/N)$
- 核心性质：**容量越高 → 流形越解缠 → 特征学习越丰富**
- 近似公式：$\alpha_M \approx (1 + R_M^{-2}) / D_M$，其中 $R_M$ 为流形半径，$D_M$ 为流形维度

### 3. GLUE：几何度量族

GLUE (Geometry Linked to Untangling Efficiency) 将容量分解为若干可解释的几何度量：

| 度量 | 含义 | 对容量的影响 |
|------|------|-------------|
| **流形维度** $D_M$ | 流形内部变化的自由度（类似 Gaussian width） | 降维 → 容量↑ |
| **流形半径** $R_M$ | 噪声-信号比（类内变化/类中心范数） | 半径缩小 → 容量↑ |
| **中心对齐** $\rho_M^c$ | 不同流形中心的相关性 | 降低 → 容量↑ |
| **轴对齐** $\rho_M^a$ | 不同流形变化方向的相关性 | 降低 → 容量↑ |
| **中心-轴对齐** $\psi_M$ | 流形中心与其他流形变化方向的相关性 | 关系更复杂 |

### 4. 理论保证 (Theorem 3.1)

在两层非线性网络 + teacher-student 设定下，证明了：

1. **容量追踪丰富度**：在比例渐近极限下，容量 $\alpha(\eta, \psi_1, \psi_2)$ 关于学习率 $\eta$ 严格单调递增
2. **容量连接预测精度**：存在单调递增可逆函数 $h$，使得 $\text{Acc}(\eta) = h(\alpha(\eta))$

这从理论上严格证明了流形容量确实能量化特征学习程度。

## 实验关键数据

### 实验一：与传统度量的对比 (2层NN + 合成数据)

- 通过逆缩放因子 $\bar{\eta}$ 在 lazy (小 $\bar{\eta}$) 和 rich (大 $\bar{\eta}$) 之间插值
- **容量**能准确区分不同 $\bar{\eta}$ 对应的丰富程度，而 NTK-label alignment 和 representation-label alignment 在某些设定下给出错误排序
- 容量还能检测**初始化时**任务相关特征的多少（wealthy vs poor regime），这是权重变化等方法做不到的

### 实验二：学习策略的差异 (Section 4.1)

- 在半径-维度等值线图上追踪训练轨迹，发现不同丰富度对应不同策略：
    - Lazy → 中等 rich：同时压缩半径和维度
    - 中等 rich → 极 rich：**牺牲半径以进一步压缩维度**
- 不同初始化财富度也导致不同策略：wealthy 初始化主要压缩半径；poor 初始化需要同时操作两者

### 实验三：学习阶段 (Section 4.2)

VGG-11 在 CIFAR-10 上训练，尽管训练 / 测试精度快速饱和，流形几何仍揭示至少四个阶段：

1. **Clustering 阶段**：流形初步压缩
2. **Structuring 阶段**：对齐度增加
3. **Separating 阶段**：对齐度降低，流形互相推开
4. **Stabilizing 阶段**：中心对齐进一步降低

### 实验四：RNN 中的结构感应偏置 (Section 5.1)

- 不同初始权重秩的 RNN 训练后容量值趋同，但几何组织大不相同
- 低秩初始化（poorer-richer）→ 大半径 + 小维度
- 高秩初始化（wealthier-lazier）→ 小半径 + 大维度
- 说明存在**流形几何层面的结构偏置**

### 实验五：OOD 泛化 (Section 5.2)

- VGG-11 / ResNet-18 在 CIFAR-10 预训练，用 CIFAR-100 线性探测
- 中等 rich 最佳；**ultra-rich regime 下 OOD 精度剧烈下降**
- 几何解释：ultra-rich 时流形半径膨胀 + 中心-轴对齐增加 → 容量下降
- ResNet-18 中则是维度增加导致容量下降，体现架构差异

## 亮点与洞察

1. **超越二分法**：首次系统性地用表征几何的视角将 feature learning 分解为多种子类型（学习策略 × 学习阶段），而非简单的 lazy/rich
2. **理论-实验一体**：在两层网络上给出了严格的渐近理论（Theorem 3.1），同时在 VGG/ResNet/RNN 等实际架构上验证
3. **跨领域适用**：框架同时覆盖了计算神经科学（RNN 神经回路偏置）和机器学习（OOD 泛化），是表征几何方法的典范应用
4. **可操作的度量**：GLUE 族度量提供了**可解释的诊断工具**——发现容量下降后，可以具体归因到半径、维度还是对齐的变化
5. **Spotlight 论文**，说明审稿人对其原创性和影响力的认可

## 局限与展望

1. **理论仅限两层 + 一步梯度**：Theorem 3.1 只在一步梯度更新后成立，多步训练的渐近行为尚未证明（Gaussian equivalence 可能不保持）
2. **实验规模**：仅使用 VGG-11/ResNet-18 和 CIFAR-10/100，未验证在更大模型(如 transformer)、更复杂任务（NLP、大规模视觉）上的适用性
3. **凸包近似**：将流形建模为凸包在数学上等价于线性分类分析，但可能忽略高阶非线性结构
4. **计算成本**：均场容量的计算需求解二次规划，对超大规模表征的可扩展性有待验证
5. **因果性缺失**：容量追踪丰富度是相关性描述，尚未建立"操纵几何 → 改善学习"的因果干预框架

## 相关工作与启发

- **Chizat et al. (2019)**：通过缩放因子在 lazy/rich 之间插值，是本文实验设定的基础
- **Ba et al. (2022)**：两层网络一步梯度的理论分析，本文将其从回归推广到分类
- **Chung et al. (2018), Chou et al. (2025)**：流形容量理论和 GLUE 的原始提出者
- **Jacot et al. (2018)**：NTK 理论，lazy regime 的理论基础
- **Neural Collapse (Papyan et al. 2020)**：极端 rich regime 下的表征结构，可看作本框架的特例

**启发**：该框架为理解大模型训练动态提供了一种新视角——不是看 loss 曲线或 NTK 变化，而是追踪表征空间中流形的几何演化。未来可探索将 GLUE 应用于 LLM 的 concept manifold 分析。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次系统性地超越 lazy-rich 二分法，提出几何视角的特征学习分类学)
- 实验充分度: ⭐⭐⭐⭐ (理论+合成+CNN+RNN+OOD，覆盖面广，但模型规模偏小)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，图文匹配好，直觉解释充分)
- 价值: ⭐⭐⭐⭐⭐ (为表征学习理论提供了新的分析范式，跨领域适用性强)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Johnson-Lindenstrauss Lemma Beyond Euclidean Geometry](../../NeurIPS2025/others/johnson-lindenstrauss_lemma_beyond_euclidean_geometry.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[NeurIPS 2025\] On a Geometry of Interbrain Networks](../../NeurIPS2025/others/on_a_geometry_of_interbrain_networks.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)
- [\[ACL 2025\] Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](../../ACL2025/others/graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)

</div>

<!-- RELATED:END -->
