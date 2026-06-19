---
title: >-
  [论文解读] Rethinking Unsupervised Outlier Detection via Multiple Thresholding
description: >-
  [ECCV 2024][自监督学习][无监督异常检测] 提出 Multi-T（多阈值）模块，通过生成两个阈值分别隔离目标数据集中的 inlier 和 outlier，利用识别出的 inlier 训练干净的正常流形、利用 outlier 进行特征去噪，从而大幅提升已有离群值评分方法的性能。 无监督异常检测的核心问题 无监督异常…
tags:
  - "ECCV 2024"
  - "自监督学习"
  - "无监督异常检测"
  - "多阈值学习"
  - "离群值评分"
  - "特征归一化"
  - "自监督信号"
---

# Rethinking Unsupervised Outlier Detection via Multiple Thresholding

**会议**: ECCV 2024  
**arXiv**: [2407.05382](https://arxiv.org/abs/2407.05382)  
**代码**: 有（[https://github.com/zhliu-uod/Multi-T](https://github.com/zhliu-uod/Multi-T)）  
**领域**: 自监督学习  
**关键词**: 无监督异常检测, 多阈值学习, 离群值评分, 特征归一化, 自监督信号

## 一句话总结

提出 Multi-T（多阈值）模块，通过生成两个阈值分别隔离目标数据集中的 inlier 和 outlier，利用识别出的 inlier 训练干净的正常流形、利用 outlier 进行特征去噪，从而大幅提升已有离群值评分方法的性能。

## 研究背景与动机

### 无监督异常检测的核心问题

无监督异常检测（UOD）的目标是在无标签数据集上为每个样本分配离群值分数。当前主流方法（如 DeepSVDD、OCSVM、LVAD 等）聚焦于学习一个判别性的评分函数 $F(\cdot)$，但忽略了一个关键步骤：**如何将连续分数转化为二值标签**。

### 现有范式的两大局限

**训练集被污染**：多数方法假设 inlier 占多数，直接在整个目标数据集上学习正常流形。但当 outlier 比例升高时，数据集均值会向 outlier 偏移（mean-shift 问题），导致学到的"正常流形"被污染。

**评分方法受限于单次推理**：现有检测器通常是非迭代的，由于缺乏预测标签，无法利用数据集的自监督信号来迭代改进自身。如果能先给数据"打标签"再回过头来优化检测，效果会好得多。

### 为什么单阈值不可行？

传统单阈值方法试图找到一个阈值 $\phi$ 来精确分离 inlier 和 outlier 的分数分布。但由于初始评分函数的不完美和 outlier 比例的影响，分数分布中必然存在**重叠区域 $A$**：

$$D = F_{\text{init}}(\mathbf{X}) = I \cup A \cup O$$

在重叠区域内，单阈值无论取在哪里都会造成误判。

### 多阈值的核心思路

与其试图精确分割，不如**放弃重叠区域**，生成两个阈值：
- $\phi_{\text{in}}$：低于此值的样本肯定是 inlier → 用于训练干净的正常流形
- $\phi_{\text{out}}$：高于此值的样本肯定是 outlier → 用于 Shell Normalization 去噪特征

这样虽然会有一些"灰色地带"的样本无法被归类，但 inlier 和 outlier 的集合都是高纯度的，用它们来增强后续的检测方法会更可靠。

## 方法详解

### 整体框架

Multi-T 是一个训练无关（training-free）的即插即用模块，工作流程为三阶段：
1. **准备**：特征提取 + 初始离群值评分函数
2. **Multi-T 模块**：生成两个阈值，分离 inlier 和 outlier
3. **集成**：将识别的 inlier 和 outlier 馈入已有检测方法，获得增强的评分函数

### 关键设计

1. **初始离群值评分函数（LVAD-S）**：

    - **功能**：为每个样本计算初始的离群值分数
    - **核心思路**：使用 Ergodic-set 归一化后的欧氏距离到数据集均值：
    $F_{\text{init}}(\mathbf{X}) = \{\text{Dist}(\text{E-norm}(\mathbf{x}_i), \text{E-norm}(\mathbf{m}_X))\}_{i=1}^n$
      其中 E-norm 的参考向量 $\mathbf{v}_E$ 是所有维度所有样本的全局标量均值
    - **设计动机**：E-norm 对 outlier 比例具有不变性（因为它用全局标量均值而非逐维均值），是一个稳健的初始评分基础

2. **阶段一：识别无污染的 Inlier（迭代 3-sigma 过滤）**：

    - **功能**：从初始分数分布中提取高纯度 inlier 集合
    - **核心思路**：
        - 将分数序列升序排列，用线性回归拟合 inlier 分布（因为 inlier 的排序分数近似线性增长）
        - 识别线性区域边界：$I = \{\hat{F}(\mathbf{x}_i) \mid i < \max_i\{g(a_i) > \hat{F}(\mathbf{x}_i)\}\}$
        - 迭代应用 3-sigma 规则过滤异常值：$\phi_{\text{out}}^b = \text{mean}(I^b) + 3 \cdot \text{std}(I^b)$
        - 移除超出阈值的样本，重复直到收敛
        - 最终 inlier 阈值：$\phi_{\text{in}} = \max(I^{b^*})$
    - **设计动机**：Shell Theory 表明高维空间中 inlier 的距离分数满足类高斯分布，3-sigma 规则可以有效过滤异常值。迭代过程逐步剥离 outlier 对均值和标准差的偏移影响

3. **阶段二：自适应 Outlier 阈值选择**：

    - **功能**：根据 outlier 比例 $\gamma$ 自适应选择合适的 outlier 阈值
    - **核心思路**：比较两种归一化方法得到的排序相似性来估计 $\gamma$：
        - Shell Normalization（S-norm）：用预测的 outlier 均值作为参考向量，对 $\gamma$ 敏感
        - Ergodic-set Normalization（E-norm）：用全局标量均值，对 $\gamma$ 无关
        - 计算两者排序索引的 Pearson 相关系数 $\rho$：
       - $\rho > 0.3$（高 $\gamma$）→ 用收敛阈值 $\phi_{\text{out}}^*$（更激进）
       - $0.1 \leq \rho \leq 0.3$（中 $\gamma$）→ 用第一轮阈值 $\phi_{\text{out}}^1$
       - $\rho < 0.1$（低 $\gamma$）→ 用全局 3-sigma $\phi_{\text{out}}^0$（最保守）
    - **设计动机**：当 $\gamma$ 高时，S-norm 表现好，两种归一化的排序高度一致，可以大胆使用更多样本作为 outlier；当 $\gamma$ 低时，S-norm 不可靠，应保守处理。这一机制使得方法在不同 outlier 比例下都能稳健工作

### 损失函数 / 训练策略

Multi-T 模块本身**不需要训练**，是纯统计方法。集成方式有两种：

**直接使用**（距离到 inlier 均值）：
$$F_{\text{Multi-T}}(\mathbf{X}) = \{\text{Dist}(\text{S-norm}(\mathbf{x}_i, \mathbf{v}'_S), \text{S-norm}(\mathbf{m}_{X'_{\text{in}}}, \mathbf{v}'_S))\}_{i=1}^n$$

**与现有方法集成**：
$$F_{M+\text{Multi-T}}(\mathbf{X}) = M.\text{fit}(\{\text{S-norm}(\mathbf{x}_i) \mid \mathbf{x}_i \in X'_{\text{in}}\}).\text{predict}(\{\text{S-norm}(\mathbf{x}_i) \mid \mathbf{x}_i \in X\})$$

即用干净 inlier 训练模型 $M$，用 Shell 归一化后的全部数据预测，两步都受益于 Multi-T 的贡献。

## 实验关键数据

### 主实验：与 SOTA 离群值检测方法的 AUC 对比

| 方法 | STL-10 (ResNet) | STL-10 (CLIP) | CIFAR-10 (CLIP) | MIT-Places (CLIP) | MNIST |
|------|:---:|:---:|:---:|:---:|:---:|
| IF | 0.836 | 0.943 | 0.891 | 0.868 | 0.776 |
| ECOD | 0.907 | 0.981 | 0.935 | 0.943 | 0.734 |
| LVAD | 0.954 | 0.968 | 0.917 | 0.919 | 0.867 |
| DeepSVDD | 0.622 | 0.597 | 0.509 | 0.549 | 0.513 |
| **Multi-T** | **0.968** | **0.989** | **0.957** | **0.974** | **0.897** |
| DeepSVDD + Multi-T | 0.925 | 0.921 | 0.819 | 0.832 | 0.732 |
| DeepSVDD 提升 | +48.7% | +54.3% | +60.9% | +51.6% | +37.9% |
| OCSVM + Multi-T | 0.957 | 0.965 | 0.916 | 0.924 | 0.863 |

Multi-T 在几乎所有数据集和特征提取器组合上达到 SOTA，且能大幅提升弱方法的性能。

### 消融实验：Multi-T 对多种方法的增强效果（STL-10）

| 方法 | 无 Multi-T | 有 Multi-T | 提升 |
|------|:---:|:---:|:---:|
| IF (ResNet) | 0.836 | 0.899 | +7.5% |
| ECOD (ResNet) | 0.907 | 0.919 | +1.3% |
| ABOD (ResNet) | 0.665 | 0.883 | +32.8% |
| PCA (ResNet) | 0.865 | 0.945 | +9.2% |
| GMM (ResNet) | 0.859 | 0.952 | +10.8% |
| IF (CLIP) | 0.943 | 0.983 | +4.2% |
| PCA (CLIP) | 0.984 | 0.994 | +1.0% |

Multi-T 对所有方法（统计型和深度学习型）都有提升，弱方法获益更大。

### 阈值学习质量评估（STL-10）

| 离群值评分函数 | 阈值方法 | $F_{0.1}$ | $F_{10}$ | 平均 |
|------|------|:---:|:---:|:---:|
| LVAD-S | 最高 $F_{0.1}$ 基线 | 0.911 | 0.454 | 0.682 |
| LVAD-S | 最高 $F_{10}$ 基线 | 0.382 | 0.967 | 0.674 |
| LVAD-S | **Multi-T (Ours)** | **0.840** | **0.869** | **0.855** |
| +GT Norm | **Multi-T (Ours)** | **0.917** | **0.980** | **0.949** |

传统方法在 $F_{0.1}$ 和 $F_{10}$ 之间严重失衡，Multi-T 两个指标均衡且领先。

### 关键发现

- **DeepSVDD 的逆袭**：原始 AUC 仅 0.622（STL-10），加上 Multi-T 后飙升至 0.925（+48.7%），说明 DeepSVDD 的瓶颈不在模型本身而在训练数据质量
- **效率优势**：Multi-T 处理 10,000 个 ResNet-50 样本仅需 1.2 秒，比深度学习基线快数个数量级
- **特征提取器的影响**：CLIP 特征一致优于 ResNet-50，Multi-T 在两者上都有效，AUC 最高达 0.994（PCA+CLIP+Multi-T）
- **跨 outlier 比例的稳健性**：实验在 $\gamma \in [0.05, 0.4]$ 范围内取平均，Multi-T 表现稳定

## 亮点与洞察

1. **思维范式转换**：从"学更好的评分函数"转向"用阈值挖掘数据自监督信号"，用预测的标签反哺原有方法。最简单的距离方法+Multi-T 就能超越复杂的 SOTA 检测器
2. **训练无关设计**：Multi-T 纯统计实现，无需训练、无超参数调优，1.2 秒即可处理大规模数据
3. **排序相关性估计 outlier 比例**：利用两种归一化方法的结构一致性和互补性来隐式估计 $\gamma$，避免了需要先验知识
4. **双阈值比单阈值更实用**：放弃精确分割的幻想，转而追求 inlier 和 outlier 两端的高纯度，是务实且有效的设计

## 局限与展望

- 两个阈值之间的"灰色区域"样本未被利用，浪费了部分数据
- 阈值选择中 $\rho$ 的分界点（0.1, 0.3）是经验设定，对极端分布可能不够鲁棒
- 初始评分函数的质量仍然影响 Multi-T 的上限——如果初始分数区分度极差，排序/线性拟合步骤可能失效
- 实验主要在图像分类数据集上验证，未扩展到工业缺陷检测、时序异常等实际场景

## 相关工作与启发

- **与 Shell Renormalization 的关系**：Multi-T 使用 Shell Normalization 做特征去噪但改进了 outlier 选择机制，用 Ergodic-set Normalization 对比来估计 $\gamma$
- **与半监督异常检测的区别**：Multi-T 不需要预定义的训练/测试划分，所有操作在同一个无标签目标数据集上完成
- **启发方向**：多阈值思路可推广到其他需要从噪声标签中提取干净子集的场景（如噪声标签学习、数据清洗）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 从"评分"到"阈值"的视角切换新颖，多阈值+自监督信号利用的思路富有启发性
- **实验充分度**: ⭐⭐⭐⭐⭐ 6 个数据集、多种特征提取器、广泛的 outlier 比例范围、20+ 种对比方法
- **写作质量**: ⭐⭐⭐⭐ 论文结构清晰，但部分符号系统略显复杂
- **价值**: ⭐⭐⭐⭐⭐ 训练无关、即插即用、提升巨大，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](../../ICML2026/self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[ECCV 2024\] FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning](flowcon_out-of-distribution_detection_using_flow-based_contrastive_learning.md)
- [\[ECCV 2024\] SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning](scpnet_unsupervised_cross-modal_homography_estimation_via_intra-modal_self-super.md)
- [\[NeurIPS 2025\] CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing](../../NeurIPS2025/self_supervised/cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)
- [\[CVPR 2025\] ScaleLSD: Scalable Deep Line Segment Detection Streamlined](../../CVPR2025/self_supervised/scalelsd_scalable_deep_line_segment_detection_streamlined.md)

</div>

<!-- RELATED:END -->
