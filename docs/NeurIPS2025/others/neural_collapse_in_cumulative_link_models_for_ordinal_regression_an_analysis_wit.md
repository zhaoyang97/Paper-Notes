---
title: >-
  [论文解读] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model
description: >-
  [NeurIPS 2025][Neural Collapse] 将Neural Collapse (NC)理论扩展到基于累积链接模型(CLM)的序数回归(OR)任务中，在无约束特征模型(UFM)框架下证明了Ordinal Neural Collapse (ONC)的三个标志性质：类内均值坍缩(ONC1)、特征坍缩到一维子空间(ONC2)、以及潜变量按类别顺序排列(ONC3)，并在零正则极限下揭示了潜变量与阈值之间的简洁几何关系。
tags:
  - "NeurIPS 2025"
  - "Neural Collapse"
  - "序数回归"
  - "累积链接模型"
  - "无约束特征模型"
  - "正则化"
  - "阈值模型"
---

# Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model

**会议**: NeurIPS 2025  
**arXiv**: [2506.05801](https://arxiv.org/abs/2506.05801)  
**作者**: Chuang Ma, Tomoyuki Obuchi, Toshiyuki Tanaka (Kyoto University, RIKEN AIP)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: Neural Collapse, 序数回归, 累积链接模型, 无约束特征模型, 正则化, 阈值模型  

## 一句话总结

将Neural Collapse (NC)理论扩展到基于累积链接模型(CLM)的序数回归(OR)任务中，在无约束特征模型(UFM)框架下证明了Ordinal Neural Collapse (ONC)的三个标志性质：类内均值坍缩(ONC1)、特征坍缩到一维子空间(ONC2)、以及潜变量按类别顺序排列(ONC3)，并在零正则极限下揭示了潜变量与阈值之间的简洁几何关系。

## 研究背景与动机

### 问题背景
Neural Collapse (NC)是深度分类网络中发现的一个关键现象：经过充分训练后，倒数第二层特征和最终分类器权重呈现出极为简单的对称几何结构（Simplex ETF）。NC已被扩展到多标签分类、多元回归、扩散模型等任务中，但在序数回归(OR)任务中是否存在类似现象尚未被研究。

### 已有工作的不足
- NC理论主要集中在标准分类中，在OR任务中的行为完全未知
- OR与分类的关键区别在于：标签之间具有有序结构，不同误分类的代价不同；与回归的区别在于标签值仅有序关系而无定量信息
- 已有NC扩展（如NRC、linguistic collapse）均未涉及有序标签的特殊结构
- 阈值模型中潜变量的角色类似于分类网络中的logit，但其特殊的有序约束使得NC分析需要全新的理论框架

### 核心动机
探索在基于CLM的深度OR任务中，是否存在类似NC的几何坍缩现象，并通过UFM框架给出严格的理论证明和实验验证。

## 方法详解

### 问题建模
- **序数回归**：输入空间X，有序标签集Y={1,2,...,Q}，训练集D={(x_i, y_i)}
- **累积链接模型(CLM)**：通过潜变量z和阈值b=(b_0,b_1,...,b_Q)建模，类别概率为P(y<=q|z)=g(b_q - z)，其中g为严格单调递增的逆链接函数
- **DNN特征提取**：z = w^T h_theta(x)，w为分类器权重，h_theta为特征提取器
- **损失函数**：负对数似然 + L2正则化

### UFM框架转化
将特征向量h_theta(x_i)视为自由可学习变量h_{q,i}，利用目标函数对正交变换的不变性，将高维优化问题分解为多阶段优化。核心技巧是固定w的方向a，仅优化其模w和各类特征的投影z_q。

### ONC三大性质（Theorem 4.2）

假设逆链接函数g可微且其导数g'对数凹，在lambda_w, lambda_h > 0条件下：

1. **ONC1（类内均值坍缩）**：同一类别的所有最优特征坍缩为同一向量。证明关键：通过Theorem 4.1证明L(z,a,b)关于z的凸性——若p(x)对数凹，则其积分P(b-x)-P(a-x)也对数凹。结合L2正则项的严格凸性，保证每个子问题的最优解唯一。
2. **ONC2（一维子空间坍缩）**：各类均值h_q*平行于分类器w*。证明：将h分解为平行和正交分量，损失仅依赖平行分量，正交分量的最优值为零。
3. **ONC3（序数结构坍缩）**：最优潜变量满足z_1* <= z_2* <= ... <= z_Q*，且在g'严格对数凹时不等式严格成立。

### 状态方程(EOS)与相变（Theorem 4.3）
- 当w* > 0时，最优解满足一组非线性方程（EOS）
- 存在相变：当lambda_h * lambda_w >= C时为平凡解(w*=0)；当lambda_h * lambda_w < C时为非平凡解
- 相变边界C由阈值和类别比例精确确定
- 零正则极限下，对称逆链接函数（logit/probit）给出简洁关系：z_q* = (b_q + b_{q-1})/2
- 缩放律：w*的模与(lambda_h/lambda_w)^{1/4}成正比

### 阈值处理策略
- **固定阈值**：边界阈值对称固定(-b_Q = b_0)，内部阈值均匀间隔
- **可学习阈值**：通过softplus参数化保证严格有序性，与模型参数联合优化

## 实验关键数据

### 实验1：ONC指标在训练过程中的演化

在ER（表格数据集，5个有序类别）和UTKFace（人脸年龄估计，ResNet101，年龄按5年分组）上验证ONC的涌现：

| 指标 | ER-固定阈值 | ER-可学习阈值 | UTKFace-固定阈值 | UTKFace-可学习阈值 |
|------|-----------|-------------|----------------|-----------------|
| ONC1指标 | 持续下降趋近0 | 持续下降趋近0 | 持续下降趋近0 | 持续下降趋近0 |
| ONC2-1指标 | 快速趋近0 | 快速趋近0 | 快速趋近0 | 快速趋近0 |
| ONC2-2指标 | 快速趋近0 | 快速趋近0 | 快速趋近0 | 快速趋近0 |
| ONC3指标 | 始终小值且持续下降 | 收敛到非零值 | 始终小值且持续下降 | 收敛到非零值 |
| MAE(训练) | 趋近0 | 趋近0 | 下降 | 下降但更慢 |
| 分类精度 | 趋近1 | 趋近1 | 显著更高 | 较低 |
| 最小灵敏度 | 趋近1 | 趋近1 | >0 | 始终=0 |

关键发现：ONC1和ONC2在固定/可学习阈值下均涌现；ONC3仅在固定阈值下以z_q* = (b_q+b_{q-1})/2形式涌现。

### 实验2：固定阈值 vs 可学习阈值的实际性能对比

在UTKFace数据集上的对比揭示了固定阈值的实际优势：

| 性能维度 | 固定阈值 | 可学习阈值 |
|---------|---------|-----------|
| 训练收敛速度 | 更快更稳定 | 较慢 |
| 分类精度 | 显著提升 | 较低 |
| 最小灵敏度 | >0（所有类别均被覆盖） | =0（至少一个类别被完全忽略） |
| 少数类表现 | 更公平的潜空间分配 | minority collapse |
| ONC3涌现 | 是 | 否 |

固定阈值通过对潜空间的均匀分配，为所有类别（包括少数类）提供了更公平的预测概率分配，在类别不平衡场景下表现远优于可学习阈值。这一发现对实践者有重要指导意义。

## 亮点

- **理论开创性**：首次将NC理论扩展到序数回归，在UFM框架下严格证明了ONC三大性质，填补了NC在有序标签任务中的理论空白
- **优雅的零正则极限结果**：推导出z_q* = (b_q + b_{q-1})/2这一简洁关系，揭示了阈值与潜变量之间的几何对称性
- **相变发现**：证明了正则化参数空间中存在平凡/非平凡解之间的相变，且给出了精确的相变边界
- **实践价值**：发现固定阈值策略不仅是理论分析的需要，更在不平衡数据上带来显著的分类精度提升和少数类保护
- **定量ONC指标设计**：提出四个可测量的ONC指标（ONC1/ONC2-1/ONC2-2/ONC3），为实证验证提供了系统化工具

## 局限与展望

- **固定阈值假设**：理论证明依赖阈值固定的假设，虽然实验表明ONC1-2在可学习阈值下也涌现，但缺乏理论保证
- **两相假设未严格证明**：Theorem 4.3中假设了w*关于正则化参数的连续性和单调性，尚未严格证明系统仅存在两个相
- **ONC3指标局限**：当前ONC3指标仅适用于均匀间隔的固定阈值，不适用于可学习阈值场景
- **UFM简化**：UFM将特征视为自由变量，忽略了DNN实际训练动态和数据依赖性
- **数据集规模有限**：主要在中小规模表格数据集和UTKFace上验证，缺乏大规模复杂任务的验证
- **未转化为实际算法**：尚未将ONC insights转化为具体的正则化项或损失函数设计并验证其效果

## 与相关工作的对比

- **Papyan et al. (2020)**：在分类中发现NC四大标志，本文将其扩展到OR并发现ONC呈现不同的几何结构（一维有序排列 vs Simplex ETF）
- **Andriopoulos et al. (2024)**：在多元回归中发现NRC，特征坍缩到目标子空间；本文在OR中发现特征坍缩到一维子空间并保持有序性
- **Thrampoulidis et al. (2022)**：研究类别不平衡下的NC，发现NC1仍成立但全局几何改变；本文在OR不平衡场景中利用固定阈值保护少数类
- **Zhou et al. (2022)**：证明多种损失函数下NC的普遍性；本文证明CLM负对数似然损失下ONC的涌现
- **Dang et al. (2023, 2024)**：在深度线性和ReLU UFM中研究不平衡NC；本文在单层UFM中建立OR的NC理论

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将NC理论扩展到序数回归，理论和概念均有创新，但框架延续UFM传统
- 实验充分度: ⭐⭐⭐⭐ — 在表格和图像数据集上系统验证，设计了定量指标，但数据集规模和多样性可进一步加强
- 写作质量: ⭐⭐⭐⭐⭐ — 结构严谨，证明完整，从理论到实验逻辑链清晰
- 价值: ⭐⭐⭐⭐ — 填补了NC在OR中的理论空白，固定阈值的实践发现有应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](../../CVPR2025/others/feature_selection_for_latent_factor_models.md)
- [\[ACL 2025\] An Analysis of Datasets, Metrics and Models in Keyphrase Generation](../../ACL2025/others/an_analysis_of_datasets_metrics_and_models_in_keyphrase_generation.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)

</div>

<!-- RELATED:END -->
