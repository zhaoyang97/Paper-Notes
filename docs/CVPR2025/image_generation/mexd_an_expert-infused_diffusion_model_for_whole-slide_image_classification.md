---
title: >-
  [论文解读] MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification
description: >-
  [CVPR 2025][图像生成][病理全切片图像] MExD 首次将生成式扩散模型应用于全切片图像（WSI）分类，通过动态混合专家（Dyn-MoE）聚合器筛选关键实例并提供条件信息，结合扩散分类器（Diff-C）从噪声中迭代还原类别标签，在Camelyon16、TCGA-NSCLC和BRACS三个基准上达到SOTA。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "病理全切片图像"
  - "多实例学习"
  - "扩散模型分类"
  - "混合专家"
  - "数据不平衡"
---

# MExD: An Expert-Infused Diffusion Model for Whole-Slide Image Classification

**会议**: CVPR 2025  
**arXiv**: [2503.12401](https://arxiv.org/abs/2503.12401)  
**代码**: [https://github.com/JWZhao-uestc/MExD](https://github.com/JWZhao-uestc/MExD)  
**领域**: 图像生成  
**关键词**: 病理全切片图像, 多实例学习, 扩散模型分类, 混合专家, 数据不平衡

## 一句话总结
MExD 首次将生成式扩散模型应用于全切片图像（WSI）分类，通过动态混合专家（Dyn-MoE）聚合器筛选关键实例并提供条件信息，结合扩散分类器（Diff-C）从噪声中迭代还原类别标签，在Camelyon16、TCGA-NSCLC和BRACS三个基准上达到SOTA。

## 研究背景与动机

1. **领域现状**：WSI分类通常采用多实例学习（MIL）的"分解-聚合"策略——将WSI切分为patch提取特征，再聚合为slide-level表示进行分类。主流方法包括注意力池化（ABMIL）、Transformer聚合（TransMIL）、图模型（WiKG）等。
2. **现有痛点**：(1) 严重的数据不平衡——一个WSI中阳性patch（含癌细胞）远少于阴性patch，模型倾向预测多数类；(2) 大量非信息区域引入噪声；(3) 简单的聚合方法难以捕捉patch间的复杂关系。
3. **核心矛盾**：现有方法全部基于判别式范式，在统一模型中处理所有patch，缺乏针对少数类的专门关注机制。判别式特征整合天然受噪声和无关patch干扰。
4. **本文目标**：(1) 如何在极端不平衡下有效提取少数类信息？(2) 如何抗噪声地整合patch特征？
5. **切入角度**：将WSI分类重新形式化为"条件生成任务"——不是从特征预测标签，而是以特征为条件从噪声中生成标签分布。同时用MoE机制为每个类别分配专门的"专家"来过滤实例。
6. **核心 idea**：用MoE路由筛选关键实例作为条件，通过扩散模型迭代生成one-hot类别标签，实现从判别式到生成式的WSI分类范式转变。

## 方法详解

### 整体框架
输入WSI经切片后→Patch Feature Extractor（冻结的预训练ViT/CTransPath）提取N个patch嵌入→Dyn-MoE Aggregator 通过K+1个专家路由筛选稀疏实例集并生成先验预测 $\rho_\theta$ 和专家洞察集 $g_\alpha$→Diffusion Classifier（Diff-C）以 $\rho_\theta$ 和 $g_\alpha$ 为条件，从噪声迭代去噪生成类别分布→输出分类结果。

### 关键设计

1. **Dynamic Mixture-of-Experts (Dyn-MoE) 聚合器**:

    - 功能：稀疏化实例集、为每个类别分配专门专家、生成先验预测和专家洞察
    - 核心思路：先通过Adapter（2层Transformer + PPEG卷积块）赋予全局依赖 $\{l_i\}_{i=1}^N$。然后设置K+1个路由器（K个正类专家+1个负类专家），每个路由器是两类MLP+softmax，对每个实例产生两个score。负类专家保留max score索引=0的实例，正类专家保留索引=1的。每个子集取top-k路由分数的实例（采样比α控制）。每个专家对其保留实例做均值池化得到类中心特征 $e_r$，通过(K+1)类分类器得到预测 $y^{ex}$ 和置信度 $c_r$。所有稀疏化后的实例拼接可学习class embedding $d$ 后经Adapter+线性分类器生成先验预测 $\rho_\theta$。
    - 设计动机：传统MIL在统一模型中处理所有patch，少数类信号被淹没。MoE为每个类别分配专用"通道"，正类专家只关注可能的阳性实例，有效缓解不平衡。动态路由+top-k筛选将实例数减半以上，大幅降低噪声。

2. **Diffusion Classifier (Diff-C)**:

    - 功能：将分类任务转化为条件生成任务，从噪声中迭代还原类别标签
    - 核心思路：将GT标签编码为one-hot向量 $f \in \mathbb{R}^{1 \times (K+1)}$ 作为初始信号。借鉴CARD方法，将先验预测 $\rho_\theta$ 作为前向扩散终点的条件期望（而非标准高斯噪声），前向过程 $q(f_t|f_0, \rho_\theta) = \mathcal{N}(f_t; \sqrt{\bar\alpha_t}f_0 + (1-\sqrt{\bar\alpha_t})\rho_\theta, (1-\bar\alpha_t)I)$。反向去噪使用3层MLP作为去噪网络 $\mathcal{D}$，以专家洞察的加权特征 $Z = \sum_r c_r \cdot e_r$ 和先验预测 $\rho_\theta$ 为条件，迭代步进 $f_T \to f_{T-\Delta} \to \cdots \to f_0'$。
    - 设计动机：生成式方法在处理噪声和数据分布方面天然优于判别式。扩散过程的迭代去噪相当于对类别预测的多步精化，每一步都利用专家知识来修正。先验预测作为扩散终点而非纯噪声，大幅加速收敛。

3. **两阶段训练策略**:

    - 功能：确保Dyn-MoE和Diff-C稳定收敛
    - 核心思路：Stage 1 训练Dyn-MoE，联合损失 $\mathcal{L}_a$ 包含：(a) 先验预测的交叉熵 $\Phi(f_0, \rho_\theta)$，(b) 负类专家的交叉熵 $\Phi(\dot{y}_0, y_0^{ex})$，(c) 选择性正类专家的加权交叉熵 $\sum_r \lambda_r \Phi(\dot{y}_r, y_r^{ex})$（仅当该正类专家对应的label与GT相同时 $\lambda_r=1$，否则为0）。Stage 2 冻结Dyn-MoE，训练去噪网络 $\mathcal{D}$，标准噪声估计损失 $\mathcal{L}_e = \|\epsilon - \epsilon_\theta(Z, f_t, \rho_\theta, t)\|^2$。
    - 设计动机：Dyn-MoE需要先学到有意义的路由和先验预测，才能为Diff-C提供高质量条件。联合训练可能导致路由不稳定。选择性专家损失确保每个专家只对"应该属于它"的样本负责。

### 损失函数 / 训练策略
- Stage 1: $\mathcal{L}_a = \frac{1}{R}\sum(\Phi(f_0, \rho_\theta) + \Phi(\dot{y}_0, y_0^{ex}) + \sum_r \lambda_r \Phi(\dot{y}_r, y_r^{ex}))$
- Stage 2: $\mathcal{L}_e = \|\epsilon - \epsilon_\theta(Z, f_t, \rho_\theta, t)\|^2$
- Patch特征离线提取，支持即插即用

## 实验关键数据

### 主实验（CTransPath特征）

| 方法 | Camelyon16 AUC | TCGA-NSCLC AUC | BRACS AUC |
|------|---------------|----------------|-----------|
| ABMIL | 92.41 | 95.87 | 80.44 |
| TransMIL | 95.03 | 95.89 | 85.18 |
| IBMIL | 96.41 | 97.45 | 85.84 |
| MHIM-MIL | 96.14 | 96.73 | 84.79 |
| **MExD** | **98.87** | **98.13** | **88.08** |

### 消融实验

| 配置 | Camelyon16 AUC | BRACS AUC | 说明 |
|------|---------------|-----------|------|
| Full MExD | 98.87 | 88.08 | 完整模型 |
| w/o Diff-C (仅Dyn-MoE) | ~96.5 | ~86 | 去掉扩散分类器，退化为判别式 |
| w/o MoE路由 | 显著下降 | 显著下降 | 无法有效处理不平衡 |
| w/o 先验预测条件化 | 收敛变慢 | 精度下降 | 纯噪声起点不如先验起点 |

### 关键发现
- MExD在Camelyon16上达到98.87% AUC，比最强baseline（IBMIL 96.41%）提升2.46%
- 在三分类任务BRACS上提升最显著（88.08% vs 85.84%），说明MoE在多类别不平衡场景效果更突出
- ViT (MoCo V3)特征上MExD同样表现最优，验证了框架的特征提取器无关性
- 扩散分类器的迭代精化确实优于单步判别，尤其在困难样本上
- F1-score在Camelyon16上达到97.29%，ACC 97.48%，全面领先

## 亮点与洞察
- **判别→生成的范式转变**：首次在WSI分类中引入生成式方法，将分类视为从噪声到标签的条件生成过程。这个思路突破了MIL领域的判别式思维定式，扩散的迭代精化过程自然地拥有抗噪声能力。
- **MoE+扩散的深度融合**：MoE不仅是聚合器，还为扩散模型提供两类条件信息——先验预测（确定扩散终点）和专家洞察（引导去噪方向）。这种"条件的条件"设计比简单的特征拼接更有机。
- **选择性专家损失**：正类专家只在label匹配时激活损失，避免了错误梯度对专家路由学习的干扰，这个设计在其他MoE应用中也可借鉴。

## 局限与展望
- 扩散推理需要多步迭代（T步），推理效率远低于判别式方法；虽然MLP去噪网络轻量，但仍有延迟
- 两阶段训练增加了超参数调优的复杂度
- MoE的路由是硬路由（argmax），可能丢失部分信息；软路由或可逆路由是否更好未探索
- 仅使用1D信号的扩散（one-hot向量），未利用扩散模型在高维空间的生成优势
- 未讨论模型的可解释性——哪些patch被分配到了哪个专家？路由决策是否与病理学家的判断一致？

## 相关工作与启发
- **vs ABMIL/TransMIL**: 这些判别式方法用注意力/Transformer聚合所有patch，缺乏针对少数类的专门机制；MExD的MoE路由显式为每个类别分配专家
- **vs DTFD-MIL**: DTFD通过伪bag划分缓解不平衡，但本质仍是判别式；MExD通过生成式框架从根本上改变了信息流
- **vs IBMIL**: IBMIL是当前MIL SOTA之一，MExD在其基础上通过MoE稀疏化+扩散分类进一步提升
- **vs CARD**: Diff-C借鉴了CARD的条件扩散分类框架，但引入了MoE专家洞察作为额外条件，实现了WSI特定的适配

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个将扩散模型用于WSI分类的工作，MoE+扩散的融合设计新颖
- 实验充分度: ⭐⭐⭐⭐ 三个benchmark全面评估，两种特征提取器验证，但消融不够详细
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，数学形式化完整，但公式较多增加阅读负担
- 价值: ⭐⭐⭐⭐ 开辟了WSI分类的新范式，SOTA结果有力验证了生成式方法的潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](../../ICCV2025/image_generation/dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[CVPR 2026\] Advancing Image Classification with Discrete Diffusion Classification Modeling](../../CVPR2026/image_generation/advancing_image_classification_with_discrete_diffusion_classification_modeling.md)
- [\[NeurIPS 2025\] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](../../NeurIPS2025/image_generation/camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)

</div>

<!-- RELATED:END -->
