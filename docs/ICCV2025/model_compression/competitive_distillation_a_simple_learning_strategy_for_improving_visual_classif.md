---
title: >-
  [论文解读] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification
description: >-
  [ICCV 2025][模型压缩][竞争蒸馏] 提出竞争蒸馏策略，在多网络联合训练中，每个迭代动态选择表现最好的网络作为教师，配合随机扰动机制引入类似遗传算法的变异操作，显著提升视觉分类性能。 知识蒸馏已被广泛用于加速网络训练和提升性能。现有方案主要有两类：（1）传统模型蒸馏，使用固定的教师网络指导学生网络…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "竞争蒸馏"
  - "知识蒸馏"
  - "互学习"
  - "随机扰动"
  - "集体智能"
---

# Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification

**会议**: ICCV 2025  
**arXiv**: [2506.23285](https://arxiv.org/abs/2506.23285)  
**代码**: 无  
**领域**: 模型压缩与知识蒸馏  
**关键词**: 竞争蒸馏, 知识蒸馏, 互学习, 随机扰动, 集体智能

## 一句话总结

提出竞争蒸馏策略，在多网络联合训练中，每个迭代动态选择表现最好的网络作为教师，配合随机扰动机制引入类似遗传算法的变异操作，显著提升视觉分类性能。

## 研究背景与动机

知识蒸馏已被广泛用于加速网络训练和提升性能。现有方案主要有两类：（1）传统模型蒸馏，使用固定的教师网络指导学生网络；（2）深度互学习（DML），多个网络互为教师和学生，相互对齐预测分布。然而，DML存在一个反直觉的问题：**表现最好的网络不应该向表现差的网络学习**，但DML的双向对齐机制迫使所有网络彼此学习，导致学习方向不够合理。

作者的核心动机是：在每个训练迭代中，**应该动态确定学习方向**——只让表现最优的网络充当教师，其他网络向它学习，从而避免优秀网络被拖后腿。这一思想来源于集体智能中的竞争过程。

## 方法详解

### 整体框架

竞争蒸馏组织一组网络 $\Theta = \{\Theta_i | i=1,2,...,n\}$（$n \geq 2$）共同训练同一任务。每个迭代中，通过竞争优化选出当前最佳网络作为教师，其余网络作为学生从教师处蒸馏知识。同时引入随机扰动模拟遗传算法中的"变异"操作，帮助网络跳出局部最优。

### 关键设计

1. **竞争优化 (Competitive Optimization)**:
   每个训练迭代中，所有网络在当前batch上计算交叉熵损失 $L_C$，**损失最低的网络被选为教师** $\Theta_T^t$，其余网络作为学生 $\Theta_S^t$。教师网络仅使用分类损失更新参数，而学生网络需要额外学习蒸馏损失和特征损失：
    $\Theta_i^{t+1} \leftarrow \begin{cases} \Theta_i^t - \gamma \frac{\partial L_{C_i}}{\partial \Theta_i^t}, & \text{if } \Theta_i^t = \Theta_T^t \\ \Theta_i^t - \gamma \left(\frac{\partial L_{C_i}}{\partial \Theta_i^t} + \frac{\partial L_{D_i}}{\partial \Theta_i^t} + \frac{\partial L_{F_i}}{\partial \Theta_i^t}\right), & \text{otherwise} \end{cases}$
   教师和学生角色在**每个迭代中动态切换**，所有网络都有机会成为教师。

2. **随机扰动 (Stochastic Perturbation)**:
   受遗传算法中变异算子启发，在每个迭代中随机选择一个网络，对其输入图像施加扰动。扰动从预定义的处理池 $P = \{P_r | r=1,...,R\}$ 中随机选取，包括图像融合、拼接、噪声注入、数据变形、随机裁剪等。这些扰动参数比常规数据增强更激进（如裁剪比例设为[0.3, 0.7]）。
   核心洞察：**正向变异的效果会被保留**（因为变异后表现更好的网络会成为教师，知识被传递给所有学生），而**负向变异会被自动忽略**（表现变差的网络不会被选为教师，其错误知识不会传播）。

3. **多层次知识传递**:
   学生网络从教师处接收两种形式的监督信号：
    - 蒸馏损失 $L_D$：通过KL散度对齐学生与教师的软标签分布
    - 特征损失 $L_F$：通过L2损失对齐学生与教师的特征图

### 损失函数 / 训练策略

- **教师网络损失**: $L_{\theta_i}^T = L_{C_i}$（仅分类损失）
- **学生网络损失**: $L_{\theta_i}^S = L_{C_i} + \alpha L_{D_i} + \beta L_{F_i}$，其中 $\alpha = \beta = 1$
- 分类损失 $L_C$ 为标准交叉熵
- 蒸馏损失 $L_D = D_{KL}(p_t \| p_i)$，衡量学生与教师预测分布差异
- 特征损失 $L_F = \sum_{j=1}^M \|F_i - F_t\|_2^2$，匹配中间层特征
- 优化器：SGD + Nesterov momentum，初始学习率 0.1，batch size 128

## 实验关键数据

### 主实验

| 数据集 | 网络组合 | 独立训练 | DML | 竞争蒸馏 | 提升(vs Ind.) |
|--------|---------|---------|-----|---------|-------------|
| CIFAR-100 | ResNet-32 × 2 | 68.73 | 70.87 | 71.45 | +2.72 |
| CIFAR-100 | ResNet-56 + ResNet-152 | 73.76/76.66 | 75.31/78.64 | 76.76/79.80 | +3.00/+3.14 |
| CIFAR-100 | CeiT-B × 2 | 86.28 | 87.76 | 88.05 | +1.77 |
| ImageNet | Vim-S × 2 | 80.46 | 81.37 | 82.04 | +1.58 |
| ImageNet | CeiT-B + ViT-B | 85.47/84.25 | 86.11/85.34 | 86.56/86.27 | +1.09/+2.02 |
| Market-1501 | ResNet-56 + ResNet-152 | 88.57/94.77 | 90.43/95.32 | 91.28/96.17 | +2.71/+1.40 |

### 消融实验

| 配置 | Net1 Acc.(CIFAR-100) | 说明 |
|------|---------------------|------|
| 竞争优化 (Backbone) | 71.13 | 基础竞争选择机制 |
| + 特征损失 $L_F$ | 71.24 | 添加特征对齐 |
| + 随机扰动 | 71.37 | 添加变异操作 |
| + $L_F$ + 扰动 (完整) | 71.45 | 全部组件，效果最佳 |
| DML + $L_F$ + 扰动 | 71.09 | DML框架下相同组件表现更差 |

### 关键发现

- 竞争蒸馏在CNN（ResNet、MobileNet、WRN）、Transformer（ViT、CeiT）、Mamba（Vim）三类架构上均有效
- 三网络组（n=3）比两网络组（n=2）带来更大提升
- 异构网络组合（如ResNet-32 + WRN28-12）也能显著受益
- 在行人重识别（Market-1501）等跨任务场景同样有效

## 亮点与洞察

- **极其简洁的想法**：仅改变"谁当教师"的选择规则，就能显著超越DML等方法，体现了学习方向对训练效果的关键影响
- **随机扰动的精巧设计**：利用竞争选择机制自动过滤负面扰动、保留正面扰动，无需手动设计扰动策略
- **方法的通用性强**：适用于CNN/Transformer/Mamba，适用于分类/ReID/检测，且支持同构/异构网络组合

## 局限与展望

- 需要同时训练多个网络，GPU显存和计算开销是独立训练的n倍
- 训练完成后仅使用一个网络推理，其余网络训练资源被"浪费"
- 竞争选择仅基于当前batch的loss，可能因数据噪声导致次优选择
- 未探讨更大规模模型（如ViT-L、LLM）上的效果

## 相关工作与启发

- 与DML的核心区别在于**单向**vs**双向**学习方向：DML让所有网络双向对齐，而竞争蒸馏保证知识只从优到劣单向流动
- 随机扰动的思想类似于进化策略中的变异-选择机制，可以考虑引入更多进化算法策略（如交叉、精英保留）
- 可以考虑将竞争机制与自蒸馏结合，在单网络内部构造"虚拟竞争者"

## 评分

- **新颖性**: ⭐⭐⭐ 核心想法简单直观，创新幅度不大但效果扎实
- **实验充分度**: ⭐⭐⭐⭐ 覆盖多种架构、多种任务、详细消融
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，图示直观
- **价值**: ⭐⭐⭐⭐ 方法简单易用，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[NeurIPS 2025\] PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](../../NeurIPS2025/model_compression/preference-driven_knowledge_distillation_for_few-shot_node_classification.md)
- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](../../ECCV2024/model_compression/improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)
- [\[ECCV 2024\] Anytime Continual Learning for Open Vocabulary Classification](../../ECCV2024/model_compression/anytime_continual_learning_for_open_vocabulary_classification.md)

</div>

<!-- RELATED:END -->
