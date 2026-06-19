---
title: >-
  [论文解读] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation
description: >-
  [AAAI 2026][AI安全][联邦学习] 提出 CoRe-Fed 框架，通过嵌入级对比对齐与贡献感知聚合两个协同模块，同时解决联邦学习中的表示公平性和协作公平性问题，在异构数据分布下显著提升全局模型的公平性与泛化能力。 联邦学习（FL）允许多客户端在不共享原始数据的前提下协作训练模型，但传统 FL 算法在异构数据分布下…
tags:
  - "AAAI 2026"
  - "AI安全"
  - "联邦学习"
  - "公平性"
  - "表示对齐"
  - "对比学习"
  - "知识蒸馏"
---

# CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation

**会议**: AAAI 2026  
**arXiv**: [2602.00647](https://arxiv.org/abs/2602.00647)  
**代码**: [有](https://github.com/Noorain1/CoRe-Fed)  
**领域**: AI安全  
**关键词**: 联邦学习, 公平性, 表示对齐, 对比学习, 知识蒸馏

## 一句话总结

提出 CoRe-Fed 框架，通过嵌入级对比对齐与贡献感知聚合两个协同模块，同时解决联邦学习中的表示公平性和协作公平性问题，在异构数据分布下显著提升全局模型的公平性与泛化能力。

## 研究背景与动机

联邦学习（FL）允许多客户端在不共享原始数据的前提下协作训练模型，但传统 FL 算法在异构数据分布下面临三类偏差问题：

**性能偏差（Performance Bias）**：客户端数据集存在统计异质性和标签关联性，导致模型在部分客户端上表现优异而忽略其他客户端

**表示偏差（Representation Bias）**：数据分布不均导致不同客户端的嵌入向量在全局潜在空间中对齐不良，如 CIFAR-10 汽车/卡车图片在不同客户端产生角度偏差

**协作偏差（Collaborative Bias）**：标准聚合策略（如 FedAvg）可能抑制弱势客户端的更新，或让噪声/不对齐的贡献与高质量贡献具有相同权重

现有工作要么只关注层级公平性、要么只处理表示公平性，极少同时考虑表示对齐和聚合公平性之间的系统性关联。CoRe-Fed 的核心洞察是：**表示公平性和协作公平性是相互强化的**——表示公平性提高嵌入质量，协作公平性保证高质量嵌入在聚合中不被稀释。

## 方法详解

### 整体框架

CoRe-Fed 包含两个协同模块在服务器端执行：

1. **表示对齐模块**：通过对比学习 + 知识蒸馏对齐客户端嵌入
2. **贡献感知聚合模块**：基于参与频率和表示相似度动态调整客户端权重

工作流程：客户端本地训练 → 提取嵌入向量 → 服务器计算全局嵌入 → 对比损失量化表示相似度 → 构建对齐向量指导知识蒸馏 → 基于参与频率和对齐分数进行公平聚合。

### 关键设计

**嵌入提取与归一化**：每个客户端 i 对本地数据集进行特征提取，得到 L2 归一化的平均嵌入向量。全局嵌入由参与客户端集合的嵌入平均得到。

**对比学习对齐**：采用温度缩放的 NT-Xent 损失对齐客户端嵌入与全局嵌入，同时与其他客户端嵌入进行对比。温度参数 tau_c = 0.07 控制分布锐度。

**嵌入级知识蒸馏**：计算对齐向量（余弦相似度对齐分数乘以全局嵌入），将客户端嵌入软对齐到全局方向：

- 更新公式：z_tilde_i = z_i + beta * (z_tilde_g^(i) - z_i)
- beta 在 [0,1] 控制蒸馏强度，既保留客户端本地特征又趋向全局语义空间

**参与频率估计**：在动态滑动窗口 tau = M/|C_t| 内统计客户端参与频率，保证低活跃客户端不被遗忘。

**Sigmoid 调制公平权重**：联合参与频率和表示对齐度计算聚合权重：

- 权重公式：w_i = [(1/f_i)^gamma * sigma(k * rho_i)] / Sum[(1/f_l)^gamma * sigma(k * rho_l)]
- (1/f_i)^gamma 放大低参与客户端影响力
- sigma(k * rho_i) 偏好与全局模型对齐良好的客户端
- 形成奖惩机制：低频参与 + 高对齐 = 高权重

**梯度复用策略**：对临时离线客户端，在滑动窗口内复用其历史梯度，使其仍能影响全局模型更新。

### 损失函数 / 训练策略

- 本地训练使用标准交叉熵损失
- 服务器端通过对比损失进行表示对齐
- 全局模型更新结合公平权重和（可能复用的）梯度
- 关键超参数：gamma 属于 {0.5, 2}（公平指数），k 属于 {0.5, 2}（sigmoid 斜率），beta = 0.5（蒸馏系数），tau_c = 0.07（温度）
- SGD 优化器，学习率 eta = 0.1，衰减因子 0.999/轮，本地 epoch E = 1

## 实验关键数据

### 主实验

在 FMNIST 和 CIFAR-10 上，Dirichlet(alpha=0.5) 非 IID 划分，100 客户端每轮选 20 个参与：

| 算法 | FMNIST Acc | D_Cosine | D_Manhattan | CIFAR-10 Acc | D_Cosine | D_Manhattan |
|---|---|---|---|---|---|---|
| FedRDN | 0.870 | 0.746 | 116.8 | 0.569 | 1.077 | 180.9 |
| FedMDFG | 0.874 | 0.587 | 88.1 | 0.681 | 0.766 | 116.3 |
| FedMGDA+ | 0.849 | 0.421 | 79.5 | 0.549 | 0.719 | 48.4 |
| Ditto | 0.862 | 0.536 | 106.5 | 0.663 | 1.251 | 104.2 |
| qFedAvg | 0.884 | 0.401 | 76.2 | 0.628 | 0.702 | 52.9 |
| **CoRe-Fed** | **0.891** | **0.294** | **73.5** | **0.722** | **0.430** | **36.0** |

CoRe-Fed 在 CIFAR-10 准确率上比最佳基线提升 6.0%，角度距离减少 43.9%，曼哈顿距离减少 69%。

### 消融实验

| 场景 | 设置 | 与完整模型对比 |
|---|---|---|
| Co-Fed | 仅贡献感知聚合 | 公平性略有提升但准确率下降 |
| Re-Fed | 仅对比学习+知识蒸馏 | 表示偏差降低但聚合仍被主导客户端覆盖 |
| CoRe-Fed | 两者结合 | 准确率最高、距离最低 |

超参数 k 和 gamma 的权衡：k=2.0, gamma=0.5 相比 k=0.5, gamma=2.0 准确率提升 0.81%、余弦距离减少 0.16%、曼哈顿距离减少 0.32%。过高的 gamma 会过度放大不活跃客户端的陈旧更新。

### 关键发现

- 两种公平性是**互补的**：单独使用任一模块均不如联合使用效果好
- CoRe-Fed 在小 batch size（50）下表现尤为稳定，而 FedMGDA+ 和 FedRDN 出现明显震荡
- 每客户端准确率分析显示 CoRe-Fed 兼具高平均准确率和低客户端间方差

## 亮点与洞察

1. **统一框架的设计哲学**：首次将表示公平性和协作公平性在一个框架中统一处理，论证两者的相互强化关系
2. **Sigmoid 调制权重机制**巧妙地结合了参与频率倒数和对齐余弦相似度，构建了直觉清晰的奖惩机制
3. **梯度复用策略**使临时离线客户端仍能在一定窗口内对全局模型产生影响，提升实际场景适用性
4. **嵌入级蒸馏而非模型级蒸馏**，计算开销小、理论直观

## 局限与展望

1. 实验仅在 FMNIST 和 CIFAR-10 两个相对简单的数据集上验证，缺乏大规模实验
2. 模型架构局限于 MLP 和简单 CNN，未验证 ResNet/ViT 等现代架构
3. 梯度复用窗口 tau 的设置较为启发式，缺乏理论最优性分析
4. 未讨论恶意客户端（Byzantine robustness）场景下的鲁棒性
5. 对齐向量中标量乘法的表达力有限，可考虑更丰富的变换

## 相关工作与启发

- **FedMDFG**（Pan et al. 2023）：通过多方向公平梯度做聚合，是本文主要对比对象
- **FedRDN**（Yan et al. 2025）：数据增强缓解特征漂移，但未考虑聚合公平性
- **Shapley 值方法**（Tastan et al. 2024）：使用最后一层梯度的 Shapley 值近似来加权聚合
- 启发：嵌入空间的对齐可以作为跨客户端协调的通用桥梁，有望推广到异构模型联邦学习

## 评分

- 新颖性: 4/5 - 联合处理两种公平性的统一框架是新贡献
- 技术深度: 3/5 - 各组件较成熟，组合方式是主要创新
- 实验充分度: 3/5 - 消融和分析详尽，但数据集和模型规模偏小
- 写作质量: 4/5 - 问题动机清晰，图示直观
- 综合: 3.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[ICLR 2026\] Bridging Fairness and Explainability: Can Input-Based Explanations Promote Fairness in Hate Speech Detection?](../../ICLR2026/ai_safety/bridging_fairness_and_explainability_can_input-based_explanations_promote_fairne.md)
- [\[AAAI 2026\] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning](healsplit_towards_self-healing_through_adversarial_distillation_in_split_federat.md)
- [\[AAAI 2026\] Credal Ensemble Distillation for Uncertainty Quantification](credal_ensemble_distillation_for_uncertainty_quantification.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)

</div>

<!-- RELATED:END -->
