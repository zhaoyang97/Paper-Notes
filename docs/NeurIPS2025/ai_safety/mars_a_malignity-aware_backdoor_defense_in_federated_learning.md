---
title: >-
  [论文解读] MARS: A Malignity-Aware Backdoor Defense in Federated Learning
description: >-
  [NeurIPS 2025][AI安全][后门攻击防御] 提出 MARS 防御方法，通过计算神经元的后门能量（Backdoor Energy）来感知模型的恶意程度，并利用 Wasserstein 距离聚类有效识别联邦学习中的后门模型。 联邦学习（FL）的分布式特性使其容易受到后门攻击。现有防御方法主要依赖三类经验性统计度量：…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "后门攻击防御"
  - "联邦学习"
  - "Wasserstein距离"
  - "后门能量"
  - "聚类检测"
---

# MARS: A Malignity-Aware Backdoor Defense in Federated Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.20383](https://arxiv.org/abs/2509.20383)  
**代码**: [GitHub](https://github.com/yunming181920/MARS)  
**领域**: AI Safety / Federated Learning  
**关键词**: 后门攻击防御, 联邦学习, Wasserstein距离, 后门能量, 聚类检测

## 一句话总结

提出 MARS 防御方法，通过计算神经元的后门能量（Backdoor Energy）来感知模型的恶意程度，并利用 Wasserstein 距离聚类有效识别联邦学习中的后门模型。

## 研究背景与动机

联邦学习（FL）的分布式特性使其容易受到后门攻击。现有防御方法主要依赖三类经验性统计度量：范数约束、OOD 检测和一致性检测。然而，最新 SOTA 攻击（如 3DFed、CerP、DarkFed）通过约束后门更新的范数、分布和一致性来模仿良性更新，使这三类防御全部失效。

作者通过实验验证了这一失败：（1）后门更新的范数可以比良性更新更小；（2）PCA 投影后后门和良性更新不可区分；（3）后门更新之间的余弦相似度甚至低于某些良性更新之间的相似度。

核心洞察：现有度量与后门攻击**松耦合**，缺乏感知恶意意图的能力。因此需要一个与后门攻击**紧耦合**的度量。

## 方法详解

### 整体框架

MARS 包含三个步骤：（1）计算每个神经元的后门能量（BE）；（2）提取最突出的 BE 值形成浓缩后门能量（CBE）；（3）使用 Wasserstein 距离聚类识别后门模型。

### 关键设计

1. **后门能量（Backdoor Energy, BE）**: 直觉上，BE 衡量每个神经元对后门攻击的关联程度。理想定义需要干净数据和触发器，但在 FL 中不可获取。作者利用 Lipschitz 常数作为 BE 的上界近似：$BE_k^{(l)}(F) = \|f_k^{(l)}\|_{Lip}$。该近似不依赖干净数据或触发器，仅需模型参数即可计算。理论支撑来自 Theorem 4.1，证明了 BE 的上界。

2. **浓缩后门能量（Concentrated Backdoor Energy, CBE）**: 后门可视为捷径，只有少量神经元与后门相关。因此从每层提取 top-κ%（默认 5%）的 BE 值，拼接成一维向量，最大化后门信息密度，减少无关神经元干扰。

3. **Wasserstein 距离聚类（K-WMeans）**: 传统 K-Means 使用欧几里得或余弦距离，对元素顺序敏感。由于 FL 中不同后门模型的 top BE 可能出现在不同神经元位置，即使值整体更大也无法被正确聚类。Wasserstein 距离关注元素的概率分布而非顺序，更适合本场景。Toy example 验证：两个后门 CBE L1=[1,2,3,4,5] 和 L2=[5,5,3,2,2] 的 Wasserstein 距离为 0.40，远小于与良性 L3=[1,1,1,1,1] 的距离 2.00/2.40。

### 损失函数 / 训练策略

- 聚类后的集群选择：不假设良性客户端占多数，而是选择中心范数较小的集群
- 当两个集群的 Wasserstein 距离不超过阈值 ε 时，认为所有模型均为良性，保留两个集群
- 超参数默认设置：κ=5, ε=0.03

## 实验关键数据

### 主实验

| 数据集 | 攻击方式 | 指标 | MARS | 最佳Baseline | 提升 |
|---------|----------|------|------|-------------|------|
| MNIST | 3DFed | ASR↓ | 9.72% | 16.69%(FedCLP) | 显著降低 |
| MNIST | 3DFed | TPR↑ | 100% | 0%(多数方法) | 完美检测 |
| CIFAR-10 | CerP | ASR↓ | 10.03% | 10.01%(Multi-Krum) | 持平 |
| CIFAR-10 | 3DFed | ASR↓ | 9.86% | 7.55%(FedCLP) | 有竞争力 |
| CIFAR-100 | MRA | CAD↑ | - | - | 全面领先 |

### 消融实验

| 配置 | 说明 |
|------|------|
| κ (top%) | 控制从每层提取的 BE 比例，默认 5% |
| ε (阈值) | 控制集群距离判定，默认 0.03 |
| 距离度量 | Wasserstein > 欧几里得 ≈ 余弦（通过 toy example 验证） |

### 关键发现

- MARS 在面对 SOTA 攻击（3DFed）时 TPR 达 100%，FPR 接近 0%
- 在攻击者比例超过 50% 的极端场景下仍有效
- 即使在无攻击场景下，MARS 不会降低模型精度（满足 Fidelity 目标）

## 亮点与洞察

- 从 Lipschitz 常数推导 BE 的上界是巧妙的理论贡献，避免了对触发器和干净数据的依赖
- Wasserstein 距离替代欧几里得/余弦距离是关键创新，解决了 FL 中 CBE 元素顺序不一致的问题
- 提出的防御不需要假设攻击者比例<50%，实用性更强

## 局限与展望

- 仅在计算机视觉任务（MNIST/CIFAR）上验证，未涉及 NLP 等领域
- Lipschitz 常数的计算对大规模模型可能有效率问题
- 自适应攻击场景的讨论可以更深入
- 超参数 κ 和 ε 的敏感性分析值得进一步探索

## 相关工作与启发

- 与 CLP（Channel Lipschitzness based Pruning）的区别在于：MARS 用 Lipschitz 常数做检测而非剪枝，且引入了 CBE 和 Wasserstein 聚类
- 防御设计思路从"检测异常统计量"转向"感知恶意意图"，这一范式转变值得借鉴
- Wasserstein 距离在分布比较中的应用可推广到其他安全领域

## 评分

- **新颖性**: ⭐⭐⭐⭐ 后门能量+Wasserstein聚类的组合是新颖的防御思路
- **实验充分度**: ⭐⭐⭐⭐ 3个数据集、3种SOTA攻击、8种baseline防御，覆盖全面
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，motivation 部分的失败分析很有说服力
- **价值**: ⭐⭐⭐⭐ 对FL安全有实际意义，defense与attack的军备竞赛中提供了新视角

## 补充细节

- 威胁模型假设攻击者可以占多数（>50%），比现有防御的假设更强
- 防御部署在中心服务器端，仅需访问模型参数，不需要任何客户端的训练数据
- 实验中 100 个客户端，20 个攻击者，每轮选 20 个参与（4个攻击者）
- 在无攻击场景（innocent scenario）下，MARS 不会丢弃任何客户端，保持 FedAvg 的收敛速度
- BackdoorIndicator（Usenix Security 2024）也被纳入对比，MARS 仍然表现更优
- 针对 MARS 设计了定制化自适应攻击，验证了防御的鲁棒性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[ICCV 2025\] Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](../../ICCV2025/ai_safety/client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [\[CVPR 2025\] Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection](../../CVPR2025/ai_safety/detecting_backdoor_attacks_in_federated_learning_via_direction_alignment_inspect.md)
- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)

</div>

<!-- RELATED:END -->
