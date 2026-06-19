---
title: >-
  [论文解读] Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery
description: >-
  [ECCV2024][模型压缩][Generalized Continual Category Discovery] 提出 CAMP 方法，通过可学习投影器蒸馏与类别中心适应网络的协同组合，在广义持续类别发现（GCCD）场景中显著提升了新类别学习与旧知识保持之间的平衡。 广义持续类别发现（Generalized Contin…
tags:
  - "ECCV2024"
  - "模型压缩"
  - "Generalized Continual Category Discovery"
  - "知识蒸馏"
  - "continual learning"
  - "Category Adaptation"
  - "Projected Distillation"
---

# Category Adaptation Meets Projected Distillation in Generalized Continual Category Discovery

**会议**: ECCV2024  
**arXiv**: [2308.12112](https://arxiv.org/abs/2308.12112)  
**代码**: [GitHub](https://github.com/grypesc/CAMP)  
**领域**: 模型压缩  
**关键词**: Generalized Continual Category Discovery, knowledge distillation, continual learning, Category Adaptation, Projected Distillation

## 一句话总结

提出 CAMP 方法，通过可学习投影器蒸馏与类别中心适应网络的协同组合，在广义持续类别发现（GCCD）场景中显著提升了新类别学习与旧知识保持之间的平衡。

## 背景与动机

广义持续类别发现（Generalized Continual Category Discovery, GCCD）是一个结合了持续学习（Continual Learning）和广义类别发现（Generalized Category Discovery）的实际场景：模型需要从序列到达的、部分标注的数据集中持续学习，同时发现新类别并保持对旧类别的识别能力。

现有方法普遍采用**特征蒸馏**（Feature Distillation, FD）来缓解灾难性遗忘。然而，作者指出 FD 存在一个根本性矛盾：

- FD 通过限制特征空间中类别分布的漂移来防止遗忘
- 但这种刚性约束严重降低了模型的**可塑性**（plasticity），使其难以有效区分新类别
- 现有的类别适应方法（如 SDC、Feature Adaptation）在 FD 下效果有限，因为 FD 导致的漂移模式难以预测

这促使作者思考：能否找到一种方法，既允许旧类别分布自由漂移以提升可塑性，又能准确预测并补偿这种漂移以防止遗忘？

## 核心问题

1. **可塑性-稳定性困境**：特征蒸馏虽然减少遗忘，但限制了模型学习新类别的能力
2. **漂移不可预测**：在标准 FD 下，旧类别在特征空间中的漂移是不规则的，难以被适应网络准确预测
3. **无样例设定**：大多数方法依赖存储旧类别的样例数据（exemplars），这在实际场景中代价高昂

## 方法详解

CAMP（Category Adaptation Meets Projected distillation）包含三个训练阶段：

### 阶段一：特征提取器训练

损失函数由三部分组成：

1. **自监督学习损失** $\mathcal{L}_{SSL}$：对所有数据（含未标注）使用 SimCLR 对比学习损失 + 伪标签交叉熵损失
2. **监督学习损失** $\mathcal{L}_{SL}$：对标注数据使用 SupCon 监督对比损失 + 交叉熵损失
3. **投影蒸馏损失** $\mathcal{L}_{KD}$：使用可学习 MLP 投影器 $\phi^{t \to t-1}$ 将当前特征映射回旧特征空间

投影蒸馏的关键创新在于：不直接约束 $\mathcal{F}^t(x) \approx \mathcal{F}^{t-1}(x)$，而是学习一个投影器使得：

$$\mathcal{L}_{KD} = \sum_{i \in B} \| \phi^{t \to t-1}(\mathcal{F}^t(x_i)) - \mathcal{F}^{t-1}(x_i) \|^2$$

这样模型特征空间可以自由演化，只需保证存在一个可学习的映射连接新旧空间。最终损失为：

$$\mathcal{L} = (1-\alpha)((1-\beta)\mathcal{L}_{SSL} + \beta\mathcal{L}_{SL}) + \alpha\mathcal{L}_{KD}$$

### 阶段二：半监督聚类

使用半监督 K-Means 对当前任务数据聚类：
- 用标注数据初始化已知类别的中心
- 用 K-Means++ 从未标注数据中发现新类别中心
- 用 elbow method 确定类别数量

### 阶段三：类别中心适应

训练辅助适应网络 $\psi^{(t-1) \to t}$ 预测旧类别中心从旧空间到新空间的漂移：

$$\mathcal{L}_{PA} = \sum_{i \in B} \| \mathcal{F}^t(x_i) - \psi^{(t-1) \to t}(\mathcal{F}^{t-1}(x_i)) \|^2$$

训练完成后，更新旧类别中心：$p_i^t = \psi^{(t-1) \to t}(p_i^{t-1})$

**核心洞察**：投影蒸馏使得旧类别的漂移变得**规则且可预测**，这是 CAMP 两个组件协同工作的关键。单独使用投影蒸馏会增加遗忘（因为允许更多漂移），单独使用类别适应在标准 FD 下效果有限，但两者结合后性能显著提升。

### 网络架构选择

- 蒸馏投影器 $\phi$：2 层 MLP（384 维 + ReLU）
- 适应网络 $\psi$：线性层（384 维）
- 特征提取器：ViT-Small（DINO 预训练，冻结前 11 块）

## 实验关键数据

### GCCD 场景（5 个数据集，无样例设定）

| 数据集 | CAMP (All) | 第二名 (All) | 提升 |
|--------|-----------|-------------|------|
| CIFAR100 | **52.1%** | GCD+FD 36.6% | +15.5% |
| Stanford Cars | **48.8%** | GCD+EWC 30.4% | +18.4% |
| CUB200 | **58.9%** | GCD+EWC 50.3% | +8.6% |
| FGVCAircraft | **39.9%** | MetaGCD 28.6% | +11.3% |
| DomainNet | **36.7%** | SimGCD 36.5% | +0.2% |

### 无样例 Class Incremental Learning

| 数据集 | CAMP | 第二名 (FeTrIL) | 提升 |
|--------|------|----------------|------|
| CIFAR100 (5 tasks) | **65.0%** | 58.5% | +6.5% |
| CIFAR100 (10 tasks) | **56.7%** | 46.3% | +10.4% |
| ImageNet-Subset (5 tasks) | **73.1%** | 63.6% | +9.5% |

### 消融实验（CUB200）

- 完整 CAMP：Known 62.6% / Novel 44.2%
- 去掉投影蒸馏：性能大幅下降
- 去掉类别适应：性能明显下降
- 去掉自监督损失：Known +2.0% 但 Novel -2.8%

### 蒸馏器与适应器组合分析

- 仅用 MLP 蒸馏器（无适应）：比基线提升 4.7%
- MLP 蒸馏器 + 线性适应器：比基线提升 **20.2%**
- 这证明了两个组件的协同效应远超各自独立贡献之和

## 亮点

1. **协同效应的深刻洞察**：投影蒸馏和类别适应单独使用效果一般甚至相反，但组合后产生显著协同效应——这是全文最核心的贡献
2. **无需样例**：CAMP 在无样例设定下性能超过许多使用样例的方法，大幅降低内存开销
3. **可视化直觉**：通过 2D 瓶颈实验清晰展示了为何投影蒸馏使漂移更可预测
4. **广泛适用性**：方法同时适用于 GCCD 和传统 CIL 场景
5. **设计简洁**：只需存储每个类别一个中心向量，比存储特征或样例高效得多

## 局限与展望

1. **DomainNet 上新类别精度不佳**：GCD+FD 在 DomainNet 上 Novel 准确率（39.7%）远超 CAMP（29.7%），说明在域偏移大的场景下投影蒸馏可能过度放松约束
2. **类别数估计依赖 elbow method**：这是一个启发式方法，在类别数差异大时可能不准确
3. **任务边界假设**：方法假设明确的任务切换边界，现实中数据流往往是连续的
4. **适应网络的累积误差**：多任务序列中，中心适应可能累积误差（$\psi^{1\to2} \to \psi^{2\to3} \to \ldots$）
5. **仅在 ViT-Small 上验证**：对更大模型或不同架构的泛化性未充分探索

## 与相关工作的对比

| 方法 | 核心策略 | 需要样例 | GCCD 支持 |
|------|---------|---------|----------|
| GCD+FD | 特征蒸馏 | 否 | 是 |
| PA | Proxy Anchor + FD + 样例 | 是 | 是 |
| IGCD | 密度支持集 + 回放 | 是 | 是 |
| MetaGCD | 元学习 + 首任务数据 | 是（首任务） | 是 |
| GM | 双模型合并 | 否 | 是 |
| FeTrIL | 冻结特征 + 分类器训练 | 否 | 仅 CIL |
| **CAMP** | **投影蒸馏 + 类别适应** | **否** | **是** |

与最相关的类别适应工作相比：SDC 使用向量场估计漂移但不涉及投影蒸馏；Feature Adaptation Network 需要存储每类多个特征且需要样例。CAMP 只存储每类一个中心向量，更加高效。

## 启发与关联

1. **"让漂移发生但可预测"的思路**可推广到其他持续学习场景：与其限制模型变化，不如学习变化的映射
2. **投影蒸馏**的理念来自表示学习中的可迁移性研究（如 BYOL、SimSiam），说明自监督方法的技术可以反哺持续学习
3. 类别适应网络本质上学习的是**特征空间之间的仿射变换**，这与 domain adaptation 中的特征对齐思路相通
4. 在模型压缩视角下，该方法展示了如何在不扩展模型容量的前提下，通过辅助网络实现知识的高效迁移

## 评分

- 新颖性: ⭐⭐⭐⭐ （投影蒸馏 + 类别适应的协同组合是新颖的，但各组件均有先例）
- 实验充分度: ⭐⭐⭐⭐⭐ （5 个 GCCD 数据集 + 3 个 CIL 数据集 + 详细消融 + 可视化分析）
- 写作质量: ⭐⭐⭐⭐ （动机清晰，2D 可视化直观，但部分符号较冗长）
- 价值: ⭐⭐⭐⭐ （在 GCCD 和无样例 CIL 上全面领先，思路对社区有启发意义）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery](../../CVPR2026/model_compression/talon_test-time_adaptive_learning_for_on-the-fly_category_discovery.md)
- [\[ECCV 2024\] Anytime Continual Learning for Open Vocabulary Classification](anytime_continual_learning_for_open_vocabulary_classification.md)
- [\[ECCV 2024\] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)
- [\[ECCV 2024\] Adversarially Robust Distillation by Reducing the Student-Teacher Variance Gap](adversarially_robust_distillation_by_reducing_the_student-teacher_variance_gap.md)
- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](../../ICML2026/model_compression/energy-structured_low-rank_adaptation_for_continual_learning.md)

</div>

<!-- RELATED:END -->
