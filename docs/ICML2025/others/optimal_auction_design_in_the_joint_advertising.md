---
title: >-
  [论文解读] Optimal Auction Design in the Joint Advertising
description: >-
  [ICML2025][联合广告] 本文针对联合广告场景（零售商与供应商共同竞标广告位）提出最优拍卖机制：单槽位下给出Myerson式闭式最优解，多槽位下设计BundleNet神经网络以bundle为单位构建IC约束，在保证近似激励兼容的同时最大化平台收入。 - 联合广告：是近年兴起的在线广告模式（Facebook等平台已部署…
tags:
  - "ICML2025"
  - "联合广告"
  - "最优拍卖"
  - "BundleNet"
  - "激励兼容"
  - "神经网络机制设计"
---

# Optimal Auction Design in the Joint Advertising

**会议**: ICML2025  
**arXiv**: [2507.07418](https://arxiv.org/abs/2507.07418)  
**代码**: 未公开  
**领域**: 拍卖设计 / 自动化机制设计  
**关键词**: 联合广告, 最优拍卖, BundleNet, 激励兼容, 神经网络机制设计

## 一句话总结

本文针对联合广告场景（零售商与供应商共同竞标广告位）提出最优拍卖机制：单槽位下给出Myerson式闭式最优解，多槽位下设计BundleNet神经网络以bundle为单位构建IC约束，在保证近似激励兼容的同时最大化平台收入。

## 研究背景与动机

- **联合广告**是近年兴起的在线广告模式（Facebook等平台已部署）：传统广告中只有零售商单方出价，而联合广告中零售商(retailer)和供应商(supplier/brand)共同为一个广告位出价，平台可向双方收费
- 现有方法存在两大问题：
  1. **VCG/JAMA类方法**仅在单维度（品牌或店铺）建立激励兼容(IC)，忽略了bundle结构，分配效率有优化空间
  2. **JRegNet**（基于RegretNet的神经网络方法）泛化能力差，在大规模复杂二部图场景下性能甚至不如VCG
- 核心挑战：联合广告中bundle由两个出价者的策略共同决定，bundle本身无独立策略，直接定义bundle级IC约束非常困难

## 方法详解

### 问题建模

联合广告系统中，零售商集合 $R$ 和供应商集合 $S$ 互不相交，广告集合 $E \subseteq R \times S$ 构成二部图 $G=(R,S,E)$。每个广告 $e=(r,s)$ 链接一个零售商和一个供应商。$m$ 个槽位的点击率为 $\boldsymbol{\lambda}=(\lambda_1,\dots,\lambda_m)$，满足 $\lambda_1 \geq \cdots \geq \lambda_m$。

拍卖机制 $\mathcal{M}=(x,p)$ 包含分配规则和支付规则：

$$x^e(v) = x_r^e(v) = x_s^e(v), \quad p^e(v) = p_r^e(v) + p_s^e(v)$$

需同时满足**激励兼容(IC)**（如实出价是占优策略）和**个体理性(IR)**（参与不亏损）。

### 单槽位最优机制（Theorem 4.3）

在正则分布假设下，定义虚拟价值函数：

$$c_i(v_i) = v_i - \frac{1-F_i(v_i)}{f_i(v_i)}$$

bundle $e=(r,s)$ 的虚拟价值为：$c^e(v_r, v_s) = c_r(v_r) + c_s(v_s)$

最优机制具有**阶跃函数**形式：当出价超过临界值 $\hat{v}_i(v_{-i})$ 时分配，支付恰好等于临界值。临界值由以下条件确定——参与者 $i$ 所在最优bundle的虚拟价值需同时：(1) 不低于平台保留价 $v_0$；(2) 不低于所有不含 $i$ 的bundle的虚拟价值。

### 多槽位BundleNet

#### Bundle级IC约束（核心创新）

不同于JRegNet对每个出价者单独构建IC约束，本文提出**以bundle为单位**的IC约束：

$$rgt^e(w) = \mathbb{E}\left[\max_{v_r'} \{u_r^e(\text{misreport}) - u_r^e(\text{truth})\} + \max_{v_s'}\{u_s^e(\text{misreport}) - u_s^e(\text{truth})\}\right]$$

**Lemma 5.1** 保证：$\sum_{i \in R \cup S} rgt_i(w) \leq \sum_{e \in E} rgt^e(w)$，即bundle级约束趋零时个体IC约束也趋零。

#### 网络架构

1. **图特征融合**：将二部图节点特征聚合为边特征
    - Divided Bids: $DB^e = [X_r, X_s] \in \mathbb{R}^{2m}$（拼接）
    - Stacked Bids: $SB^e = X_r + X_s \in \mathbb{R}^m$（求和）
2. **分配网络**：MLP处理 $SB^E$ → 双随机矩阵方法（行/列softmax取min）确保分配可行
3. **支付网络**：MLP处理 $DB^E$ → Sigmoid输出归一化支付系数 $\tilde{p} \in [0,1]$，乘以分配加权出价得到实际支付，从而结构性保证IR

#### 训练优化

采用增广拉格朗日方法求解约束优化：

$$\mathcal{L}_\rho(w;\mu) = -\text{rev}(w) + \sum_{e \in E}\mu_e \cdot \widehat{rgt}^e(w) + \frac{\rho}{2}\sum_{e \in E}(\widehat{rgt}^e(w))^2$$

交替更新网络参数 $w$、误报值 $v'$、和拉格朗日乘子 $\mu$。

## 实验关键数据

| 场景 | 方法 | U2 | U3 | U4 | U5 | E2 | E3 | E4 | E5 |
|------|------|------|------|------|------|------|------|------|------|
| 单槽位 | **BundleNet** | 0.529 | 0.668 | 0.781 | 0.880 | 0.425 | 0.546 | 0.635 | 0.722 |
| 单槽位 | Optimal | 0.525 | 0.671 | 0.783 | 0.882 | 0.425 | 0.548 | 0.647 | 0.738 |
| 单槽位 | JRegNet | 0.562 | 0.729 | 0.779 | 0.788 | 0.473 | 0.589 | 0.631 | 0.694 |
| 单槽位 | RVCG | 0.381 | 0.600 | 0.746 | 0.861 | 0.282 | 0.465 | 0.593 | 0.704 |

| 多槽位 | 方法 | U5×5 | U8×5 | U10×5 | U12×5 |
|--------|------|------|------|-------|-------|
| 5槽位 | **BundleNet** | 1.498 | 2.089 | 2.405 | 2.650 |
| 5槽位 | JRegNet | 1.497 | 1.935 | 1.962 | 1.997 |
| 5槽位 | RVCG | 0.842 | 1.883 | 2.280 | 2.587 |

- 单槽位下BundleNet几乎完美逼近理论最优解，JRegNet虽收入有时更高但偏离最优机制
- 多槽位下BundleNet全面优于RVCG和JRegNet，且随bundle数增加优势更明显
- 所有方法IC violation均 < 0.001

## 亮点与洞察

1. **理论贡献扎实**：首次给出联合广告单槽位最优拍卖的充要条件（Theorem 4.3），将Myerson引理推广到bundle场景
2. **Bundle级IC约束巧妙**：不直接对bundle定义策略（因其无独立策略），而是通过Lemma 5.1证明bundle级约束可包络个体级约束，化解了核心难点
3. **网络设计贴合问题结构**：利用二部图的边/节点对应关系设计Graph Feature Fusion，分配网络用双随机矩阵保证可行性，支付网络用Sigmoid结构性保证IR
4. **实验验证闭环**：单槽位可与理论最优对比验证方法正确性，多槽位展示实际优越性

## 局限与展望

1. **正则分布假设**：单槽位最优结果依赖分布的正则性条件（虚拟价值单调递增），非正则分布下需要ironing处理
2. **仅考虑单参数出价者**：每个出价者只有一个私有价值，未扩展到多物品/多参数设定
3. **二部图结构随机生成**：实验中的零售商-供应商关系矩阵随机生成，未在真实大规模广告系统上端到端验证
4. **IC近似保证**：虽然实验IC violation很低，但缺乏理论上的近似IC界
5. **可扩展性**：未报告计算开销和在更大规模（如数百bundles）下的性能

## 相关工作与启发

- **Myerson (1981)**：经典单参数最优拍卖理论，本文将其推广到联合广告的bundle场景
- **RegretNet (Dütting et al., 2024)**：用神经网络近似最优机制的开创性工作，BundleNet在此基础上改进了架构和约束
- **JRegNet (Zhang et al., 2024)**：联合广告的RegretNet变体，本文的直接竞争对手
- **JAMA (Ma et al., 2024)**：基于AMA的联合广告方法，提出了联合广告设定

## 评分

- 新颖性: ⭐⭐⭐⭐ (bundle级IC约束和单槽位最优刻画有原创性)
- 实验充分度: ⭐⭐⭐⭐ (单/多槽位、多种分布全面覆盖，但缺少真实系统实验)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，理论和实验衔接好)
- 价值: ⭐⭐⭐⭐ (联合广告是实际部署的新场景，理论+算法双重贡献)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Lightspeed Geometric Dataset Distance via Sliced Optimal Transport](lightspeed_geometric_dataset_distance_via_sliced_optimal_transport.md)
- [\[ACL 2025\] Evaluating Design Decisions for Dual Encoder-based Entity Disambiguation](../../ACL2025/others/evaluating_design_decisions_for_dual_encoder-based_entity_disambiguation.md)
- [\[ICML 2025\] Hierarchical Refinement: Optimal Transport to Infinity and Beyond](hierarchical_refinement_optimal_transport_to_infinity_and_beyond.md)

</div>

<!-- RELATED:END -->
