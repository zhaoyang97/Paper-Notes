---
title: >-
  [论文解读] Fair Dataset Distillation via Cross-Group Barycenter Alignment
description: >-
  [ICML 2026][AI安全][数据蒸馏] 本文揭示数据集蒸馏 (DD) 会放大原始数据中的偏差——根源是「子组样本量不平衡」与「子组表征分离度」的交互作用，并提出 COBRA：用各子组表征的（与组大小无关的）barycenter 作为蒸馏目标，可在多个 DD 框架上同时降低 EOD、提高准确率。
tags:
  - "ICML 2026"
  - "AI安全"
  - "数据蒸馏"
  - "群体公平"
  - "子组重心对齐"
  - "EOD"
  - "表征对齐"
---

# Fair Dataset Distillation via Cross-Group Barycenter Alignment

**会议**: ICML 2026  
**arXiv**: [2605.00185](https://arxiv.org/abs/2605.00185)  
**代码**: 无公开链接  
**领域**: 数据集蒸馏 / 公平机器学习 / AI 安全  
**关键词**: 数据蒸馏, 群体公平, 子组重心对齐, EOD, 表征对齐

## 一句话总结
本文揭示数据集蒸馏 (DD) 会放大原始数据中的偏差——根源是「子组样本量不平衡」与「子组表征分离度」的交互作用，并提出 COBRA：用各子组表征的（与组大小无关的）barycenter 作为蒸馏目标，可在多个 DD 框架上同时降低 EOD、提高准确率。

## 研究背景与动机
**领域现状**：数据集蒸馏把成千上万真实样本压缩成几张/几十张合成图像，使下游模型在合成集上训练即可接近全量训练性能。主流方法（DC/IDC/DM/CAFE/MTT 等）共同套路是：对每个类别 $y$ 选一种「表征」$\phi(x;\theta)$（梯度、嵌入、特征、轨迹），把合成集的类条件统计 $\Phi_{S_y}$ 对齐到真实集的 $\Phi_{T_y}$。

**现有痛点**：当训练集中存在受保护属性 $A\in\mathcal{A}$（性别、肤色、年龄等）的子组结构，且子组之间的表征模式不同时，简单地把 $\Phi_{S_y}$ 对齐到「所有样本的平均统计」$\Phi_{T_y}=\sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$ 会被多数子组主导，少数子组在合成集中几乎没影。下游模型在合成集训练后对少数子组的 conditional 准确率显著掉，EOD（equalized odds difference）变大。

**核心矛盾**：作者反复强调一个被忽视的事实——「按子组等比例采样」（Uniform DD）也无法根治问题，因为偏差来自**两个独立因素的乘积**：(i) 子组大小不平衡 $\pi_{a\mid y}$ 偏离均匀，(ii) 子组在表征空间分得很开 $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|$ 大。任一存在就够导致 residual $\Delta_{a\mid y}^*=\Phi_{T_{a\mid y}}-\Phi_{S_y}^*$ 失衡。

**本文目标**：(1) 形式化拆解 DD 中偏差放大的两个来源并给出 upper bound；(2) 设计一个紧致上界的目标，让所有子组得到「等距离」的合成表征；(3) 保持与现有 DD 框架（DC/DM/CAFE/IDC）兼容，仅替换匹配目标。

**切入角度**：在 MSE 距离下，标准 DD 的不动点是 $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$（加权平均），这天然偏向大组；改用**与权重 $\pi$ 无关的 barycenter**$m_y^* = \arg\min_m \sum_a d(\Phi_{T_{a\mid y}}, m)$ 作为目标，就能让合成集到每个子组的距离差距收紧。

**核心 idea**：用「子组重心 (cross-group barycenter)」取代「子组加权均值」作为类条件蒸馏目标，把对组大小的依赖直接从目标里抹掉，使最差子组的 residual 不会被多数组拉爆。

## 方法详解

### 整体框架
COBRA 想解决的是「数据蒸馏把原始偏差放大」这件事，做法是只动一处——把每个类别的蒸馏对齐目标从「子组加权均值」换成「子组重心」。具体走两步：先在每个类 $y$ 内按子组 $a\in\mathcal{A}$ 算出类条件子组统计 $\Phi_{T_{a\mid y}}=\frac{1}{|T_{a\mid y}|}\sum_{x\in T_{a\mid y}}\phi(x;\theta_T)$，再求这些子组统计之间的 barycenter $m_y^*$ 当作蒸馏目标；然后把合成集的类条件统计 $\Phi_{S_y}$ 对齐到 $m_y^*$ 而非原始的全样本均值 $\Phi_{T_y}$。因为 $\phi$ 可以是梯度、嵌入、特征或轨迹中任意一种表征，这套替换对 backbone 不挑，能原样嵌进 DC、DM、CAFE、IDC 等主流 DD 方法。

### 关键设计

**1. 偏差机制的形式化拆解：先证明偏差来自哪两个因子**

要修偏差得先知道偏差从哪来。作者在 MSE 距离下推导标准 DD 的 SGD 更新，得到合成集统计的不动点 $\Phi_{S_y}^* = \sum_a \pi_{a\mid y}\Phi_{T_{a\mid y}}$——这是个按子组大小 $\pi_{a\mid y}$ 加权的均值，天然偏向多数子组。每个子组到这个目标的残差是 $\Delta_{a\mid y}^* = \sum_{a'\neq a}\pi_{a'\mid y}(\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}})$，可放缩为 $\|\Delta_{a\mid y}^*\|_2 \leq \sum_{a'\neq a}\pi_{a'\mid y}\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|_2$。这个上界里同时含两个因子：子组大小不平衡 $\pi_{a'\mid y}$ 与子组在表征空间的分离度 $\|\Phi_{T_{a\mid y}}-\Phi_{T_{a'\mid y}}\|$，二者是相乘关系、缺一不可。这正是对前作 FairDD 的关键纠偏——FairDD 把问题全归到「子组不平衡」一个因子上，而本文用 Figure 2 的双轴控制实验证明：固定不平衡只调分离度、或固定分离度只调不平衡，两条曲线都能单独把 EOD 拉高，所以只修不平衡注定不够。

**2. 跨组 Barycenter $m_y^*$：与组大小解耦的新对齐目标**

既然 $\pi$ 加权均值是偏差的源头，就把 $\pi$ 从目标里直接抹掉。作者取 $d(u,v)=\|u-v\|_Q^2$（$Q$ 正定），求内层优化 $m_y^* = \arg\min_m \sum_a \|\Phi_{T_{a\mid y}}-m\|_Q^2$，其闭式解恰好是 $m_y^* = \frac{1}{|\mathcal{A}|}\sum_a \Phi_{T_{a\mid y}}$——一个子组级的 uniform 平均，完全不含 $\pi_{a\mid y}$，与 vanilla DD 的 $\pi$ 加权平均形成直接对照。选 uniform 权 $w_a=1/|\mathcal{A}|$ 是为切断对子组大小的依赖，而 barycenter 的几何意义又使这个目标到所有子组的总距离最小，相当于一个「对各子组最公允的中心」，让合成表征到每个子组的距离尽量拉平、压住最大那个残差。

**3. 理论保证：最差子组的残差不会比 vanilla 更糟**

换目标得保证不会顾此失彼。Theorem 4.1 定义 $s_y = m_y^\text{van}-m_y^*$ 为 imbalance 造成的偏移向量，只要最差子组 $a^\dagger$ 满足 $\langle \Delta_{a^\dagger\mid y}^C, s_y\rangle_Q \leq 0$（即它与 imbalance 偏移方向反向），就有 $\max_a \|\Delta_{a\mid y}^C\|_Q \leq \max_a \|\Delta_{a\mid y}^V\|_Q$——COBRA 的最差子组残差不超过 vanilla 的。这个条件几何上很温和：受亏待最严重的子组本就该落在加权目标的反方向上，符合直觉。意义在于，FairDD 把 per-group loss 平均化但参数更新仍可能跑偏，而 COBRA 直接收紧 worst-case residual 这一与 EOD 子组级误差差异直接挂钩的几何量，把公平性保证从「loss 平均」抬到了「表征对齐」层面。

### 损失函数 / 训练策略
最终损失 $\mathcal{L}_\text{COBRA}(T,S)=\sum_y D(m_y^*,\Phi_{S_y})$，相比 vanilla DD 仅替换了对齐目标。由于 barycenter 在 $\|u-v\|_Q^2$ 下有闭式解，内层无需额外迭代求解，效率与 vanilla 接近；IPC、网络架构、初始化、外层距离 $D$ 等超参全部沿用 backbone DD（DC/DM/CAFE/IDC）的默认设置，内层距离 $d$ 也可换成 cosine/MMD 做 ablation。

## 实验关键数据

### 主实验

| 数据集 | Backbone | IPC | Vanilla EOD/Acc | FairDD EOD/Acc | COBRA EOD/Acc |
|--------|---------|-----|-----------------|----------------|---------------|
| CIFAR10-S | DM | 100 | 82.87 / 45.4 | 25.17 / 61.2 | **9.37 / 62.4** |
| CIFAR10-S | DC | 50 | 71.85 / 39.5 | 35.65 / 46.2 | **26.18 / 46.6** |
| C-MNIST (BG) | DM | 50 | 100.0 / 48.8 | — | **7.46 / 96.8** |
| BFFHQ (真实) | DM | 100 | 63.47 / 65.8 | — | **7.87 / 74.2** |
| Full baseline | — | — | EOD 48.96 / Acc 69.71 (CIFAR10-S) | — | — |

（数值取自 Table 1，COBRA 在所有 IPC 与 backbone 下都同时降低 EOD 并提高/保持 Acc）

### 消融实验

| 配置 | 关键结果 | 说明 |
|------|---------|------|
| 距离 $d$ 选择 | MSE/cosine/MMD 都可工作，MSE 闭式解最高效 | 框架对 $d$ 鲁棒 |
| Backbone（DC/DM/CAFE/IDC） | 在 4 种 DD 范式上都生效 | 与 backbone 正交、即插即用 |
| 真实 vs 合成 baseline | Vanilla DD 比 Full 训练 EOD 更高 → DD 放大原始偏差 | 实证印证 Section 1 的主张 |
| 子组等比例采样 (Uniform DD) | 当子组在表征上离得近时反而更糟 | 单一修正 imbalance 不够 |
| varying imbalance / separation | 两条曲线分别独立拉高 EOD | 验证「两因子交互」是偏差源 |

### 关键发现
- **DD 放大原始偏差**：在 CIFAR10-S 上 Vanilla DD 的 EOD 远高于 Full 训练（如 IPC=100 时 82.87 vs Full 48.96），首次明确量化「蒸馏不仅没保住公平性，反而比训全集还更不公平」。
- **IPC 越小放大越严重**：IPC 减小时少数组容量不足，spurious correlation 更被强化，EOD 急剧上升。
- **barycenter 是 fairness 的几何解**：worst-case residual 直接关联 EOD 的子组级误差差异，把公平问题从「加权 loss 平均」上升到「表征几何中心」级。
- **跨数据集普适性**：从合成的 C-MNIST/CIFAR10-S 到真实的 UTKFace/BFFHQ，都保持显著增益，且兼容 4 种 backbone DD 方法。

## 亮点与洞察
- 把 DD 的 fairness 问题严格拆解为 imbalance×separation 的乘积，并给出 Figure 2 的双轴控制实验，研究方法论非常清晰，可作为后续 fair DD 工作的标准 baseline 范式。
- barycenter as target 的想法借自最优传输/聚类，引入到 DD 里却几乎没增计算量（MSE 下闭式解），有「学术 elegance + 工程 friendly」双重优势。
- 理论保证条件 $\langle\Delta,s\rangle_Q \leq 0$ 是几何上温和、可解释的——表示最差子组与 imbalance 偏移方向反向，符合直觉中「最受亏待的子组应当落在加权目标的反方向上」。
- 可立刻嵌入任何 representation-matching 类 DD（DC/DM/CAFE/IDC/MTT），社区采纳成本极低。

## 局限与展望
- 假设受保护属性 $A$ 在训练时可观察、子组标签可用——现实场景如医学数据常因隐私无法标注子组。
- 当某子组在表征空间是「均值不显眼但方差大」的形态时，单纯均值 barycenter 可能掩盖分布差异，应考虑分布级 (Wasserstein) barycenter。
- 理论 Theorem 4.1 是 worst-case，不直接给出 expected EOD 改进的紧致界；与下游模型选择的耦合尚未刻画。
- 仅讨论 EOD 一种公平准则，对 demographic parity、equal opportunity 等其他准则的影响尚未实验。
- 在 ImageNet 级大数据 + 大 IPC 上的可扩展性、与 trajectory matching (MTT) 的兼容性还需进一步验证。

## 相关工作与启发
- **vs FairDD (Zhou et al., 2025)**：他们做 per-group loss 平均，仅修 imbalance；本文证明 separation 因子单独存在也会放大 EOD，COBRA 在所有数据集/IPC 上一致优于 FairDD。
- **vs 标准 DD (DC/DM/CAFE/IDC/MTT)**：仅改对齐目标 $m_y^*$，不动 backbone 算法，证明 fairness 增益与表征匹配范式正交。
- **vs Long-tail DD (Cui 2024, Lu 2024, Zhao 2025)**：那些工作关注类不平衡，本文关注子组级 protected-attribute 不平衡，是更细粒度的 fairness。
- **vs Barycenter 公平 ML (Gordaliza 2019, Charpentier 2023)**：他们在原始数据空间做 OT barycenter 预处理，本文在表征空间做 barycenter 蒸馏，更轻量且适配 DD 流水线。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把「imbalance×separation 交互」这一被忽视的双因子结构精确刻画并给出几何解，是 fair DD 的关键概念推进
- 实验充分度: ⭐⭐⭐⭐ 覆盖 4 个 backbone × 多 IPC × 5 个数据集（合成 + 真实），消融研究系统
- 写作质量: ⭐⭐⭐⭐ 偏差机制推导清晰、Theorem + Figure 互证，论证流畅
- 价值: ⭐⭐⭐⭐ 即插即用、社区采纳门槛低，对 high-stakes 部署的公平性具有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Optimal Transport under Group Fairness Constraints](optimal_transport_under_group_fairness_constraints.md)
- [\[ICML 2026\] Scaling Unsupervised Multi-Source Federated Domain Adaptation through Group-Wise Discrepancy Minimization](scaling_unsupervised_multi-source_federated_domain_adaptation_through_group-wise.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2026\] Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods](extending_fair_null-space_projections_for_continuous_attributes_to_kernel_method.md)
- [\[AAAI 2026\] Fair Model-Based Clustering](../../AAAI2026/ai_safety/fair_model-based_clustering.md)

</div>

<!-- RELATED:END -->
