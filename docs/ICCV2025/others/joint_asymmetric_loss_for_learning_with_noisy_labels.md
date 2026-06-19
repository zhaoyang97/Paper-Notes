---
title: >-
  [论文解读] Joint Asymmetric Loss for Learning with Noisy Labels
description: >-
  [ICCV 2025][噪声标签学习] 将非对称损失函数扩展到更复杂的被动损失场景，提出非对称均方误差（AMSE），严格建立其满足非对称条件的充要条件，并将 AMSE 嵌入 APL 框架构建联合非对称损失（JAL），在 CIFAR-10/100 等多个数据集上全面超越现有鲁棒损失函数方法。 问题定义 在带噪声标签的数据上训练…
tags:
  - "ICCV 2025"
  - "噪声标签学习"
  - "非对称损失函数"
  - "鲁棒损失"
  - "APL框架"
  - "噪声容忍"
---

# Joint Asymmetric Loss for Learning with Noisy Labels

**会议**: ICCV 2025  
**arXiv**: [2507.17692](https://arxiv.org/abs/2507.17692)  
**代码**: [github.com/cswjl/joint-asymmetric-loss](https://github.com/cswjl/joint-asymmetric-loss)  
**领域**: 其他  
**关键词**: 噪声标签学习, 非对称损失函数, 鲁棒损失, APL框架, 噪声容忍

## 一句话总结

将非对称损失函数扩展到更复杂的被动损失场景，提出非对称均方误差（AMSE），严格建立其满足非对称条件的充要条件，并将 AMSE 嵌入 APL 框架构建联合非对称损失（JAL），在 CIFAR-10/100 等多个数据集上全面超越现有鲁棒损失函数方法。

## 研究背景与动机

### 问题定义

在带噪声标签的数据上训练深度神经网络是机器学习中的关键挑战：
- 人类标注不可避免地引入错误标签
- 直接在噪声标签上进行监督学习会严重降低模型性能
- 需要设计噪声容忍的损失函数，使得在噪声数据上的风险最小化器同时也能最小化干净数据的风险

三种主要噪声类型：

**对称噪声**：真实标签以均匀概率翻转到其他类别

**非对称噪声**：噪声率依赖于类别

**实例相关噪声**：噪声率依赖于具体样本

### 已有方法的不足

**对称损失的欠拟合问题**：MAE 等对称损失虽然理论上噪声容忍，但过于严格的对称条件限制了拟合能力，优化困难

**APL 框架的局限**：Ma et al. 提出的 Active Passive Loss (APL) 通过组合主动损失（如 NCE）和被动损失（如 MAE）来改善拟合，但目前 APL 框架中的被动损失全部基于对称条件

**非对称损失的不兼容**：Zhou et al. 提出的非对称损失函数（ALFs）理论上优于对称损失（仅需更宽松的"干净标签占主导"条件），但现有非对称损失（如 AGCE、AUL）全部是**主动损失**——在被动损失场景下实现非对称条件一直是未解决的难题

**AGCE 不能替代 NCE**：实验表明用非对称的 AGCE 替换 APL 中的主动 NCE 并不有效（AGCE+MAE 在 0.8 对称噪声下仅 44.61%），NCE/NFL 在 APL 中不可替代

### 核心动机

**关键洞察**：APL 框架的核心在于主动损失（NCE/NFL）和被动损失的互补性。现有瓶颈不在主动损失（NCE/NFL 已经很好），而在被动损失——如果能设计出一个比 MAE 更优的**非对称被动损失**，就能在保持 NCE/NFL 不变的前提下大幅提升 APL 框架的整体性能。

关键区分：
- **主动损失**：仅显式最大化标签类别的概率（如 CE：$L = -\log f(x)_y$），对非标签类别的损失为零
- **被动损失**：也显式最小化至少一个其他类别的概率（如 MAE：$L = \sum_k |e_k - f(x)_k|$）

## 方法详解

### 整体框架

1. 提出 AMSE：一个新的非对称被动损失函数
2. 建立 AMSE 满足非对称条件的充要条件
3. 将 AMSE 嵌入 APL 框架，替换原有的对称被动损失，构建 JAL

### 关键设计

#### 1. **非对称均方误差（AMSE）**

- **功能**：设计一个同时满足非对称条件和被动损失定义的新损失函数
- **核心思路**：在标准 MSE 的标签向量上引入缩放参数 $a$：

$$L_{\text{AMSE}}(f(\mathbf{x}), y) = \frac{1}{K} \|a \cdot \mathbf{e}_y - f(\mathbf{x})\|_2^2 = \sum_{k=1}^{K} \frac{1}{K} |a \cdot e_k - f(\mathbf{x})_k|^2$$

其中 $a \geq 1$ 是超参数。当 $a = 1$ 时退化为标准 MSE。

**定理 4.1（充要条件）**：对于给定权重 $w_1, \ldots, w_K$（其中 $w_m > w_n = \max_{i \neq m} w_i$），损失函数 $L(f(\mathbf{x}), y) = \frac{1}{K}\|a \cdot \mathbf{e}_y - f(\mathbf{x})\|_q^q$ 是非对称的，当且仅当：

$$\frac{w_m}{w_n} \geq \frac{a^{q-1} + \sum_{i \neq m} \frac{w_i}{w_n}}{(a-1)^{q-1}} \cdot \mathbb{I}(q > 1) + \mathbb{I}(q \leq 1)$$

对于 $q = 2$（AMSE）的情况，条件简化为：$\frac{w_m}{w_n} \geq \frac{a + \sum_{i \neq m} w_i/w_n}{a - 1}$

例如，10 类数据集，0.8 对称噪声时：$\frac{0.2}{0.8/9} \geq \frac{a + 9}{a - 1}$，即 $a \geq 9$

- **设计动机**：标准 MSE（$a=1$）是对称损失，拟合能力受限。通过引入 $a > 1$，将标签目标从 one-hot 向量 $\mathbf{e}_y$ 放大为 $a \cdot \mathbf{e}_y$，使得正确类别的惩罚更大，从而破坏对称性但保持非对称性条件下的噪声容忍性。$a$ 值越大约束越严格、鲁棒性越强，但可能降低拟合能力。

#### 2. **联合非对称损失（JAL）**

- **功能**：将 AMSE 嵌入 APL 框架替换被动对称损失
- **核心思路**：

基于交叉熵的 JAL-CE：
$$L_{\text{JAL-CE}} = \alpha \cdot L_{\text{NCE}} + \beta \cdot L_{\text{AMSE}}$$

基于 Focal Loss 的 JAL-FL：
$$L_{\text{JAL-FL}} = \alpha \cdot L_{\text{NFL}} + \beta \cdot L_{\text{AMSE}}$$

**噪声容忍性证明**：Zhou et al. 已证明对称损失是完全非对称的，且非对称损失的组合仍是非对称的。由于 NCE/NFL 是对称的（因此也是非对称的），AMSE 也是非对称的，所以 JAL 也是非对称的，因而噪声容忍。

- **设计动机**：NCE/NFL 在 APL 中的主动损失角色不可替代（实验证实），关键在于被动损失端。AMSE 作为非对称被动损失比对称的 MAE/NNCE 更优，嵌入 APL 后实现了"鲁棒且充分的学习"。

#### 3. **参数 $a$ 的选择策略**

- **功能**：根据噪声率和类别数确定 $a$ 的合理取值范围
- **核心思路**：根据定理 4.1 的充要条件计算 $a$ 的下界，然后选择适度的值。例如 CIFAR-10 + 0.8 对称噪声 → $a \geq 9$，实验选择 $a \in [10, 20, 30, 40]$
- **设计动机**：$a$ 过小不满足非对称条件、鲁棒性不足；$a$ 过大约束过强、拟合能力下降。需要在鲁棒性和拟合能力之间取得平衡。

### 损失函数 / 训练策略

- JAL-CE：$\alpha \cdot \text{NCE} + \beta \cdot \text{AMSE}$，$\alpha = 1, \beta = 1$
- CIFAR-10：8层 CNN，CIFAR-100：ResNet-34
- SGD 优化器，学习率 0.01，权重衰减 $10^{-4}$，训练 120 epochs

## 实验关键数据

### 主实验

**CIFAR-10 对称/非对称噪声（最后 epoch 测试精度 %）**：

| 方法 | 对称 0.4 | 对称 0.8 | 非对称 0.2 | 非对称 0.4 |
|------|---------|---------|-----------|-----------|
| CE | 58.05 | 19.74 | 83.05 | 73.85 |
| NCE+RCE (APL) | 85.89 | 54.99 | 88.62 | 77.94 |
| ANL-CE | 87.16 | 62.28 | 89.09 | 77.99 |
| ANL-FL | 86.94 | 61.89 | 89.29 | 77.89 |
| **JAL-CE** | **87.53** | **65.43** | **89.11** | **79.54** |
| **JAL-FL** | **87.43** | **64.84** | **89.36** | **79.51** |

**CIFAR-100 对称/非对称噪声**：

| 方法 | 对称 0.4 | 对称 0.6 | 非对称 0.3 | 非对称 0.4 |
|------|---------|---------|-----------|-----------|
| NCE+RCE | 58.48 | 46.73 | 55.86 | 41.50 |
| ANL-CE | 61.58 | 52.09 | 60.57 | 45.73 |
| **JAL-CE** | **64.11** | **56.73** | **64.90** | **56.17** |
| **JAL-FL** | **64.55** | **56.44** | **65.18** | **56.26** |

### 消融实验

**AMSE vs 现有被动损失（CIFAR-10）**：

| 被动损失 | 对称 0.4 | 对称 0.8 | 非对称 0.2 | 非对称 0.4 |
|---------|---------|---------|-----------|-----------|
| NCE（主动+被动基线） | 69.37 | 41.20 | 72.20 | 65.33 |
| AMSE（仅被动） | 87.54 | 64.97 | 83.88 | 58.07 |
| JAL-CE（NCE+AMSE） | 87.53 | 65.43 | 89.11 | 79.54 |

**JAL 的鲁棒性与拟合能力**：

| 配置 | 说明 |
|------|------|
| NCE 单独 | 拟合极差（69.37@sym0.4），但有基本鲁棒性 |
| AMSE 单独 | 对称噪声下表现极好，但非对称下拟合不充分 |
| JAL（NCE+AMSE） | 两者互补：对称噪声下保持 AMSE 的强鲁棒性，非对称下 NCE 补充拟合能力 |

**实例相关噪声（IDN）**：

| 方法 | CIFAR-10 IDN 0.4 | CIFAR-10 IDN 0.6 | CIFAR-100 IDN 0.4 | CIFAR-100 IDN 0.6 |
|------|-----------------|-----------------|-------------------|-------------------|
| ANL-CE | 85.74 | 69.83 | 60.88 | 48.12 |
| **JAL-CE** | **86.46** | **75.62** | **63.24** | **51.69** |

**CIFAR-10N/100N 人工标注噪声**：

| 方法 | CIFAR-10 Worst | CIFAR-100 Noisy |
|------|---------------|----------------|
| ANL-FL | 80.56 | 57.09 |
| **JAL-CE** | **81.33** | **59.54** |

### 关键发现

1. **高噪声下优势显著**：在 CIFAR-10 0.8 对称噪声下，JAL-CE 比 ANL-CE 提升 3.15%（65.43 vs 62.28）
2. **CIFAR-100 非对称噪声下大幅领先**：0.4 非对称噪声下 JAL 比 ANL 提升约 10%（56.17 vs 45.73）
3. **NCE 在 APL 中不可替代**：AGCE+MAE（44.61@sym0.8）远不如 NCE+RCE（54.99@sym0.8），证实了主动损失必须使用 NCE/NFL
4. **AMSE 的参数 a 有理论指导**：$a$ 的下界可由充要条件直接计算，$a=20$ 或 $a=30$ 在多数场景下表现最优
5. **JAL 在真实噪声数据上同样有效**：CIFAR-10N/100N 人工标注噪声下一致取得 top-2 性能

## 亮点与洞察

1. **理论贡献扎实**：严格建立了 AMSE 满足非对称条件的充要条件（定理 4.1），证明覆盖了 $q > 1$ 和 $q \leq 1$ 两种情况
2. **填补理论空白**：首次将非对称损失扩展到被动损失场景，解决了 ALF 与 APL 框架不兼容的问题
3. **设计极简**：AMSE 仅在标准 MSE 基础上增加一个缩放参数 $a$，实现简单、理论明确
4. **互补性分析**：通过消融实验清晰展示了 NCE（主动）和 AMSE（被动）的互补关系——对称噪声下 AMSE 主导鲁棒性，非对称噪声下 NCE 主导拟合能力
5. **参数选择有据**：$a$ 的下界有理论公式指导，不是纯经验调参

## 局限与展望

1. **$a$ 值依赖噪声率先验**：充要条件中需要知道噪声率 $\eta$，实际中需要估计或设置较保守的值
2. **极高噪声率下仍有困难**：在 CIFAR-100 + 0.8 对称噪声下（22.80%），JAL 性能显著低于中等噪声率场景
3. **仅考虑损失函数层面**：未与样本选择、标签修正等方法结合，可能存在进一步提升空间
4. **架构依赖**：实验使用较简单的 8层 CNN / ResNet-34，在更复杂架构上的表现未验证
5. **理论假设**：非对称条件要求"干净标签占主导"（$1-\eta_x > \max_{k \neq y} \eta_{x,k}$），在极端噪声比例下不满足

## 相关工作与启发

- 与 APL/ANL 的关系：JAL 是 APL 框架的自然扩展——保持主动端（NCE/NFL）不变，仅替换被动端为更优的 AMSE
- 与 ALF 的关系：ALF 仅解决了主动非对称损失，JAL 解决了被动非对称损失这一缺失的拼图
- 损失函数设计思路：通过缩放标签目标（$a \cdot \mathbf{e}_y$）来破坏对称性是一种巧妙且简洁的策略
- 互补学习启发：主动+被动的互补框架可能适用于其他需要平衡鲁棒性和拟合能力的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将非对称损失扩展到被动场景是明确的理论贡献，AMSE 设计简洁
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖合成噪声（对称/非对称/实例相关）和真实噪声（CIFAR-N/WebVision/Clothing1M），7种噪声设置 × 多个数据集
- **写作质量**: ⭐⭐⭐⭐ — 理论推导严谨，定理陈述清晰，但符号较多需要耐心阅读
- **价值**: ⭐⭐⭐⭐ — AMSE 实现简单且有理论保证，对噪声标签学习社区有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Debiased Sample Selection for Learning with Noisy Labels](../../CVPR2026/others/debiased_sample_selection_for_learning_with_noisy_labels.md)
- [\[ECCV 2024\] Foster Adaptivity and Balance in Learning with Noisy Labels](../../ECCV2024/others/foster_adaptivity_and_balance_in_learning_with_noisy_labels.md)
- [\[ICML 2025\] Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation](../../ICML2025/others/bipartite_ranking_from_multiple_labels_on_loss_versus_label_aggregation.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)

</div>

<!-- RELATED:END -->
