---
title: >-
  [论文解读] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data
description: >-
  [ICLR 2026][可解释性][序数嵌入] 提出LORE——首个同时从序数三元组比较中联合学习嵌入表示和内在维度的框架：用非凸Schatten-p拟范数(p<1)正则化替代传统的预设维度策略，通过迭代重加权(IRNN)算法求解并证明收敛到稳定点；在合成数据、LLM模拟感知实验和3个众包数据集上，LORE在维度恢复上远超所有基线方法，同时保持高三元组准确率和语义可解释性。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "序数嵌入"
  - "内在维度恢复"
  - "Schatten-p拟范数"
  - "三元组比较"
  - "感知空间"
  - "低秩正则化"
---

# LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data

**会议**: ICLR 2026  
**arXiv**: [2602.04192](https://arxiv.org/abs/2602.04192)  
**代码**: [GitHub](https://github.com/vivek2000anand/lore_iclr)  
**领域**: 表示学习/感知建模  
**关键词**: 序数嵌入, 内在维度恢复, Schatten-p拟范数, 三元组比较, 感知空间, 低秩正则化

## 一句话总结
提出LORE——首个同时从序数三元组比较中联合学习嵌入表示和内在维度的框架：用非凸Schatten-p拟范数(p<1)正则化替代传统的预设维度策略，通过迭代重加权(IRNN)算法求解并证明收敛到稳定点；在合成数据、LLM模拟感知实验和3个众包数据集上，LORE在维度恢复上远超所有基线方法，同时保持高三元组准确率和语义可解释性。

## 研究背景与动机

**领域现状**：序数嵌入(Ordinal Embedding, OE)从三元组比较("A与B更相似，还是A与C更相似？")中学习感知空间的多维表示，广泛应用于心理物理学（味觉、嗅觉、美学偏好等主观感知）。相比绝对量化评分（如Likert量表），三元组比较不依赖语言描述、不受个体尺度偏差影响。

**核心痛点**：
   - 所有现有OE方法（SOE、FORTE、t-STE、CKL、OENN）均需用户预先指定嵌入维度d'
   - 缺乏判断"真实维度"的准则→实践中通常设置过高的维度
   - 过高维度掩盖真实结构（如10维嵌入实际只需2维→"甜度"被碎片化到多个轴）
   - 科学发现追求简约性（Occam's razor）：低维表示更易解释、计算更高效
   - 唯一尝试恢复维度的Künstle方法需枚举候选维度并逐一训练→不可扩展

**切入角度**：将维度发现融入OE优化本身→用Schatten-p拟范数正则化自动平衡三元组准确率与嵌入秩→无需预设维度。

## 方法详解

### 整体框架

LORE 把"该用几维"这件原本要靠用户预设的事，直接塞进序数嵌入的优化目标里：给定 $N$ 个对象和三元组集合 $T=\{(a,i,j)\}$（表示 $a$ 与 $i$ 比 $a$ 与 $j$ 更相似），它在一个足够宽的 $d'$ 维空间里优化嵌入矩阵 $Z\in\mathbb{R}^{N\times d'}$，目标同时包含一个平滑的三元组拟合损失和一个 Schatten-$p$ 低秩正则项，让模型在拟合比较关系的同时自动把冗余维度的奇异值压到零，最终落在的非零奇异值个数就是恢复出来的内在维度 $d\ll N$。整套优化由一个迭代重加权算法求解，每步靠一次 SVD 完成。

### 关键设计

**1. Schatten-$p$ 拟范数正则：用非凸惩罚精确逼近秩**

要让模型"自动选维度"，本质是要最小化嵌入矩阵 $Z$ 的秩，但秩函数本身是 NP-hard、不可直接优化。传统做法退而用核范数（$p=1$，即所有奇异值之和 $\sum_i\sigma_i(Z)$）做凸松弛，可核范数对大、小奇异值一视同仁地均匀压缩，会连同真正承载结构的大奇异值也削弱，引入显著偏差。LORE 改用 $0<p<1$ 的 Schatten-$p$ 拟范数 $\sum_{i}\sigma_i(Z)^p$（论文默认 $p=0.5$）：它对大奇异值惩罚很轻、对接近零的小奇异值惩罚极重，于是能在保留主结构的同时干净地"杀死"冗余维度，是比核范数准得多的低秩近似。代价是引入额外的非凸性——这一点交给后面的迭代算法消化。

**2. Softplus 平滑三元组损失：消除零梯度平台**

标准的 hinge 形式三元组损失在已满足的约束上梯度恒为零，优化容易卡死在平台区。LORE 把它换成处处可微的 softplus 形式

$$\sum_{(a,i,j)\in T}\log\big(1+\exp(1+d(z_a,z_i)-d(z_a,z_j))\big),$$

让整个拟合项没有零梯度平台、梯度信号在所有三元组上都存在，优化更顺。唯一的不可微点是嵌入坍塌（所有点重合到一起），用方差足够大的宽初始化即可避开。

**3. 直接优化嵌入而非 Gram 矩阵：换来可扩展性**

GNMDS、CKL、FORTE 等方法优化的是 $N\times N$ 的 Gram 矩阵 $G=ZZ^\top$，复杂度随对象数 $N$ 平方增长。LORE 直接优化 $N\times d'$ 的嵌入 $Z$，把每步复杂度降到正比于 $Nd'$，从而能扩展到更大的数据集；而且直接拿到嵌入坐标，后续读取可解释的语义轴也更自然。

**4. 迭代重加权算法（IRNN）与收敛保证：把非凸目标拆成一串 SVD 子问题**

非凸的 Schatten-$p$ 项无法直接梯度下降，LORE 用迭代重加权近邻算法（Iterative Reweighted Nuclear Norm, IRNN）求解：每一步先沿三元组损失梯度走一步并做奇异值分解 $U,S,V^\top=\mathrm{SVD}\big(Z^k-\tfrac{1}{\mu}\nabla f(Z^k)\big)$，再对奇异值做带阈值的收缩 $S^k=\max\{S-\tfrac{p}{\mu}\sigma^{p-1},\,0\}$，最后重构 $Z^{k+1}=U S^k V^\top$，如此迭代直到目标值或嵌入变化低于阈值，每步复杂度 $O\!\big(d'(|T|+Nd')\big)$。论文进一步证明该序列收敛到稳定点，即 $\sum_{k=1}^{\infty}\|Z^{k+1}-Z^k\|_F<+\infty$。虽只是稳定点而非全局最优，但序数嵌入已有理论（如 Bower 等证明 $d=2$ 时局部最优即全局最优）表明这类稳定点通常已足够接近最优解。收敛后非零奇异值的个数，就是 LORE 恢复出的内在维度 $d$。

### 损失函数 / 训练策略

完整目标即三元组拟合项加正则项 $\min_Z \Psi(Z)=\text{softplus 三元组损失}+\lambda\sum_{i=1}^{\min\{N,d'\}}\sigma_i(Z)^p$。三个固定量基本无需调：$p=0.5$（先验研究验证的最优值）、$\mu=0.1$（需大于三元组损失的 Lipschitz 常数）、初始化用方差 $\geq 5$ 的高斯随机。真正需要调的只有正则权重 $\lambda$，论文取 $\lambda\approx 0.01$，且在很宽的范围内都稳定，这也是后面实验中"无需精细调参即可跨数据集恢复维度"的关键。

## 实验关键数据

### 1. 合成数据（已知真实维度）
- 系统变化4个因素：查询比例、内在秩、感知数量、噪声水平
- **LORE是唯一能恢复真实内在秩的方法**，其他所有方法默认使用最大允许维度
- λ≈0.01在所有条件下均表现稳定→无需精细调参
- 随内在秩增加，LORE能跟踪变化→其他方法完全不变

### 2. LLM模拟感知实验
- 用SBERT嵌入50种食物→截断SVD控制内在维度(1-10)→生成噪声三元组
- LORE准确跟踪内在秩，且三元组准确率显著优于基线
- Dim-CV不仅维度估计更差，运行时间高出**数量级**（log尺度差异！）

### 3. 众包真实数据（3个数据集）

| 数据集 | LORE维度 | 其他方法维度 | LORE准确率 | 最佳基线准确率 |
|--------|---------|------------|-----------|-------------|
| Food-100 | **3.3** | 15 | 82.45% | 82.79% |
| Materials | **2.23** | 15 | **84.08%** | 83.94% |
| Cars | **3.0** | 15 | 52.12% | 54.06% |

- LORE用远低于基线的维度(~3 vs 15)达到相当甚至更高的准确率
- Dim-CV严重欠拟合（Food: 77.67%, Cars: 50.43%）→保守的假设检验策略失败
- LORE运行速度排第二（仅次于FORTE）

### 4. 语义可解释性
- Food-100数据集LORE学到的前3个轴对应可解释的食物属性：
    - 轴1: 甜 → 咸
    - 轴2: 密实 → 轻盈
    - 轴3: 碳水化合物丰富 → 蛋白质/蔬菜
- 无需语义监督即自动发现→对科学发现极有价值

## 与现有方法的系统对比

| 方法 | 优化对象 | 恢复维度 | 可扩展 | 高准确率 | 可解释轴 |
|------|---------|---------|--------|---------|---------|
| GNMDS | Gram矩阵 | ✗ | ✗ | ✗ | ✗ |
| CKL | Gram矩阵 | ✗ | ✗ | ✓ | ✓ |
| FORTE | Gram矩阵 | ✗ | ✓ | ✓ | ✗ |
| t-STE | 嵌入 | ✗ | — | ✓ | ✗ |
| SOE | 嵌入 | ✗ | ✓ | ✓ | ✗ |
| Dim-CV | 多嵌入 | 部分 | ✗ | ✗ | — |
| **LORE** | **嵌入** | **✓** | **✓** | **✓** | **✓** |

## 局限性
- 缺乏精确秩恢复或全局最优的理论保证（仅保证收敛到稳定点）
- 高内在秩时因固定三元组数量和维度诅咒，恢复精度下降
- Cars数据集上所有方法准确率均不高(~52-54%)→极端噪声数据的挑战

## 亮点与洞察
- **心理物理学的核心问题回答**："感知空间有几维？"是心理物理学的根本问题→LORE是首个数据驱动、端到端回答该问题的方法
- **非凸正则化的精妙应用**：Schatten-p (p<1)虽引入额外非凸性→但迭代重加权算法将其分解为一系列凸子问题→保证收敛→在低秩恢复上远优于凸核范数松弛
- **"维度本身就是科学发现"**：知道味觉空间是2维还是10维→直接揭示人类感知的内在结构→这比嵌入本身可能更有价值
- **实用性强**：仅一个需调超参(λ≈0.01)→跨数据集稳定→即将集成到cblearn库→降低使用门槛
- **跨领域潜力**：不限于心理物理学→任何只有相对比较数据（无绝对量度）的场景均适用→如推荐系统、美学评估、材料感知

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个联合学习OE维度和嵌入的方法，Schatten-p在OE中首次应用
- 实验充分度: ⭐⭐⭐⭐⭐ 合成+LLM模拟+3个真实众包数据集，系统消融4个因素
- 写作质量: ⭐⭐⭐⭐⭐ 问题motivation清晰，数学推导严谨，图表信息量大
- 价值: ⭐⭐⭐⭐ 对感知科学和表示学习有重要理论与实用贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning to Weight Parameters for Training Data Attribution](learning_to_weight_parameters_for_training_data_attribution.md)
- [\[ICLR 2026\] Estimating Dimensionality of Neural Representations from Finite Samples](estimating_dimensionality_of_neural_representations_from_finite_samples.md)
- [\[ICML 2026\] Dimensionality Controls When Modularity Helps in Continual Learning](../../ICML2026/interpretability/dimensionality_controls_when_modularity_helps_in_continual_learning.md)
- [\[ICLR 2026\] Conjuring Semantic Similarity](conjuring_semantic_similarity.md)
- [\[ICLR 2026\] Sequences of Logits Reveal the Low Rank Structure of Language Models](sequences_of_logits_reveal_the_low_rank_structure_of_language_models.md)

</div>

<!-- RELATED:END -->
