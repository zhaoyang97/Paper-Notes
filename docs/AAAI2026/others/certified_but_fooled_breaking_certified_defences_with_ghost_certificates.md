---
title: >-
  [论文解读] Certified but Fooled! Breaking Certified Defences with Ghost Certificates
description: >-
  [AAAI 2026][对抗攻击] 提出 GhostCert，一种基于显著性区域的对抗攻击方法，能在保持扰动不可感知的同时误导分类器并伪造大半径的认证证书（ghost certificates），在 ImageNet 上对包括 DensePure 在内的 SOTA 认证防御取得显著优于 Shadow Attack 的攻击成功率和更大的伪造认证半径。
tags:
  - "AAAI 2026"
  - "对抗攻击"
  - "认证防御"
  - "随机平滑"
  - "证书伪造"
  - "区域扰动"
---

# Certified but Fooled! Breaking Certified Defences with Ghost Certificates

**会议**: AAAI 2026  
**arXiv**: [2511.14003](https://arxiv.org/abs/2511.14003)  
**代码**: [github.com/ghostcert](https://github.com/ghostcert)  
**领域**: 其他  
**关键词**: 对抗攻击, 认证防御, 随机平滑, 证书伪造, 区域扰动

## 一句话总结

提出 GhostCert，一种基于显著性区域的对抗攻击方法，能在保持扰动不可感知的同时误导分类器并伪造大半径的认证证书（ghost certificates），在 ImageNet 上对包括 DensePure 在内的 SOTA 认证防御取得显著优于 Shadow Attack 的攻击成功率和更大的伪造认证半径。

## 研究背景与动机

### 问题定义

**认证防御**（如 Randomized Smoothing）承诺提供可证明的鲁棒性保证：在 $\ell_2$ 扰动球内，平滑分类器的预测保持不变。但这些保证的实际安全性有多可靠？

**证书伪造（Certificate Spoofing）**：攻击者不仅要误导分类器做出错误预测，还要操纵认证过程为错误输入颁发高置信度的鲁棒性证书。

### 现有攻击的不足

**Shadow Attack**（ICLR 2020）是唯一针对认证防御的先前工作，但存在关键缺陷：
1. 使用**大幅度全局扰动**将输入移到远离决策边界的区域
2. 依赖总变差（TV）、颜色通道均值等正则化来维持"自然"外观，形成**复杂的多目标优化**
3. 对集成模型效果大幅下降（ASR 仅约 40%）
4. 生成的对抗样本**视觉上不够自然**（$\|\delta\|_2$ 通常 >10）

### 核心动机

大幅度扰动并非必要——通过将扰动**约束在语义相关的显著性区域**，可以用更小的扰动实现更高的攻击成功率和更大的伪造证书半径。

## 方法详解

### 整体框架

GhostCert 的流程分三步：
1. **区域提案生成**：使用 SAM（Segment Anything Model）生成语义分割掩码
2. **显著性区域选择**：结合 GradCAM/Attention 的显著性信息选择 top-k 区域
3. **约束扰动优化**：在选定区域内使用 PGD 优化对抗扰动

### 关键设计

#### 1. **显著性区域掩码（Salient-Region Mask）**

将 GradCAM 的梯度显著性与 SAM 的语义分割边界结合：

- 对每个 SAM 分割掩码 $M_i$，计算与显著性图 $\mathcal{S}$ 的重叠分数：

$$\text{score}(M_i) = \frac{\sum_{x,y} M_i(x,y) \cdot \mathcal{S}(x,y)}{\sum_{x,y} M_i(x,y) + \sum_{x,y} \mathcal{S}(x,y)}$$

- 选择得分最高的 $k=5$ 个掩码（包括未被任何掩码覆盖的区域 $U$），合并为最终掩码：$m = \sum_{M \in \mathcal{T}} M$

**设计动机**：GradCAM 的显著性图产生无定形区域，忽略自然图像边界，导致不自然的伪影。SAM 的分割掩码提供语义连贯的边界。两者结合确保扰动在语义上有意义且视觉上自然。

#### 2. **约束扰动优化**

**非定向攻击**优化目标：

$$\max_\delta \sum_{i=1}^N L(f_\theta(x + \Delta_i + \delta \odot m), y) \quad \text{s.t.} \|\delta \odot m\|_2 \leq \epsilon$$

其中 $\Delta_i$ 是高斯噪声（标准差 $\sigma$），$\odot$ 为逐元素乘法，$m$ 是选定区域掩码。

**定向攻击**则将损失改为最大化目标类别的概率。

使用 PGD 进行梯度上升：$\delta \leftarrow \delta + \lambda \cdot \frac{g}{\|g\|_2}$，然后投影：$\delta \leftarrow (\epsilon \frac{\delta}{\|\delta\|_2}) \odot m$。

#### 3. **适配不同防御方法**

针对三种认证防御分别调整优化目标：

- **Randomized Smoothing（RS）**：$\max_\delta \sum_i L(f(x + \epsilon_i + \delta), y)$
- **Smoothed Ensemble**：$\max_\delta \sum_i L(\bar{f}(x + \epsilon_i + \delta), y)$
- **DensePure（去噪平滑）**：$\max_\delta \sum_i L(f(D_\theta(x + \epsilon_i + \delta)), y)$

对 Transformer 模型使用 Attention maps 替代 GradCAM。

### 损失函数 / 训练策略

- 使用**交叉熵损失**
- PGD 步长 $\lambda = 0.0001$
- 每次攻击使用 $N=1000$ 个 Monte Carlo 样本估计平滑概率
- 扰动预算 $\epsilon \in \{2, 4, 6, 8, 10\}$
- Randomized Smoothing 的失败概率 $\alpha = 0.001$

## 实验关键数据

### 主实验

**非定向攻击成功率（ASR）对比**：

| 防御方法 | $\sigma$ | $\epsilon$ | GhostCert | Shadow (bounded) | Shadow (原始) |
|---------|--------|----------|-----------|-----------------|-------------|
| Single RS (ResNet50) | 0.25 | 10 | **~98%** | ~60% | ~65% |
| Ensemble RS | 0.25 | 10 | **~100%** | ~40% | ~40% |
| Ensemble RS | 0.5 | 10 | **~85%** | ~35% | ~30% |
| DensePure | 0.25 | 10 | **~100%** | ~50% | ~55% |
| DensePure | 0.5 | 10 | **~55%** | ~35% | ~40% |

**伪造认证半径**：GhostCert 一致性地产生更大或可比的伪造认证半径，且通常**超过源图像的真实认证半径**（"strongly certified"）。

### 消融实验

| 配置 | ASR（非定向，$\epsilon$=10） | ASR（定向，$\epsilon$=10） | 说明 |
|------|--------------------------|--------------------------|------|
| GhostCert（完整，k=5） | **90%** | **30%** | 完整方法 |
| 随机像素掩码（50%） | 90% | 5% | 定向攻击差很多 |
| 随机区域提案（k=5，无显著性） | 45% | 0% | 显著性选择关键 |
| k=3 | 90% | 15% | 区域数不足 |
| k=7 | 90% | 35% | 与 k=5 相近 |

### 关键发现

1. **GhostCert 在集成防御上的优势最为显著**：Shadow Attack 在集成模型上 ASR 从 ~65% 骤降至 ~40%，而 GhostCert 保持 ~100%
2. **更低扰动 + 更大伪造半径**：GhostCert 在 $\|\delta\|_2=4$ 时即可达到 Shadow Attack 在 $\|\delta\|_2=13$ 时的效果
3. **人类用户研究**：在 Amazon Mechanical Turk 上，GhostCert 生成的图像在所有扰动水平下都被认为更自然（74% at $\epsilon$=2, 62% at $\epsilon$=10）
4. **DoS 攻击**：当 ASR 较低时，GhostCert 的输入更频繁地导致认证方法**弃权**（abstain），形成拒绝服务攻击
5. **重要声明**：攻击不"invalidate"证书——证书关于有界范数内输入不是对抗样本的断言仍然正确。攻击揭示的是认证框架的**实用安全性边界**

## 亮点与洞察

1. **约束优化 vs 多目标优化**的范式转变：Shadow Attack 将语义约束作为正则项加入损失（多目标），GhostCert 通过区域掩码将其转为约束（缩小搜索空间），更高效
2. **SAM + GradCAM 的组合**是关键创新：GradCAM 告诉"哪里重要"，SAM 告诉"如何尊重边界"
3. **新的威胁维度**：证书伪造不仅是分类错误，还产生"虚假安全感"——这对安全关键应用（自动驾驶、医疗）尤为危险
4. 揭示了认证防御的一个根本局限：$\ell_2$ 范数约束的认证在语义空间中可能不够

## 局限与展望

1. **白盒威胁模型**：需要完全访问模型参数，现实中攻击者未必有此权限
2. 对 DensePure 在 $\sigma=0.5$ 时的 ASR 显著下降，说明更强的去噪器仍具防御潜力
3. 每次攻击需 1000 个 Monte Carlo 样本，计算成本较高
4. 用户研究规模有限（~50 人），可能存在偏差
5. 未探索对其他认证方法（如确定性认证）的攻击效果
6. SAM 模型本身的计算开销未被纳入考量

## 相关工作与启发

- **Shadow Attack**（Ghiasi et al., ICLR 2020）是直接对比目标，GhostCert 在所有维度上全面超越
- **Randomized Smoothing**（Cohen et al., 2019）提供的认证半径公式 $R = \frac{\sigma}{2}[\Phi^{-1}(p_A) - \Phi^{-1}(p_B)]$ 是攻击优化的直接目标
- **DensePure**（Xiao et al., 2022）使用扩散模型去噪器是当前最强认证防御，GhostCert 首次对其进行了系统评估
- 本文的区域约束思想可能启发对抗样本研究中更自然的扰动模型

## 评分

- 新颖性: ⭐⭐⭐⭐ （SAM+GradCAM 区域选择思路新颖，证书伪造场景重要）
- 实验充分度: ⭐⭐⭐⭐⭐ （三种防御 × 三种攻击 × 多个σ/ε + 定向/非定向 + 用户研究 + 消融）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，实验展示直观）
- 价值: ⭐⭐⭐⭐ （对认证防御社区是重要的安全警示，推动了对认证可靠性的反思）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables](faster_certified_symmetry_breaking_using_orders_with_auxiliary_variables.md)
- [\[AAAI 2026\] Certified Branch-and-Bound MaxSAT Solving (Extended Version)](certified_branch-and-bound_maxsat_solving_extended_version.md)
- [\[AAAI 2026\] Axis-Aligned Document Dewarping](axis-aligned_document_dewarping.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)
- [\[AAAI 2026\] DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design](deeprwcap_neural-guided_random-walk_capacitance_solver_for_ic_design.md)

</div>

<!-- RELATED:END -->
