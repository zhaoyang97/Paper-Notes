---
title: >-
  [论文解读] Private Model Personalization Revisited
description: >-
  [ICML 2025][AI安全][差分隐私] 提出 Private FedRep 算法，在用户级差分隐私 (DP) 约束下通过交替最小化框架学习共享低维嵌入 $U^* \in \mathbb{R}^{d \times k}$（$k \ll d$），将隐私误差项相比先前工作 Jain et al. 降低 $\widetilde{O}(dk)$ 倍，且适用于更广泛的 sub-Gaussian 分布（而非仅限高斯），并通过 Johnson-Lindenstrauss 变换给出维度无关的分类风险界。
tags:
  - "ICML 2025"
  - "AI安全"
  - "差分隐私"
  - "模型个性化"
  - "共享表示学习"
  - "联邦学习"
  - "Johnson-Lindenstrauss"
---

# Private Model Personalization Revisited

**会议**: ICML 2025  
**arXiv**: [2506.19220](https://arxiv.org/abs/2506.19220)  
**代码**: 待确认  
**领域**: AI安全  
**关键词**: 差分隐私, 模型个性化, 共享表示学习, 联邦学习, Johnson-Lindenstrauss

## 一句话总结

提出 Private FedRep 算法，在用户级差分隐私 (DP) 约束下通过交替最小化框架学习共享低维嵌入 $U^* \in \mathbb{R}^{d \times k}$（$k \ll d$），将隐私误差项相比先前工作 Jain et al. 降低 $\widetilde{O}(dk)$ 倍，且适用于更广泛的 sub-Gaussian 分布（而非仅限高斯），并通过 Johnson-Lindenstrauss 变换给出维度无关的分类风险界。

## 研究背景与动机

**领域现状**：模型个性化（Model Personalization）是应对用户数据统计异质性的核心策略。在共享表示学习框架中，$n$ 个用户协作学习一个低维嵌入 $U^* \in \mathbb{R}^{d \times k}$（$k \ll d$），每个用户再训练自己的局部模型 $v_i \in \mathbb{R}^k$，使得 $w_i^* = U^* v_i^*$。FedRep（Collins et al., 2021）是该框架下的代表性非隐私算法。

**现有痛点**：Jain et al. (2021) 提出了第一个有隐私保证的模型个性化算法，但存在三个主要限制：(1) 要求用户数据服从标准高斯分布，假设过强；(2) 需要集中式处理——嵌入更新涉及所有用户数据的联合优化，难以转化为联邦算法；(3) 隐私误差项中包含 $\widetilde{O}(d^3 k^2)$ 级别的高维依赖，在高维场景下效用损失严重。

**核心矛盾**：隐私保护与个性化效用之间的 trade-off 在高维设置下尤为尖锐——嵌入维度 $d$ 越高，需要注入的 DP 噪声越多，而现有方法的噪声校准不够紧致，导致隐私代价过大。

**本文目标** (1) 设计原生联邦的隐私模型个性化算法；(2) 在更宽松的数据分布假设下（sub-Gaussian）给出更紧的效用保证；(3) 为分类场景推导与输入维度 $d$ 无关的风险界。

**切入角度**：作者观察到 FedRep 的交替最小化结构天然适合联邦设置——用户局部更新 $v_i$，服务器聚合嵌入梯度。关键创新是使用不相交数据批次分别更新局部向量和计算嵌入梯度，从而收紧梯度 Frobenius 范数的上界，减少所需的 DP 噪声量。

**核心 idea**：通过不相交批次 + QR 正交化 + 裁剪聚合的联邦交替最小化框架，在 sub-Gaussian 假设下实现比 SOTA 紧 $\widetilde{O}(dk)$ 倍的隐私-效用权衡。

## 方法详解

### 整体框架

输入为 $n$ 个用户的数据集 $S_i = \{(x_{i,j}, y_{i,j})\}_{j=1}^m$，输出为共享嵌入矩阵 $U^{\text{priv}} \in \mathbb{R}^{d \times k}$（正交列）和每个用户的局部向量 $v_i^{\text{priv}} \in \mathbb{R}^k$。算法分为三个阶段：(1) 隐私初始化（Algorithm 2）获得初始嵌入 $U_{\text{init}}$；(2) Private FedRep 迭代（Algorithm 1）交替更新局部向量和嵌入；(3) 每个用户独立求解自己的最终局部模型。

### 关键设计

1. **不相交批次的交替最小化**:

    - 功能：每轮迭代中，每个用户从自己的数据中采样两个不相交的批次 $B_{i,t}$ 和 $B'_{i,t}$，分别用于更新局部向量 $v_{i,t}$ 和计算嵌入梯度 $\nabla_{i,t}$
    - 核心思路：用户先在 $B_{i,t}$ 上求解 $v_{i,t} = \arg\min_v \hat{L}(U_t, v; B_{i,t})$，再在独立的 $B'_{i,t}$ 上计算 $\nabla_{i,t} = \nabla_U \hat{L}(U_t, v_{i,t}; B'_{i,t})$。由于两个批次独立，梯度范数的上界更紧：$\|\nabla_{i,t}\|_F \leq \widetilde{O}((R+\Gamma)\Gamma\sqrt{dk})$
    - 设计动机：原始 FedRep 复用同一批次导致局部向量和梯度之间存在依赖关系，使得梯度范数的高概率上界更松散。不相交批次打破了这种依赖，直接减少了 DP 噪声的注入量

2. **裁剪聚合 + QR 正交化**:

    - 功能：服务器收到各用户的梯度后，先裁剪到 $\psi$ 范数球内，汇聚并加入高斯噪声 $\xi_{t+1} \sim \mathcal{N}^{d \times k}(0, \hat{\sigma}^2)$，然后通过 QR 分解将更新后的嵌入重新正交化
    - 核心思路：嵌入更新为 $\hat{U}_{t+1} = U_t - \eta(\frac{1}{n}\sum_i \text{clip}(\nabla_{i,t}, \psi) + \xi_{t+1})$，然后 $U_{t+1}, P_{t+1} = \text{QR}(\hat{U}_{t+1})$。噪声标准差 $\hat{\sigma} = C\psi\sqrt{T\log(1/\delta)}/(n\epsilon)$
    - 设计动机：QR 正交化保证 $U_t$ 始终有正交列，这一性质确保噪声不会在迭代间传播累积——因为 $\|U_t\|_2 = 1$ 恒成立，每轮的梯度范数上界不受历史噪声影响

3. **隐私初始化（基于子空间恢复）**:

    - 功能：基于 Duchi et al. 的子空间恢复估计器，通过 U-统计量 $Z_i = \frac{2}{m(m-1)}\sum_{j_1 \neq j_2} y_{i,j_1} y_{i,j_2} x_{i,j_1} x_{i,j_2}^\top$ 估计二阶矩矩阵，加噪后取 top-$k$ SVD 获得初始嵌入
    - 核心思路：这个 U-统计量的期望为 $U^* V^{*\top} V^* U^{*\top}$（加上噪声和单位阵的贡献），其 top-$k$ 特征向量跨越 $U^*$ 的列空间。裁剪 $\psi_{\text{init}} = \widetilde{O}((R^2 + \Gamma^2)d)$ 后加入高斯噪声保证 DP
    - 设计动机：Algorithm 1 的收敛需要初始嵌入 $U_0$ 与真实 $U^*$ 的主角度距离小于 $\sqrt{1-c}$。随机初始化在高维空间中几乎正交于 $U^*$，无法满足此条件

### 损失函数 / 训练策略

回归场景使用二次损失 $\ell(U, v, (x,y)) = (y - \langle x, Uv \rangle)^2$。标签生成模型为 $y = x^\top U^* v_i^* + \zeta$，其中 $\zeta$ 为 $R$-sub-Gaussian 噪声。学习率 $\eta = 1/(2\Lambda^2)$，迭代次数 $T = \Theta(\Lambda^2 \log(n^3)/\lambda^2)$，batch size $b = \lfloor m/(2T) \rfloor$。整个算法通过 billboard model 实现用户级 DP：服务器广播 $U^{\text{priv}}$，各用户独立求解局部模型。

## 实验关键数据

### 主实验

| 方法 | 隐私误差项 | 数据假设 | 联邦支持 |
|------|-----------|---------|---------|
| Jain et al. (2021) | $\widetilde{O}(d^3 k^2 / (n^2 \epsilon^2 \sigma_{\min,*}^4))$ | 标准高斯 | 否（集中式） |
| **Private FedRep（本文）** | $\widetilde{O}(d^2 k / (n^2 \epsilon^2 \sigma_{\min,*}^4))$ | sub-Gaussian | **是** |
| 提升倍数 | $\widetilde{O}(dk)$ | 更宽松 | — |

### 消融实验

| 配置 | excess risk 关键项 | 说明 |
|------|-------------------|------|
| 完整 Private FedRep | $\widetilde{O}(d^2k/(n^2\epsilon^2\sigma_{\min,*}^4) + d/(nm\sigma_{\min,*}^4)) + k/m$ | 隐私项 + 统计项 + 局部项 |
| 去掉不相交批次 | 梯度范数上界变松 $\widetilde{O}(dk)$ 倍 | 需要更多噪声 |
| 去掉 QR 正交化 | 噪声跨迭代累积 | 收敛不稳定 |
| 分类场景 (JL 变换) | $d$ 无关的 margin-based 风险界 | 通过降维至 $O(\log n/\epsilon^2)$ |

### 关键发现

- **隐私误差的主要改进来自不相交批次**：将梯度 Frobenius 范数的上界从 $\widetilde{O}(d^{3/2}k)$ 收紧到 $\widetilde{O}(\sqrt{dk})$ 级别（消除了 $v$ 和梯度之间的统计依赖），这直接降低了裁剪阈值 $\psi$ 和所需的噪声标准差 $\hat{\sigma}$
- **sub-Gaussian 扩展是非平凡的**：在非高斯分布下，距离的几何递减不再以高概率在每轮都成立（因为协方差矩阵的集中性更弱），作者采用归纳论证（induction-based argument）来处理这种间歇性的不收缩
- **分类的维度无关结果依赖 JL 变换**：通过将 $d$ 维空间随机投影到 $O(\log n)$ 维，在 margin loss 下实现了与 $d$ 完全无关的风险界，但代价是需要有界范数假设

## 亮点与洞察

- **不相交批次的巧妙之处**：这是一个看似微小但效果显著的修改——仅仅是将一个批次拆成两个独立使用，就消除了关键的统计依赖，带来了 $\widetilde{O}(dk)$ 倍的效用提升。这种"去耦合"思想在其他涉及隐私梯度裁剪的场景中也可能有用
- **QR 正交化防止噪声传播**：在隐私联邦优化中，噪声的迭代累积是核心难题。通过每步投影回 Stiefel 流形（正交矩阵集），巧妙地将噪声影响局限在当前迭代，这个 trick 值得在其他低秩优化问题中借鉴
- **Billboard model 的实用性**：相比需要安全聚合（secure aggregation）的方案，billboard model 允许服务器公开广播结果，各用户独立训练局部模型，实现简单且不泄露局部向量

## 局限与展望

- **仅分析了线性回归和二分类**：实际的联邦个性化场景涉及深度网络和非凸优化，理论结果的适用性需要进一步验证
- **对 $\sigma_{\min,*}$ 的依赖**：excess risk 中的 $\sigma_{\min,*}^{-4}$ 依赖意味着当用户多样性低（即 $V^*$ 的条件数大）时，效用会急剧退化。这在用户任务高度同质的场景中可能是瓶颈
- **初始化需要额外的隐私预算**：Algorithm 2 的初始化步骤消耗 $(\epsilon/2, \delta/2)$ 的隐私预算，使得留给主算法的预算减半。是否存在免初始化或共享预算的方案值得探索
- **缺少实证实验**：全文为纯理论结果，没有在实际数据集上验证算法的实际表现。合成数据或 FEMNIST 等联邦基准上的实验会大大增强说服力

## 相关工作与启发

- **vs Jain et al. (2021)**：两者都是共享表示学习框架下的 DP 个性化，但 Jain 的算法需集中式精确最小化（嵌入更新需访问所有用户数据），隐私误差项多 $\widetilde{O}(dk)$ 倍，且限于高斯分布。本文通过联邦梯度+不相交批次全面超越
- **vs Collins et al. (2021) FedRep**：本文是 FedRep 的隐私化版本，除了加入 DP 机制外，还扩展到了有标签噪声的非可实现设置（non-realizable case）。原始 FedRep 的分析仅适用于无噪声标签
- **vs JL-based 隐私学习**：Bassily et al. (2022) 等工作用 JL 变换降维实现隐私学习，本文将该技术引入联邦个性化场景的分类设置，获得了维度无关的风险保证

## 评分

- 新颖性: ⭐⭐⭐⭐ 不相交批次去耦合和 JL 分类结合是有意义的技术创新
- 实验充分度: ⭐⭐ 纯理论工作，缺少任何实证验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，定理陈述规范，技术贡献的定位明确
- 价值: ⭐⭐⭐⭐ 推进了隐私联邦个性化的理论前沿，但实际影响需等待实验验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICML 2025\] Faster Rates for Private Adversarial Bandits](faster_rates_for_private_adversarial_bandits.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](understanding_model_ensemble_in_transferable_adversarial_attack.md)

</div>

<!-- RELATED:END -->
