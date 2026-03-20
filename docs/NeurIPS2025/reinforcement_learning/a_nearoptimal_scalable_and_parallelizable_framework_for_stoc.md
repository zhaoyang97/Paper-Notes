# A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond

**会议**: NeurIPS 2025  
**arXiv**: [2502.07514](https://arxiv.org/abs/2502.07514)  
**代码**: 无  
**领域**: AI Safety / 在线学习 / 多臂老虎机  
**关键词**: adversarial corruptions, multi-armed bandits, elimination-based algorithm, regret bound, parallelizable  

## 一句话总结
提出 BARBAT 框架，改进了经典的 BARBAR 算法，通过固定 epoch 长度和逐 epoch 调整失败概率，将对抗腐蚀下随机多臂老虎机的 regret 从 $O(\sqrt{K}C)$ 降至近最优的 $O(C)$（消除了 $\sqrt{K}$ 因子），并成功扩展到多智能体、图老虎机、组合半老虎机和批量老虎机等多种场景。

## 背景与动机
多臂老虎机（MAB）是在线学习中最基本的问题之一。近年来安全性备受关注，研究者开始考虑对手可以篡改奖励观测值的"对抗腐蚀"设定。现有方法分两大类：

1. **FTRL 系列**（Follow-the-Regularized-Leader）：能在随机和对抗环境中都达到最优 regret，但每轮需要解一个约束凸优化问题，在半老虎机等场景下计算开销极大，且难以并行化到多智能体或批量设定。
2. **消除型方法**（如 BARBAR）：计算高效且易并行化，但 BARBAR 的 regret 中包含 $O(\sqrt{K}C)$ 项，存在 $\sqrt{K}$ 因子冗余，与下界 $\Omega(C)$ 不匹配。

核心痛点在于：能否用消除型方法达到与 FTRL 同级别的最优 regret？这是 Gupta et al. 提出的公开问题。

## 核心问题
在对抗腐蚀下的随机多臂老虎机问题中，如何设计消除型算法使其 regret 达到近最优（即 corruption 依赖项为 $O(C)$ 而非 $O(\sqrt{K}C)$），同时保持消除型方法计算高效和易并行化的优势？

## 方法详解

### 整体框架
BARBAT（**B**ad **A**rms get **R**ecourse, **B**est **A**rm gets **T**rust）是对 BARBAR 的改进版本。核心思路是在每个 epoch 中：
- 输入：当前存活臂集合、上一轮估计的次优性 gap
- 过程：按照预设概率采样各臂，累计观测奖励
- 输出：更新后的经验奖励估计和 gap 估计

整体采用 epoch 迭代结构，每个 epoch 结束后重新估计每个臂的次优性 gap，逐步聚焦于最优臂。

### 关键设计

1. **固定 epoch 长度（Data-independent epoch length）**：BARBAR 的 epoch 长度依赖于上一轮的数据（即估计的 gap），这使得对手可以通过在某个 epoch 集中攻击来间接延长下一个 epoch 的长度，导致 $O(\sqrt{K} C)$ 的额外 regret。BARBAT 改为使用与数据无关的 epoch 长度 $N_m = \lceil K \lambda_m 2^{2(m-1)} \rceil$，对手无法通过腐蚀来操纵 epoch 长度。多出来的采样次数（即 $N_m - \sum_{k \neq k_m} n_k^m$）全部分配给当前估计的最优臂 $k_m$，体现 "Best Arm gets Trust" 思想。

2. **逐 epoch 变化的失败概率（Epoch-varying failure probabilities）**：BARBAR 使用全局统一的失败概率 $\delta$，而 BARBAT 为每个 epoch $m$ 设置不同的 $\delta_m = 1/(K \zeta_m)$。这种精细设计消除了对时间范围 $T$ 的先验知识需求，同时使得总失败概率被良好控制。

3. **BARBAR 为何付出 $O(\sqrt{K}C)$ 的直觉说明**：Gupta et al. 构造了一个反例——如果对手在某个 epoch $c$ 投入所有腐蚀预算 $C = N_c$，BARBAR会丢失之前积累的信息，下一个 epoch 长度变为 $N_{c+1} = O(KC)$，每个臂被均匀采样约 $O(C)$ 次，产生 $O((K-1)C)$ regret。而 BARBAT 由于固定了 epoch 长度为 $N_m = K \lambda_m 2^{2(m-1)}$，对手要攻击整个 epoch 需要 $C = N_c = K \lambda 2^{2(c-1)}$，比攻击 BARBAR 需要多 $K$ 倍的腐蚀预算，因此只需付出 $O(C)$ 的腐蚀代价。

### 扩展场景

BARBAT 体现了良好的可扩展性和可并行化特性，被推广到以下四个场景：

- **MA-BARBAT（多智能体）**：$V$ 个智能体协作，每个 epoch 结束后广播各自奖励，个体 regret 降低 $V$ 倍——$R_v(T) = O(C/V + \sum_{\Delta_k>0} \log^2(VT) / (V\Delta_k))$。通信代价仅为 $O(V \log(VT))$。
- **BB-BARBAT（批量老虎机）**：适配 $L$ 个 batch 的限制，epoch 长度设为 $O(T^{m/(L+1)})$，首次研究对抗腐蚀下的批量老虎机，并给出下界。
- **SOG-BARBAT（强可观测图老虎机）**：利用图结构的 out-domination set 来减少实际需要采样的臂数，通过 OODS 算法高效计算该集合，regret 中与 gap 相关的项只依赖 $O(\alpha \ln(K/\alpha))$ 个最小 gap 的臂。
- **DS-BARBAT（$d$-set 半老虎机）**：处理组合动作空间，最优动作变为 $d$ 个臂的集合，regret 为 $O(dC + \sum_{k=d+1}^{K} \log^2(T)/\Delta_k)$。

### 损失函数 / 训练策略
这是一篇理论方法论文，不涉及神经网络训练。关键的技术工具包括：
- Chernoff-Hoeffding 不等式控制奖励估计误差
- Freedman 鞅集中不等式处理腐蚀分量
- 归纳法建立 gap 估计的上下界递推关系
- 定义 "offset level" $D_m$ 和 "discounted offset rate" $\rho_m$ 统一分析腐蚀和失败事件的影响

## 实验关键数据

| 场景 | 设置 | BARBAT 时间 | 最强基线时间 | BARBAT 优势 |
|------|------|-------------|-------------|-------------|
| 多智能体 MAB (K=12) | V=10, T=50000 | 0.13 s | IND-FTRL: 1.53 s | ~12x 加速 |
| 多智能体 MAB (K=16) | V=10, T=50000 | 0.13 s | IND-FTRL: 1.91 s | ~15x 加速 |
| d-set 半老虎机 (K=12) | d=3, T=50000 | 0.36 s | HYBRID: 14.63 s | ~40x 加速 |
| d-set 半老虎机 (K=16) | d=4, T=50000 | 0.37 s | LBINF_GD: 12.69 s | ~34x 加速 |

在所有场景中，BARBAT 的累积 regret 曲线在高腐蚀水平下（C=5000）都明显低于基线方法，验证了理论分析中 corruption 项的改进。

### 消融实验要点
- 论文主要通过对比不同 corruption 水平（C=2000 vs C=5000）展示鲁棒性差异
- MA-BARBAT 协作带来的 $V$ 倍 regret 降低在实验中得到验证
- 计算效率优势在半老虎机场景最为突出（40x 加速），因为 FTRL 在此场景需求解昂贵的凸优化

## 亮点
- **关键洞察精炼**：用"固定 epoch 长度"这一简单修改消除 $\sqrt{K}$ 因子，解决了 BARBAR 的核心缺陷——对手可以通过腐蚀间接操控 epoch 长度
- **消除 $T$ 的先验假设**：通过逐 epoch 设置 $\delta_m$，无需预知时间范围，这对实际部署非常有利
- **可扩展性设计典范**：一个核心框架自然地适配四种不同的老虎机变体，每次扩展仅需局部修改
- **计算效率压倒性优势**：相比 FTRL 在半老虎机上 40 倍的速度提升，源于消除了每轮的凸优化
- **不需要唯一最优臂假设**：FTRL 方法（除 MAB 外）需要此假设，但 BARBAT 系列不需要

## 局限性 / 可改进方向
- regret 中有 $\log^2(T)$ 因子，而非最优的 $\log(T)$，作者明确指出缩小此差距是未来方向
- 批量老虎机的上界与下界仍有 gap，最优 regret 仍为开放问题
- 多智能体版本需要中心化通信（每个智能体广播给所有人），未考虑去中心化或通信受限场景
- 图老虎机部分的 out-domination set 计算虽然是多项式时间，但实际复杂度为 $O(|V|^2 |E|)$
- 论文未扩展到线性老虎机等重要结构化设定，作者将此列为未来方向

## 与相关工作的对比

| 方法 | corruption 项 | 计算效率 | 可并行化 | 无需唯一最优臂 |
|------|--------------|---------|---------|--------------|
| BARBAR | $O(\sqrt{K}C)$ | ✓ | ✓ | ✓ |
| Tsallis-FTRL | $O(C)$ | ✗ | ✗ | ✗（MAB 外） |
| Shannon-FTRL | $O(C)$ | ✓（有闭式解） | ✓ | ✓ |
| **BARBAT** | $O(C)$ | ✓ | ✓ | ✓ |

核心差异：BARBAT 是唯一同时满足近最优 regret、计算高效、可并行化、无需唯一最优臂假设四个条件的消除型方法。Shannon-FTRL 虽在 MAB 上有闭式解，但在半老虎机等组合场景下仍需解优化问题。

## 启发与关联
- **epoch 固定化思想**可迁移到其他需要抵御对抗攻击的 epoch-based 在线学习算法中
- **discounted offset rate** 的分析技巧是一种优雅的处理历史腐蚀累积影响的方式，可借鉴

## 评分
- 新颖性: ⭐⭐⭐⭐ 核心思想（固定 epoch + epoch-varying δ）虽简洁但针对性强，完整解决了一个公开问题
- 实验充分度: ⭐⭐⭐⭐ 覆盖三种场景、多种基线、计算时间对比，但缺少更大规模实验
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清晰，直觉解释（BARBAR 为何付出 √K·C 的例子）写得极好，证明完整
- 价值: ⭐⭐⭐⭐ 解决公开问题且框架统一性强，但领域相对小众
