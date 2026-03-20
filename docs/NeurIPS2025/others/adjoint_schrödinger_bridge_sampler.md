# Adjoint Schrödinger Bridge Sampler

**会议**: NeurIPS 2025  
**arXiv**: [2506.22565](https://arxiv.org/abs/2506.22565)  
**代码**: https://github.com/facebookresearch/adjoint_samplers  
**领域**: 扩散模型 / 采样方法 / 分子模拟  
**关键词**: Schrödinger Bridge, Diffusion Sampler, Boltzmann分布, Adjoint Matching, 随机最优控制

## 一句话总结
提出 Adjoint Schrödinger Bridge Sampler (ASBS)，通过将 Schrödinger Bridge 问题重新解释为随机最优控制问题，消除了先前扩散采样器的 memoryless 条件限制，支持任意源分布（如高斯、谐波先验），使用可扩展的 matching 目标无需重要性权重估计，在多粒子能量函数和分子构象生成上全面超越先前方法。

## 研究背景与动机

1. **领域现状**：从 Boltzmann 分布 $\nu(x) \propto e^{-E(x)}$ 采样是计算科学的核心问题（贝叶斯推断、统计物理、化学），传统 MCMC 方法混合慢、能量评估开销大。近年扩散采样器（Diffusion Sampler）通过学习 SDE 的漂移 $u_t^\theta$ 将样本传输到目标分布。
2. **现有痛点**：由于 Boltzmann 分布只知道未归一化的能量函数而无显式样本，先前基于 matching 的扩散采样器（PDDS、iDEM）需要通过重要性权重估计目标样本，计算开销大。Adjoint Sampling (AS) 虽然用 Adjoint Matching 避免了重要性权重，但受限于 memoryless 条件——源分布必须是 Dirac delta $\mu(x) = \delta$。
3. **核心矛盾**：memoryless 条件排除了高斯先验、谐波先验等有用的源分布选择。已知非 memoryless 过程可以提升传输效率，但现有方法要么需要 memoryless，要么需要昂贵的非 matching 方法。
4. **本文要解决什么？** 如何在不需要 memoryless 条件、不需要重要性权重的前提下，用可扩展的 matching 目标学习扩散采样器？
5. **切入角度**：将 Schrödinger Bridge 问题的最优性条件重新解释为一个 SOC 问题，引入 corrector 函数 $\nabla \log \hat{\varphi}_1$ 来消除非 memoryless 引起的初始值函数偏差。
6. **核心idea一句话**：通过交替优化 Adjoint Matching（学漂移 $u$）和 Corrector Matching（学去偏 corrector $h$），等价于 IPF 算法，收敛到 Schrödinger Bridge 全局最优解。

## 方法详解

### 整体框架
ASBS 学习一个 SDE $dX_t = [f_t(X_t) + \sigma_t u_t^\theta(X_t)] dt + \sigma_t dW_t$，将源分布 $\mu$ 的样本传输到目标 Boltzmann 分布 $\nu$。与 AS 不同，源分布可以是任意分布（高斯、谐波先验等），基础漂移 $f_t = 0$（布朗运动）。算法交替训练两个网络：漂移网络 $u_\theta$ 和 corrector 网络 $h_\phi$。

### 关键设计

1. **SB 问题的 SOC 特征化 (Theorem 3.1)**:
   - 做什么：证明 Schrödinger Bridge 的动力学最优漂移 $u_t^*$ 可以通过求解一个带特殊终端代价的 SOC 问题获得
   - 核心思路：SB 的最优性方程涉及耦合的 SB 势函数 $\varphi_t, \hat{\varphi}_t$，直接求解困难。关键观察是前向 SB 势 $\varphi_t$ 的积分形式恰好类似 SOC 最优性条件，因此 SB 问题等价于终端代价 $g(x) = \log \frac{\hat{\varphi}_1(x)}{\nu(x)}$ 的 SOC 问题
   - 设计动机：将难以直接求解的 SB 问题转为可用 Adjoint Matching 的 SOC 问题，保留了 AS 的可扩展性

2. **Corrector Matching 去偏 (Eq. 15)**:
   - 做什么：学习 corrector 函数 $h_\phi \approx \nabla \log \hat{\varphi}_1$ 来消除非 memoryless 引起的偏差
   - 核心思路：$\nabla \log \hat{\varphi}_1$ 是反向时间方向上的动力学最优漂移在 $t=1$ 时的 Markovian 投影，可通过回归 $\min_h \mathbb{E}_{p_{0,1}^{u^{(k)}}} [\|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1|X_0)\|^2]$ 学习。关键：这个目标只依赖模型自身的样本，不需要目标分布样本
   - 设计动机：当源分布是 Dirac delta 时 $\nabla \log \hat{\varphi}_1 = \nabla \log p_1^{\text{base}}$ 已知，不需要额外学习；但对任意源分布必须显式学习 corrector

3. **交替优化 = IPF (Theorem 3.2)**:
   - 做什么：证明 Adjoint Matching 和 Corrector Matching 交替优化等价于 Iterative Proportional Fitting
   - 核心思路：AM 解的是前向半桥——固定源分布 $\mu$ 最小化 $D_{KL}(p \| q^{\bar{h}^{(k-1)}})$；CM 解的是后向半桥——固定目标分布 $\nu$ 最小化 $D_{KL}(p^{u^{(k)}} \| q)$。交替执行等价于 IPF，保证全局收敛 $\lim_{k \to \infty} u^{(k)} = u^*$
   - 设计动机：IPF 的收敛性保证了 ASBS 无需调参就能收敛到 SB 全局最优解

### 损失函数 / 训练策略
- **AM 损失**：$\min_u \mathbb{E}[\|u_t(X_t) + \sigma_t(\nabla E + h^{(k-1)})(X_1)\|^2]$，回归能量梯度 + corrector
- **CM 损失**：$\min_h \mathbb{E}[\|h(X_1) - \nabla_{x_1} \log p^{\text{base}}(X_1|X_0)\|^2]$
- 初始化 $h^{(0)} = 0$，第一个阶段等价于标准 AS
- 使用 replay buffer 存储历史样本，Adam 优化器
- 分子系统使用等变图神经网络 (EGNN)，谐波先验作源分布

## 实验关键数据

### 主实验

| 能量函数 | 指标 | ASBS | AS | 其他最佳 | 提升 |
|---------|------|------|-----|---------|------|
| MW-5 (d=5) | Sinkhorn ↓ | **0.15** | 0.32 | 0.44 (SCLD) | -53% vs AS |
| DW-4 (d=8) | $\mathcal{W}_2$ ↓ | **0.43** | 0.62 | 0.68 (PIS) | -31% vs AS |
| LJ-13 (d=39) | $\mathcal{W}_2$ ↓ | **1.59** | 1.67 | 1.61 (iDEM) | -5% vs AS |
| LJ-55 (d=165) | $\mathcal{W}_2$ ↓ | **4.00** | 4.04 | 4.60 (DDS) | -1% vs AS |
| 丙氨酸二肽 | $D_{KL}(\phi)$ ↓ | **0.02** | 0.09 | 0.03 (DDS) | -78% vs AS |

### 消融实验 (构象生成)

| 配置 | SPICE Coverage ↑ | GEOM Coverage ↑ | 说明 |
|------|-----------------|-----------------|------|
| ASBS + harmonic prior | **73.04%** | **50.23%** | 完整模型，领域先验 |
| ASBS + Gaussian prior | 67.58% | 41.23% | 去掉领域先验 |
| AS (Dirac prior) | 56.75% | 36.23% | 基线，memoryless |
| RDKit ETKDG | 56.94% | 50.81% | 化学启发式方法 |
| ASBS + harmonic + RDKit warmup | **85.82%** | **66.79%** | 最强配置 |

### 关键发现
- **在所有合成能量函数上全面超越 AS 和其他方法**，尤其在低维 (MW-5, DW-4) 提升显著（30-50%），高维 (LJ-55) 提升较小
- **谐波先验比高斯先验和 Dirac delta 更好**：说明领域特定的源分布确实提升了传输效率，验证了放松 memoryless 条件的价值
- **丙氨酸二肽上所有 5 个扭转角的 KL 散度都接近 0**：远优于 AS 和其他方法，Ramachandran 图几乎与 MD 真值重合
- **计算效率**：每次梯度更新的能量评估/模型评估次数与 AS 接近，仅多一个 corrector 网络的开销

## 亮点与洞察
- **SB → SOC 的理论洞见极为优雅**：将看似需要耦合边界条件的 SB 问题转化为可用 Adjoint Matching 的 SOC 问题，corrector 函数精确补偿了非 memoryless 偏差。这个思路可推广到其他 SB 应用场景
- **交替优化 = IPF 的证明为收敛性提供了理论保障**：不同于大多数深度学习方法的经验收敛，ASBS 有严格的全局收敛证明（前提是每阶段达到 critical point）
- **谐波先验的使用展示了领域知识整合的价值**：先前被 memoryless 条件排除的分子模拟领域标准先验现在可以自然融入扩散采样框架

## 局限性 / 可改进方向
- **每个 SB stage 需要 AM + CM 两步全训练**：虽然理论上保证收敛，实际中 stage 数和每 stage 训练步数需要调参
- **高维问题提升有限**：LJ-55 (d=165) 上仅比 AS 提升 1%，说明维度增大后非 memoryless 的优势减弱
- **corrector 网络增加了内存和计算开销**：虽然理论上可忽略，实际部署时额外网络的维护是工程负担
- **仅在 $f_t = 0$ 的 Brownian motion 基过程上验证**：虽然理论适用于一般 $f_t$，VP-SDE 等其他选择未实验验证

## 相关工作与启发
- **vs Adjoint Sampling (AS)**: AS 是 ASBS 的特例（$\mu = \delta, h = \nabla \log p_1^{\text{base}}$），ASBS 通过 corrector matching 将其推广到任意源分布，保留了所有可扩展性优势
- **vs PDDS/iDEM**: 这些方法也用 matching 目标但依赖重要性权重估计目标样本，ASBS 完全不需要
- **vs Sequential SB (SSB)**: 同样解一般 SB 问题，但 SSB 基于 SMC，每步需大量能量评估，可扩展性差
- **vs 数据驱动 SB (DSB, I²SB 等)**: 这些方法需要目标分布显式样本，不适用于 Boltzmann 采样

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SB 的 SOC 特征化和交替 matching 算法都是全新贡献，理论优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 合成函数 + 分子模拟 + 大规模构象生成，覆盖面广，对比全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，动机清晰，关键公式有直觉解释
- 价值: ⭐⭐⭐⭐⭐ 扩散采样器设计空间的重要突破，对分子模拟等领域有直接应用价值
