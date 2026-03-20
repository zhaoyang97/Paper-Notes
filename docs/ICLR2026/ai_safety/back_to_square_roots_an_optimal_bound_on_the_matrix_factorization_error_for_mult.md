# Back to Square Roots: An Optimal Bound on the Matrix Factorization Error for Multi-Epoch Differentially Private SGD

**会议**: ICLR 2026  
**arXiv**: [2505.12128](https://arxiv.org/abs/2505.12128)  
**代码**: 无（使用 jax-privacy 库进行基线比较）  
**领域**: AI安全 / 差分隐私  
**关键词**: differential privacy, matrix factorization, DP-SGD, multi-epoch participation, banded factorization, optimal error bounds  

## 一句话总结
提出 Banded Inverse Square Root (BISR) 矩阵分解方法，通过对逆相关矩阵（而非相关矩阵本身）施加带状结构，首次在多轮参与差分隐私 SGD 中实现渐近最优的分解误差界，并配套低存储优化变体 BandInvMF。

## 研究背景与动机
1. **领域现状**：矩阵分解机制（Matrix Factorization Mechanism）是差分隐私训练中通过注入相关噪声来提升模型效用的重要方法，已被 Google 用于生产级 on-device 语言模型训练。
2. **现有痛点**：在多轮训练（multi-epoch）中，同一数据点被多次使用，需要刻画分解误差与参与次数的关系。但现有上下界之间存在显著差距——Banded Square Root (BSR) 的误差界中对带宽 $p$ 的依赖是隐式的，无法判断其是否最优。
3. **核心矛盾**：理论上不清楚多轮参与下分解误差的最优增长率是什么，实践中缺少既高效又有显式误差刻画的分解方法。
4. **本文要解决什么？** 给出多轮参与下矩阵分解误差的紧界（tight bound），并提供一个计算高效、理论最优的显式分解方法。
5. **切入角度**：不是像 BSR 那样让相关矩阵 $C$ 带状化，而是让 $C^{-1}$ 带状化——这一视角转换带来了显式误差刻画和高效实现的双重优势。
6. **核心 idea 一句话**：在逆相关矩阵上施加带状结构，使噪声注入可通过卷积高效实现，同时获得关于带宽的显式最优误差界。

## 方法详解

### 整体框架
在差分隐私 SGD 训练中，通过矩阵分解 $A = BC$ 将公开系数矩阵分解，私有估计为 $\widehat{AX} = B(CX + Z)$。核心目标是最小化分解误差 $\mathcal{E}(B,C)$，它由 $\|B\|_F$ 和 $C$ 的灵敏度（sensitivity）共同决定。

### 关键设计

1. **BISR 分解（Definition 1）**：
   - 先计算工作负载矩阵 $A$ 的正定对角矩阵平方根 $C$（即 $C^2 = A$）
   - 计算 $C^{-1}$，将其截断为 $p$-带状矩阵
   - 再求逆得到 $C^p$，分解为 $A = B^p C^p$
   - 关键优势：$C^{-1}$ 带状意味着噪声注入只需与 $p$ 个系数做卷积，可用 FFT 加速

2. **新的下界（Theorem 3）**：
   - 对任意分解，当 $\alpha = 1$ 时误差为 $\Omega(\sqrt{k}\log n + k)$
   - 当 $\alpha < 1$（有 weight decay）时误差为 $\Omega_\alpha(\sqrt{k})$
   - 证明方法：概率方法 + 参与向量范数界

3. **BISR 上界（Theorem 4）**：
   - 误差显式依赖带宽 $p$、参与次数 $k$、矩阵大小 $n$、分离参数 $b$
   - 选择最优 $p^* = O(b \log b)$ 后，上界匹配下界，证明 BISR 渐近最优

4. **BandInvMF（低存储优化）**：
   - 保持逆矩阵的带状 Toeplitz 结构，但通过数值优化（而非闭式解）确定系数
   - 单带宽时即可实现 $O(n^{1/4})$ 误差（普通分解是 $O(\sqrt{n})$），显著改善
   - 以 BISR 系数为初始化，约 20 步收敛

5. **算法实现（Algorithm 1）**：
   - 每步只需存储 $p$ 个噪声向量的缓冲区
   - 噪声注入：$\hat{x}_i = x_i + \zeta \sum_{t=0}^{\min(p,i)-1} c_t Z_{i-t}$
   - 兼容 momentum 和 weight decay

## 实验关键数据

### 表1：CIFAR-10 测试精度（(9, 10⁻⁵)-DP，10 epochs）

| 方法 | Epoch 1 | Epoch 5 | Epoch 10 |
|------|---------|---------|----------|
| DP-SGD (Amp.) | 12.7±2.2 | 39.8±1.2 | 44.6±0.7 |
| BSR (Amp.) | 28.3±0.7 | 48.0±2.0 | 49.8±0.3 |
| **BISR (Amp.)** | **32.3±0.7** | **52.8±2.0** | **61.8±0.3** |
| Band-MF (Amp.) | 27.7±2.0 | 46.8±0.8 | 50.0±0.4 |
| Band-Inv-MF (Amp.) | 23.6±2.8 | 48.6±1.0 | 57.4±1.2 |
| DP-SGD (Non-Amp.) | 19.5±3.0 | 37.7±1.2 | 39.0±0.7 |
| **BISR (Non-Amp.)** | **31.8±1.5** | **51.1±1.0** | **56.2±0.2** |

### 表2：RMSE 比较（矩阵分解误差，n=16384）

| 方法 | k=4, α=1,β=0 | k=16, α=1,β=0 | k=16, α=1,β=0.9 |
|------|---------------|----------------|------------------|
| BSR | 与 BISR 相当 | 明显差于 BISR | 差于 BISR |
| BLT | 与 BISR 相当 | 与 BISR 相当 | 仅支持 prefix-sum |
| BandMF | 略优（小矩阵） | 略优但不可扩展 | 计算成本过高 |
| **BISR** | **最优或接近最优** | **显著优于 BSR** | **一致性最佳** |

> BISR 在 k=16 高参与次数下优势尤为明显；BandMF 虽 RMSE 略低但不可扩展至 n>4096

## 亮点与洞察
- **视角转换的力量**：从"让 $C$ 带状化"转为"让 $C^{-1}$ 带状化"，获得了显式误差刻画——这种看似微小的结构改变带来了理论突破。
- **理论与实践闭环**：BISR 同时实现了理论最优性（上下界匹配）和实践竞争力（与 BLT/BandMF 精度相当），且实现极其简单（卷积操作）。
- **低存储 regime 的洞察**：RMSE 更低不等于模型精度更高——Band-Inv-MF 的 RMSE 优于 BISR，但两者训练精度相近，说明分解误差与模型效用的关系非简单单调。
- **实用性突出**：仅需 $p$ 个系数的卷积，存储和计算成本远低于需要求解优化问题的 BandMF。

## 局限性
1. **渐近最优 ≠ 有限规模最优**：BISR 在渐近意义上最优，但有限矩阵大小下 BandMF 等数值优化方法仍可能略优。
2. **常数因子未优化**：上下界匹配在阶（order）的意义上，常数项的差距尚未完全消除。
3. **RMSE-精度脱节**：更低的分解 RMSE 不一定转化为更高的模型精度，特别是在使用 amplification by subsampling 时。
4. **BLT 比较受限**：BLT 仅实现了 prefix-sum 矩阵，无法在 momentum/weight decay 设置下对比。

## 补充实验：IMDB 情感分析（BERT-base）
- 在 (9, 10⁻⁵)-DP 下微调 BERT-base，BISR 在 amplified 和 non-amplified 设置下均优于 BSR 和 Band-MF
- BISR (Amplified) 10 epoch 后显著领先 DP-SGD，体现矩阵分解机制的优势
- 低存储 regime 下 Band-Inv-MF 与 BISR 精度接近，但 BISR 无需优化求解

## 相关工作
- **矩阵分解机制**：Choquette-Choo et al. (2023a) 定义了多轮参与下的最优分解问题；BLT (Dvijotham et al., 2024) 提供了 buffer-based 方法；BandMF (McKenna, 2025) 通过数值优化求解最优带状分解。
- **平方根分解**：Henzinger et al. (2024) 提出，Kalinin & Lampert (2024) 扩展为 BSR 并建立了首个上下界，但带宽依赖不显式。
- **隐私会计**：本文使用 MCMC accountant (Choquette-Choo et al., 2024b) 和 bins-and-balls 子采样 (Chua et al., 2025) 进行隐私分析。
- **联邦学习中的 MF**：Zhang et al. (2025) 和 Bienstock et al. (2025) 将矩阵分解扩展到联邦学习场景。

## 评分
- **创新性**: ★★★★☆ — 逆矩阵带状化的视角转换优雅且有效，配套的理论紧界具有重要贡献。
- **实用性**: ★★★★☆ — 实现简单高效，卷积操作可并行化，已有 JAX 实现。
- **理论深度**: ★★★★★ — 闭合了多轮参与分解误差的理论差距，上下界渐近匹配。
- **实验充分性**: ★★★★☆ — RMSE 和训练精度双重评估，覆盖多种优化器设置和数据集，但大规模 LLM 实验缺失。
- **表达清晰度**: ★★★★☆ — 数学推导严谨，算法描述清晰，Figure 1 的可视化很直观。
