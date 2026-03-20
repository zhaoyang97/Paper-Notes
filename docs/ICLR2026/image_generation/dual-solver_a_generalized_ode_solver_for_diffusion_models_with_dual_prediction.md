# Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction

**会议**: ICLR 2026  
**arXiv**: [2603.03973](https://arxiv.org/abs/2603.03973)  
**代码**: 无  
**领域**: 扩散模型 / 采样加速  
**关键词**: ODE solver, learnable sampler, prediction interpolation, domain selection, low-NFE  

## 一句话总结
提出 Dual-Solver，通过三组可学习参数（预测类型插值 $\gamma$、积分域选择 $\tau$、残差调整 $\kappa$）泛化扩散模型多步采样器，用冻结预训练分类器（MobileNet/CLIP）的分类损失学习参数（无需教师轨迹），在 3-9 NFE 低步区间全面优于 DPM-Solver++ 等方法。

## 研究背景与动机

1. **领域现状**：推理加速是扩散模型的核心挑战。ODE 求解器（DPM-Solver、DEIS 等）利用扩散动力学结构设计高效采样。学习型求解器（BNS-Solver、DS-Solver）优化时间步和采样参数以进一步提升质量。

2. **现有痛点**：(a) 传统求解器固定预测类型（noise/data/velocity）和积分域（对数/线性），但不同选择在不同 NFE 下表现不一，没有通用最优方案；(b) 学习型求解器需要大量教师轨迹或高 NFE 采样生成目标样本，准备开销大。

3. **核心矛盾**：预测类型和积分域的选择影响采样质量，但最优选择依赖 backbone 和 NFE——需要自适应方法。

4. **本文要解决什么？** 统一不同预测类型和积分域到一个连续参数化框架，并用无需目标样本的分类损失来学习最优参数。

5. **切入角度**：观察到 noise prediction、velocity prediction、data prediction 可以通过线性组合互换，积分在 $\log$ SNR 域 vs 线性 $t$ 域也是连续插值，这些都可以参数化并端到端学习。

6. **核心 idea 一句话**：将预测类型、积分域、残差项全部参数化，然后用冻结分类器的分类准确率作为无目标样本的训练信号。

## 方法详解

### 整体框架

Dual-Solver 保留标准的 predictor-corrector 结构，但在三个维度上引入可学习参数：
1. **$\gamma$（预测插值）**：在 noise/velocity/data prediction 之间连续插值
2. **$\tau$（域选择）**：在 log-SNR 域和线性时间域之间连续插值
3. **$\kappa$（残差调整）**：调整多步更新的残差项，保持二阶精度

### 关键设计

1. **预测类型插值**：$\hat{p}_\gamma = (1-\gamma) \hat{\epsilon}_\theta + \gamma \hat{x}_\theta$，$\gamma$ 连续选择最优预测类型组合
2. **积分域插值**：在 $\lambda = \log(\alpha_t/\sigma_t)$ 和 $t$ 之间以 $\tau$ 参数连续过渡
3. **分类学习策略**：用冻结的 MobileNet/CLIP 对生成图像做分类，用分类 loss 反向传播更新求解器参数。无需高 NFE 目标样本
4. 所有改动保持二阶局部精度

### 损失函数 / 训练策略
- 分类损失（交叉熵 + 条件类别标签）
- 冻结预训练分类器，仅更新 $\gamma, \tau, \kappa$
- 适用于 DiT, GM-DiT, SANA, PixArt-α 等多种 backbone

## 实验关键数据

### 主实验（ImageNet 256, DiT-XL/2, 3-9 NFE）

| 方法 | NFE=5 FID ↓ | NFE=7 FID ↓ |
|------|-----------|-----------|
| DPM-Solver++ | ~15 | ~8 |
| BNS-Solver | ~10 | ~5 |
| **Dual-Solver** | **~6** | **~3.5** |

在 SANA T2I（NFE=3）上视觉质量显著优于 DPM-Solver++ 和 BNS-Solver。

### 关键发现
- $\gamma$ 在不同时间步的最优值不同——早期偏向 noise prediction，晚期偏向 data prediction
- $\tau$ 在不同 backbone 上最优值不同——验证了"无通用最优"假设
- 分类学习策略无需教师轨迹，准备开销为零，比回归学习更实用

## 亮点与洞察
- **三维参数化**统一了大量采样器的设计选择——DPM-Solver++ 是 $\gamma=0, \tau=0$ 的特例。
- **分类学习**替代回归学习是关键创新——不需要生成高 NFE 目标样本，只需一个冻结分类器。这种思路可以迁移到任何可微指标的优化。

## 局限性 / 可改进方向
- 参数依赖 backbone 和 NFE，每个配置需要重新学习
- 分类损失可能偏向可分类性而非视觉质量
- 仅在 3-9 NFE 区间验证，更高 NFE 下优势是否保持未知

## 相关工作与启发
- **vs DPM-Solver++**: Dual-Solver 是其泛化版本，通过学习参数自适应选择最优配置
- **vs BNS/DS-Solver**: 同为学习型求解器但 Dual-Solver 不需要目标样本
- 可以与一致性蒸馏等方法正交使用

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一参数化框架+分类学习是有新意的组合
- 实验充分度: ⭐⭐⭐⭐ 多 backbone (DiT/SANA/PixArt) + 多 NFE 全面
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰
- 价值: ⭐⭐⭐⭐ 低 NFE 区间的实用改进，即插即用
