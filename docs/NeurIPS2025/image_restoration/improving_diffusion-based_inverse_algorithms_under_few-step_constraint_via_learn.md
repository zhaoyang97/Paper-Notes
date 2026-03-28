# Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation

**会议**: NeurIPS 2025  
**arXiv**: [2503.10103](https://arxiv.org/abs/2503.10103)  
**代码**: 待确认  
**领域**: 图像恢复 / 扩散模型  
**关键词**: 扩散逆问题, 少步加速, 线性外推, 可学习系数, 即插即用

## 一句话总结
提出 Learnable Linear Extrapolation (LLE)——用可学习的线性组合系数将当前和历史 clean data estimate 组合，以增强任何符合 Sampler-Corrector-Noiser 范式的扩散逆问题算法在少步（3-5步）下的表现，仅需 50 个样本、几分钟训练，跨 9+ 算法 × 5 个任务一致提升。

## 研究背景与动机
1. **领域现状**：扩散模型在逆问题（去模糊、超分、修复、压缩感知等）上表现优异，但需要大量采样步数（100-1000步）才能获得高质量结果。快速 ODE 求解器在无条件生成中有效，但在逆问题中因异构公式和近似误差效果不佳。
2. **现有痛点**：扩散逆问题的 corrector 步（数据一致性增强）引入额外误差，少步时这些误差累积导致质量显著下降。不同算法（DPS, DDNM, DDRM, MCG 等）各有不同的 corrector 设计，需要一种通用的增强方法。
3. **核心矛盾**：实际应用需要少步出结果（3-5步推理），但少步暴露了 corrector 误差的累积效应。
4. **本文要解决什么**：设计一种轻量级、通用的"补丁"，能增强任何扩散逆算法在少步约束下的表现。
5. **切入角度**：所有扩散逆问题算法都遵循 Sampler → Corrector → Noiser 的范式，可以统一表示并用线性外推增强。
6. **核心idea一句话**：学习少量线性组合系数，将历史 clean data estimates 与当前估计组合，弥补少步带来的近似误差。

## 方法详解

### 整体框架
将 9+ 种扩散逆问题算法统一为 canonical form：每步包含 Sampler（$\mathbf{x}_{0,t_i} = \Phi_{t_i}(\mathbf{x}_{t_i})$）→ Corrector（$\hat{\mathbf{x}}_{0,t_i} = \mathbf{h}_{t_i}(\mathbf{x}_{0,t_i}, \mathcal{A}, \mathbf{y})$）→ Noiser（$\mathbf{x}_{t_{i-1}} = \Psi_{t_i}(\hat{\mathbf{x}}_{0,t_i})$）。LLE 在 Corrector 之后插入一个可学习的线性外推步。

### 关键设计

1. **统一的 Canonical Form**:
   - 做什么：将 DDNM, DPS, DDRM, MCG, PGDM, ReSample, DDPG, DiffPIR, PSLD 等 9+ 算法统一写为 Sampler-Corrector-Noiser 三步范式
   - 设计动机：不同算法的区别在于 Corrector 的设计（数据一致性如何实现），但框架一致，允许通用增强

2. **Learnable Linear Extrapolation (LLE)**:
   - 做什么：在第 $i$ 步，将 corrected 估计 $\hat{\mathbf{x}}_{0,t_i}$ 与之前所有步的估计线性组合
   - 核心思路：$\tilde{\mathbf{x}}_{0,t_i} = \gamma_{t_i,S-i}^{\perp}\hat{\mathbf{x}}_{0,t_i} + \sum_{j=0}^{S-i-1}\gamma_{t_i,j}^{\perp}\tilde{\mathbf{x}}_{0,t_{S-j}}$
   - 可学习参数：每步只有几个系数 $\gamma$，总参数量极少
   - 设计动机：历史估计包含之前步的信息，线性组合可以修正当前步的误差

3. **解耦系数（线性逆问题）**:
   - 做什么：对线性逆问题，将系数分为 range space 和 null space 两组
   - 核心思路：测量空间中的分量可以被精确约束（数据一致），零空间需要扩散先验补充。两者需要不同的外推策略
   - 效果：inpainting +0.96 PSNR, SR +0.26 PSNR

### 损失函数 / 训练策略
$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{LPIPS}$。仅需 50 个训练样本，RTX 3090 上 2-20 分钟训练完成。

## 实验关键数据

### 主实验（CelebA-HQ, 有噪声 $\sigma=0.05$）

| 任务 | 步数 | DDNM | DDNM+LLE | DPS | DPS+LLE |
|------|------|------|----------|-----|---------|
| Deblur | 3 | 27.80/0.758 | **28.08/0.784** | 23.59/0.650 | **24.59/0.675** |
| Inpainting | 5 | 22.76/0.550 | **26.35/0.659** | 25.49/0.647 | **27.51/0.748** |
| Super-Res | 3 | 27.09/0.773 | **27.84/0.770** | 25.49/0.647 | **24.57/0.666** |
| CS 50% | 5 | 18.20/0.474 | **19.41/0.536** | 17.27/0.591 | **18.44/0.605** |

### 消融实验
| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 解耦 vs 不解耦系数 | 解耦在 inpainting 上 +0.96 PSNR | range/null space 需不同处理 |
| $\omega=0.1$ (LPIPS 权重) | 平衡 PSNR (32.5) vs LPIPS (0.19) | 最优 trade-off |
| 推理开销 | DDNM: 2.0→2.0 min, DPS: 5.07→5.13 min | 几乎零额外开销 (<2%) |
| 跨数据集迁移 | CelebA→FFHQ 效果保持 | 泛化性好 |
| 跨任务迁移 | 有一定性能下降 | 线性搜索空间限制 |

### 关键发现
- LLE 在 9+ 算法上一致提升 PSNR/SSIM，从未降低性能
- 训练极轻量（50 样本 + 几分钟），推理几乎无额外开销
- Inpainting 受益最大（DDNM 从 22.76→26.35 PSNR，+3.6 dB）
- 跨数据集迁移有效，但跨任务迁移有限

## 亮点与洞察
- **即插即用的通用增强**：不修改任何原始算法的设计，只在 Corrector 后插入轻量线性组合。这种"补丁"思路可推广到其他迭代算法。
- **少样本训练**：仅 50 个样本就能训练好系数，说明线性外推的参数空间非常小且容易优化。这对数据稀缺的医学影像等领域特别有价值。
- **统一视角**：将 9+ 种异构算法统一为 Sampler-Corrector-Noiser 范式本身就是重要贡献，为未来的算法设计和分析提供了清晰框架。

## 局限性 / 可改进方向
- 搜索空间限于历史估计的线性组合，非线性组合可能进一步提升
- 跨任务迁移有限——不同任务的最优系数差异大
- 仅在 VP 扩散上验证，flow-matching 等新范式未探索
- 理论分析较弱——为什么线性外推有效的理解仍不深入

## 相关工作与启发
- **vs DPM-Solver (Lu et al., 2022)**：DPM-Solver 用高阶 ODE 求解器加速无条件生成，LLE 专注于有 corrector 的逆问题场景——两者互补
- **vs DDNM (Wang et al., 2022)**：DDNM 的零空间投影在少步时退化严重，LLE 直接补偿其误差
- **vs DPS (Chung et al., 2023)**：DPS 的梯度引导在少步时方向不准，LLE 平滑历史信息改善

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一范式 + 轻量线性增强的idea简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 9+ 算法 × 5 任务 × 多数据集，极其全面
- 写作质量: ⭐⭐⭐⭐ 统一框架的描述清晰
- 价值: ⭐⭐⭐⭐ 对扩散逆问题的实际应用有直接帮助
