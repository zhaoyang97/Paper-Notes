---
title: >-
  [论文解读] Linearly Constrained Diffusion Implicit Models
description: >-
  [NeurIPS 2025][3D视觉][扩散模型] 提出 CDIM，一种基于 DDIM 的线性逆问题求解算法，通过将残差能量与前向扩散过程的 $\chi^2$ 分布对齐来自适应控制投影步数和步长，实现比 DPS 快 10-50 倍的推理速度，同时在无噪声情况下精确满足测量约束。 从噪声线性测量 $\mathbf{y} =…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "扩散模型"
  - "inverse problems"
  - "linear constraints"
  - "DDIM"
  - "accelerated sampling"
---

# Linearly Constrained Diffusion Implicit Models

**会议**: NeurIPS 2025  
**arXiv**: [2411.00359](https://arxiv.org/abs/2411.00359)  
**代码**: [https://grail.cs.washington.edu/projects/cdim/](https://grail.cs.washington.edu/projects/cdim/)  
**领域**: 3D Vision / Image Restoration  
**关键词**: diffusion models, inverse problems, linear constraints, DDIM, accelerated sampling

## 一句话总结

提出 CDIM，一种基于 DDIM 的线性逆问题求解算法，通过将残差能量与前向扩散过程的 $\chi^2$ 分布对齐来自适应控制投影步数和步长，实现比 DPS 快 10-50 倍的推理速度，同时在无噪声情况下精确满足测量约束。

## 研究背景与动机

从噪声线性测量 $\mathbf{y} = \mathbf{A}\mathbf{x} + \boldsymbol{\sigma}$ 中恢复信号 $\mathbf{x}$ 是一个基础问题，涵盖超分辨率、修复、去噪等任务。使用预训练扩散模型作为先验来求解这类逆问题已成为热门方向。

**现有痛点**：

**投影步骤过多**：DPS 等方法在每个扩散步交替进行无条件去噪和约束投影，但每步投影都需要一次网络前向传播，导致推理极慢（>70 秒）

**加速采样下约束失效**：直接将 DPS 与 DDIM 加速采样结合，由于步长大导致投影不充分，产出模糊或发散的结果

**无法精确满足约束**：DPS 使用固定的小步长投影，在无噪声情况下也无法保证 $\mathbf{A}\hat{\mathbf{x}}_0 = \mathbf{y}$

**步长选择困难**：投影步长太大导致发散或拉出分布，太小收敛不充分

**核心矛盾**：如何在大幅减少投影步数的同时，既不过度投影（拉出分布）也不欠投影（无法满足约束）？

**核心 idea**：利用前向扩散过程中残差能量 $\|\mathbf{A}\mathbf{x}_t - \mathbf{y}\|^2$ 的解析 $\chi^2$ 分布来指导投影策略——只有当残差超出"合理区域"时才做投影，步长通过一维搜索确定使残差恰好落在合理区域边界。

## 方法详解

### 整体框架

CDIM 的推理流程（Algorithm 1）：
1. 初始化 $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$
2. 对每个时间步 $t = T, T-\delta, ..., 1$：
    - 做一步无条件 DDIM 去噪得到 $\mathbf{x}_{t-\delta}$
    - 计算合理区域边界 $\rho_t(\mathbf{y}) = \mu_{t-\delta}(\mathbf{y}) + c \cdot \sigma_{t-\delta}(\mathbf{y})$
    - 当 $\|\mathbf{A}\mathbf{x}_{t-\delta} - \mathbf{y}\|^2 > \rho_t$ 时：计算 Tweedie 估计 $\hat{\mathbf{x}}_0$，求梯度 $\mathbf{g}$，通过一维搜索找最优步长 $\eta^*$，更新 $\mathbf{x}_{t-\delta}$

### 关键设计

1. **代理残差与 $\chi^2$ 分布**：

    - **功能**：为投影提供解析可计算的停止准则
    - **核心思路**：真实目标 $L_t = \|\mathbf{A}\hat{\mathbf{x}}_0(\mathbf{x}_t) - \mathbf{y}\|^2$ 的分布难以计算（依赖数据分布）。但代理残差 $R_t = \|\mathbf{A}\mathbf{x}_t - \mathbf{y}\|^2$ 在前向过程中服从非中心广义 $\chi^2$ 分布，其均值和方差有解析公式：
        - $\mu_t(\mathbf{y}) = (\sqrt{\bar\alpha_t} - 1)^2 \|\mathbf{y}\|^2 + (1 - \bar\alpha_t) \text{tr}(\mathbf{A}\mathbf{A}^\top)$
        - $\sigma_t^2(\mathbf{y})$ 类似可解析计算
    - **Proposition 1**：当 $|R_t - \mathbb{E}[R_t|\mathbf{y}]| \leq \gamma$ 时，$|L_t - \mathbb{E}[L_t|\mathbf{y}]| \leq \gamma/\bar\alpha_t + O(\sqrt{1-\bar\alpha_t}/\bar\alpha_t)$。即控制代理残差就能控制真实目标
    - **设计动机**：$R_t$ 的计算不需要网络评估（仅矩阵乘法），因此可以做步长搜索而不增加 NFE（Neural Function Evaluations）

2. **自适应步长选择**：

    - **功能**：在不增加 NFE 的情况下找到最优投影步长
    - **核心思路**：预计算梯度 $\mathbf{g} = \nabla_{\mathbf{x}_{t-\delta}} \|\mathbf{A}\hat{\mathbf{x}}_0 - \mathbf{y}\|^2$，然后一维搜索 $\eta^* = \arg\min_\eta |\rho_{t-\delta}(\mathbf{y}) - \|\mathbf{A}(\mathbf{x}_{t-\delta} - \eta\mathbf{g}) - \mathbf{y}\|^2|$
    - 目标函数是 $\eta$ 的二次函数，搜索高效且稳定
    - **设计动机**：将残差拉回合理区域边界（而非中心）效果更好

3. **停止准则与超参数 $c$**：

    - **功能**：决定何时停止投影
    - **核心思路**：当 $\|\mathbf{A}\mathbf{x}_{t-\delta} - \mathbf{y}\|^2 \leq \rho_{t-\delta}(\mathbf{y})$ 时停止
    - $c$ 越小（合理区域越窄），结果越好但投影步数越多。实验中使用 $c = 0.1$
    - 高噪声时合理区域很大，几乎不需要投影；低噪声时合理区域收窄，投影增加

4. **泊松噪声处理**：

    - **功能**：将方法扩展到非高斯噪声
    - **核心思路**：通过 Pearson 残差 $R(\mathbf{A}\hat{\mathbf{x}}_0, \mathbf{y}) = \frac{\lambda(\mathbf{y} - \mathbf{A}\hat{\mathbf{x}}_0)}{\sqrt{\lambda\hat{\mathbf{x}}_0}}$ 将泊松噪声转化为近似标准正态噪声，然后使用相同框架处理

### 损失函数 / 训练策略

- 无需训练，使用预训练扩散模型
- 仅有两个超参数：去噪步数 $T'$ 和停止准则常数 $c$
- 默认 $T' = 50$，$c = 0.1$
- 当 $t \to 0$ 时，$\hat{\mathbf{x}}_0 \to \mathbf{x}_t$，残差 $L_t$ 变为简单凸二次函数，可以精确收敛到零

## 实验关键数据

### 主实验

**FFHQ 256×256 (σ=0.05)**：

| 方法 | 超分辨 FID | 修复(box) FID | 高斯模糊 FID | 修复(random) FID | 时间(秒) |
|------|----------|-------------|------------|----------------|---------|
| DPS | 39.35 | 33.12 | 44.05 | 21.19 | 70.42 |
| FPS-SMC | 26.62 | 26.51 | 29.97 | 33.10 | 116.90 |
| DDRM | 62.15 | 42.93 | 74.92 | 69.71 | 2.0 |
| **CDIM (T'=25)** | 33.87 | 27.51 | 34.18 | 29.67 | **2.4** |
| **CDIM (T'=50)** | **31.54** | **26.09** | **29.68** | 28.52 | 6.4 |

CDIM 在保持可比或更优质量的同时，推理速度比 DPS 快 ~28 倍（2.4s vs 70.4s）。

### 消融实验

| 去噪步数 T' | 总投影步数 | 质量 | 说明 |
|------------|----------|------|------|
| T'=25 | 31 | 好 | 默认快速设置 |
| T'=10 | 15 | 合理 | 仍然可用 |
| T'=4 | 10 | 可接受 | <1秒推理，14 NFEs |

| 停止准则 c | 效果 | 投影步数 |
|-----------|------|---------|
| c=4 | 约束满足较差 | 最少 |
| c=1 | 好 | 中等 |
| c=0.1 | 最优 | 较多但可控 |

**DPS + DDIM 对比**：
- DPS + DDIM (小步长): MAE 4%，约束不满足，头发模糊
- DPS + DDIM (大步长): MAE 0.3%，约束好转但结果发散
- CDIM: MAE 0.05%，约束精确满足且结果自然

### 关键发现

1. **精确约束满足**：在无噪声情况下，CDIM 能精确满足 $\mathbf{A}\mathbf{x}_0 = \mathbf{y}$，即使观测是分布外的也能做到
2. **加速不仅来自 DDIM**：DPMC 使用 DDIM 策略但仍需 200 次额外 NFE，说明单靠 DDIM 不够
3. **高噪声时投影少，低噪声时投影多**：符合直觉——早期阶段 Tweedie 估计不准确，不应过度投影
4. **极端加速仍可用**：T'=4 步（<1秒）仍能产出合理结果

## 亮点与洞察

1. **优雅的数学框架**：利用前向过程的解析分布来指导反向采样的投影策略，理论动机清晰
2. **代理残差的巧妙设计**：$R_t$ 不需要网络评估即可计算，使步长搜索零成本
3. **精确约束满足**：首次在扩散模型逆问题中实现精确约束满足（DPS 无法做到）
4. **泊松噪声扩展**：通过 Pearson 残差统一处理非高斯噪声

## 局限与展望

- **仅限线性约束**：对非线性约束，$\mathbb{E}[h(\mathbf{x}_0)] \neq h(\mathbb{E}[\mathbf{x}_0])$，无法扩展 Tweedie 估计
- 需要 $\mathbf{A}$ 的显式形式来计算 $\|\mathbf{A}\mathbf{x}_t - \mathbf{y}\|^2$
- 泊松噪声处理在极端噪声或 $\hat{\mathbf{x}}_0$ 接近零时 Pearson 残差的正态近似会失效
- 梯度 $\mathbf{g}$ 的计算仍需网络评估，虽然大幅减少了次数但并未完全消除

## 相关工作与启发

- **DPS (2023)**：CDIM 的主要对比基线；DPS 的每步小投影策略在加速采样下失效
- **DDRM/DDNM**：基于 SVD 的方法速度快但质量有限
- **DSG**：使用类似的优化更新但采用软约束而非精确约束
- **启发**：前向过程的统计特性是指导反向过程的宝贵信息源，利用解析分布来替代启发式策略是一个有前景的方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 利用 $\chi^2$ 分布指导投影的思路非常优雅且原创
- 实验充分度: ⭐⭐⭐⭐ 多任务多数据集对比 + 消融 + 新应用（时光重拍/点云），但定量评估可更丰富
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨清晰，图示直观
- 价值: ⭐⭐⭐⭐⭐ 在扩散模型逆问题速度-质量权衡上开辟新的 Pareto 前沿

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[CVPR 2025\] Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models](../../CVPR2025/3d_vision/difix3d_improving_3d_reconstructions_with_single-step_diffusion_models.md)
- [\[NeurIPS 2025\] UMAMI: Unifying Masked Autoregressive Models and Deterministic Rendering for View Synthesis](umami_unifying_masked_autoregressive_models_and_deterministic_rendering_for_view.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](../../ICCV2025/3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](../../ICCV2025/3d_vision/repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)

</div>

<!-- RELATED:END -->
