---
title: >-
  [论文解读] Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers
description: >-
  [NeurIPS 2025][时间序列][降阶建模] 提出Φ-ROM框架，将可微分PDE求解器嵌入非线性降阶模型的训练过程中，通过求解器反馈直接约束潜在空间动态，使模型在泛化到未见参数/初始条件、长时间外推、稀疏观测数据恢复等方面显著优于纯数据驱动ROM和其他物理信息方法。 问题背景 降阶建模（ROM）旨在通过将高维PDE系…
tags:
  - "NeurIPS 2025"
  - "时间序列"
  - "降阶建模"
  - "可微分求解器"
  - "物理信息神经网络"
  - "隐式神经表示"
  - "偏微分方程"
---

# Physics-informed Reduced Order Modeling of Time-dependent PDEs via Differentiable Solvers

**会议**: NeurIPS 2025  
**arXiv**: [2505.14595](https://arxiv.org/abs/2505.14595)  
**作者**: Nima Hosseini Dashtbayaz (UWO), Hesam Salehipour (Autodesk Research), Adrian Butscher (Autodesk Research), Nigel Morris (Autodesk Research)  
**代码**: [phi-rom.github.io](https://phi-rom.github.io)  
**领域**: 时间序列  
**关键词**: 降阶建模, 可微分求解器, 物理信息神经网络, 隐式神经表示, 偏微分方程  

## 一句话总结

提出Φ-ROM框架，将可微分PDE求解器嵌入非线性降阶模型的训练过程中，通过求解器反馈直接约束潜在空间动态，使模型在泛化到未见参数/初始条件、长时间外推、稀疏观测数据恢复等方面显著优于纯数据驱动ROM和其他物理信息方法。

## 研究背景与动机

### 问题背景
降阶建模（ROM）旨在通过将高维PDE系统压缩到低维潜在流形上实现加速仿真，广泛应用于设计优化、最优控制、逆问题等many-query工程场景。传统方法使用PCA等线性降维，新近研究采用自编码器等非线性流形ROM，并通过Neural ODE等网络学习潜在空间的时间演化。

### 已有工作的不足
- **数据驱动ROM的根本缺陷**：现有非线性ROM（如DINo）完全依靠数值求解器生成数据集进行训练，一旦数据生成完毕就丢弃求解器。学到的潜在动态不保证与真实物理一致，导致误差累积、长时间外推失败、对新参数/初始条件泛化差
- **已有物理信息方法的局限**：(i) PINN-ROM在损失函数中增加PINN残差项，但在非线性PDE上表现极差（受谱偏差等优化困难影响）；(ii) CROM在推理期间直接在物理空间求解PDE，但无法实现真正的降维加速，且对不输出完整物理场的INR不可行
- **求解器信息被浪费**：尽管高保真数值求解器编码了离散化后的真实物理，但在所有已有框架中被完全排除在训练过程之外

### 核心动机
将数值求解器作为训练的一部分直接嵌入ROM，使潜在空间动态受真实物理约束，从而在不牺牲降维优势的前提下获得更好的泛化、外推和数据效率。

## 方法详解

### 整体框架
Φ-ROM建立在DINo框架之上，包含两个核心组件：

1. **条件INR解码器 $D_\theta$**：将低维潜在坐标 $\alpha \in \mathbb{R}^k$ 映射为物理场 $\hat{\mathbf{u}} = D_\theta(\alpha, \mathcal{X})$，采用auto-decoding方案（无编码器，通过反演优化获取潜在坐标），天然支持任意网格和不规则观测
2. **动态网络 $\Psi_\phi$**：以Neural ODE形式学习潜在空间的时间演化 $\Psi_\phi(\alpha) = \dot{\alpha}$

### 核心创新：物理信息动态损失
关键突破在于用可微分PDE求解器 $\mathcal{S}$ 直接计算潜在空间的"目标"时间导数。具体步骤：

1. 对解码器关于 $\alpha$ 求Jacobian：$J_D(\alpha)\dot{\alpha} = d\hat{\mathbf{u}}/dt$
2. 用求解器 $\mathcal{S}$ 计算重构场 $\hat{\mathbf{u}}$ 的真实时间导数 $d\hat{\mathbf{u}}/dt$
3. 通过伪逆投影到潜在空间：$\dot{\alpha}^* = J_D^\dagger(\alpha) \cdot d\hat{\mathbf{u}}/dt$
4. 定义动态损失：$L_{dyn} = \ell(\Psi_\phi(\alpha), \dot{\alpha}^*)$

### 训练目标
联合优化重构损失和动态损失：

$$L_{\Phi\text{-ROM}} = \lambda L_{rec} + (1-\lambda) L_{dyn}$$

其中 $\lambda \in [0.5, 0.8]$ 控制正则化强度。由于 $\mathcal{S}$ 可微，$L_{dyn}$ 的梯度通过求解器反向传播到解码器参数 $\theta$ 和潜在流形 $\Gamma$，起到物理正则化效果。

### 超降维加速
直接计算完整Jacobian和求解最小二乘问题的成本随空间网格 $N$ 增长。论文采用**随机超降维**策略：
- 对每个训练快照，随机子采样 $\gamma N$（$\gamma=0.1$）个空间点
- 仅在子采样点上构建Jacobian和求解最小二乘
- 使用前向模式自动微分计算Jacobian

### 稀疏数据训练
由于INR解码器天然支持任意网格，训练数据可在不规则稀疏网格 $\mathcal{X}_{tr}$ 上提供，而求解器在其专用网格 $\mathcal{X}_\mathcal{S}$ 上计算。重构损失在 $\mathcal{X}_{tr}$ 上计算，动态损失在 $\mathcal{X}_\mathcal{S}$ 上计算，实现灵活的数据同化。

### 参数化动态网络
对于参数化PDE（如不同Reynolds数），将参数 $\beta$ 经可训练线性变换后与 $\alpha$ 拼接输入动态网络：$\dot{\alpha} = \Psi(\alpha, \beta)$，使模型能跨参数泛化。

## 实验关键数据

### 实验1：物理信息策略对比（Diffusion & Burgers'）

在2D扩散方程和1D Burgers方程上，对比Φ-ROM与DINo（纯数据驱动）、PINN-ROM、CROM三种物理信息方法：

| 方法 | Diffusion $[0,T_{tr}]$ | Diffusion $[T_{tr},T_{te}]$ | Burgers' $[0,T_{tr}]$ | Burgers' $[T_{tr},T_{te}]$ |
|------|:---:|:---:|:---:|:---:|
| **Φ-ROM** | **0.080** | **0.034** | 0.021 | **0.028** |
| DINo | 0.089 | 0.051 | 0.021 | 0.060 |
| PINN-ROM | 0.081 | 0.042 | 0.088 | 0.348 |
| FD-CROM | 0.131 | 0.351 | **0.001** | 0.044 |
| AD-CROM | 0.093 | 0.106 | 0.121 | 0.196 |
| ↓AD-CROM | 0.456 | 0.856 | 0.090 | 0.212 |

关键发现：PINN-ROM和AD-CROM在非线性Burgers方程上严重失败（外推误差0.348和0.196），FD-CROM虽然训练窗口内精度极高但外推退化，Φ-ROM在所有外推场景中表现最优。

### 实验2：复杂PDE泛化性能（N-S & KdV & LBM）

在2D Navier-Stokes湍流衰减（64x64网格，256条轨迹训练）、2D KdV方程（512条轨迹）和2D绕圆柱流（LBM，Reynolds数参数化）上的对比：

| 问题 | 设定 | Φ-ROM 测试插值 | DINo 测试插值 | Φ-ROM 测试外推 | DINo 测试外推 |
|------|------|:---:|:---:|:---:|:---:|
| N-S | 全网格训练 | **0.170** | 0.580 | **0.373** | 1.543 |
| N-S | 5%稀疏训练 | **0.192** | 0.584 | **0.397** | 1.450 |
| N-S | 2%稀疏训练 | **0.189** | 0.594 | **0.394** | 1.517 |
| KdV | 全网格训练 | **0.233** | 0.459 | **0.486** | 0.728 |
| KdV | 5%稀疏训练 | **0.248** | 0.543 | **0.499** | 0.851 |
| LBM | 全网格(域外β) | **0.115** | 0.457 | **0.180** | 0.566 |
| LBM | 2%稀疏(域外β) | **0.188** | 0.412 | **0.303** | 0.507 |

N-S外推中Φ-ROM比DINo好4倍以上（0.373 vs 1.543）。在仅2%观测点的稀疏训练下，Φ-ROM在N-S上仍保持接近全网格训练的精度（0.394 vs 0.373），而DINo严重退化。

## 亮点

- **方法论创新**：首次将可微分PDE求解器嵌入非线性ROM训练循环，通过求解器梯度反传直接物理约束潜在空间，概念简洁而效果显著
- **全面优越性**：在5个不同PDE和5种不同数值方法（有限差分、谱方法、有限体积、Lattice Boltzmann）上均展现一致的泛化和外推优势，验证了框架的鲁棒通用性
- **稀疏数据能力**：仅用2%-5%的空间观测点训练即可恢复全场解，为场重建和数据同化提供实用框架
- **开源与可扩展**：提供基于JAX的开源实现，可方便扩展到新PDE和新求解器

## 局限与展望

- **需要可微分求解器**：要求PDE求解器在JAX/PyTorch等框架中实现并支持自动微分，限制了对legacy代码的即用性
- **训练成本增加**：相比纯数据驱动方法，每步训练需额外执行求解器前向+反向传播和Jacobian计算
- **仅限一阶时间导数PDE**：当前框架假设 $\dot{u} = \mathcal{N}(u;\beta)$ 形式，未覆盖波动方程等高阶时间导数PDE和稳态PDE
- **大规模3D问题**：超降维在高维空间的可扩展性尚未验证，需进一步优化
- **解码器精度瓶颈**：DINo在训练集内精度更高（如N-S: 0.036 vs 0.064），说明物理正则化在一定程度上牺牲了训练集拟合精度

## 与相关工作的对比

- **DINo (Yin et al. 2023)**：Φ-ROM的直接基线，同样使用INR解码器+Neural ODE动态网络，但纯数据驱动训练导致泛化差、外推误差累积严重
- **CROM (Chen et al. 2021)**：在推理时在物理空间直接求解PDE，但无真正降维加速，且对复杂PDE（需要多物理场如压力+速度）不可行；子采样后精度骤降
- **PINN-ROM**：用自动微分计算PDE残差作为正则化，但在非线性PDE上受谱偏差等优化困难影响严重失败（Burgers外推误差0.348）
- **Lee & Parish (2025)**：引入参数化动态网络，本文改进为加入可训练线性变换显著提升参数泛化
- **传统投影ROM (Benner et al. 2015)**：基于线性子空间，无法捕捉非线性动态的流形结构

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将可微分求解器嵌入ROM训练的想法自然但此前未被实现，超降维投影设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ — 5个PDE、5种数值方法、多种训练/测试设定（稀疏、参数外推、时间外推），消融充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机表达到位，数学公式与直觉解释平衡良好
- 价值: ⭐⭐⭐⭐ — 提供了物理信息ROM的有效新范式，开源代码增强了实际影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](../../ICLR2026/time_series/towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[NeurIPS 2025\] Parallelization of Non-linear State-Space Models: Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling](parallelization_of_non-linear_state-space_models_scaling_up_liquid-resistance_li.md)
- [\[ICML 2026\] Spatiotemporal Imputation with Graph-Informed Flow Matching](../../ICML2026/time_series/spatiotemporal_imputation_with_graph-informed_flow_matching.md)
- [\[ICML 2026\] Generalizing Multi-scale Time-Series Modeling with a Single Operator](../../ICML2026/time_series/generalizing_multi-scale_time-series_modeling_with_a_single_operator.md)
- [\[ICML 2025\] A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments](../../ICML2025/time_series/a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca.md)

</div>

<!-- RELATED:END -->
