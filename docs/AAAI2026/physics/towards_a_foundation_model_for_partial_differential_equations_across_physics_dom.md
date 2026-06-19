---
title: >-
  [论文解读] Towards a Foundation Model for Partial Differential Equations Across Physics Domains
description: >-
  [AAAI 2026][物理/科学计算][偏微分方程] 提出 PDE-FM，一个结合空间-频谱双模态 tokenization、FiLM 物理调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准 12 个异构物理域数据集上平均降低 VRMSE 46%。
tags:
  - "AAAI 2026"
  - "物理/科学计算"
  - "偏微分方程"
  - "foundation model"
  - "神经算子"
  - "Mamba"
  - "FNO"
  - "multi-physics"
  - "The Well benchmark"
---

# Towards a Foundation Model for Partial Differential Equations Across Physics Domains

**会议**: AAAI 2026  
**arXiv**: [2511.21861](https://arxiv.org/abs/2511.21861)  
**代码**: 无  
**领域**: 科学计算  
**关键词**: PDE, foundation model, neural operator, Mamba, FNO, multi-physics, The Well benchmark

## 一句话总结
提出 PDE-FM，一个结合空间-频谱双模态 tokenization、FiLM 物理调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准 12 个异构物理域数据集上平均降低 VRMSE 46%。

## 研究背景与动机

### 领域现状

**领域现状**：Neural operator（FNO、GNOT、Transformer-based 等）在特定 PDE 类型上取得优异性能，但都是领域特定的——在单一数据集上训练，仅适用于窄类 PDE。NLP/Vision 领域已广泛采用 foundation model 的"一次预训练，多任务迁移"范式，但科学计算领域尚未实现类似突破。

### 现有痛点

**现有痛点**：(1) 现有 neural operator 边界条件或物理规律变化时性能骤降，无法跨物理域迁移；(2) 物理系统的独特挑战——多分辨率多尺度、守恒定律约束、连续时空演化、非线性算子耦合——使得统一建模极为困难；(3) Transformer-based 方法在高分辨率网格上的 $O(N^2)$ 复杂度限制了可处理的问题规模。

### 核心矛盾

**核心矛盾**：不同物理域（流体、辐射、弹性、天体物理）的 PDE 具有完全不同的方程形式、边界条件和物理守恒律，用单一模型统一建模看似不可能。但物理系统底层共享"局部结构+全局约束"的二元性，为统一架构提供了可能。

### 解决思路

**本文目标**：设计一个统一的 foundation model 架构，在异构 PDE 系统上预训练后无需架构修改即可迁移。**切入角度**：用空间 token 捕获局部结构、频谱 token 编码全局约束、FiLM 调制注入物理条件。**核心idea**：模块化设计——可替换的 tokenizer、backbone、decoder 和条件注入机制，通过系统化消融找到最优组合。

## 方法详解

### 整体框架
输入为 PDE 状态 $u \in \mathbb{R}^{C \times H \times W}$，经空间+频谱双模态 Tokenization 编码后，通过 FiLM 物理调制注入边界条件等元数据，Cross-Attention 融合两种 token，Mamba backbone 建模时空演化，最后 FNO Decoder 输出预测。支持多数据集联合预训练。

### 关键设计

1. **双模态 Tokenization（空间+频谱）**:

    - 功能：同时编码局部空间结构和全局频谱特性
    - 核心思路：空间 token $T_{spatial} = \text{PatchConv}(u) \in \mathbb{R}^{N_p \times d}$ 用 patch 卷积提取局部特征；频谱 token $T_{spectral} = \text{Linear}(\text{FFT}_m(u)) \in \mathbb{R}^{1 \times d}$ 保留低频模态的全局结构信息（仅保留前 $m$ 个频率分量）。Cross-Attention 实现双向信息融合
    - 设计动机：PDE 解同时具有局部空间梯度结构和全局频谱特性（如周期边界、守恒量），单一 tokenization 无法兼顾；单个频谱 token 作为"全局摘要"控制上下文分配

2. **FiLM 物理条件调制**:

    - 功能：将物理元数据（边界条件、本构参数、时间网格）注入模型
    - 核心思路：利用物理条件 $c$ 通过仿射变换调制 token：$\tilde{T}_{spatial} = T_{spatial} \odot (1 + \gamma(c)) + \beta(c)$，其中 $\gamma, \beta$ 为可学习映射
    - 设计动机：不同物理域的 PDE 具有不同的参数（Reynolds 数、Mach 数等），FiLM 以极轻量的方式（两个向量）注入这些条件信息，避免为每种物理配置设计专用分支

3. **Mamba State-Space Backbone + FNO Decoder**:

    - 功能：高效建模长序列时空演化并保持频谱平滑性
    - 核心思路：Mamba 层 $T^{(l+1)} = T^{(l)} + \text{MambaLayer}(T^{(l)})$ 以 $O(N_p d)$ 线性复杂度替代 Transformer 的 $O(N_p^2)$，支持大网格和长上下文。FNO 频谱 decoder $\hat{u}(x) = \sum_{|k| \leq m} W_k \cdot \mathcal{F}[z](k) e^{2\pi i k \cdot x}$ 保留频谱平滑先验
    - 设计动机：Mamba 的选择性状态空间结构天然适合时序演化建模（PDE 求解本质上是时间推进）；FNO decoder 利用频谱先验避免空间 aliasing

### 损失函数与训练策略
双目标损失：$\mathcal{L} = \text{VRMSE} + \lambda \sum_k w(k) \|\hat{U}(k) - U(k)\|^2$（高频加权），可选守恒量约束。多数据集采样：$p(i) \propto (\epsilon + \bar{\mathcal{L}}_i)^\alpha \cdot |\mathcal{D}_i|^\tau$ 结合难度感知和温度缩放。数据集特定 1×1 卷积适配器统一通道数。

## 实验关键数据

### 主实验

在 The Well 基准的 12 个跨物理域数据集上评测。

| 数据集 | FNO VRMSE | CNextU-net | **PDE-FM** | 降低 |
|--------|:---:|:---:|:---:|:---:|
| rayleigh_benard | 0.8395 | 0.6699 | **0.0415** | 95.1% |
| shear_flow | 1.189 | 0.808 | **0.0345** | 97.1% |
| gray_scott_RD | 0.1365 | 0.1761 | **0.0183** | 86.6% |
| post_neutron_star | 0.3866 | — | **0.2995** | 22.5% |
| turbulence_gravity | 0.2429 | 0.2096 | **0.0796** | 67.2% |

12 个数据集中 6 个 SOTA，5 个第二。平均 VRMSE 降低 46%。

### 消融实验

| 配置 | Mean VRMSE | 说明 |
|------|:---:|------|
| Full (Mamba+FNO+SpecTok+XAttn+FiLM) | **0.2581** | 最优配置 |
| w/o 频谱 Token | 0.3012 | 掉 16.7%，全局结构丢失 |
| w/o FiLM 调制 | 0.2891 | 掉 12.0%，物理条件未注入 |
| Transformer 替代 Mamba | 0.2743 | 掉 6.3%，且复杂度更高 |
| w/o FNO decoder | 0.2956 | 掉 14.5%，频谱平滑先验缺失 |

### 关键发现
- Rayleigh-Bénard 和 shear_flow 改进最为显著（>95% VRMSE 降低），这些是强湍流场景，全局频谱建模优势突出
- 频谱 token 对性能贡献最大（16.7%），验证了全局-局部双模态设计的必要性
- 难度感知采样策略有效缓解负迁移——在 active_matter 等困难数据集上改善最明显

## 亮点与洞察
- **真正的跨物理域 foundation model**：从流体湍流到中子星合并、超新星用同一模型，证明了物理系统的可统一建模性
- **Mamba + FNO 的互补组合**：Mamba 提供线性复杂度的时序建模，FNO 保持频谱域的物理约束，两者协同
- **空间-频谱双模态 tokenization**：单个频谱 token 作为全局摘要控制上下文分配，设计简洁但效果显著

## 局限与展望
- Ablation 仅在短训练（8 epochs, 600 steps）上进行，可能未充分反映组件贡献
- 在 active_matter 和 helmholtz_staircase 上不如 U-net 变体
- 模型复杂度高（Tokenizer+CrossAttn+Mamba+FNO），训练成本未报告
- 3D 数据集效果不如 2D 充分

## 相关工作与启发
- **vs FNO**: 领域特定、无预训练、$O(N\log N)$ 复杂度；PDE-FM 跨域预训练、$O(Nd)$ 线性复杂度
- **vs PhysiX**: 部分跨域但无统一预训练策略；PDE-FM 的难度感知采样和 FiLM 调制更系统化
- **vs OmniArch**: 类似目标但架构不同；PDE-FM 的 Mamba backbone 更高效
- FiLM 调制是引入物理元数据的轻量有效方式，可推广到其他科学计算任务

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 跨物理域 PDE foundation model 的首次系统尝试
- 实验充分度: ⭐⭐⭐⭐ 12 个异构数据集覆盖全面，但 ablation 训练不够充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，模块化设计文档完善
- 价值: ⭐⭐⭐⭐⭐ 对科学计算基础模型方向有开创性意义
---
title: >-
  [论文解读] Towards a Foundation Model for Partial Differential Equations Across Physics Domains
description: >-
  [AAAI 2026][科学计算][偏微分方程] 提出 PDE-FM，一个结合空间-频谱 tokenization、物理感知 FiLM 调制和 Mamba 状态空间 backbone 的模块化 PDE foundation model，在 The Well 基准的 12 个跨物理域数据集上平均降低 VRMSE 46%。
tags:
  - AAAI 2026
  - 科学计算
  - 偏微分方程
  - foundation model
  - 神经算子
  - Mamba
  - FNO
  - multi-physics
  - The Well benchmark
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] OmniArch: Building Foundation Model For Scientific Computing](../../ICML2025/physics/omniarch_building_foundation_model_for_scientific_computing.md)
- [\[ICML 2026\] Foundation Inference Models for Ordinary Differential Equations](../../ICML2026/physics/foundation_inference_models_for_ordinary_differential_equations.md)
- [\[ICML 2025\] Closed-form Symbolic Solutions: A New Perspective on Solving Partial Differential Equations](../../ICML2025/physics/closed-form_solutions_a_new_perspective_on_solving_differential_equations.md)
- [\[AAAI 2026\] SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs](saot_an_enhanced_locality-aware_spectral_transformer_for_solving_pdes.md)
- [\[AAAI 2026\] SVD-NO: Learning PDE Solution Operators with SVD Integral Kernels](svd-no_learning_pde_solution_operators_with_svd_integral_kernels.md)

</div>

<!-- RELATED:END -->
