---
title: >-
  [论文解读] UniSim: A Unified Simulator for Time-Coarsened Dynamics of Biomolecules
description: >-
  [ICML 2025][计算生物][分子动力学] UniSim 是首个面向跨域（小分子/肽链/蛋白质）全原子时间粗化分子动力学的深度生成模型，通过三阶段管线——多头预训练统一原子表示、随机插值向量场模型学习长时间步状态推进、力引导核参数高效适配不同化学环境——实现跨分子域的可迁移动力学模拟。 领域现状：经典分子动力学（MD）…
tags:
  - "ICML 2025"
  - "计算生物"
  - "分子动力学"
  - "时间粗化"
  - "跨域预训练"
  - "随机插值"
  - "力引导"
  - "全原子模拟"
---

# UniSim: A Unified Simulator for Time-Coarsened Dynamics of Biomolecules

**会议**: ICML 2025  
**arXiv**: [2506.03157](https://arxiv.org/abs/2506.03157)  
**代码**: [https://github.com/yaledeus/UniSim](https://github.com/yaledeus/UniSim)  
**领域**: 分子动力学 / 计算生物学  
**关键词**: 分子动力学, 时间粗化, 跨域预训练, 随机插值, 力引导, 全原子模拟

## 一句话总结

UniSim 是首个面向跨域（小分子/肽链/蛋白质）全原子时间粗化分子动力学的深度生成模型，通过三阶段管线——多头预训练统一原子表示、随机插值向量场模型学习长时间步状态推进、力引导核参数高效适配不同化学环境——实现跨分子域的可迁移动力学模拟。

## 研究背景与动机

**领域现状**：经典分子动力学（MD）模拟需要飞秒级（$\Delta t \approx 10^{-15}$ s）极小时间步长来保证数值积分稳定性，限制了对蛋白质折叠等长时间尺度过程的模拟能力。量子力学方法精确但计算成本极高，经验力场方法快但精度不足。

**现有痛点**：近年深度学习方法（FBM、Timewarp、ITO 等）通过学习"时间粗化"推进映射 $\mathbf{X}_t \to \mathbf{X}_{t+\tau}$（$\tau \gg \Delta t$）大幅加速模拟，但存在两大瓶颈：(1) 几乎所有方法局限于单一分子域（如仅肽链或仅蛋白质），缺乏跨域迁移能力；(2) 部分模型依赖手工设计的域特定表示（如亮氨酸的 $\gamma$-碳标记），无法识别含非天然氨基酸的蛋白质。

**核心矛盾**：MD 轨迹数据稀缺，但分子系统种类繁多且化学环境多变（温度/压力/溶剂），需要一个通用模型而非为每个分子系统单独训练。

**本文目标**：构建一个可跨小分子、肽链、蛋白质迁移的统一全原子时间粗化模拟器，并能通过参数高效微调适配不同化学环境。

**切入角度**：利用跨域 3D 分子数据进行多头预训练获取统一原子表示，再基于随机插值生成框架学习状态推进，最后通过力引导核实现"一次训练、多环境适配"。

**核心 idea**：预训练统一表示 + 随机插值向量场 + 力引导核 = 跨域可迁移的时间粗化 MD 模拟器。

## 方法详解

### 整体框架

UniSim 包含四个阶段：(a) 在多域 3D 分子数据上多头预训练原子表示模型 $\varphi$；(b) 基于随机插值框架训练向量场模型 $\phi = \{v, \eta_z\}$ 学习 $\mathbf{X}_t \to \mathbf{X}_{t+\tau}$；(c) 训练力引导核 $\zeta$ 适配不同化学环境（冻结 $\varphi, \phi$ 参数）；(d) 推理时迭代求解 SDE 生成轨迹。底层网络架构采用 SO(3)-等变图神经网络 TorchMD-NET。

### 关键设计

1. **梯度-环境子图（Gradient-Environment Subgraph）**：解决跨域分子尺度差异问题（小分子几十原子 vs 蛋白质数千原子）。对超过 1000 原子的大分子，随机选择中心原子 $c$，构建梯度子图 $\mathcal{G}_g = \{j : \|\mathbf{x}_j - \mathbf{x}_c\|_2 < \delta_{\min}\}$ 和环境子图 $\mathcal{G}_e = \{j : \|\mathbf{x}_j - \mathbf{x}_c\|_2 < \delta_{\max}\}$（$\delta_{\min} = 8$Å, $\delta_{\max} = 20$Å），仅将 $\mathcal{G}_e$ 输入模型，仅 $\mathcal{G}_g$ 中的原子参与损失计算。当 $\delta_{\max} - \delta_{\min}$ 足够大时，$\mathcal{G}_e$ 外原子对 $\mathcal{G}_g$ 的影响可忽略——物理上合理且计算高效。

2. **原子嵌入扩展（Atomic Embedding Expansion）**：蛋白质中同种元素存在离散的化学模式（如碳的 CA、CB），键长键角高度规则，仅用元素周期表作为词表粒度过粗。方法：定义基础词表 $\mathbf{A}_b \in \mathbb{R}^{A \times H}$ 和扩展词表 $\mathbf{A}_e \in \mathbb{R}^{A \times D \times H}$（$D$ 为每种元素的模式数），通过邻居信息计算模式概率 $\mathbf{w}_i = \text{softmax}(\text{lin}(\mathbf{A}_b[i], \mathbf{n}_i))$，最终嵌入 $\mathbf{z}_i = \text{lin}(\mathbf{A}_b[i], \mathbf{w}_i^\top \mathbf{A}_e[i], \mathbf{n}_i)$。消融实验证实去掉扩展嵌入后 PWD 从 0.332 退化到 0.389。

3. **力引导核（Force Guidance Kernel）**：冻结 $\varphi, \phi$ 全部参数，新增 TorchMD-NET $\Psi$ 和输出网络 $\psi$ 拟合中间力场 $\nabla \varepsilon_t$。目标：生成分布 $p_t(\cdot) \propto q_t(\cdot) \exp(-\alpha \varepsilon_t(\cdot))$，修改去噪器为 $\eta_z'(t, \mathbf{X}) = \eta_z(t, \mathbf{X}) + \alpha \gamma(t) \nabla \varepsilon_t(\mathbf{X})$。$\psi$ 采用插值形式 $(1-t)\psi_0 + t\psi_1 + t(1-t)\psi_2$ 保证端点与 MD 力场连续。化学环境变化通过 MD 势能 $\varepsilon$ 反映在生成分布中——"一次预训练、多环境适配"。

### 损失函数 / 训练策略

- **预训练**：力对齐 $\mathcal{L}_o = \|\nabla_{\mathbf{X}}(\sum_i \mathbf{H}_\text{out}[i]) + \mathbf{F}\|_2^2$ (off-equilibrium) + 去噪 $\mathcal{L}_e$ (equilibrium)，多头区分不同力场
- **向量场**：$\mathcal{L}_v = \mathbb{E}[\|v(t, \mathbf{X}_t) - (\mathbf{X}_1 - \mathbf{X}_0)\|^2 + \|\eta_z(t, \mathbf{X}_t) - \mathbf{Z}\|^2]$
- **力引导**：端点力拟合 + 中间力场拟合（四项联合损失）
- 推理后对肽链/蛋白质进行 OpenMM 能量最小化构象细化（平均 69.3 步）
- 训练环境：8× RTX 3090，一周内完成

## 实验关键数据

### 主实验：肽链（PepMD 14 测试肽，JS距离↓）

| 模型 | PWD↓ | RG↓ | TIC↓ | TIC-2D↓ | VAL-CA↑ | CONTACT↓ |
|------|------|-----|------|---------|---------|----------|
| FBM | 0.361 | 0.411 | 0.510 | 0.736 | 0.539 | 0.205 |
| Timewarp | 0.362 | 0.386 | 0.514 | 0.745 | 0.028 | 0.195 |
| ITO | 0.367 | 0.371 | 0.495 | 0.748 | 0.160 | 0.174 |
| SD | 0.727 | 0.776 | 0.541 | 0.782 | 0.268 | 0.466 |
| UniSim/g | 0.332 | 0.332 | 0.510 | 0.738 | 0.505 | 0.162 |
| **UniSim** | **0.328** | **0.330** | **0.510** | **0.731** | **0.575** | **0.157** |

### 蛋白质（ATLAS 14 测试蛋白）

| 模型 | PWD↓ | RG↓ | TIC↓ | VAL-CA↑ | CONTACT↓ |
|------|------|-----|------|---------|----------|
| FBM | 0.519 | 0.597 | 0.621 | 0.012 | 0.252 |
| ITO | 0.588 | 0.775 | 0.624 | 0.052 | 0.428 |
| SD | 0.604 | 0.762 | 0.605 | 0.001 | 0.235 |
| UniSim/g | 0.508 | 0.569 | 0.543 | 0.071 | 0.171 |
| **UniSim** | **0.506** | **0.554** | **0.542** | **0.079** | **0.173** |

### 消融实验

| 消融项 | 关键变化 |
|-------|---------|
| 去掉原子嵌入扩展 | PWD: 0.332→0.389, CONTACT: 0.162→0.228 |
| 力引导($\alpha$增大) | VAL-CA提升显著，但分布多样性有下降趋势 |
| SDE步数($T$增大) | 多数指标退化——小$T$足以平衡精度和效率 |
| 小分子 TIC (UniSim/g→UniSim) | 0.408→0.368，力引导改善跨域迁移 |

### 关键发现

- 跨域预训练不损害单域性能：UniSim 在肽链上全面优于从头训练的 FBM
- 力引导核关键作用：VAL-CA 从 0.505→0.575（肽链），显著提升构象有效性
- 蛋白质领域取得突破：CONTACT 从 FBM 的 0.252 降到 0.173（改善 31%）
- 推理效率：ESS/s 约为传统 MD 的 25 倍
- Alanine-Dipeptide 案例中成功恢复 C5、C7eq、$\alpha_R'$、$\alpha_R$、$\alpha_L$ 五个亚稳态

## 亮点与洞察

- 首个跨域全原子时间粗化模拟器，成功将统一预训练范式引入分子动力学领域
- 梯度-环境子图设计巧妙，在物理合理的前提下解决跨域尺度差异的计算瓶颈
- 力引导核的"冻结主干+训练适配器"思路与 NLP 的 LoRA 异曲同工

## 局限与展望

- 自回归生成的预测误差累积导致大蛋白长程模拟不稳定，需 OpenMM 后处理
- 评估轨迹长度较短（$10^3$步），可能无法发现更多亚稳态
- 预训练数据规模受 MD 轨迹稀缺限制
- 未与 AlphaFold3 等结构预测模型集成

## 相关工作与启发

- FBM (Yu et al. 2024)：UniSim 的力引导模块直接继承自 FBM，但扩展到跨域场景
- DPA-2 (Zhang et al. 2024)：多任务预训练思路类似，但面向材料系统
- 启发：统一预训练+域适配范式可推广到其他科学模拟领域

## 评分

⭐⭐⭐⭐ — 首个跨域全原子时间粗化模拟器，技术创新扎实（梯度子图、原子嵌入扩展、力引导核），三个分子域实验全面优于基线。蛋白质长程模拟的误差累积和后处理依赖是主要局限。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](../../NeurIPS2025/computational_biology/remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/computational_biology/unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICLR 2026\] Unified Biomolecular Trajectory Generation via Pretrained Variational Bridge](../../ICLR2026/computational_biology/unified_biomolecular_trajectory_generation_via_pretrained_variational_bridge.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)

</div>

<!-- RELATED:END -->
