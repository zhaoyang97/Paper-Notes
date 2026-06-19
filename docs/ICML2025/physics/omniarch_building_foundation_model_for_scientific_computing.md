---
title: >-
  [论文解读] OmniArch: Building Foundation Model For Scientific Computing
description: >-
  [ICML 2025][物理/科学计算][foundation model] OmniArch 是首个在 1D-2D-3D PDE 上进行统一预训练的科学计算基础模型，通过 Fourier 编解码器解决多尺度问题、Temporal Mask 机制处理多物理量耦合、PDE-Aligner 实现物理先验对齐，在 PDEBench 的 11 类 PDE 上达到了 SOTA 性能。
tags:
  - "ICML 2025"
  - "物理/科学计算"
  - "foundation model"
  - "偏微分方程"
  - "神经算子"
  - "Multi-scale"
  - "Physics-Informed"
---

# OmniArch: Building Foundation Model For Scientific Computing

**会议**: ICML 2025  
**arXiv**: [2402.16014](https://arxiv.org/abs/2402.16014)  
**代码**: [https://openi.pcl.ac.cn/cty315/OmniArch](https://openi.pcl.ac.cn/cty315/OmniArch)  
**领域**: 科学计算  
**关键词**: foundation model, PDE Solver, Fourier Neural Operator, Multi-scale, Physics-Informed

## 一句话总结
OmniArch 是首个在 1D-2D-3D PDE 上进行统一预训练的科学计算基础模型，通过 Fourier 编解码器解决多尺度问题、Temporal Mask 机制处理多物理量耦合、PDE-Aligner 实现物理先验对齐，在 PDEBench 的 11 类 PDE 上达到了 SOTA 性能。

## 研究背景与动机
偏微分方程（PDE）求解是众多科学与工程应用（飞行器设计、天气预报、半导体制造）的核心基础。传统方法（有限元法、有限体积法等）需要大量手工编程且计算开销极高，即便在高性能计算集群上也耗时巨大。神经算子方法（如 FNO、DeepONet）可以学习函数空间之间的映射，但每个模型只能解决特定类型的 PDE，无法跨物理系统迁移。

现有工作中，MPP、Poseidon、DPOT 等尝试了统一预训练，但存在三大核心矛盾：(1) **多尺度**——不同 PDE 涉及 1D/2D/3D 数据、不同网格分辨率和形状，现有方法多受限于固定映射网格；(2) **多物理量**——不同系统含不同数量的物理量（速度、密度、压力等），需同时建模它们的耦合关系；(3) **物理对齐**——预测需符合已知物理定律（守恒律、边界条件等），而非仅拟合数据。

本文切入角度是：**能否像大语言模型一样，用一个统一的基础模型同时求解 1D、2D、3D 的多种 PDE**？核心 idea 是用 Fourier 域编码消除维度差异 + Transformer 自回归建模时间演化 + 对比学习对齐物理先验。

## 方法详解

### 整体框架
OmniArch 采用「预训练 + 微调」范式。预训练阶段：不同维度（1D/2D/3D）的物理场数据经 Fourier 编码器转到频域，通过 TopK 模式截断统一表征长度，然后由共享的 Transformer 骨干建模时间动态，最后 Fourier 解码器恢复空间域预测。微调阶段：引入 PDE-Aligner 利用方程文本描述进行物理对齐的对比学习。

### 关键设计

1. **Fourier 编解码器（解决多尺度）**:

    - 功能：将不同维度、不同分辨率的物理场统一编码到频域
    - 核心思路：对物理场 $u(x^{(d)}, t)$ 先做线性投影 $\Psi$ 对齐维度，再进行 FFT，然后用 TopK 选取最显著的 $K$ 个频率成分：$\hat{u}_K(k,t) = \text{TopK}(\text{FFT}(\Psi[u(x^{(1)},t), \ldots, u(x^{(D)},t)]^\top))$。解码时对预测的 $K$ 个模式做 zero-padding 恢复目标形状，再 IFFT 回空间域。由于不同网格的数据经截断后有相同长度的频域表示，实现了跨尺度的统一输入
    - 设计动机：FFT 的复杂度为 $O(N \log N)$，低于卷积的 $O(N^2)$；频域中高频（细节变化）和低频（整体趋势）自然分离，且全局信息天然加权，适合处理复杂边界条件和异构网格

2. **Temporal Mask + Transformer 骨干（解决多物理量）**:

    - 功能：用 Transformer 自回归机制建模多物理量的时间演化
    - 核心思路：将每个时间步的所有物理量嵌入分组为 $\mathbf{Z}_t = \{\mathbf{U}_t, \mathbf{V}_t\}$，设计 Temporal Mask $\mathbf{M}$ 使得每个时间步的 token 可以 attend 到当前和之前所有时间步的所有物理量，但不能看到未来。具体地，对于 $C$ 个物理量，mask 规则为：$\mathbf{M}(i,j) = 0$ 当 $\lfloor j/C \rfloor \le \lfloor i/C \rfloor$，否则 $-\infty$。这与标准因果 mask 不同——同一时间步的物理量之间可以完全互相 attend
    - 设计动机：Navier-Stokes 方程中速度和压力是耦合的，必须同时处理（满足连续性方程等约束），sequential token processing 无法正确建模这种同步约束。此设计连接了 Transformer 自回归与传统多步法求解的类比

3. **PDE-Aligner（物理对齐微调）**:

    - 功能：在微调阶段利用 PDE 方程的文本描述对预测进行物理约束
    - 核心思路：使用预训练的 BERT 编码 PDE 方程文本 $E_{\text{text}}(\mathcal{P})$，同时从初始状态和当前状态的频域表示中提取物理演化特征（相位差 $\Delta\phi$ 捕捉波传播和色散特性，振幅比 $R$ 量化跨尺度能量传递）。对齐损失为 $L_{\text{Align}} = L_{\text{eq}} + \lambda L_E$，其中 $L_{\text{eq}}$ 是文本-物理特征的对比损失，$L_E = |\sum_K R - 1|$ 保证 Parseval 定理（频域能量守恒）。微调总损失为 $L_{\text{ft}} = L_{\text{sim}} - L_{\text{eq}}$
    - 设计动机：PDE 方程是物理现象最自然的"监督信号"。在频域做对齐更有效，因为守恒律约束能量在各模式间的分布，不同 PDE 有特征性的频谱指纹

### 损失函数 / 训练策略
- **预训练损失**：nRMSE 归一化损失 $L_{\text{sim}}^u = \frac{1}{|B|}\sqrt{\sum_{(x,t)\in B}\left(\frac{u^{\text{pred}}(x,t)-u(x,t)}{\sigma_u}\right)^2}$，按物理量取平均
- **微调损失**：$L_{\text{ft}} = L_{\text{sim}} - L_{\text{eq}}$，同时优化预测精度和物理一致性
- **骨干架构**：LLaMA 架构的 Transformer（从零训练），有 Base 和 Large 两个版本
- **PDE-Aligner 文本编码**：使用预训练的 BERT-base-cased 模型

## 实验关键数据

### 主实验

| PDE 类型 | FNO | MPP-AVIT-L | DPOT-L | OmniArch-L + Aligner | 提升 |
|----------|-----|-----------|--------|----------------------|------|
| 1D CFD | 1.4100 | – | – | **0.0200** | 98.7% |
| 1D Advection | 0.0091 | – | – | **0.0041** | 4.65% |
| 1D Burgers | 0.0174 | – | – | **0.0032** | 66.3% |
| 2D CFD | 0.2060 | 0.0178 | 0.0112 | **0.0125** | – |
| 2D Reaction | 0.1203 | 0.0098 | 0.0263 | **0.0084** | 14.3% |
| 2D SWE | 0.0044 | 0.0022 | 0.0451 | **0.0012** | 45.5% |
| 2D Incom. | 0.2574 | – | – | **0.0827** | 67.9% |
| 3D Maxwell | 0.1906 | – | – | **0.1671** | 12.3% |

### 消融实验

| 配置 | 2D Incom. | 2D CFD | 3D CFD |
|------|-----------|--------|--------|
| Causal Mask | 0.0277 | 0.0198 | 0.1842 |
| No Mask | 0.0285 | 0.0205 | 0.1923 |
| **Temporal Mask** | **0.0227** | **0.0148** | **0.1494** |

| 配置 | 1D PDEs | 2D PDEs | 3D PDEs |
|------|---------|---------|---------|
| 仅预训练 | 0.0103 | 0.0440 | 0.3399 |
| 微调 w/o Aligner | 0.0073 | 0.0345 | 0.3432 |
| **微调 w/ Aligner** | **0.0056** | **0.0262** | **0.2697** |
| 提升 | 23.3% | 24.1% | 21.4% |

### 关键发现
- **1D-2D-3D 统一预训练有效**：OmniArch 是首个在三个维度统一预训练的模型，在 11 类 PDE 上整体超越所有专家模型和预训练模型
- **Temporal Mask 显著优于因果 mask**：改进幅度 18-20%，尤其在 3D CFD（5 个物理量耦合）上优势最明显
- **PDE-Aligner 一致提升约 22%**：且在不同维度上提升比例相似（1D 23.3%、2D 24.1%、3D 21.4%），说明物理对齐与维度无关
- **零样本泛化**：在未见过的 PDE（Shock、KH、OTVortex）上，误差比 MPP 低 4-7 倍
- **多尺度推理**：因 Fourier 截断机制，可处理不同分辨率输入无需重训练，128-256 分辨率性能最优
- **In-context learning**：类似 LLM 的涌现能力，给定几个时间步的观测即可学习新的神经算子

## 亮点与洞察
- 将 NLP 领域基础模型的成功范式（预训练 + 微调 + 对齐）迁移到 PDE 求解领域，概念上简洁且有力
- Fourier 域编码是解决多尺度问题的优雅方案——频率截断自然实现跨分辨率的统一表示
- Temporal Mask 的设计抓住了多物理量系统的本质——耦合变量必须同步处理
- PDE-Aligner 用方程文本做物理对齐，巧妙借鉴了 CLIP 式对比学习，频域特征（相位差+振幅比）作为物理指纹的设计新颖
- 零样本和 in-context learning 的涌现能力令人印象深刻，暗示模型学到了可迁移的物理算子而非数据模式

## 局限与展望
- **3D 性能仍有提升空间**：3D CFD 和 Maxwell 的 nRMSE 仍较高（0.37、0.17），作者也承认 3D 系统对模型构成挑战
- **可解释性不足**：虽然 PDE-Aligner 增强了物理对齐，但模型本质仍是数据驱动的黑盒
- **计算和数据瓶颈**：scalability 受限于计算资源和可用的训练数据，特别是在复杂突变系统中
- **尚未验证实际工程问题**：所有实验在标准 benchmark 上进行，真实工程应用（复杂几何、非结构化网格）的效果未知
- **PDE-Aligner 需要方程文本**：对于未知方程的系统无法直接使用物理对齐

## 相关工作与启发
- **vs FNO**: OmniArch 在保持 FNO 频域处理优势的同时，通过预训练获得跨 PDE 的迁移能力
- **vs MPP/DPOT**: 这些方法仅支持 2D 预训练，OmniArch 首次实现 1D-2D-3D 统一，且零样本泛化远优
- **vs Poseidon**: Poseidon 支持任意时间步的单步推理但精度不足，OmniArch 用自回归多步推理获得更高精度
- **PDE-Aligner 启发**：用自然语言描述物理规律并通过对比学习对齐，这个方向值得深入——未来可能扩展到用 LLM 理解和生成 PDE 约束

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个 1D-2D-3D 统一预训练，Temporal Mask 和 PDE-Aligner 的设计都有原创性
- 实验充分度: ⭐⭐⭐⭐ 11 类 PDE 全面评测，含零样本、in-context learning、多尺度、逆问题等丰富实验
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但部分公式较密集，3D 实验分析不够深入
- 价值: ⭐⭐⭐⭐⭐ 为 PDE 求解的基础模型方向树立了重要里程碑，统一架构思想有深远影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Towards a Foundation Model for Partial Differential Equations Across Physics Domains](../../AAAI2026/physics/towards_a_foundation_model_for_partial_differential_equations_across_physics_dom.md)
- [\[ICML 2025\] Finetuning Stellar Spectra Foundation Models with LoRA](finetuning_stellar_spectra_foundation_models_with_lora.md)
- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](../../NeurIPS2025/physics/transfer_learning_beyond_the_standard_model.md)
- [\[AAAI 2026\] Data Verification is the Future of Quantum Computing Copilots](../../AAAI2026/physics/data_verification_is_the_future_of_quantum_computing_copilots.md)
- [\[ICLR 2026\] Augmenting Representations with Scientific Papers](../../ICLR2026/physics/augmenting_representations_with_scientific_papers.md)

</div>

<!-- RELATED:END -->
