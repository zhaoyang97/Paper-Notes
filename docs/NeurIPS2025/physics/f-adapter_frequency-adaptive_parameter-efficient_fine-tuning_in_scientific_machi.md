---
title: >-
  [论文解读] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning
description: >-
  [NeurIPS 2025][物理/科学计算][参数高效微调] 本文首次系统研究了科学机器学习中预训练大型算子模型(LOM)的参数高效微调(PEFT)，发现 LoRA 在傅里叶层中存在深度放大的近似误差下界，而 Adapter 保留了通用逼近能力；据此提出频率自适应 Adapter（F-Adapter），按频谱能量分配 Adapter 容量，在 3D Navier-Stokes 预测任务上仅调参不到 2% 即达到 SOTA。
tags:
  - "NeurIPS 2025"
  - "物理/科学计算"
  - "参数高效微调"
  - "傅里叶神经算子"
  - "频率自适应"
  - "科学机器学习"
  - "大型算子模型"
---

# F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning

**会议**: NeurIPS 2025  
**arXiv**: [2509.23173](https://arxiv.org/abs/2509.23173)  
**代码**: [有](https://github.com/)  
**领域**: 科学计算  
**关键词**: 参数高效微调, 傅里叶神经算子, 频率自适应, 科学机器学习, 大型算子模型

## 一句话总结
本文首次系统研究了科学机器学习中预训练大型算子模型(LOM)的参数高效微调(PEFT)，发现 LoRA 在傅里叶层中存在深度放大的近似误差下界，而 Adapter 保留了通用逼近能力；据此提出频率自适应 Adapter（F-Adapter），按频谱能量分配 Adapter 容量，在 3D Navier-Stokes 预测任务上仅调参不到 2% 即达到 SOTA。

## 研究背景与动机
**领域现状**: 参数高效微调(PEFT)在 NLP 和 CV 大模型适配中已被广泛验证（LoRA、Adapter、Prompt Tuning 等），但在科学机器学习(SciML)中尚未被系统探索。近年来大型算子模型(LOM)如 DPOT（10亿参数）通过多 PDE 预训练获得了强大的泛化能力。

**现有痛点**: LOM 参数量庞大（DPOT-H 达 1B），全量微调代价高昂（训练显存约 25-37 GB，参数量 100%）；而直接将 NLP/CV 中常用的 LoRA 迁移到 LOM 上效果很差，不同 rank 设置下 L2RE 均约为 0.63-0.64，远不如全量微调的 0.54。

**核心矛盾**: PDE 解的解流形具有宽频带、级联耦合的频谱特征，不同于自然语言/图像的低秩结构。LoRA 的线性低秩约束在傅里叶层中会产生深度放大的逼近误差下界，无法有效匹配 PDE 解的频谱特性。

**本文目标**: 能否设计出一种尊重 PDE 解频率自适应结构和物理先验的 PEFT 方法，使其既高效又保持频谱保真度？

**切入角度**: 从理论分析入手——证明 LoRA 的固有缺陷（Proposition 3.2 的深度放大下界）和 Adapter 的优势（Proposition 3.2 的指数衰减误差），再结合 PDE 解的能量频谱集中特性（低频主导能量），设计频率感知的 Adapter。

**核心 idea**: 低频频段包含 PDE 解的大部分能量，应分配更大的 Adapter 瓶颈维度；高频频段稀疏且噪声敏感，用轻量级 Adapter 即可。

## 方法详解

### 整体框架
F-Adapter 是一种即插即用的 PEFT 模块，插入到 LOM（如 DPOT）中每个傅里叶域混合层。整体流程：
1. 输入张量经 3D rFFT 变换到频域
2. 频谱按径向壳分为 B 个不重叠频带（通常 B=4）
3. 通道维度分为 K 个块
4. 每个 (block, band) 组合配备三个 F-Adapter（input/mid/output），瓶颈宽度由频带位置决定
5. 经 iRFFT 变换回物理空间并残差相加

### 关键设计
1. **频率自适应瓶颈分配**: 瓶颈宽度按公式 $r_b = \lfloor r_{min} + (r_{max} - r_{min})(1 - f_b/M)^p \rfloor$ 分配，其中 $f_b$ 为频带中心频率。低频带获得更大的 $r_b$（接近 $r_{max}$），高频带收缩至 $r_{min}$。这一设计直接来源于 PDE 解的频谱能量集中理论（Proposition 3）。
2. **瓶颈 MLP 微架构**: 每个 Adapter 是标准的 down-activation-up 瓶颈残差结构：$\tilde{z} = z + s_b \cdot W^{up}_b \cdot \sigma(W^{down}_b \cdot z + b^{down}_b) + b^{up}_b$，使用 GELU 激活，$s_b$ 为可学习标量。
3. **零初始化策略**: $W^{up}_b$ 和 $b^{up}_b$ 零初始化，确保训练初期 Adapter 为恒等映射，不扰动预训练权重。Down-projection 使用 Kaiming-uniform 初始化。
4. **实虚分离处理**: 频域中复数张量的实部和虚部分别通过 Adapter 处理，避免复数运算的额外复杂度。

### 理论支撑
- **LoRA 的下界**（Proposition 3.2）: Block-wise LoRA 的最坏情况算子范数误差下界为全局矩阵的第 $(Kr+1)$ 个奇异值，随深度 $K$ 增长误差累积。
- **Adapter 的指数收敛**（Proposition 3.2）: 傅里叶域 Adapter 的逼近误差为 $O(K^{d/2-\alpha}) + O(K^{d/2} e^{-cm})$，随瓶颈宽度 $m$ 指数衰减。
- **PDE 解的频谱稀疏性**（Proposition 3）: 高频模态的累积能量以 $O(K^{d-2s})$ 多项式衰减，低频模态主导能量。

### 损失函数 / 训练策略
- 使用 AdamW 优化器训练 500 epochs
- 仅微调 F-Adapter 参数（不到 2% 的总参数），冻结预训练主干
- 评估指标为 L2 相对误差 (L2RE)

## 实验关键数据

### 主实验

| 方法 | % Params | L2RE (Rand M=1.0) | L2RE (Rand M=0.1) | L2RE (Turb) |
|------|----------|--------------------|--------------------|-------------|
| LoRA (r=32) | 1.37% | 0.6395 | 0.6211 | 0.6842 |
| AdaLoRA | 0.69% | 0.6726 | 0.6275 | 0.6795 |
| HydraLoRA | 0.85% | 0.6333 | 0.6164 | 0.6888 |
| Vanilla Adapter (d=8) | 1.16% | 0.5496 | 0.4893 | 0.4696 |
| FiLM Adapter | 1.30% | 0.5655 | 0.5054 | 0.4987 |
| **F-Adapter (Ours)** | **1.91%** | **0.5329** | **0.4639** | **0.4523** |
| Full Fine-Tuning | 100% | 0.5391 | 0.4002 | 0.2382 |

在 SWE-2D 上 F-Adapter 的 L2RE 达到 0.0116（Vanilla Adapter 为 0.0902，LoRA 为 0.1081）；在 MHD-3D 数据稀缺场景下达到 0.6341（Vanilla Adapter 为 0.7226）。

### 消融实验

| 方法 | Rand M=1.0 | Rand M=0.1 | Turb M=1.0 |
|------|------------|------------|------------|
| F-Inverse-Adapter（反转分配） | 0.5664 | 0.4983 | 0.4747 |
| Vanilla Adapter | 0.5496 | 0.4893 | 0.4696 |
| **F-Adapter** | **0.5329** | **0.4639** | **0.4523** |

频域 Adapter 变体对比：Chebyshev Adapter 精度略低但延迟增加 3 倍；Fourier Adapter 显存增加 29%、速度下降 10 倍且精度大幅下降；WaveAct Adapter 速度和显存接近但精度不及 F-Adapter。

### 关键发现
1. LoRA 及其所有变体（AdaLoRA、HydraLoRA、RandLoRA、SVFT）在 FNO 基架构上全面失败，L2RE 均在 0.60 以上
2. Adapter 方法显著优于 LoRA，且性能随瓶颈宽度增加而提升
3. 频率自适应分配（低频大维度、高频小维度）始终优于均匀分配和反转分配
4. 在非 FNO 的 Transformer 架构(Poseidon)上，F-LoRA（频率自适应 + LoRA）达到 SOTA（L2RE=0.2746），说明频率感知理念具有跨架构通用性

## 亮点与洞察
1. **首次系统研究 SciML PEFT**: 填补了科学机器学习中大模型高效微调的空白，揭示了 NLP/CV 中的 PEFT 方法不能直接迁移到 SciML
2. **理论与实践的紧密结合**: 先从理论证明 LoRA 的固有缺陷和 Adapter 的优势，再用 PDE 解频谱理论指导架构设计，最后实验验证
3. **频谱 drop-high 实验**: 通过逐步去除高频分量的诊断实验，直观展示了低频能量主导的现象
4. **跨架构泛化**: F-LoRA 变体在 Transformer 架构上也有效，展示了频率感知思想的普适性

## 局限与展望
1. F-Adapter 的参数量（~1.91%）略高于 LoRA（~1.37%），虽然显存开销相当
2. 与全量微调仍有差距（特别是 Turbulence 数据集：0.4523 vs 0.2382），高频细节恢复仍不完美
3. 频带数 B 和分配曲线参数 p 需要手动调节，自适应确定这些超参数可以进一步研究
4. 目前仅在 FNO 系列和 Poseidon 上验证，更多架构（如 U-Net 变体、图神经网络）的扩展有待探索
5. 非 FNO 架构上需要额外的频率估计步骤（如局部 FFT），增加了实现复杂度

## 相关工作与启发
- **LoRA 系列**: LoRA、AdaLoRA、HydraLoRA、RandLoRA、SVFT 等都基于低秩线性更新，在 LLM 上表现好但在 FNO 上失败，说明频域操作的非线性特性需要非线性适配
- **DPOT**: 当前最大的公开 LOM（1B 参数），采用 Fourier-Attention 架构和去噪预训练
- **Poseidon**: 纯 Transformer 基的算子模型，F-LoRA 在其上的成功说明频率感知 PEFT 可泛化到非 FFT 架构
- **启发**: 领域特定的归纳偏置（如频谱稀疏性）应该被显式编码到 PEFT 架构中，而非简单套用通用方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统研究 SciML PEFT + 频率自适应设计 + 扎实理论分析
- 实验充分度: ⭐⭐⭐⭐ 多数据集多基线对比 + 丰富消融 + 跨架构验证
- 写作质量: ⭐⭐⭐⭐ 理论-实验-设计逻辑清晰，动机自然
- 价值: ⭐⭐⭐⭐⭐ 为 SciML 大模型高效适配开辟新方向，理论和方法贡献均突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] TITAN: A Trajectory-Informed Technique for Adaptive Parameter Freezing in Large-Scale VQE](titan_a_trajectory-informed_technique_for_adaptive_parameter_freezing_in_large-s.md)
- [\[AAAI 2026\] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](../../AAAI2026/physics/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)
- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[NeurIPS 2025\] Physics-Guided Machine Learning for Uncertainty Quantification in Turbulence Models](physics-guided_machine_learning_for_uncertainty_quantification_in_turbulence_mod.md)

</div>

<!-- RELATED:END -->
