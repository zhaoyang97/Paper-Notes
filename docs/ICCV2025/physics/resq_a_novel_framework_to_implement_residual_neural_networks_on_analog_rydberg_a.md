---
title: >-
  [论文解读] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers
description: >-
  [ICCV 2025][物理/科学计算][量子机器学习] 提出 ResQ——首个利用模拟 Rydberg 原子量子计算机的连续时间哈密顿演化来原生实现残差神经网络（ResNet）的框架，通过分段参数化激光脉冲编码输入特征和训练参数，在 MNIST/FashionMNIST/医疗数据集的分类任务上相比同等规模经典模型平均提升50%。
tags:
  - "ICCV 2025"
  - "物理/科学计算"
  - "量子机器学习"
  - "残差网络"
  - "Rydberg原子"
  - "模拟量子计算"
  - "神经ODE"
---

# ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers

**会议**: ICCV 2025  
**arXiv**: [2506.21537](https://arxiv.org/abs/2506.21537)  
**代码**: [github.com/positivetechnologylab/ResQ](https://github.com/positivetechnologylab/ResQ)  
**领域**: 量子计算 / 物理启发式计算  
**关键词**: 量子机器学习, 残差网络, Rydberg原子, 模拟量子计算, 神经ODE

## 一句话总结

提出 ResQ——首个利用模拟 Rydberg 原子量子计算机的连续时间哈密顿演化来原生实现残差神经网络（ResNet）的框架，通过分段参数化激光脉冲编码输入特征和训练参数，在 MNIST/FashionMNIST/医疗数据集的分类任务上相比同等规模经典模型平均提升50%。

## 研究背景与动机

### 问题定义

量子机器学习（QML）旨在利用量子计算加速或增强机器学习模型。当前 QML 主要基于数字门（gate-based）量子系统，但这类系统在实现残差神经网络（ResNet）时面临根本限制——量子门只允许酉线性变换，无法直接实现跳跃连接中的加法操作 $F(x)+x$。

### 已有方法的不足

**门基方案的局限**：数字量子电路是离散操作序列，难以表达 ResNet 和 Neural ODE 等具有连续动力学的模型

**混合方案不完全量子化**：Liang 等人的混合量子-经典模型仍依赖经典计算完成残差连接

**连续变量 QNN**：Killoran 等人利用连续变量基实现 ResNet，但未利用 Rydberg 原子系统的独特能力

**现有 Rydberg 原子工作**：Lu 等人的二值 MNIST 分类使用数字-模拟混合方案，不适用于现有仅支持模拟功能的商用设备

### 核心洞察

ResNet 可以通过 Neural ODE 表示为连续时间微分方程 $\frac{dx}{dt} = F(x, \theta)$。而模拟 Rydberg 原子量子计算机天然通过薛定谔方程 $i\hbar\frac{d}{dt}|\Psi(t)\rangle = H(t)|\Psi(t)\rangle$ 进行连续时间演化。两者在数学形式上具有**天然对应关系**——这是门基量子系统所不具备的。

## 方法详解

### 整体框架

ResQ 的分类工作流：
1. PCA 降维预处理数据
2. 将 PCA 特征缩放到物理硬件约束范围内
3. 将输入和参数编码到 Rydberg 哈密顿量的全局和局部控制参数中
4. 执行哈密顿演化
5. 测量所有量子比特，平均 $|1\rangle$ 态概率作为预测输出
6. 基于交叉熵损失的梯度训练

### 关键设计

#### 1. **Rydberg 原子哈密顿量参数化**
- **功能**：将 ResNet 的参数和数据编码到量子系统的物理控制量中
- **核心思路**：全局 Rydberg 哈密顿量：

$$H(t) = \frac{\Omega(t)}{2}\sum_i(e^{i\phi(t)}|g\rangle_i\langle r|_i + h.c.) - \Delta(t)\sum_i \hat{n}_i + \sum_{i<j}\frac{C_6}{|\vec{p}_i - \vec{p}_j|^6}\hat{n}_i\hat{n}_j$$

加上局部失谐项：$H_{local}(t) = -\delta(t)\sum_i h_i \hat{n}_i$

其中 $\Omega(t)$（Rabi频率）、$\Delta(t)$（失谐）、$\delta(t)$（局部失谐）作为时间序列参数化，$h_i \in [0,1]$ 为逐站点耦合系数。

- **设计动机**：这些物理参数天然对应 ResNet 的可学习权重，时间演化对应网络的层

#### 2. **分段脉冲参数化（Piecewise Pulse Parameterization）**
- **功能**：将每个激光参数分成多个分段线性脉冲间隔
- **核心思路**：每个脉冲间隔 $i$ 的强度参数化为 $\theta_j \omega_i + \theta_{j+1}$，其中 $\theta_j, \theta_{j+1}$ 是可训练参数，$\omega_i$ 是输入特征
    - 保持时间0.15μs + 过渡时间0.05μs = 0.2μs/间隔
    - 最大运行时间4.0μs → 最多19个间隔
    - 三个全局参数 × 各需2个参数/间隔 = $6M$ 个可训练参数
- **设计动机**：线性组合使每个输入特征在整个脉冲持续时间内产生影响，同时保持可训练的偏移

#### 3. **局部失谐特征编码**
- **功能**：利用逐站点耦合系数 $h_i$ 进行特征编码和参数化
- **核心思路**：偶数量子比特的 $h_i$ 编码输入特征，奇数量子比特的 $h_i$ 作为可训练参数
    - $N$ 个量子比特 → $N/2$ 个额外输入 + $N/2$ 个额外参数
    - 总计：$3 + N/2$ 个输入特征，$6M + N/2$ 个可训练参数
- **设计动机**：特征数量与量子比特数呈**线性**扩展，无需增加电路深度，这是重要的可扩展性优势

### 损失函数 / 训练策略

- **损失函数**：二元交叉熵损失
- **梯度计算**：随机脉冲梯度方法（Stochastic Pulse Gradient）——对模拟量子程序的无偏梯度估计
    - 重复执行模拟路径，随机插入量子比特旋转
    - 每次梯度评估使用20个采样点
- **测量**：重复1000次测量估计概率分布
- **优化器**：Adam
- 使用 $N=4$ 个原子，5个 PCA 特征
- 精心设计的特征分配：高方差 PCA 特征分配给全局参数（$\Omega, \Delta$），低方差特征分配给局部项

## 实验关键数据

### 主实验（与同等参数量经典模型对比）

| 任务 | ResQ | C-NN (1×) | C-ResNet (1×) | C-NODE (1×) | C-NN (100×) |
|------|------|-----------|---------------|-------------|-------------|
| PID 糖尿病 | **最优** | ↓52% | ↓57% | ↓36% | 可比 |
| MNIST 0v1（简单）| 高精度 | ↓显著 | ↓显著 | ↓显著 | 可比 |
| MNIST 4v9（困难）| 竞争力 | ↓显著 | ↓显著 | ↓ | 略优 |
| FashionMNIST（简单）| 竞争力 | ↓37% | ↓ | ↓ | 略优 |
| FashionMNIST（困难）| 竞争力 | ↓ | ↓ | ↓ | 可比 |

总体：ResQ 相比同参数量的经典模型**平均提升50%**（C-NN: 56%, C-ResNet: 57%, C-NODE: 36%）

### 原子配置消融

| 配置 | 链形 | 环形/方形 | 三角形 | 最佳间距 |
|------|------|---------|--------|---------|
| 性能趋势 | 竞争力 | 略优 | 竞争力 | 12μm（中等间距）|
| 交互强度 | - | - | - | 弱交互更优 |

### 脉冲间隔消融

| 间隔数 | 1 | 3 | 5 | 7+ |
|--------|---|---|---|-----|
| PID 准确率 | 基线 | 略优 | 持平 | 持平 |
| 说明 | 3个间隔已足够，增加间隔收益递减 |

### 硬件噪声鲁棒性

- 在 QuEra Aquila 256量子比特真实量子计算机上执行推理
- 准确率和 F1 分数与理想模拟结果在**1%以内**
- 仅在靠近决策边界（0.5）的样本可能因硬件噪声翻转预测

### 关键发现

1. **ResQ 在参数效率上显著优于经典模型**：相同参数量下平均提升50%，可媲美10-100×参数量的经典模型
2. **弱交互（大间距）更适合分类**：强量子纠缠反而难以训练
3. **天然的噪声鲁棒性**：ResQ 的设计使其对模拟量子硬件的噪声具有内在抗性
4. **常数执行时间**：Neural ODE 表示使 ResNet 具有与问题规模无关的执行时间，这在经典和数字量子架构中不成立
5. **线性特征扩展**：输入特征数随量子比特数线性增长，无需增加电路深度

## 亮点与洞察

1. **深刻的物理-计算对应**：ResNet → Neural ODE → 薛定谔方程 → Rydberg 原子演化的对应链条优雅且有洞察力
2. **在真实量子硬件上验证**：不仅是模拟，还在 QuEra Aquila 的256量子比特设备上进行了实际推理
3. **参数效率优势的量子来源**：$N$ 个量子比特可表示 $2^N$ 维状态空间，提供指数级表达能力
4. **工程实用的设计选择**：非零下界强制动力学存在、PCA 特征按重要性分配到全局/局部参数等

## 局限与展望

1. **仅支持二分类**：当前框架基于单个概率阈值，多分类需要扩展
2. **仅使用4个原子**：模拟更多原子的计算成本极高，实际可用的计算规模有限
3. **训练在经典模拟器上完成**：真实量子硬件仅用于推理，量子训练仍是挑战
4. **PCA 降维限制**：5个 PCA 特征可能丢失重要信息，限制了在复杂数据集上的表现
5. **局部失谐的实验性支持**：QuEra Aquila 目前仅实验性支持局部失谐功能
6. **与 ICCV 的视觉领域关联较弱**：尽管使用了 MNIST/FashionMNIST 等视觉数据集

## 相关工作与启发

- Neural ODE（Chen et al., NeurIPS 2018）奠定了 ResNet 与 ODE 的理论联系
- Rydberg 原子系统是目前唯一支持连续时间哈密顿演化 + 局部/全局控制的量子硬件
- 随机脉冲梯度方法为模拟量子程序的训练提供了理论基础
- 量子计算的指数级状态空间可能为未来的大规模视觉任务提供新的计算范式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次在模拟 Rydberg 原子系统上全量子化实现 ResNet，物理洞察深刻
- **实验充分度**: ⭐⭐⭐⭐ — 包含真实量子硬件验证和多个数据集，但限于4原子和二分类
- **写作质量**: ⭐⭐⭐⭐ — 物理背景和方法设计讲解清晰，适合跨领域读者
- **价值**: ⭐⭐⭐ — 学术创新性强，但实际应用场景和可扩展性有限，与视觉领域关联较弱

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Accelerating Inference for Multilayer Neural Networks with Quantum Computers](../../ICLR2026/physics/accelerating_inference_for_multilayer_neural_networks_with_quantum_computers.md)
- [\[CVPR 2025\] ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks](../../CVPR2025/physics/atp_adaptive_threshold_pruning_for_efficient_data_encoding_in_quantum_neural_net.md)
- [\[ICML 2025\] Compact Matrix Quantum Group Equivariant Neural Networks](../../ICML2025/physics/compact_matrix_quantum_group_equivariant_neural_networks.md)
- [\[NeurIPS 2025\] DeltaPhi: Physical States Residual Learning for Neural Operators in Data-Limited PDE Solving](../../NeurIPS2025/physics/deltaphi_physical_states_residual_learning_for_neural_operators_in_data-limited_.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](../../ICML2025/physics/differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)

</div>

<!-- RELATED:END -->
