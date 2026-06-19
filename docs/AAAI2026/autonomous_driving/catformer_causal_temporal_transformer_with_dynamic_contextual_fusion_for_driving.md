---
title: >-
  [论文解读] CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction
description: >-
  [AAAI2026][自动驾驶][driving intention prediction] 提出 CaTFormer，通过因果时序 Transformer 显式建模驾驶员行为与环境上下文之间的因果交互，在 Brain4Cars 数据集上以 98.6% F1 达到 SOTA。 驾驶意图预测是自动驾驶安全的关键：提前数秒预判驾…
tags:
  - "AAAI2026"
  - "自动驾驶"
  - "driving intention prediction"
  - "causal inference"
  - "Transformer"
  - "dual-stream fusion"
  - "counterfactual reasoning"
---

# CaTFormer: Causal Temporal Transformer with Dynamic Contextual Fusion for Driving Intention Prediction

**会议**: AAAI2026  
**arXiv**: [2507.13425](https://arxiv.org/abs/2507.13425)  
**代码**: [srwang0506/CaTFormer](https://github.com/srwang0506/CaTFormer)  
**领域**: 自动驾驶  
**关键词**: driving intention prediction, causal inference, Transformer, dual-stream fusion, counterfactual reasoning

## 一句话总结

提出 CaTFormer，通过因果时序 Transformer 显式建模驾驶员行为与环境上下文之间的因果交互，在 Brain4Cars 数据集上以 98.6% F1 达到 SOTA。

## 背景与动机

驾驶意图预测是自动驾驶安全的关键：提前数秒预判驾驶员的变道/转弯意图，系统可主动预警或采取规避措施。现有方法存在两大不足：

1. **融合方式粗糙**：大多数方法对车内（驾驶员状态）和车外（交通场景）特征仅做简单拼接或线性聚合，未显式建模两者间的因果依赖。驾驶员状态变化直接影响车辆行为，但这种因果关系被忽略。
2. **时序建模不足**：早期 3D CNN-LSTM 架构难以捕捉长距离、非连续的时序依赖。即使引入 Transformer，也主要依赖架构自身的隐式注意力，缺乏对因果关系的显式推理。

此外，已有因果推理工作（如基于因果模型的域泛化）覆盖的场景范围有限，未能在统一框架内同时解决时序对齐、虚假相关性消除和多视角自适应融合问题。

## 核心问题

如何在双流 Transformer 架构中**显式建模**驾驶员行为与环境上下文之间的**因果时序依赖**，同时消除虚假相关性，实现鲁棒的驾驶意图预测？

## 方法详解

CaTFormer 采用双流架构处理车内外同步视频帧序列，包含三个核心模块：

### 1. Reciprocal Delayed Fusion (RDF) — 互惠延迟融合

- **核心思想**：通过时间延迟机制建模帧间时序优先关系。在时间步 $t$，注意力机制仅访问前一帧 $t-1$ 的信息，显式建立车内外特征流之间的时序因果约束。
- **双向依赖注意力 (BDA)**：当前帧的车内和车外特征双向注意其单帧延迟对应物，使用 $H=8$ 个注意力头捕捉多样化关联，通过拼接和线性映射聚合。
- **通道门控**：对 BDA 输出施加两层通道门控（FC → ReLU → FC → Sigmoid → Hadamard 积），自适应增强信息性通道、抑制噪声通道。
- **正则化**：RMSNorm 稳定数值 + Dropout 防止过拟合。

### 2. Counterfactual Residual Encoding (CRE) — 反事实残差编码

这是本文最核心的创新点，利用反事实推理消除虚假相关性：

- **直接因果效应计算**：在每个时间步，计算两种注意力分布：
    - **观测注意力** $A^{\text{obs}}$：用实际车外特征计算正常的跨流注意力
    - **反事实注意力** $A^{\text{cf}}$：将所有车外特征替换为其时间均值（中性基线），计算注意力
    - 两者之差 $\Delta = A^{\text{obs}} - A^{\text{cf}}$ 量化了环境上下文对车内表示的直接因果贡献
- **正交投影**：将因果残差投影到全局基线的正交补空间，确保因果模式反映真实意图相关依赖而非数据集偏差。
- **动态残差门控**：可学习的门控系数 $g_T$ 根据预测价值自适应调整残差贡献，关键机动动作被放大，常规模式被抑制。
- **自适应意图编码**：从车外摘要通过 softmax 分类提取粗粒度意图分布，再嵌入为连续的 intention token $z_{\text{intent}}$，作为全局语义锚点为下游融合提供自顶向下指导。

### 3. Feature Synthesis Network (FSN) — 特征合成网络

- 对车内、车外、交互三个分支分别进行残差非线性变换（双层 FC-ReLU-FC），结合速度特征 $s$。
- 通过可学习的置信度权重 $w_i$ 自适应控制各分支贡献：$\ell_{\text{joint}} = \sum_{i} w_i (W_i r_i)$。
- 权重通过 softmax 归一化，实现场景自适应的动态融合。

### 训练策略

统一损失函数结合四个流级别头（in, out, ctx, joint）的平均交叉熵损失和意图预测损失：

$$\mathcal{L} = \frac{1}{4}\sum_{i \in \mathcal{H}} \text{CE}(\ell_i, y) + \alpha \cdot \text{CE}(\ell_{\text{intent}}, y)$$

其中 $\alpha = 0.1$。Adam 优化器，初始学习率 $10^{-3}$，训练 160 epochs，batch size 16。

## 实验关键数据

### 数据集

- **Brain4Cars**：包含车外（480×720）和车内（1088×1920）视频，共 594 个有效事件，覆盖高速公路和城市场景。五类动作：直行、左转、右转、左变道、右变道。

### 主要结果（Brain4Cars）

| 方法 | 模态 | Precision | Recall | F1 |
|------|------|-----------|--------|-----|
| DCNN | 相机+速度 | 91.8 | 92.5 | 92.1 |
| IDIPN | 仅相机 | 94.2 | 94.9 | 94.5 |
| FedPRM | 相机+GPS+速度 | 99.0 | 92.0 | 95.2 |
| **CaTFormer（仅相机）** | 仅相机 | 96.7 | 98.5 | **97.6** |
| **CaTFormer（+速度）** | 相机+速度 | 98.7 | 98.5 | **98.6** |

仅用相机就达到 97.6% F1，超越所有多模态方法；加入速度后 98.6%，比最佳 FedPRM 高 3.4%。

### 早期预警能力（截断实验）

| 观测窗口 | CaTFormer | IDIPN | TIFN |
|----------|-----------|-------|------|
| [-5, 0] | 98.6 | 94.5 | 87.9 |
| [-5, -1] | 97.4 | 84.1 | 80.9 |
| [-5, -2] | 90.1 | 74.2 | 71.0 |
| [-5, -3] | 78.4 | 62.0 | 55.0 |
| [-5, -4] | 63.7 | 55.4 | 44.6 |

在所有提前预测窗口上均显著优于对比方法，提前 2 秒预测仍有 90.1% F1。

### 消融实验

| 配置 | F1 [-5,0] | F1 [-5,-2] |
|------|-----------|------------|
| Base（双流 Transformer） | 95.8 | 85.4 |
| Base+RDF | 97.1 | 87.1 |
| Base+CRE | 97.0 | 86.9 |
| Base+FSN | 96.6 | 86.3 |
| CaTFormer (R+C+F) | **98.6** | **90.1** |

三个模块各自贡献 +1～1.3% 提升，组合后协同增益达 +2.8%（完整窗口）和 +4.7%（提前 2 秒）。

## 亮点

1. **反事实因果推理**：CRE 模块通过对比观测注意力与反事实注意力差异来识别真实因果效应，并用正交投影和动态门控进一步纯化，方法论上很优雅。
2. **极强的早期预警能力**：提前 2 秒预测仍有 90.1% F1，远超对比方法（74.2%），对实际安全预警系统意义重大。
3. **少模态高性能**：仅用相机就超越了使用 GPS+地图+速度的多模态方法，说明因果建模比简单堆叠传感器更有效。
4. **可解释的注意力可视化**：时序注意力热图和决策边际显著性图展示了从动态事件理解到静态决策归因的完整推理路径。

## 局限与展望

1. **单一数据集验证**：仅在 Brain4Cars（594 个事件）上评估，数据规模较小，缺乏在大规模数据集和更多场景（恶劣天气、夜间）上的验证。
2. **五类动作粗粒度**：仅覆盖直行、转弯、变道，未涉及加减速、停车、紧急避让等细粒度意图。
3. **光流计算开销**：外部流依赖 Farneback 光流预计算，增加了预处理时间，可考虑用端到端可学习的运动特征替代。
4. **实时性未讨论**：未报告推理延迟，14.53M 参数 + 光流预计算的实时性有待验证。
5. **反事实基线选择**：CRE 中使用时间均值作为中性基线是否最优？可探索更复杂的反事实构造策略。

## 与相关工作的对比

| 维度 | CaTFormer | TIFN | CEMFormer | IDIPN |
|------|-----------|------|-----------|-------|
| 融合方式 | 因果延迟双向注意力 | STU 状态更新 | 跨视角 Transformer | 交互解耦 |
| 因果推理 | 反事实残差编码 | 无 | 无 | 无 |
| 最佳 F1 | **98.6** | 87.9 | 87.1 | 94.5 |
| 意图编码 | 自适应 intention token | 无 | 无 | 无 |
| 多视角融合 | FSN 动态加权 | 语义分割注意力 | 统一跨视角 | 解耦预测 |

CaTFormer 的核心区分点在于**显式因果建模**：RDF 建立时序因果约束，CRE 消除虚假相关性，FSN 自适应融合，三者形成完整的因果推理链条，而非依赖隐式的注意力相关性。

## 启发与关联

- **反事实注意力减法**的思路可推广到其他需要消除虚假相关性的视觉任务（如 VQA、视频理解）。
- **延迟融合机制**为多模态时序对齐提供了轻量级但有效的方案，比复杂的时序对齐网络简洁。
- 结合因果推理和 Transformer 的范式可延伸用于行人意图预测、交通流预测等相关任务。
- CRE 的正交投影去偏思路与因果表示学习中的去混淆方法（如 backdoor adjustment）有异曲同工之处。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 反事实残差编码 + 互惠延迟融合的组合较新颖
- 实验充分度: ⭐⭐⭐ — 消融完备但仅单一小数据集
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、公式严谨、可视化丰富
- 价值: ⭐⭐⭐⭐ — 因果推理思路有普适性，早期预警性能优异

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SocialMOIF: Multi-Order Intention Fusion for Pedestrian Trajectory Prediction](../../CVPR2025/autonomous_driving/socialmoif_multi-order_intention_fusion_for_pedestrian_trajectory_prediction.md)
- [\[AAAI 2026\] Meta Dynamic Graph for Traffic Flow Prediction](meta_dynamic_graph_for_traffic_flow_prediction.md)
- [\[CVPR 2026\] Efficient Equivariant Transformer for Self-Driving Agent Modeling](../../CVPR2026/autonomous_driving/efficient_equivariant_transformer_for_self-driving_agent_modeling.md)
- [\[AAAI 2026\] RAST: A Retrieval Augmented Spatio-Temporal Framework for Traffic Prediction](rast_a_retrieval_augmented_spatio-temporal_framework_for_traffic_prediction.md)
- [\[AAAI 2026\] Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults](multimodal_data_fusion_to_capture_dynamic_interactions_between_built_environment.md)

</div>

<!-- RELATED:END -->
