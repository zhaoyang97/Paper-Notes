---
title: >-
  [论文解读] Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention
description: >-
  [CVPR 2025][目标检测][事件相机] 提出首个面向大规模基准的混合 SNN-ANN 目标检测模型，设计注意力桥接模块（ASAB）将 SNN 的稀疏脉冲表示通过时空注意力转换为 ANN 可处理的密集特征，在 Gen1/Gen4 数据集上以仅 6.6M 参数大幅超越 SNN 方法并接近 ANN/RNN 方法的精度，同时 SNN 部分可部署在 Intel Loihi 2 神经形态芯片上实现低功耗推理。
tags:
  - "CVPR 2025"
  - "目标检测"
  - "事件相机"
  - "混合SNN-ANN"
  - "注意力桥接"
  - "神经形态硬件"
  - "高效推理"
---

# Efficient Event-Based Object Detection: A Hybrid Neural Network with Spatial and Temporal Attention

**会议**: CVPR 2025  
**arXiv**: [2403.10173](https://arxiv.org/abs/2403.10173)  
**代码**: 无  
**领域**: 目标检测 / 事件视觉  
**关键词**: 事件相机, 混合SNN-ANN, 注意力桥接, 神经形态硬件, 高效推理

## 一句话总结

提出首个面向大规模基准的混合 SNN-ANN 目标检测模型，设计注意力桥接模块（ASAB）将 SNN 的稀疏脉冲表示通过时空注意力转换为 ANN 可处理的密集特征，在 Gen1/Gen4 数据集上以仅 6.6M 参数大幅超越 SNN 方法并接近 ANN/RNN 方法的精度，同时 SNN 部分可部署在 Intel Loihi 2 神经形态芯片上实现低功耗推理。

## 研究背景与动机

**领域现状**：事件相机以 ~10μs 的时间分辨率异步捕获像素级亮度变化，具有低延迟、高动态范围（140 dB）和无运动模糊的优势，适合快速运动和低光场景。目标检测是事件视觉的重要应用场景。当前主要有三类处理方案：(1) ANN 方法直接复用传统检测架构但模型大、不利于边端部署；(2) SNN 方法天然匹配事件数据的稀疏性，可在神经形态硬件上低功耗运行，但精度显著落后 ANN；(3) RNN 方法擅长时序建模但计算量大。

**现有痛点**：ANN 方法（>60M 参数）过于庞大，难以部署到边端或神经形态设备；SNN 方法将 ANN 骨干（VGG、ResNet 等）直接转为 SNN 后精度大幅下降（mAP 仅 0.15-0.31）；已有的混合 SNN-ANN 方法仅在简单任务（分类、跟踪）上验证，未在大规模检测基准上测试，且 SNN→ANN 的桥接设计过于简单（如直接累加），丢失了时空信息。

**核心矛盾**：SNN 的稀疏脉冲表示与 ANN 需要的密集特征图之间存在表示鸿沟——简单的累加或均值操作会严重丢失事件数据中的时空关系。

**本文目标** 如何设计一个高效的 SNN→ANN 桥接机制，在保留事件的时空信息的同时将稀疏脉冲转为密集表示，从而让小型混合网络在大规模检测基准上达到接近 ANN 的精度？

**切入角度**：作者观察到事件数据的稀疏空间分布具有不规则性（适合可变形卷积），且不同时间步的脉冲之间存在需要显式建模的时序关系（适合注意力机制）。此外，事件发放率本身可作为空间注意力的指导信号——高事件率区域通常对应运动物体，应被重点关注。

**核心 idea**：设计包含时空感知时序注意力（SAT）和事件率空间注意力（ERS）的桥接模块，将 SNN 脉冲有效转为 ANN 密集特征，实现小模型大精度。

## 方法详解

### 整体框架

混合骨干网络由三部分组成：(1) SNN 模块 $f_{snn}$：由多个 Conv-BN-PLIF 块组成，以高时间分辨率（5ms bins）处理事件张量，输出稀疏脉冲特征 $\mathbf{E}_{spike} \in \mathbb{R}^{T \times C \times H' \times W'}$；(2) ASAB 桥接模块 $\beta_{asab}$：通过 SAT 和 ERS 注意力将脉冲转为密集特征图 $\mathbf{F}_{out} \in \mathbb{R}^{C \times H' \times W'}$；(3) ANN 模块 $f_{ann}$：标准卷积块提取高层空间特征，输出送入 YOLOX 检测头。可选地加入 DWConvLSTM 形成多时间尺度 RNN 变体。

### 关键设计

1. **时空感知时序注意力（SAT）**:

    - 功能：从稀疏脉冲中捕获时序关系并转化为空间特征
    - 核心思路：首先做通道-时间分组（Channel-wise Temporal Grouping），将 $T \times C$ 维度重排为 $C \times T$ 以逐通道处理。然后用时间分离的可变形卷积（TSDC）独立提取每个时间步的局部空间上下文——由于脉冲具有不规则的空间分布，固定网格卷积核难以捕获其形状，可变形卷积通过学习偏移量自适应调整采样位置。最后通过 softmax 时序注意力对不同时间步的特征计算关联权重：$\mathbf{A}_{score} = \text{softmax}(\mathbf{A}_q \mathbf{A}_k)$，加权求和后沿时间维度用 1×1 卷积聚合为单帧输出。
    - 设计动机：可变形卷积匹配脉冲的不规则空间结构；时间分离处理保持各时间步空间信息的独立性；时序注意力显式建模时间步之间的关联强度。

2. **事件率空间注意力（ERS）**:

    - 功能：利用事件发放率作为空间注意力权重，突出运动活跃区域
    - 核心思路：将脉冲张量 $\mathbf{E}_{spike}$ 沿时间维度求和得到事件率图 $\mathbf{S}_{rate} \in \mathbb{R}^{C \times H' \times W'}$，经 sigmoid 归一化后与 SAT 输出做 Hadamard 乘积：$\mathbf{E}_{feature} = \text{sigmoid}(\mathbf{S}_{rate}) \odot \mathbf{A}_{out}$。事件率高的区域（通常对应运动物体边缘）获得更大权重。
    - 设计动机：事件相机的核心特性是"运动触发"，事件率直接反映了场景中的运动信息，是天然的空间显著性信号。用它做注意力权重符合事件数据的物理特性。

3. **多时间尺度 RNN 变体**:

    - 功能：在 ANN 部分加入 DWConvLSTM 捕获慢动态
    - 核心思路：SNN 处理 5ms 短时步的快速动态，DWConvLSTM 在 ASAB 输出的更大时步上（50ms）处理长时序依赖。深度可分离卷积 LSTM 降低了参数量。
    - 设计动机：自动驾驶场景中既有快速运动（突然出现的行人）也有缓慢变化（远处缓行车辆），多时间尺度建模能同时覆盖两种动态。

### 损失函数 / 训练策略

检测头使用 YOLOX 框架，损失包含 IoU 损失、类别损失和回归损失。Gen1 数据集上 batch size=24, lr=2e-4，4 张 3090 训练约 8 小时（50 epoch）。RNN 变体序列长度 21，训练 40 万步约 6 天。SNN 部分使用替代梯度（surrogate gradient）方法联合端到端训练。

## 实验关键数据

### 主实验

Gen1 和 Gen4 自动驾驶检测数据集 mAP 对比：

| 方法 | 类型 | 参数量 | Gen1 mAP | Gen4 mAP |
|------|------|--------|----------|----------|
| EMS-RES34 (最佳SNN) | SNN | 14.4M | 0.31 | - |
| Events-RetinaNet | ANN | 33M | 0.34 | 0.18 |
| RVT-B (w/o LSTM) | Transformer | 16.2M | 0.32 | - |
| **Proposed (混合)** | **Hybrid** | **6.6M** | **0.35** | **0.27** |
| RVT-B | TF+RNN | 19M | 0.47 | - |
| **Proposed+RNN** | **Hybrid+RNN** | **7.7M** | **0.43** | - |

混合模型以 6.6M 参数在无 RNN 方法中达最佳精度；加入 RNN 后以 7.7M 参数接近 19M 的 RVT-B。

### 消融实验

| 配置 | mAP(.5) | mAP | 说明 |
|------|---------|-----|------|
| w/o ASAB (简单累加) | 0.53 | 0.30 | 简单桥接严重损失信息 |
| w/o 时序注意力 Φ_ta | 0.57 | 0.33 | 时序建模缺失 |
| w/o 可变形卷积 | 0.59 | 0.34 | 用标准卷积替代 |
| w/o ERS 空间注意力 | 0.59 | 0.34 | 失去空间显著性信号 |
| **Full model** | **0.61** | **0.35** | 所有组件完整 |

### 关键发现

- ASAB 模块的重要性：去掉 ASAB 改用简单累加，mAP 从 0.35 掉到 0.30（-14%），证明精心设计的桥接对混合网络至关重要。
- 可变形卷积对稀疏脉冲的空间建模有效——标准卷积在固定网格上采样不匹配脉冲的不规则分布。
- SNN 部分部署到 Intel Loihi 2 上仅需 1.73W 功耗，int8 量化后精度几乎无损（0.348→0.343），每步执行时间 2.06ms。
- 混合模型的 MAC 中仅 <5% 来自注意力和 MLP（vs RVT 的 67%），更适合边端/神经形态部署。

## 亮点与洞察

- **ASAB 桥接模块的设计哲学**：不是简单地将 SNN 脉冲累加为帧，而是通过注意力机制让网络学会"哪些时间步和空间位置的脉冲更重要"，实现了信息保留与表示转换的平衡。这个桥接思路可推广到任何 SNN-ANN 混合架构。
- **事件率作为注意力信号**：用事件相机的本征物理量（事件率 = 运动强度）直接做空间注意力，零参数开销，这种利用传感器特性做注意力的思路非常巧妙。
- 实际部署验证（Loihi 2 硬件实验）为"混合网络可以实际落地"提供了可信证据，这在学术论文中罕见但价值极高。

## 局限与展望

- RNN 变体与纯 RNN 方法（如 RVT-B 0.47）仍有 4 个点的精度差距，桥接后的信息损失可能是瓶颈。
- 目前仅在汽车检测数据集上验证，对更多类别的泛化性待探索。
- SNN 部分仅 2-3 层卷积较浅，更深的 SNN 是否能提取更好的低层特征？但这也会增加神经形态硬件的部署复杂度。
- ASAB 中的时序注意力复杂度为 $O(T^2)$，当事件时步增多时可能成为瓶颈。

## 相关工作与启发

- **vs EMS-YOLO (EMS-RES)**: EMS 将 ResNet 转为 SNN 做检测但精度有限（0.31），本文混合方案在更少参数下达到 0.35，说明 SNN 不需要独自完成所有工作。
- **vs RVT**: RVT 用 Transformer+RNN 达到高精度但参数量 19M 且 67% MAC 来自注意力/MLP，本文混合方案以 6.6M 参数和 <5% 注意力 MAC 更适合硬件部署。
- **vs DashNet**: DashNet 也做 SNN-ANN 混合但用于跟踪等简单任务，本文首次在大规模检测基准上验证混合方案的可行性。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个面向基准的混合检测方案 + ASAB 设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 三类方法全面对比 + 硬件部署验证 + 充分消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融分析有说服力
- 价值: ⭐⭐⭐⭐ 为事件相机检测的高效部署提供了可行路径

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Efficient Test-Time Adaptive Object Detection via Sensitivity-Guided Pruning](efficient_test-time_adaptive_object_detection_via_sensitivity-guided_pruning.md)
- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[CVPR 2025\] Object Detection using Event Camera: A MoE Heat Conduction based Detector and A New Benchmark Dataset](object_detection_using_event_camera_a_moe_heat_conduction_based_detector_and_a_n.md)
- [\[ICML 2025\] When Every Millisecond Counts: Real-Time Anomaly Detection via the Multimodal Asynchronous Hybrid Network](../../ICML2025/object_detection/when_every_millisecond_counts_real-time_anomaly_detection_via_the_multimodal_asy.md)
- [\[ICCV 2025\] Intervening in Black Box: Concept Bottleneck Model for Enhancing Human-Neural Network Mutual Understanding](../../ICCV2025/object_detection/intervening_in_black_box_concept_bottleneck_model_for_enhancing_human_neural_net.md)

</div>

<!-- RELATED:END -->
