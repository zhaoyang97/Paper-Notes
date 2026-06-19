---
title: >-
  [论文解读] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling
description: >-
  [AAAI 2026 Oral][脉冲神经网络] 提出拓扑深度脉冲神经网络（TDSNNs），通过设计时空约束（STC）损失函数，在深层SNN中成功复现灵长类视觉皮层从V1到IT的层级拓扑组织，既保持了高任务性能（ImageNet top-1无精度下降），又显著超越现有拓扑ANN的脑相似性。 领域现状：灵长类视觉皮层呈现拓扑组…
tags:
  - "AAAI 2026 Oral"
  - "脉冲神经网络"
  - "拓扑组织"
  - "视觉皮层建模"
  - "时空约束"
  - "生物合理性"
---

# TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling

**会议**: AAAI 2026 Oral  
**arXiv**: [2508.04270](https://arxiv.org/abs/2508.04270)  
**代码**: 无  
**领域**: 脉冲神经网络 / 计算神经科学  
**关键词**: 脉冲神经网络, 拓扑组织, 视觉皮层建模, 时空约束, 生物合理性

## 一句话总结

提出拓扑深度脉冲神经网络（TDSNNs），通过设计时空约束（STC）损失函数，在深层SNN中成功复现灵长类视觉皮层从V1到IT的层级拓扑组织，既保持了高任务性能（ImageNet top-1无精度下降），又显著超越现有拓扑ANN的脑相似性。

## 研究背景与动机

**领域现状**：灵长类视觉皮层呈现拓扑组织结构——功能相似的神经元在空间上聚类，从V1的方向/空间频率/颜色选择性到IT的类别选择性（如面孔、身体区域的空间聚类）。深度学习模型已被用于建模这种拓扑结构，如TopoNet、TDANN等通过引入布线代价辅助损失来诱导拓扑特征。

**现有痛点**：现有拓扑ANN模型存在两个关键问题：（1）在分类任务上存在显著的性能退化（TopoNet在ImageNet上掉3%，最差的LLCNN-G掉16.57%）；（2）完全忽略了时间维度——生物视觉系统的时间动态处理是根本特性，但ANN无法内在捕获时序信息。

**核心矛盾**：时间动态是生物神经系统的核心特征，但现有拓扑模型要么是忽略时间的ANN，要么是仅限于浅层（如两层网络SESNN）的SNN，无法在深层网络中同时实现层级拓扑组织和高任务性能。

**本文目标** （1）如何在深层SNN中诱导从V1到IT的层级拓扑组织？（2）如何利用SNN的时间动态来缓解拓扑约束带来的性能退化？（3）拓扑结构如何改变SNN的时序信息处理机制？

**切入角度**：SNN天然具备基于脉冲的时间动态处理能力，作者设计了既约束长时间尺度（发放率）又约束短时间尺度（脉冲同步性）的损失函数STC，利用SNN的时间编码优势来弥补拓扑约束的性能代价。

**核心 idea**：用时空约束（STC）损失在深层SNN中同时优化空间拓扑和时序同步，实现零性能退化的视觉皮层拓扑建模。

## 方法详解

### 整体框架

TDSNNs的构建流程分为三个阶段：（1）将SNN每一层的神经元映射到虚拟二维皮层面（cortical sheet）上，赋予每个LIF神经元一个物理坐标；（2）通过预优化步骤调整神经元位置，使响应相似的神经元在空间上靠近；（3）使用包含任务损失和STC损失的联合目标函数从头训练SNN。输入为图像，输出为分类结果，同时各层自发形成类似V1（方向、空间频率、颜色选择性）和IT（类别选择性）的拓扑特征。

### 关键设计

1. **虚拟皮层面映射（Cortical Sheet Mapping）**:

    - 功能：将SNN每层的神经元非均匀嵌入到二维物理空间中，模拟生物视觉皮层的空间布局
    - 核心思路：对维度为 $(C, H, W)$ 的SNN层，将每个单元 $u_{c,h',w'}$ 通过注入映射 $\mathcal{M}$ 分配到大小为 $h \times w$（单位mm）的皮层面上的唯一坐标 $(x, y)$。不同层的皮层面尺寸根据对应的视觉区域设定（如V1层36.75mm，IT层70.0mm），邻域宽度也相应调整
    - 设计动机：为后续的STC损失提供空间距离基础，同时保持与生物视觉皮层的尺度对应关系

2. **时空约束（STC）损失函数**:

    - 功能：促进空间邻近的LIF神经元在长时间尺度（发放率）和短时间尺度（脉冲同步性）上具有相似的响应模式
    - 核心思路：STC由两部分组成。长时间尺度损失 $\mathcal{L}_L = \frac{1}{2}(1 - P(\mathbf{r}, \mathbf{d}))$，其中 $\mathbf{r}$ 是神经元对之间发放率的Pearson相关系数向量，$\mathbf{d}$ 是空间距离的倒数向量。短时间尺度损失 $\mathcal{L}_S = \frac{1}{2}(1 - P(\mathbf{r_{CCG}}, \mathbf{d}))$，利用交叉相关图（CCG）在时间窗口 $[-W, W]$ 内计算脉冲时序同步性 $r_{CCG}(i,j)$，并做自相关归一化。最终损失为 $\mathcal{L} = \mathcal{L}_{task} + \frac{1}{M}\sum_{k,m}[\alpha \mathcal{L}_L + \beta \mathcal{L}_S]$
    - 设计动机：生物视觉皮层同时存在基于发放率的长时间尺度表征和基于脉冲同步的短时间尺度表征。仅用长时间尺度约束（$\beta=0$）时，SNN本质上退化为类ANN的率编码模式；短时间尺度损失 $\mathcal{L}_S$ 作为脉冲时序调节器，显著增强时间编码能力

3. **神经元位置预优化（Position Pre-optimization）**:

    - 功能：在正式训练前为皮层面上的神经元建立初步的空间组织结构
    - 核心思路：先用BPTT预训练一个辅助SNN，然后用正弦光栅刺激生成响应，通过随机交换神经元位置（每个邻域500次交换尝试×20000个独立采样）来使响应相似的神经元靠近。预训练仅用于生成最终位置，权重被丢弃
    - 设计动机：由于卷积参数共享机制，大规模SNN层中的神经元无法自发形成拓扑结构——多个单元共享同一滤波器参数导致模型更新同时影响多个神经元的响应

### 损失函数 / 训练策略

总损失为任务损失（交叉熵）加STC损失的加权和，权重因子 $\alpha$ 和 $\beta$ 控制拓扑约束强度。STC损失通过在每层随机采样 $M=10$ 个固定大小的神经元簇来近似计算，降低计算开销。CCG中的时间常数 $\tau$ 与总时间步数 $T$ 成比例（$\tau = T/2 - 1$ 或 $T/2$）。使用BPTT配合代理梯度进行端到端训练，TSResnet18在ImageNet上训练300 epochs，AdamW优化器（基础学习率5e-4，余弦衰减）。

## 实验关键数据

### 主实验

| 指标 | TSResnet18 | SResnet18(非拓扑) | TopoNet(ANN) | TDANN(ANN) |
|------|-----------|-----------------|-------------|-----------|
| ImageNet Top-1 (%) | 58.34 (α50β50) / 58.72 (α10β90) | 58.49 | ~55.5 (掉3%) | - |
| V1 BrainScore | 0.6845 | 0.6823 | 0.7116 | 0.6932 |
| IT BrainScore | **0.7127** | 0.7102 | 0.5723 | 0.4259 |
| V4 BrainScore | 0.3886 | **0.3970** | 0.2923 | 0.2792 |

### 消融实验

| 配置 | ImageNet Acc (%) | V1平滑性 | 说明 |
|------|-----------------|---------|------|
| SResnet18 (非拓扑) | 58.49 | 0.5555 | 基线 |
| TSResnet18 α10-β10 | 58.53 | 0.6839 | 拓扑+轻微约束 |
| TSResnet18 α10-β90 | **58.72** | 0.6991 | 最高精度 |
| TSResnet18 α50-β0 | 58.21 | 0.7550 | 无短时间尺度约束 |
| TSResnet18 α50-β50 | 58.34 | **0.7674** | 最高平滑性 |

### 关键发现

- 短时间尺度损失 $\mathcal{L}_S$ 是关键：加入后平滑性从0.755提升到0.7674，精度从58.21%提升到58.34%——它不仅增强拓扑质量，还改善分类性能
- TDSNNs在CIFAR-100上同样表现出拓扑增益（非拓扑73.01% → 拓扑73.97%），在Spikformer上也无精度下降
- IT层的面孔-身体选择性空间重叠度：TSResnet18为0.63 vs SResnet18仅0.15，与灵长类视觉皮层的面孔-身体共定位现象一致
- 鲁棒性实验：TSResnet18在四种攻击下均优于非拓扑SResnet18（如PGD: 10.7% vs 9.97%），拓扑组织增强了决策鲁棒性
- Fisher信息分析揭示了拓扑驱动的信息层级：早期层（V1/V2）FI稳定保持信号保真度，V4显著放大判别特征，IT层FI降低实现稳定编码

## 亮点与洞察

- **SNN的时间编码优势弥补拓扑代价**：这是本文最核心的洞察。ANN引入拓扑约束必然损失性能（因为约束了空间自由度），但SNN通过时间维度的额外自由度补偿了空间约束的信息损失，实现零精度掉点
- **STC损失的双时间尺度设计很巧妙**：将率编码和时序编码统一在一个损失函数中，既有长时间尺度的功能聚类，又有短时间尺度的同步性约束，直接对应生物神经系统的两种信息编码模式
- **拓扑引发的信息层级重塑**：发现拓扑组织主要重塑深层连接（V4和IT），而非浅层——这与生物视觉系统中拓扑组织从低级到高级逐渐复杂化的特性吻合

## 局限与展望

- 仅在ResNet18/CORnet/Spikformer上验证，未扩展到更大架构（如ResNet50、ViT-Large），计算复杂度是主要瓶颈（BPTT训练深层SNN非常耗资源）
- LIF模型虽然高效，但更复杂的HH或FIF模型可能提供更好的生物真实性
- 仅考虑前馈和局部侧向连接，缺乏长距离连接和兴奋-抑制神经元群体的多样性
- 时间步数受限（前馈SNN仅4步，循环SNN仅10步），无法探索更长时间窗口的时序动态

## 相关工作与启发

- **vs TopoNet (deb2025toponets)**：TopoNet通过神经修剪平衡拓扑和性能，但在ImageNet上仍掉3%；TDSNNs利用时间编码完全消除性能退化，且IT区BrainScore远超（0.71 vs 0.57）
- **vs SESNN (zhong2024emergence)**：SESNN仅在两层网络中复现V1方向偏好图，深层拓扑会消失；TDSNNs首次在所有视觉层级（V1到IT）保持拓扑组织
- **vs TDANN (margalit2024unifying)**：TDANN作为拓扑ANN的代表，在V2/V4/IT的BrainScore均远低于TDSNNs，说明时间动态对脑相似性的重要性

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在深层SNN中实现完整视觉皮层层级拓扑建模，STC双时间尺度设计有理论深度
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖V1/IT拓扑分析、BrainScore、性能-拓扑权衡、鲁棒性、Fisher信息等多维评估
- 写作质量: ⭐⭐⭐⭐ 结构清晰，生物学背景介绍充分，但部分符号可以更简洁
- 价值: ⭐⭐⭐⭐ 为计算神经科学和深度学习的交叉领域提供了新视角，但工程应用价值有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](../../ICML2026/others/bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)
- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)

</div>

<!-- RELATED:END -->
