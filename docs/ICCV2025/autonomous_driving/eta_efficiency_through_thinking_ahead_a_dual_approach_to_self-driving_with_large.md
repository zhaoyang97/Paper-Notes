---
title: >-
  [论文解读] ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models
description: >-
  [ICCV 2025][自动驾驶][端到端自动驾驶] 提出ETA双系统框架，通过将大模型的当前帧计算转移到前序时间步并进行批量推理，使大模型特征在每帧都可用，在Bench2Drive上以50ms延迟达到69.53驾驶分数，提升SOTA 8%。 大型视觉模型（如CLIP-ViT-L、VLM）凭借强大的感知和推理能力…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "端到端自动驾驶"
  - "双系统架构"
  - "异步推理"
  - "大模型实时化"
  - "特征预测"
---

# ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models

**会议**: ICCV 2025  
**arXiv**: [2506.07725](https://arxiv.org/abs/2506.07725)  
**代码**: [https://github.com/OpenDriveLab/ETA](https://github.com/OpenDriveLab/ETA)  
**领域**: 自动驾驶  
**关键词**: 端到端自动驾驶, 双系统架构, 异步推理, 大模型实时化, 特征预测

## 一句话总结

提出ETA双系统框架，通过将大模型的当前帧计算转移到前序时间步并进行批量推理，使大模型特征在每帧都可用，在Bench2Drive上以50ms延迟达到69.53驾驶分数，提升SOTA 8%。

## 研究背景与动机

大型视觉模型（如CLIP-ViT-L、VLM）凭借强大的感知和推理能力，在自动驾驶中展现出优异性能。然而，自动驾驶系统对实时性要求严格——控制频率通常需要达到20Hz（50ms/帧），而当前最佳模型（如DriveTransformer、DriveAdapter）的延迟动辄数百毫秒，远不满足实时需求。

**双系统方法**（受认知科学System-1/System-2启发）是一种自然的解决方案：用小模型做快速反应决策（System-1），用大模型做深思熟虑分析（System-2）。然而现有双系统设计存在根本性问题：

- **DriveVLM-Dual**：大VLM异步运行但耗时300ms，大模型的推理结果只在少数帧可用。
- **AD-H**：LLM输出高级规划命令给小LLM执行，但大模型同样无法每帧响应。
- **LeapAD**：用记忆库存储大模型推理结果并在需要时检索，但动态环境中记忆库管理困难且难以泛化。

**核心矛盾**：现有双系统方法中大模型的推理结果只在某些帧可用，大部分时间系统仅依赖小模型。这意味着在大模型"缺席"的帧里，系统等于放弃了大模型带来的性能提升。

**ETA的关键洞察**：将大模型在当前帧的密集计算**转移到前序时间步**，并通过**批量推理**同时处理多帧——这样每一帧都能获得大模型特征的推理结果。核心是用空间复杂度换取时间复杂度。

## 方法详解

### 整体框架

ETA是一个异步双系统框架，包含四个核心组件：(1) 大模型处理前序帧 $\mathbf{I}_{t-\Delta}$，提取高质量特征 $\mathbf{f}_{t-\Delta}^l$；(2) 预测模块将前序帧特征预测到当前帧 $\hat{\mathbf{f}}_t^l$；(3) 小模型处理当前帧 $\mathbf{I}_t$，捕捉难以预测的实时变化 $\mathbf{f}_t^s$；(4) 动作模型融合双路特征并预测驾驶动作。

### 关键设计

#### 1. 异步批量推理——时间转移

- **功能**：将大模型在当前帧的计算推迟到前序时间步执行，并通过批量处理多帧使大模型特征在每帧都可用。
- **核心思路**：传统双系统中大模型低频运行、小模型高频运行，导致大多数帧只有小模型输出。ETA让大模型在时间步 $t-\Delta$ 处理前序帧 $\mathbf{I}_{t-\Delta}$：
  $\mathbf{f}_{t-\Delta}^l = f_{\text{large}}(\mathbf{I}_{t-\Delta})$
  
  然后将多帧的大模型推理并行化处理（同时推理 $\mathbf{I}_{t-\Delta}$ 对应的未来多帧特征），实现每帧都有相关的大模型特征可用。
- **设计动机**：自动驾驶场景的时间连续性意味着相邻帧的视觉特征高度相关。将前序帧特征预测到当前帧，虽然有信息损失，但远好于完全没有大模型特征。批量推理进一步摊薄了大模型的每帧成本。

#### 2. 特征预测模块（Forecasting）

- **功能**：基于前序帧的大模型特征，预测当前帧的大模型特征。
- **核心思路**：用预测模型 $f_{\text{forecast}}$ 以前序帧特征 $\mathbf{f}_{t-\Delta}^l$、预测的动作 $\hat{\mathbf{a}}_{t-\Delta}$ 和条件输入 $\mathbf{c}_{t-\Delta}$（速度+目标航点）为输入：
  $\hat{\mathbf{f}}_t^l = f_{\text{forecast}}(\mathbf{f}_{t-\Delta}^l, \hat{\mathbf{a}}_{t-\Delta}, \mathbf{c}_{t-\Delta})$
  
  预测的特征通过预测损失 $\mathcal{L}_{\text{forecast}} = |\mathbf{f}_t^l - \hat{\mathbf{f}}_t^l|$ 来监督（训练时使用真实大模型特征 $\mathbf{f}_t^l$，推理时不需要）。
- **设计动机**：类似世界模型的思路——根据之前的状态和动作预测未来状态。通过条件化动作和速度信息，模型能学会"如果前一帧以某速度向左转，那么当前帧特征应该怎样变化"。监督信号仅在训练时需要，不影响推理速度。

#### 3. 小模型——实时补充

- **功能**：用轻量模型处理当前帧，捕捉预测模型无法预见的突变信息。
- **核心思路**：$\mathbf{f}_t^s = f_{\text{small}}(\mathbf{I}_t)$。使用CLIP-ViT-L的前8层作为小模型（复用大模型的浅层权重），足以提取基础视觉特征且速度很快（仅占全模型推理时间的一小部分）。
- **设计动机**：某些关键变化从前序帧无法预测——如交通信号灯变化、行人突然出现等。小模型直接处理当前帧来捕捉这些不可预测的实时变化，与预测的大模型特征互补。

#### 4. 动作掩码（Action Mask）

- **功能**：引导模型关注与驾驶动作相关的图像区域。
- **核心思路**：计算patch特征与动作编码查询之间的注意力，生成掩码 $\hat{\mathbf{m}}_t$，标记自车即将驶向的图像区域。地面真值掩码通过将专家动作投影到图像上获取——包含路径点或航点的patch标记为1，其余为0。使用BCE损失进行监督。
- **设计动机**：观察到模型难以聚焦在动作相关的图像区域。掩码机制显式引导模型将注意力集中在路径和航点所在的图像patch上，促进观测特征和预测动作在特征空间中的对齐。

### 损失函数 / 训练策略

- **Base模型损失**：$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{action}} + \frac{1}{16}\mathcal{L}_{\text{mask}}$
- **Async模型损失**：$\mathcal{L}_{\text{async}} = \mathcal{L}_{\text{action}} + \frac{1}{16}\mathcal{L}_{\text{mask}} + 0.5 \mathcal{L}_{\text{forecast}}$
- 动作损失使用L1损失，预测残差而非绝对位置（稳定训练）。
- 预测损失对ground truth特征做梯度截断（stop gradient），防止坍缩。
- 大模型用CLIP-ViT-L-336px，小模型用其前8层，动作解码器12层Transformer。
- 8×A100训练40 epochs，$\Delta=0.5s$。

## 实验关键数据

### 主实验——Bench2Drive排行榜

| 方法 | 驾驶分数(DS)↑ | 成功率(SR)↑ | 效率↑ | 延迟(ms)↓ |
|------|-------------|------------|-------|----------|
| AD-MLP (sanity check) | 18.05 | 0.00% | 48.45 | **3** |
| UniAD-Base | 45.81 | 16.36% | 129.21 | 663 |
| VAD | 42.35 | 15.00% | 157.94 | 278 |
| DriveTransformer-L | 63.46 | 35.01% | 100.64 | 212 |
| DriveAdapter | 64.22 | 33.08% | 70.22 | 931 |
| **ETA Base** | **74.33** | **48.33%** | **186.04** | 102 |
| **ETA Async** | **69.53** | **38.64%** | **184.51** | **50** |

Base模型以102ms延迟取得全面最优（DS 74.33），但不满足实时要求。Async模型将延迟降到50ms（20Hz），DS仍高达69.53，超越之前最佳DriveTransformer的63.46（+8%）。

### 消融实验

| 配置 | DS↑ | SR↑ | 延迟↓ | 说明 |
|------|-----|------|-------|------|
| Async完整模型 | 69.53 | 38.64% | 50ms | 全部组件 |
| 去掉预测模块 | 54.92 | 30.45% | 50ms | DS降低21%，预测模块是核心 |
| 去掉小模型 | 42.49 | 17.27% | 31ms | DS暴跌39%，实时信息不可或缺 |
| 去掉动作掩码 | 42.67 | 17.27% | 50ms | DS暴跌39%，掩码对注意力对齐至关重要 |
| 仅小模型 | 61.30 | 35.00% | 50ms | 无大模型特征，DS低于Async |
| GT特征预测(训练+测试) | 74.12 | 48.18% | 124ms | 近似Base上界 |

分能力消融显示：小模型去掉后紧急刹车(66.67→16.36)和交通标志(59.43→30.56)下降最剧烈——这些正是不可预测、需要实时感知的场景。

### 关键发现

- **预测+小模型的协同是关键**：单独使用任一都不够。预测模块提供全局理解（如场景布局、车辆位置），小模型补充实时细节（如信号灯状态、突现行人）。
- **GT特征预测的实验(F)**验证了——推理时给GT特征但训练时未用GT来监督预测，反而性能下降到60.72。说明训练时的预测损失对学习有效的特征传播至关重要。
- **延迟-性能Pareto最优**：ETA Async在DS-延迟平面上处于Pareto前沿，没有其他方法能在50ms延迟下达到69.53的DS。

## 亮点与洞察

- **"提前思考"的范式创新**：将大模型计算从当前帧转移到前序帧，是一种优雅的时间换空间策略。这不同于传统的模型压缩或蒸馏——保留了完整的大模型，只是改变了使用方式。
- **实验设计严谨**：在挑战性极高的Bench2Drive闭环评估中进行实验，所有模型报告三次运行均值+标准差，消融实验全面覆盖各组件和各能力维度。
- **动作掩码**是一个简单但高效的设计——通过将专家动作投影为图像空间的二值掩码，为端到端模型提供了显式的空间注意力引导。

## 局限与展望

- 预测模块假设 $\Delta=0.5s$ 的时间间隔，更长的间隔可能导致预测精度大幅下降。
- 当场景发生剧烈变化（如车辆突然切入）时，预测模块可能给出严重错误的预计特征。
- 小模型使用大模型的前8层，存在参数冗余且无法独立优化。
- 仅在CARLA仿真环境评估，缺少真实世界数据上的验证。
- 当前框架仅使用单相机输入，多视角场景下的扩展未被探讨。

## 相关工作与启发

- 与DriveVLM-Dual的思路一脉相承，但ETA通过特征预测实现了大模型在每帧都可用，而非稀疏采样大模型输出。
- 与世界模型（如GAIA-1、DriveDreamer）有概念联系——预测模块本质上学习了一个轻量的隐空间世界模型。
- 启发：该框架可推广到任何需要大模型实时化的embodied AI系统（如机器人操作、无人机导航），只要场景具有时间连续性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Distilling Multi-modal Large Language Models for Autonomous Driving](../../CVPR2025/autonomous_driving/distilling_multi-modal_large_language_models_for_autonomous_driving.md)
- [\[NeurIPS 2025\] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](../../NeurIPS2025/autonomous_driving/futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)
- [\[ACL 2025\] Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [\[CVPR 2025\] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](../../CVPR2025/autonomous_driving/vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)

</div>

<!-- RELATED:END -->
