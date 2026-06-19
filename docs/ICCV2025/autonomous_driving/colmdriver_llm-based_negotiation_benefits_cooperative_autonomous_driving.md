---
title: >-
  [论文解读] CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving
description: >-
  [ICCV 2025][自动驾驶][cooperative driving] 首个全流程 LLM 驱动的协作驾驶系统，通过 Actor-Critic 范式的语言协商模块和意图引导的轨迹生成器，在多种 V2V 交互场景中实现比现有方法高 11% 的成功率。 核心问题：车辆间 (V2V) 协作自动驾驶能通过共享感知信息来弥补单车…
tags:
  - "ICCV 2025"
  - "自动驾驶"
  - "cooperative driving"
  - "V2V"
  - "LLM negotiation"
  - "actor-critic"
  - "waypoint planning"
  - "CARLA"
---

# CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving

**会议**: ICCV 2025  
**arXiv**: [2503.08683](https://arxiv.org/abs/2503.08683)  
**代码**: [cxliu0314/CoLMDriver](https://github.com/cxliu0314/CoLMDriver)  
**领域**: 自动驾驶  
**关键词**: cooperative driving, V2V, LLM negotiation, actor-critic, waypoint planning, CARLA

## 一句话总结

首个全流程 LLM 驱动的协作驾驶系统，通过 Actor-Critic 范式的语言协商模块和意图引导的轨迹生成器，在多种 V2V 交互场景中实现比现有方法高 11% 的成功率。

## 研究背景与动机

**核心问题**：车辆间 (V2V) 协作自动驾驶能通过共享感知信息来弥补单车系统的感知和预测不确定性，但面临两个关键挑战：

**传统协作方法的局限**：现有 V2V 协作方法（如 CoDriving）依赖刚性的协作协议，预定义的交互规则难以泛化到未见过的复杂交互场景。例如在多车交汇的十字路口、紧急避让等非标准场景中，固定规则无法灵活应对。

**LLM 直接用于驾驶的困难**：虽然 LLM 具备泛化推理能力，但存在两个障碍：(a) 空间规划能力不足——LLM 难以直接输出精确的轨迹坐标；(b) 推理延迟不稳定——实时驾驶对延迟要求极高，LLM 的推理时间波动会导致控制不稳定。

**本文思路**：不让 LLM 直接做空间规划，而是让 LLM 负责它擅长的事——语言协商。将驾驶系统分为"协商"和"执行"两个并行流水线：LLM 通过自然语言协商达成合作意图（如"我先通过，你减速让行"），再由专门的轨迹生成器将协商结果转化为可执行轨迹。

## 方法详解

### 整体框架

CoLMDriver 采用**并行驾驶流水线**架构，包含三大模块：

1. **协作感知模块 (Cooperative Perception)**：基于 V2Xverse/CoDriving 的 BEV 特征融合，多车共享 LiDAR 点云并通过 SpConv 提取体素特征，生成统一的鸟瞰图 (BEV) 感知结果。
2. **LLM 协商模块 (LLM-based Negotiation)**：采用 Actor-Critic 范式，车辆间通过自然语言进行多轮协商。
3. **意图引导的轨迹生成器 (Intention-Guided Waypoint Generator)**：将协商得到的驾驶意图转化为 20 个未来轨迹点。

三个模块的协作流程：感知模块提供环境理解 → VLM 分析场景生成驾驶意图 → LLM 协商模块在车辆间协调意图 → 轨迹生成器输出可执行轨迹 → PID 控制器执行。

### 关键设计 1：LLM 协商模块 (Actor-Critic 范式)

这是本文最核心的创新，将多车协商建模为 Actor-Critic 迭代过程：

**Actor（协商生成）**：每辆车拥有一个 LLM Agent (Comm_Client)，输入包括：
- 自车信息：ID、意图（如 "turn left"）、速度
- 周围车辆信息：方向、航向、距离、速度、意图
- 交通规则（右侧优先、让行紧急车辆等）
- 历史协商记录和 Critic 反馈

LLM 根据这些信息生成自然语言形式的协商提案，例如："I will slow down to let Vehicle 2 pass first, then proceed with my left turn."

**Critic（协商评估）**：由 Nego_Client 实现，从三个维度评估协商质量：

| 评估维度 | 计算方式 | 权重 |
|---------|---------|------|
| 共识得分 (Consensus) | LLM 分析对话内容，输出 0-100 的共识分数 | 3 |
| 安全得分 (Safety) | 基于预测轨迹对之间的最小距离，使用 sigmoid 函数：$\frac{1}{1+e^{-2(d_{min}-3)}}$ | 5 |
| 效率得分 (Efficiency) | 基于轨迹长度评估各车是否高效行驶 | 2 |

总分计算：$S_{total} = 3 \cdot S_{cons} + 5 \cdot \min(S_{safety}) + 2 \cdot S_{eff}$

**迭代反馈**：当安全得分 < 0.7 时，生成安全建议（如"Vehicle X 和 Vehicle Y 可能冲突，建议一方让行"）；当效率得分 < 0.4 时，生成效率建议（如"部分车辆可以加速"）。这些建议作为 Critic Feedback 注入下一轮 Actor 的 prompt 中。最多迭代 3 轮 (`local_max_round=3`)，达成共识后提前终止。

### 关键设计 2：VLM 场景理解

使用视觉语言模型 (VLM) 处理前置摄像头图像，结合系统提示（包含车辆测量数据、通信信息、感知结果），输出结构化的驾驶命令：
- **方向指令** (6 类)：`turn left / turn right / straight / lane follow / lane change left / lane change right`
- **速度指令** (4 类)：`Stop / Slow down / Hold / Accelerate`

VLM 通过 LoRA 微调（使用 ms-swift 框架），数据来自 V2Xverse 仿真数据集。

### 关键设计 3：意图引导的轨迹生成器

核心模型为 `WaypointPlanner_e2e_cmd_attn_fix_20points`，架构设计：

**输入融合**：
- 占用栅格图 (Occupancy map)：6 通道，包含障碍物和道路信息，维度 `(B, T=5, C=6, H=192, W=96)`
- BEV 感知特征：来自协作感知模块的 128 维特征图
- 导航指令：方向嵌入 (Embedding, 6→256) + 速度嵌入 (Embedding, 4→256) + 目标点编码 (MLP, 2→256)

**编码器**：采用时空卷积 (STC) 块，交替使用 2D 卷积和 Conv3D 实现空间和时间建模：
- STC Block 1: Conv2D(32→64) → Conv3D(64→64) (时间融合 5→3 帧)
- STC Block 2: Conv2D(64→128) → Conv3D(128→128) (时间融合 3→1 帧)
- 空间进一步卷积到 256 维

**解码器**：使用交叉注意力机制，将导航指令特征与 BEV 空间特征交互：
- 构造查询嵌入 (query embedding)
- 交替进行指令-查询注意力和空间-查询注意力
- 最终通过回归分支输出 20 个轨迹点 `(B, 20, 2)`，使用累积求和确保轨迹连续性

### 损失函数

采用加权 L1 损失 (WaypointL1Loss20) 监督轨迹预测，对不同时间步使用递减权重（近处权重大、远处权重小），权重从 0.14 递减到 0.04。还有 ADE/FDE 作为辅助评估指标。

### InterDrive 基准

作者提出了 **InterDrive**，一个基于 CARLA 0.9.10.1 的 V2V 交互驾驶评测基准：
- **10 类挑战性交互场景**：涵盖十字路口交汇、并线冲突、紧急避让等
- **92 个测试用例**：46 个纯协作车辆场景 (no NPC) + 46 个含背景交通参与者场景 (with NPC)
- **评估指标**：成功率 (Success Rate)、碰撞率、超时率等
- 支持 ideal（不考虑推理延迟）和 realtime（考虑推理延迟）两种评测模式

## 实验关键数据

### 主实验结果

在 InterDrive 基准上与多个 baseline 对比：

| 方法 | 类型 | 成功率 (↑) |
|------|------|-----------|
| TCP | 单车端到端 | 基线 |
| CoDriving | V2V 协作（规则驱动） | 基线 |
| LMDrive | LLM 驾驶（单车） | 基线 |
| UniAD | 端到端规划 | 基线 |
| VAD | 端到端规划 | 基线 |
| **CoLMDriver** | **V2V+LLM 协商** | **+11% vs 最佳 baseline** |

CoLMDriver 在高交互性 V2V 场景中显著优于所有方法，证明了语言协商对协作驾驶的有效性。

### 消融实验

| 配置 | 变化 |
|------|------|
| 无协商 (w/o negotiation) | 性能显著下降，验证协商模块的必要性 |
| 无 Critic 反馈 | 仅单轮协商，性能下降，验证迭代反馈的价值 |
| 无 VLM 意图 | 去掉场景理解，直接用规则生成指令，性能下降 |
| 不同协商轮数 | 3 轮协商达到较好平衡 |

### 关键发现

1. **协商质量与驾驶安全正相关**：Critic 反馈机制将安全得分从单轮协商的不稳定水平提升到稳定高分
2. **NPC 场景更具挑战性**：含背景交通的场景成功率普遍低于纯协作场景
3. **推理延迟的影响**：realtime 模式下性能有所下降，但 CoLMDriver 的并行流水线设计有效缓解了这一问题——协商与轨迹规划异步进行

## 亮点与洞察

1. **"让 LLM 做它擅长的事"**：不强迫 LLM 输出精确坐标，而是让它做自然语言协商，再由专门模型转化为轨迹。这种分工设计很优雅。

2. **Actor-Critic 协商范式**：将强化学习中的 Actor-Critic 思想迁移到多车协商，Critic 从安全/效率/共识三维度评估，自动生成改进建议，实现了可解释的闭环优化。

3. **Critic 设计巧妙**：安全得分基于轨迹最小距离的 sigmoid 函数 $\frac{1}{1+e^{-2(d-3)}}$，阈值 3 米对应车辆安全间距，物理意义明确。共识得分则利用 LLM 自身的理解能力评估对话质量。

4. **全栈开源**：从感知到规划到协商的完整代码、模型权重、训练脚本、InterDrive 基准全部开源，对社区贡献大。

5. **并行流水线**：协商过程和感知/规划过程并行，有效缓解 LLM 推理延迟对实时性的影响。

## 局限性

1. **仿真到真实的差距**：实验完全在 CARLA 仿真器中进行，真实场景中 V2V 通信延迟、带宽限制、信号丢失等问题未被考虑。

2. **LLM 推理成本高**：每步协商需要多次 LLM 调用（每车每轮一次 + Critic 评估），3 轮协商涉及大量 API 调用，部署成本和延迟仍是挑战。

3. **协商假设较强**：假设所有协作车辆都诚实参与协商、准确报告意图。真实场景中存在恶意车辆、通信欺骗等对抗性问题。

4. **场景规模有限**：InterDrive 基准仅涉及 2-3 辆协作车辆的交互，更大规模车队的协商效率和可扩展性未验证。

5. **VLM 依赖单视角图像**：场景理解仅基于前置摄像头，侧方和后方的信息可能遗漏。

## 相关工作与启发

- **V2Xverse / CoDriving**：本文感知模块的基础，提供了 V2V 协作感知框架和仿真数据集
- **TCP (Trajectory-guided Control Prediction)**：作为单车端到端驾驶的 baseline
- **LMDrive**：将 LLM 用于单车驾驶的先驱工作，但缺乏 V2V 协作能力
- **Bench2Drive**：CARLA 驾驶评测基准，本文的评测框架部分参考

**启发**：
- 多智能体协作中的自然语言协商是一个有前景的方向，可以扩展到机器人编队、无人机群等领域
- Actor-Critic 协商范式可以推广到任何需要多方达成共识的场景（如多机器人任务分配）
- 将 LLM 的语言推理与领域专用模型的精确控制解耦，是一种通用的系统设计思路

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | 首个全流程 LLM 协作驾驶系统，Actor-Critic 协商范式新颖 |
| 技术深度 | ⭐⭐⭐⭐ | 感知-协商-规划全栈设计，模块间耦合设计合理 |
| 实验充分度 | ⭐⭐⭐⭐ | 提出新基准、多个 baseline 对比、消融实验完整 |
| 写作质量 | ⭐⭐⭐⭐ | 问题定义清晰，方法描述详细 |
| 实用价值 | ⭐⭐⭐ | 仿真验证有说服力，但部署到真实场景仍需大量工作 |
| **综合** | **⭐⭐⭐⭐** | **协作驾驶领域的重要工作，LLM+自动驾驶结合的有价值探索** |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception](cooptrack_exploring_end-to-end_learning_for_efficient_cooperative_sequential_per.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICML 2026\] Threshold-Based Exclusive Batching for LLM Inference](../../ICML2026/autonomous_driving/threshold-based_exclusive_batching_for_llm_inference.md)
- [\[CVPR 2026\] Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles](../../CVPR2026/autonomous_driving/open-ended_instruction_realization_with_llm-enabled_multi-planner_scheduling_in_.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
