---
title: >-
  [论文解读] TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making
description: >-
  [NeurIPS 2025][3D视觉][具身导航] 提出任务偏好多需求驱动导航（TP-MDDN）基准和AWMSystem自主决策系统，通过指令分解、动态目标选择和任务状态监控三个LLM模块配合多维度累积语义地图，实现长程多子任务导航。 在日常生活中，人们经常需要连续完成多个需求（如打扫→休息→吃饭），每个需求还伴随个人偏好…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "具身导航"
  - "长程任务规划"
  - "多需求驱动"
  - "语义地图"
  - "大语言模型"
---

# TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making

**会议**: NeurIPS 2025  
**arXiv**: [2511.17225](https://arxiv.org/abs/2511.17225)  
**代码**: 暂无  
**领域**: 3D视觉  
**关键词**: 具身导航, 长程任务规划, 多需求驱动, 语义地图, 大语言模型

## 一句话总结

提出任务偏好多需求驱动导航（TP-MDDN）基准和AWMSystem自主决策系统，通过指令分解、动态目标选择和任务状态监控三个LLM模块配合多维度累积语义地图，实现长程多子任务导航。

## 研究背景与动机

在日常生活中，人们经常需要连续完成多个需求（如打扫→休息→吃饭），每个需求还伴随个人偏好。传统的需求驱动导航（DDN）只处理单一需求，无法反映真实世界多需求的复杂性。

现有工作的不足：

**单需求局限**：DDN和MO-DDN等方法一次只处理一个需求指令，无法高效管理多个子任务的序列执行和状态跟踪

**缺乏任务偏好**：现有DDN没有明确建模用户偏好，如"整理居住空间"可能意味着清洁工具、装饰品或收纳盒，需要偏好约束才能精确执行

**长程导航挑战**：频繁调用大型语言模型推理成本高，导航中的碰撞、越界等错误缺乏实时纠正机制

**环境记忆不足**：现有方法对环境的记忆编码有限，难以在长程任务中维持空间语义一致性

本文的核心思路是：将单次DDN扩展为包含多个子需求和明确任务偏好的长程导航基准，并设计一套模块化的自主决策系统来高效应对。

## 方法详解

### 整体框架

系统由四个核心组件组成：（1）AWMSystem自主决策系统（包含BreakLLM + LocateLLM + StatusMLLM三个基础模型模块）；（2）MASMap多维累积语义地图；（3）双节奏动作生成框架；（4）自适应错误纠正器。

### 关键设计

1. **MASMap多维累积语义地图**：融合3D点云累积和2D语义地图，无需额外训练即可平衡精度和效率。

    - **原始数据处理**：对RGB图像用Ram-Grounded-SAM做目标检测和分割，从深度图提取3D点云
    - **实时累积**：通过重叠度指标判断当前检测到的目标与历史记录是否为同一物体，如果 $os^* > 0.8$ 且 $ros^* > 0.8$ 则合并点云并更新标签；如果 $\max os^* < 0.25$ 则视为新目标加入记录
    - **全局语义地图融合**：记录目标点云中心坐标后清除3D数据以节省内存，通过2D IoU和匈牙利算法进行历史目标匹配
    - 记忆库分为长期记忆（全局累积地图+历史规划目标）和短期记忆（局部3D点云+当前子任务状态）

2. **AWMSystem三模块决策链**：

    - **BreakLLM**：将长程指令自动分解为子任务列表 $d_{sub}$ 和状态列表 $Sub_{Status}$
    - **LocateLLM**：综合目标记忆、子任务状态、执行反馈等信息决定下一个目标。引入辅助反馈机制：当同一目标连续失败次数 $n_{CFE} \geq n_{tolerance}$ 时，生成"不要再选择该目标"的提示避免执行循环
    - **StatusMLLM**：当策略网络输出Done动作时触发，利用多模态LLM判断当前子任务是否完成，输出推理结果和更新后的状态

3. **双节奏动作生成器**：将规划解耦为慢节奏和快节奏两个阶段来平衡推理深度和效率：

    - **慢节奏阶段**：提取当前目标点云→LocateLLM决策→计算可行性价值图→A*路径规划→分解为航路点序列→执行
    - **快节奏阶段**：直接使用预训练策略网络输出低级动作（MoveAhead/RotateRight等），策略网络输出Done时触发StatusMLLM评估
    - 可行性价值图结合障碍物回避值 $a_{obs}(n_i)$ 和语义目标接近值 $a_{tgt}(n_i)$

4. **自适应错误纠正器**：当检测到MoveAhead可能碰撞时，从当前位置重新规划路径。将轨迹分为初始段（精细采样间隔 $n_{block}$，支持障碍物附近精细推理）和后续段（标准采样频率 $n_{waypoint}$），重新计算可行性价值图生成修正轨迹。

### 损失函数 / 训练策略

系统为零样本设计，不需要端到端重训练。策略网络沿用DDN的预训练模型，大语言模型（Qwen2.5-VL-72B）用于推理和决策。整体评估指标包括成功率（SR）、独立成功率加权路径长度（ISPL）、成功轨迹长度（STL）和独立成功率（ISR）。

## 实验关键数据

### 主实验

**TP-MDDN基准上与SOTA方法对比：**

| 方法 | 零样本 | 大模型推理 | 显式历史 | STL↑ | ISR↑ | SR↑ | ISPL↑ |
|------|--------|-----------|---------|------|------|-----|-------|
| DDN | ✗ | ✗ | ✗ | 15.50 | 44.67 | 16.00 | 40.66 |
| MO-DDN | ✗ | ✓ | ✓ | 12.11 | 39.78 | 13.33 | 36.25 |
| InstructNav | ✓ | ✓ | ✓ | 9.50 | 42.44 | 16.00 | 39.41 |
| **AWM-Nav** | ✗ | ✓ | ✓ | **20.11** | **62.89** | **32.00** | **44.19** |

### 消融实验

**不同组件消融（部分结果）：**

| 配置 | ISR↑ | SR↑ | ISPL↑ | 说明 |
|------|------|-----|-------|------|
| GLEE分割器 | 51.11 | 21.33 | 41.05 | 精度不足 |
| YOLO分割器 | 58.00 | 29.33 | 43.69 | 次优 |
| **RAM-Grounded-SAM** | **62.89** | **32.00** | **44.19** | 最佳 |
| Qwen2.5-VL-7B推理 | 47.78 | 19.33 | 36.45 | 参数不足 |
| GPT-4o推理 | 56.44 | 28.67 | 39.95 | 上下文理解弱于72B |
| **Qwen2.5-VL-72B** | **62.89** | **32.00** | **44.19** | 最佳 |
| 无自适应纠错 | 60.67 | 27.33 | 42.46 | 路径规划缺乏鲁棒性 |
| 无StatusMLLM | 60.44 | 28.00 | 42.20 | 子任务状态判断失效 |

### 关键发现

- AWM-Nav成功率比DDN和InstructNav高出16个百分点，且在效率上也显著优于InstructNav（6.82分钟 vs 88.90分钟）
- 双节奏策略中慢节奏动作执行时间约为快节奏的22倍，但整体推理效率远优于InstructNav的每步调用LLM
- 大模型参数量对长程导航的智能规划能力影响显著（7B vs 72B差距巨大）
- 即使有强大推理能力，自适应错误纠正仍然重要（碰撞和越界是不可避免的物理约束）

## 亮点与洞察

- **完整的基准设计**：TP-MDDN明确定义了多子需求+任务偏好的长程导航任务格式，填补了领域空白
- **模块化系统设计**：三个LLM模块各司其职（分解、定位、监控），清晰解耦了长程导航的核心挑战
- **效率与性能平衡**：双节奏设计避免了每步调用LLM的巨大开销，同时保持了强推理能力
- **MASMap的轻量级设计**：3D点云检测后只保留2D中心坐标，大幅减少存储开销

## 局限与展望

- 双节奏动作生成框架存在非自愿模式切换问题，慢节奏和快节奏之间的过渡不总是平滑
- 过度依赖预训练大语言模型可能导致指令误判进而影响导航决策
- 未来可用强化学习优化模式切换策略，训练领域专用的语言模型减少对通用大模型的依赖
- 基准目前只在ProcTHOR模拟环境中测试，真实世界迁移性有待验证

## 相关工作与启发

系统设计借鉴了WMNav的世界模型思想和Voyager的自主演化机制。与InstructNav相比，本文引入了显式的任务分解和状态跟踪，使长程导航更加可控。MASMap的IoU融合策略和记忆清理机制为其他长程具身任务提供了有效的空间记忆管理方案。

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统设计新颖，TP-MDDN基准填补空白，但单个模块的创新性有限
- 实验充分度: ⭐⭐⭐⭐ 消融实验覆盖各组件，但仅在模拟环境测试
- 写作质量: ⭐⭐⭐⭐ 结构清晰，系统描述详尽，但公式较多略显冗长
- 价值: ⭐⭐⭐⭐ 为多需求长程导航提供了完整解决方案，推动具身AI发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Orientation Matters: Making 3D Generative Models Orientation-Aligned](orientation_matters_making_3d_generative_models_orientation-aligned.md)
- [\[CVPR 2026\] Multi-Scale Gaussian-Language Map for Zero-shot Embodied Navigation and Reasoning](../../CVPR2026/3d_vision/multi-scale_gaussian-language_map_for_zero-shot_embodied_navigation_and_reasonin.md)
- [\[CVPR 2026\] Task-Driven Implicit Representations for Automated Design of LiDAR Systems](../../CVPR2026/3d_vision/task-driven_implicit_representations_for_automated_design_of_lidar_systems.md)
- [\[CVPR 2026\] LiDAR Prompted Spatio-Temporal Multi-View Stereo for Autonomous Driving](../../CVPR2026/3d_vision/lidar_prompted_spatio-temporal_multi-view_stereo_for_autonomous_driving.md)
- [\[NeurIPS 2025\] GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion](geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)

</div>

<!-- RELATED:END -->
