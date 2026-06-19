---
title: >-
  [论文解读] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation
description: >-
  [ICCV 2025][机器人][几何基元提取] 提出 PASG（Primitive-Aware Semantic Grounding），一个闭环框架，通过自动化几何基元提取（关键点、功能轴、主轴）和 VLM 驱动的语义锚定，将低层几何特征与高层任务语义动态耦合，在机器人操作任务中实现了接近人工标注的性能，并构建了 Robocasa-PA 基准和微调模型 Qwen2.5VL-PA。
tags:
  - "ICCV 2025"
  - "机器人"
  - "几何基元提取"
  - "语义锚定"
  - "机器人操作"
  - "VLM"
  - "闭环框架"
---

# PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2508.05976](https://arxiv.org/abs/2508.05976)  
**代码**: 无  
**领域**: 机器人操作 / 目标检测  
**关键词**: 几何基元提取, 语义锚定, 机器人操作, VLM, 闭环框架

## 一句话总结

提出 PASG（Primitive-Aware Semantic Grounding），一个闭环框架，通过自动化几何基元提取（关键点、功能轴、主轴）和 VLM 驱动的语义锚定，将低层几何特征与高层任务语义动态耦合，在机器人操作任务中实现了接近人工标注的性能，并构建了 Robocasa-PA 基准和微调模型 Qwen2.5VL-PA。

## 研究背景与动机

机器人操作领域长期存在**高层任务语义与低层几何特征之间的断裂**问题：

- VLM 虽然在生成感知感知视觉表示方面表现出色，但在规范空间中缺乏语义接地能力
- 现有方法（如 ReKep、CoPa）依赖于 VLM 在任务级别检测基元，但缺乏验证机制，检测错误会传播并严重降低成功率
- 手动标注几何基元成本高，限制了泛化能力
- 现有框架定义不完整——通常只包含关键点，忽略了物体的主轴等重要方向信息，导致抓取和运输任务失败

核心问题：如何构建一个统一框架，自动化地从物体中提取几何基元，并与任务语义建立动态映射关系？

## 方法详解

### 整体框架

PASG 包含四个核心阶段：
1. 自动化几何基元提取（VFM + 拓扑分析）
2. 语义基元识别（VLM 推理可能的操作任务和相关基元）
3. 视觉-语义对齐（将语义基元映射到几何检测结果）
4. 动态自校正匹配（闭环优化检测遗漏或错误）

### 关键设计

1. **语义基元定义（Semantic Primitives）**：

    - 每个交互基元定义为三元组 $(E, S, F)$：几何实体、结构描述、功能角色
    - **点基元 $\mathcal{P}$**：锚点（对齐参考位置）、抓取点（末端执行器最优抓取位置）、操作点（触发机制的位置）
    - **轴基元 $\mathcal{A}$**：主轴（物体几何/对称轴）、功能轴（功能方向）、接近轴（末端执行器接近方向）
    - 设计动机：统一点和轴两种基元，覆盖操作任务中所有必要的空间约束

2. **自动化几何基元提取（Geometry Primitive Extraction）**：

    - 使用 Semantic SAM 进行多粒度语义分割
    - **关键点提取**：从分割掩码中提取中心点、角点、PCA 轴与边界的交点
    - **关键点过滤**：DBSCAN 聚类去冗余 + 最远点采样保留全局代表性特征
    - **主轴校准**：通过上下视图掩码质心连线定义 Z 轴，正交生成 X/Y 轴
    - 跨视图颜色标准化确保轴可视化的一致性

3. **动态自校正匹配机制（Dynamic Self-Refine Matching）**：

    - 核心闭环算法：分割→对齐→检测→重采样
    - 当匹配置信度 < 0.5 或基元标记为 NONE 时，触发更细粒度的分割重采样
    - 利用 Semantic SAM 的多粒度分割层级自适应调整
    - 在数据集上实现 98% 的匹配成功率，有效防止了错误传播
    - 最多迭代 $\tau_{max} = 5$ 轮

### 损失函数 / 训练策略

- 对 VLM 微调（Qwen2.5VL-PA）使用 **LoRA** 参数高效微调
- 训练数据由 PASG 管道自动生成，5,583 条用于微调，1,396 条 in-distribution 测试，1,364 条 out-of-distribution 测试
- 三类任务：类型识别、任务关联、任务到基元映射
- 基准评估使用单选题准确率

## 实验关键数据

### 主实验 (表格)

**RoboTwin 操作任务成功率（%）**

| 方法 | 锤击 | 放容器 | 双瓶拾取 | 放空杯 | 拾苹果 | 放鞋 | 平均 |
|------|------|--------|----------|--------|--------|------|------|
| 人工标注 | 79.0 | **93.0** | **95.0** | 73.0 | **85.0** | **83.0** | **84.67** |
| **PASG** | **82.0** | 89.0 | 70.0 | **76.0** | 81.0 | 69.0 | 77.83 |

**空间语义推理基准准确率（%）**

| 模型 | In-distribution 整体 | Out-of-distribution 整体 |
|------|----------------------|--------------------------|
| GPT-4V | 39.04 | 39.00 |
| GPT-4O | 43.19 | 41.79 |
| Qwen-2.5VL | 43.91 | 43.33 |
| RoboPoint | 14.40 | 15.32 |
| **Qwen2.5VL-PA** | **77.79** | **79.69** |

### 消融实验 (表格)

**数据效率实验**（微调数据量 vs 准确率提升）

| 训练数据比例 | 样本数 | In-distribution 提升 | Out-of-distribution 提升 |
|-------------|--------|---------------------|--------------------------|
| 1% | 55 | 微弱提升 | 微弱提升 |
| 5% | 279 | ~+10% (绝对) | ~+10% (绝对) |
| 10% | 558 | ~+20% (绝对) | ~+20% (绝对) |
| 50% | 2,791 | 接近100%性能 | 接近100%性能 |
| 100% | 5,583 | +33.9% (绝对) | +36.4% (绝对) |

**数据集验证**：
- 语义识别准确率：91.6%
- 严格对齐准确率：75.8%
- 对齐有效性：91.5%
- 自校正匹配成功率：98%

### 关键发现

- PASG 在锤击和放空杯任务上**超越了人工标注**（82% vs 79%，76% vs 73%），说明自动标注的多样性带来更灵活的操作策略
- PASG 生成的交互基元比人工标注更丰富多样——人工标注受成本约束只标注少数最优点，PASG 生成更多语义有意义的点
- 微调 Qwen2.5VL-PA 仅用 5% 数据就能获得约 10% 的绝对准确率提升（20.6% 相对提升）
- In-distribution 和 out-of-distribution 性能差距稳定在 ±2%，证明了良好的跨域泛化
- PASG 是首个同时具备自动化基元提取、语义锚定和自校正机制的闭环框架

## 亮点与洞察

- **从物体级别出发**而非任务级别——先全面理解物体的几何和功能属性，再映射到具体任务，比现有的任务级方法（ReKep、CoPa）更通用
- **闭环自校正**设计巧妙利用了 Semantic SAM 的多粒度分割能力，自适应提升检测质量
- **语义层次化**：从低层描述（"瓶颈边缘"）到高层意图（"倒水时对齐瓶子的关键位置"），提供了完整的语义链
- 验证了**自动标注可以媲美甚至超越人工标注**的可能性

## 局限与展望

- 在双瓶拾取和放鞋任务上显著落后于人工标注（70% vs 95%，69% vs 83%），说明在需要精确协调的任务上仍有差距
- 当前 GPT-4o 在复杂空间语义理解上仍面临挑战
- 仅在仿真环境（RoboTwin）中验证，尚未大规模部署到真实机器人场景
- 数据集来源受限于 RoboCasa 和 Objaverse，可能缺乏工业场景的物体
- 路由精度依赖 GPT-4o 的推理质量，存在一定的误差传播风险

## 相关工作与启发

- **ReKep**：使用 VFM+VLM 在任务级别检测关键点，但缺乏验证机制和语义耦合
- **CoPa**：纳入功能轴但同样缺乏自适应修正
- **OmniManip**：使用约束优化和场景渲染进行 VLM 验证，但计算成本较高
- **SOFAR**：提出方向感知的空间理解模块，但使用预定义方向先验而非自动提取
- **SoM**（Set-of-Mark）：启发了标注可视化方案

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个将几何基元提取、语义锚定和自校正融为闭环的框架，物体级语义基元定义全面
- **实验充分度**: ⭐⭐⭐⭐ — 操作任务评估、VQA 基准、数据效率实验、人工验证均有覆盖，但缺乏真实机器人实验
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，方法细节充分，比较表格（Table 1）直观展示了与现有方法的差异
- **价值**: ⭐⭐⭐⭐ — 为机器人操作中的语义-几何桥接提供了实用且可扩展的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Closed-loop Long-horizon Robotic Planning via Equilibrium Sequence Modeling](../../ICML2025/robotics/closed-loop_long-horizon_robotic_planning_via_equilibrium_sequence_modeling.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[CVPR 2026\] Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control](../../CVPR2026/robotics/iterative_closed-loop_motion_synthesis_for_scaling_the_capabilities_of_humanoid_.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[CVPR 2025\] Robotic Visual Instruction](../../CVPR2025/robotics/robotic_visual_instruction.md)

</div>

<!-- RELATED:END -->
