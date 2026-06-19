---
title: >-
  [论文解读] Realistic Synthetic Household Data Generation at Scale
description: >-
  [AAAI 2026][机器人][合成数据生成] 提出一个基于 LLM 的双向耦合生成框架，通过人物画像驱动环境生成、环境语义引导行为生成的迭代循环过程，大规模生成包含家庭环境配置、人类行为和人机交互的合成数据集，用于训练家用机器人。 训练家用机器人面临一个根本性的技术挑战：建模和理解人类行为模式与环境配置之间的双向关系：…
tags:
  - "AAAI 2026"
  - "机器人"
  - "合成数据生成"
  - "家庭环境建模"
  - "双向耦合"
  - "LLM驱动"
  - "具身AI"
---

# Realistic Synthetic Household Data Generation at Scale

**会议**: AAAI 2026  
**arXiv**: [2602.07243](https://arxiv.org/abs/2602.07243)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 合成数据生成, 家庭环境建模, 双向耦合, LLM驱动, 具身AI

## 一句话总结

提出一个基于 LLM 的双向耦合生成框架，通过人物画像驱动环境生成、环境语义引导行为生成的迭代循环过程，大规模生成包含家庭环境配置、人类行为和人机交互的合成数据集，用于训练家用机器人。

## 研究背景与动机

训练家用机器人面临一个根本性的技术挑战：**建模和理解人类行为模式与环境配置之间的双向关系**。机器人要在多样化的家庭环境中安全有效地运作，需要同时理解：

- **静态环境属性**：物体可供性（affordance）、空间关系、语义标签
- **时序人类-环境交互**：日常作息、物体操作序列、长期空间利用模式

**现有方法的三个核心局限**：

**时空依赖缺失**：无法捕捉人类活动如何影响物体放置、房间布局等

**单向而非双向耦合**：现有方法将环境生成和行为合成当作独立过程

**语义关系断裂**：人物特征、环境可供性和行为模式之间缺乏连贯的语义联系

**具体问题**：
- ProcGen 等算法方法语义多样性有限
- Holodeck 等 LLM 方法缺乏粒度化的行为建模
- Wang et al. 的 Dynamic Scene Generation 不考虑人格驱动的场景生成
- 所有现有方法都将环境和行为的生成解耦处理

## 方法详解

### 整体框架

框架由四个主要模块组成，在迭代精炼循环中运作：

1. **Environment Schematic Generator**（环境示意图生成器）
2. **Human Activity and HRI Generator**（人类活动和人机交互生成器）
3. **Bidirectional Influence Controller**（双向影响控制器）
4. **Universal Simulator Adapter**（通用仿真器适配器）

核心思想：人物画像（Persona）→ 影响环境生成 → 环境语义反过来引导活动生成 → 生成的活动再修改环境 → 循环迭代直至收敛。

### 关键设计

#### 1. **人类活动与人机交互生成**

采用三阶段层次化分解策略：

**第一阶段**：活动生成——基于家庭成员画像、环境约束、时间参数和机器人能力，生成结构化的活动序列，保持时空一致性

**第二阶段**：交互合成——处理第一阶段的活动序列，用上下文适当的对话丰富它们，考虑社会动态和文化因素

**第三阶段**：仿真器适配——将中间表示转换为兼容各种仿真环境的格式

关键技术：
- **Least-to-most prompt tuning**：渐进式提示工程
- **滚动窗口上下文机制**：维护事件一致性，确保活动逻辑递进
- **上下文记忆管理**：每个步骤都了解（1）正在执行的任务，（2）流水线步骤，（3）前一步完成的工作，（4）当前步骤需求，减少幻觉

#### 2. **环境示意图生成**

在前人工作基础上做了几个关键改进：

- **资产数据库灵活性**：不绑定特定资产库（如 Objaverse），只需元数据（描述、尺寸、枢轴点、图片）
- **房间布局错误修正**：后处理嵌套房间和断开连接的配置
- **逼真的门连接生成**：LLM 推荐基于连接房间的门类型（如开放式厨房/客厅取消隔墙）

人物画像整合：
- 每个家庭成员分配个人卧室
- 基于工作模式生成家庭办公空间
- 根据活动行为选择匹配的资产

#### 3. **迭代双向影响控制器**

这是本文的核心创新。环境生成模块产生物体清单、空间布局和可供性地图来约束活动生成；生成的活动序列反过来影响物体放置、房间使用和环境修改。

收敛标准是加权组合：

$$\text{Score} = w_2 \rho_{\text{env}}(i+1) + w_3 \gamma_{\text{act}}(i+1) + w_4 \sigma_{\text{sem}}(i+1)$$

其中：
- $\rho_{\text{env}}$：环境物体密度 = |Objects| / |Rooms|
- $\gamma_{\text{act}}$：活动时间表粒度 = Σ duration / |Activities|
- $\sigma_{\text{sem}}$：语义相似度 = 环境描述和活动描述在 SBERT 嵌入空间中的余弦相似度

迭代终止条件：达到最大迭代次数或满足用户指定的收敛阈值。

### 损失函数 / 训练策略

本框架不涉及神经网络训练。核心 "损失" 体现在收敛标准的优化上——通过迭代精炼而非梯度下降来提升生成质量。关键参数包括 LLM 的 temperature、top_p、top_k 用于控制变异性。

## 实验关键数据

### 主实验：语义对齐分析

使用多模态嵌入（SBERT + CLIP）计算成对余弦相似度：

| 嵌入对 | SBERT 相似度 |
|---|---|
| 人物-环境 | 0.68 ± 0.09 |
| 环境-行为 | 0.72 ± 0.07 |
| 人物-行为 | 0.61 ± 0.12 |
| 房屋图像 vs 家庭描述 (CLIP) | 0.74 ± 0.08 |

环境-行为的最高相关性（0.72）说明双向影响机制有效地将环境和行为联系起来。

### 真实世界对齐验证

| 数据集对比 | 余弦相似度 |
|---|---|
| HOMER（真实数据）vs 本框架 | 0.60 |
| Wang et al.（合成数据）vs 本框架 | 0.27 |

与真实数据集 HOMER（21 名参与者的自报告活动）的高对齐度（0.60 > 0.5 的中等阈值），验证了框架生成的行为模式接近真实人类行为。

### 消融实验

**迭代改进验证**：

| 迭代次数 | 互信息 MI(P,E)+MI(E,B) | 余弦相似度 |
|---|---|---|
| 1 | 0.45 ± 0.09 | 0.58 ± 0.12 |
| 2 | 0.62 ± 0.08 | 0.65 ± 0.10 |
| 3 | 0.74 ± 0.06 | 0.71 ± 0.08 |
| 4 | 0.81 ± 0.05 | 0.76 ± 0.07 |
| 5 | 0.85 ± 0.04 | 0.79 ± 0.06 |

迭代从 1 到 5，MI 从 0.45 提升至 0.85，验证了双向精炼机制的有效性。

**干预分析**（因果验证）：

| 干预类型 | p-value | Cohen's d | 效应量 |
|---|---|---|---|
| 年龄：青少年 | p<0.001 | 0.89 | 大 |
| 年龄：退休者 | p<0.001 | 1.12 | 大 |
| 整洁度：凌乱 | p=0.003 | 0.64 | 中 |
| 整洁度：整齐 | p=0.001 | 0.73 | 中 |
| 睡眠：早起 | p=0.012 | 0.51 | 中 |
| 睡眠：晚睡 | p=0.008 | 0.58 | 中 |

所有干预均达到统计显著性，Cohen's d = 0.51-1.12，确认双向耦合机制成功将人物特征差异转化为可测量的环境配置和行为模式差异。

### 关键发现

1. **双向耦合满足 mediation criterion**：MI(persona,env) + MI(env,beh) > MI(persona,beh)，证明环境是人物和行为之间有效的中介
2. **5 次迭代后互信息几乎翻倍**（0.45→0.85），说明迭代精炼确实提升了语义连贯性
3. **与真实数据的对齐度（0.60）远高于与其他合成数据的对齐度（0.27）**
4. **三成员/三房间/一天的生成**需约 150 秒和 22 次 LLM 调用，具有实用性

## 亮点与洞察

1. **双向耦合架构**是核心创新——打破了环境和行为独立生成的传统范式
2. **上下文记忆机制**减少 LLM 幻觉——在每步提供任务上下文、已完成工作和当前需求
3. **结构化输入优于自由文本**——将信息按步骤结构化提供给 LLM，比全程自由文本效果更好
4. **面向工业应用的实用设计**——支持自然语言配置、变异生成和仿真器无关性

## 局限与展望

1. **缺乏可视化的 3D 环境展示**：未展示实际生成的 3D 场景质量
2. **LLM 幻觉问题**：承认存在"不可能的活动"生成（如 LLM 生成睡觉时大声放音乐）
3. **交互冲突检测不完善**：同时进行不兼容活动（如大声音乐 vs 睡觉）
4. **评估主要基于统计指标**：余弦相似度和互信息无法完全反映生成数据的实际可用性
5. **计算开销随场景复杂度增长**：多成员/多房间/多天的场景可能导致生成时间大幅增加
6. **未展示下游任务性能**：生成的数据是否真的能提升机器人训练效果未验证

## 相关工作与启发

- **与 Holodeck 的关系**：在 Holodeck 的 LLM 驱动环境生成基础上增加了行为-环境双向耦合
- **与 Dynamic Scene Generation (Wang et al.) 的对比**：后者不考虑人格驱动因素
- **实际应用场景广阔**：扫地机器人理解日常作息、辅助机器人预测人类需求、智能家居系统适应家庭动态
- 可启发方向：将双向耦合思想应用到其他数据生成领域（如城市交通仿真）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 双向耦合机制新颖，但各模块（LLM 驱动生成）本身较常规
- **实验充分度**: ⭐⭐⭐ — 统计验证充分，但缺乏下游任务验证和 3D 场景质量评估
- **写作质量**: ⭐⭐⭐ — 结构清晰但部分内容偏冗长，算法伪代码有助于理解
- **价值**: ⭐⭐⭐⭐ — 对家用机器人训练数据的规模化生成有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Video2Robo: 3DGS-based Synthetic Data from One Video Enables Scalable Robot Learning](../../CVPR2026/robotics/video2robo_3dgs-based_synthetic_data_from_one_video_enables_scalable_robot_learn.md)
- [\[CVPR 2026\] InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy](../../CVPR2026/robotics/interndata-a1_pioneering_high-fidelity_synthetic_data_for_pre-training_generalis.md)
- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](../../CVPR2026/robotics/igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[CVPR 2026\] GeoDexGrasp: Geometry-aware Generation for Data-efficient and Physics-plausible Dexterous Grasping](../../CVPR2026/robotics/geodexgrasp_geometry-aware_generation_for_data-efficient_and_physics-plausible_d.md)
- [\[AAAI 2026\] UrbanNav: Learning Language-Guided Urban Navigation from Web-Scale Human Trajectories](urbannav_learning_language-guided_urban_navigation_from_web-scale_human_trajecto.md)

</div>

<!-- RELATED:END -->
