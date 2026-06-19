---
title: >-
  [论文解读] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs
description: >-
  [NeurIPS 2025][多模态VLM][VLM] 提出T-Rex框架，根据任务复杂度动态选择最优的空间表示提取方案（点/向量/6D位姿），并设计Chain of Grounding (CoG)引导VLM逐步推理，实现无需训练的开放词汇机器人操纵。 视觉语言模型（VLM）因其从大规模数据中获得的丰富世界知识…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLM"
  - "机器人操纵"
  - "空间表示"
  - "任务自适应"
  - "Chain of Grounding"
---

# T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs

**会议**: NeurIPS 2025  
**arXiv**: [2506.19498](https://arxiv.org/abs/2506.19498)  
**代码**: [https://github.com/](https://github.com/) (未提供)  
**领域**: 多模态VLM / 机器人操作  
**关键词**: VLM, 机器人操纵, 空间表示, 任务自适应, Chain of Grounding

## 一句话总结

提出T-Rex框架，根据任务复杂度动态选择最优的空间表示提取方案（点/向量/6D位姿），并设计Chain of Grounding (CoG)引导VLM逐步推理，实现无需训练的开放词汇机器人操纵。

## 研究背景与动机

视觉语言模型（VLM）因其从大规模数据中获得的丰富世界知识，正被越来越多地应用于机器人操纵任务。空间表示（如表示物体位置的点、表示物体朝向的向量）是连接VLM推理能力与真实世界场景的桥梁。

**核心痛点**：现有VLM引导的机器人方法（如ReKep、VoxPoser）采用**固定的空间表示提取方案**——不论任务简单还是复杂，都用同一种方式提取空间信息。这导致两个问题：

**表示能力不足**：简单的点表示无法处理需要物体朝向信息的任务（如"让毛绒玩具正面面对相机"）

**提取时间过长**：对只需质心点即可完成的简单任务也使用昂贵的6D位姿估计

**核心矛盾**：任务复杂度决定了所需空间表示的类型和粒度，而更强的表示能力通常意味着更高的系统运行成本。如何在表示能力和效率之间取得平衡？

**本文切入角度**：让VLM自己判断每个任务阶段中每个物体需要什么级别的空间表示，动态调用对应的提取工具。设计CoG方法显式引导VLM的分阶段推理过程，确保推理稳定性。

## 方法详解

### 整体框架

给定自然语言指令和场景观测，VLM通过CoG将指令分解为多阶段任务，为每个阶段的每个相关物体选择最优的空间表示提取方案，生成约束函数。底层动作序列生成器基于这些约束和追踪的空间表示生成机器人动作。

### 关键设计

1. **任务自适应异构空间表示提取**：构建一个可扩展的空间表示提取工具箱（Toolkit），包含多种大视觉模型（如Grounding DINO用于关键点、FoundationPose用于6D位姿等）。每个工具定义为 $(I_i, o_i, f_i, s_i, h_i)$，包含输入、输出类型、格式、实现摘要和历史平均执行时间。VLM根据任务和场景为每个物体选择最优工具：$t_{s,o}^* = \arg\max_{t \in \mathcal{R}} [P_{\text{succ}}(t|I,X,s,o) - \lambda h_t]$，在成功概率和提取成本间权衡。

2. **任务自适应多粒度空间表示提取**：当VLM判断某任务阶段需要更细粒度的空间表示时（如机器人狗的腿部朝向），触发"局部放大"策略：先用SAM分割目标物体区域，扩展padding后裁剪局部子图，再在子图上应用自适应提取。这种attention启发的zoom-in策略只在必要时激活，不增加简单任务的开销。

3. **Chain of Grounding (CoG)**：显式引导VLM的推理过程，分为四个顺序依赖阶段：

    - **操作提示推理**：将任务分解为多阶段，生成与表示无关的操作提示
    - **约束推理**：为每个提示推理所需的空间约束（自然语言形式）
    - **工具选择**：查询Toolkit Registry为每个物体选择最优提取工具
    - **约束代码生成**：将自然语言约束转换为可执行Python函数（返回标量cost）

### 损失函数 / 训练策略

T-Rex是zero-shot方法，**无需任何训练**。核心依赖VLM的推理能力（使用GPT-4.1）和预训练的视觉基础模型。约束函数以Python代码形式生成，通过数值优化器求解机器人动作序列。

## 实验关键数据

### 主实验

**15个真实世界开放词汇操纵任务（10次独立试验/任务）**

| 任务 | VoxPoser成功率 | ReKep成功率 | T-Rex成功率 | T-Rex时间(s) |
|------|-------------|------------|------------|-------------|
| Open Drawer | 4/10 | 2/10 | 6/10 | 14.3 |
| Pour Water | 0/10 | 3/10 | 7/10 | 24.1 |
| Close Lid of Laptop | 4/10 | 2/10 | 7/10 | 21.6 |
| Setup: Tools Insert | 0/10 | 3/10 | 7/10 | 56.3 |
| Setup: Mixed | 0/10 | 0/10 | 2/10 | 217.5 |
| **总计** | **30%** | **36.4%** | **60.7%** | **45.5** |

### 消融实验

| 配置 | 成功率(%) | 时间(s) |
|------|---------|--------|
| T-Rex完整版 | 60.7%±2.1% | 45.5±1.3 |
| w/o CoG | 52.1%±2.4% | 41.4±2.1 |
| w/o Toolkit (仅点) | 30.7%±3.7% | 33.6±3.6 |
| w/o Toolkit (VLM点+向量) | 55.0%±2.9% | 47.9±3.2 |
| w/o CoG, w/o Toolkit (仅点) | 27.9%±1.5% | 30.0±0.9 |

### 关键发现

- T-Rex总成功率60.7%，较VoxPoser(30%)提升约2倍，较ReKep(36.4%)提升约1.7倍
- 移除Toolkit（仅用点表示）成功率降至30.7%，证明异构空间表示的必要性
- 移除CoG成功率降至52.1%，CoG提供稳定的8.6%绝对提升且几乎不增加延迟
- 系统误差主要来自空间表示追踪(tracking)模块，而非VLM推理或提取
- 在需要6D位姿的任务（如毛绒玩具摆放）中优势尤为明显

## 亮点与洞察

- **任务驱动的表示选择**理念非常务实——不同任务确实需要不同粒度的空间感知
- **可扩展性强**：新工具只需在配置文件中注册几个参数即可集成
- **零训练部署**：完全依赖预训练模型的组合，适合快速原型开发
- **CoG的设计哲学**：将VLM的一次性复杂推理拆解为四个阶段的链式推理，显著降低幻觉风险

## 局限与展望

- 空间表示追踪是最大瓶颈，现有工具缺乏持续追踪能力
- 复杂多阶段任务（如Mixed setup）成功率仍然较低(2/10)
- 高度依赖VLM推理质量（GPT-4.1），更弱的VLM会导致约束生成错误
- 实验仅在桌面操纵场景中验证，未测试移动操纵等更复杂场景
- 处理时间较长（复杂任务>200s），实时性不足

## 相关工作与启发

- 与ReKep等工作互补：ReKep使用固定关键点，T-Rex的Toolkit可包含ReKep作为一种工具
- CoG的链式推理思路与Chain-of-Thought类似，但专门针对机器人操纵的"grounding"过程设计
- Toolkit的开放注册框架可启发其他需要组合多种视觉工具的应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 任务自适应表示提取的思路新颖，CoG设计合理
- 实验充分度: ⭐⭐⭐⭐ 15个真实世界任务+消融+误差分析，较为丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，形式化定义完整，图示直观
- 价值: ⭐⭐⭐⭐ 为VLM驱动的机器人系统提供了实用的工程解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[ACL 2025\] SemEval-2025 Task 1: AdMIRe -- Advancing Multimodal Idiomaticity Representation](../../ACL2025/multimodal_vlm/semeval-2025_task_1_admire_--_advancing_multimodal_idiomaticity_representation.md)
- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[NeurIPS 2025\] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models](text_to_robotic_assembly_of_multi_component_objects_using_3d_generative_ai_and_v.md)
- [\[NeurIPS 2025\] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)

</div>

<!-- RELATED:END -->
