---
title: >-
  [论文解读] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors
description: >-
  [CVPR 2025][机器人][视觉语言接地] 提出 RoboGround，一个两阶段框架：先用 Grounded VLM（GLaMM）从图像和文本指令中生成目标物体和放置区域的分割掩码，再通过 Grounded Perceiver 将掩码作为中间表示引导机器人策略网络执行操作，在复杂语义操作任务上实现 60-100% 的相对提升。
tags:
  - "CVPR 2025"
  - "机器人"
  - "视觉语言接地"
  - "分割掩码"
  - "机器人操作"
  - "Grounded Perceiver"
  - "零样本泛化"
---

# RoboGround: Robotic Manipulation with Grounded Vision-Language Priors

**会议**: CVPR 2025  
**arXiv**: [2504.21530](https://arxiv.org/abs/2504.21530)  
**代码**: [https://robo-ground.github.io](https://robo-ground.github.io)  
**领域**: 机器人操作  
**关键词**: 视觉语言接地, 分割掩码, 机器人操作, Grounded Perceiver, 零样本泛化

## 一句话总结

提出 RoboGround，一个两阶段框架：先用 Grounded VLM（GLaMM）从图像和文本指令中生成目标物体和放置区域的分割掩码，再通过 Grounded Perceiver 将掩码作为中间表示引导机器人策略网络执行操作，在复杂语义操作任务上实现 60-100% 的相对提升。

## 研究背景与动机

**领域现状**：语言条件的机器人操作策略（如 RT-1、GR-1）直接将文本指令和图像送入端到端网络预测动作。这类方法在简单指令（"pick up the cup"）上表现尚可，但面对需要语义推理的指令（如"把红色的那个放到蓝色碗旁边"）时泛化能力不足。

**现有痛点**：文本指令中的语义复杂性（外观描述、空间关系、常识推理）无法被纯端到端网络有效理解。模型需要同时解决"哪个物体"和"怎么操作"两个问题，但纯语言条件无法提供足够的视觉接地信息。

**核心矛盾**：VLM 有强大的语义理解能力但不擅长低层动作预测；策略网络擅长精确控制但缺乏语义理解。两者之间需要一个信息密度合适的中间表示。

**本文目标** 找到 VLM 语义理解和策略网络执行之间的最佳中间表示——分割掩码，并设计有效的融合机制。

**切入角度**：分割掩码既包含物体的像素级形状信息（比点/框更丰富），又足够简洁可以直接拼接到图像通道中。

**核心 idea**：用 VLM 生成分割掩码作为中间表示 + Grounded Perceiver 做掩码感知特征提取 = 语义操作能力大幅提升。

## 方法详解

### 整体框架

两阶段流水线：Stage 1 用 GLaMM（基于 CLIP+LLM+pixel decoder）处理图像和文本指令，输出目标物体掩码 $M_o$ 和放置区域掩码 $M_p$。Stage 2 将掩码与图像拼接后送入 ViTMAE 提取视觉特征，通过 Grounded Perceiver 用掩码引导的注意力提取关键区域信息，Transformer 解码器预测机械臂和夹爪动作。

### 关键设计

1. **Grounded Perceiver（掩码感知感知器）**:

    - 功能：将 196 个 patch token 压缩为 $3 \times k$ 个语义聚焦的 token
    - 核心思路：设计三组可学习查询——全局查询 $Q_g$（捕获整体场景）、目标物体查询 $Q_o$（聚焦目标）、放置区域查询 $Q_p$（聚焦放置位置）。在注意力计算时，用掩码对注意力矩阵进行 mask fill——$Q_o$ 只能关注 $M_o$ 覆盖的 patch，$Q_p$ 只能关注 $M_p$ 覆盖的 patch
    - 设计动机：简单的通道拼接虽然能传递掩码信息，但无法让策略网络的注意力显式聚焦在目标区域。消融显示两种方式有协同效应——仅拼接 26%/30%，仅 Perceiver 22%/30%，两者结合 32%/32%

2. **多模态指令数据生成流水线**:

    - 功能：自动构建大规模多样化训练数据
    - 核心思路：从 RoboCasa 扩展到 3526 个物体、176 类。三种指令类型自动生成：（1）外观指令——GPT-4 提取物体属性（颜色/形状/材质），CLIP 嵌入相似度采样干扰物；（2）空间指令——基于规则生成相对位置描述（±30° 容差）；（3）常识指令——GPT-4 生成日常场景任务
    - 设计动机：24K 演示 × 4.7 条多样化指令/演示 = 112K 训练对，确保策略网络在多种语义复杂度下泛化

3. **掩码作为中间表示（vs 点/框）**:

    - 功能：提供最丰富的视觉接地信息
    - 核心思路：图像 + 目标掩码 + 放置掩码通过线性层映射到 3 通道后送入 ViTMAE。掩码是二进制的，每个像素明确标记是否属于目标区域
    - 设计动机：消融对比显示，掩码 > 边界框 > 点标注（成功率 42% > 38% > 32%），因为掩码保留了物体的精确形状和大小信息

### 损失函数 / 训练策略

VLM 微调用分割损失 $\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{DICE}$ + 文本自回归交叉熵。策略网络用 $\mathcal{L}_{total} = \text{SmoothL1}(\hat{a}_{arm}, a_{arm}) + \text{BCE}(\hat{a}_{gripper}, a_{gripper})$。VLM 微调后 mIoU 从零样本 13.2% 提升到 48.2%。

## 实验关键数据

### 主实验

仿真 pick-and-place 任务（接触率/成功率）：

| 方法 | Easy | 外观推理 | 空间推理 | 常识推理 |
|------|------|---------|---------|---------|
| ACT | 47.3/18.3 | 18.5/3.8 | 17.5/3.5 | 15.3/2.8 |
| GR-1 | 85.3/42.8 | 49.5/13.8 | 54.5/16.3 | 43.0/11.5 |
| **RoboGround** | **89.0/43.3** | **78.5/30.5** | **81.0/33.5** | **76.3/30.0** |

零样本泛化（未见物体实例/未见类别）：

| 设置 | w/ 掩码 | w/o 掩码 | 提升 |
|------|--------|---------|------|
| 未见实例 (外观) | 75.5/29.5 | 38.0/11.5 | +100% |
| 未见类别 (外观) | 68.5/14.3 | 27.5/5.3 | +170% |

### 消融实验

| 中间表示 | Easy 成功率 | 外观 成功率 |
|---------|-----------|-----------|
| 无 | 24% | 12% |
| 点标注 | 32% | 26% |
| 边界框 | 38% | 30% |
| **分割掩码** | **42%** | **32%** |
| GT 掩码（上界） | 68% | 48% |

### 关键发现
- **掩码是最优中间表示**：比框高 4-12% 成功率，比点高 6-10%。形状信息对抓取尤其关键
- **GT 掩码 vs 预测掩码差距大**：68% vs 42%（easy），说明 VLM 掩码质量仍是瓶颈
- **接触率>>成功率**：89% 接触但只有 43% 成功，说明接触后的精确抓取仍是难题
- **数据多样性提升巨大**：仅用简单数据训练在推理任务上仅 6% 成功，加入多样指令后升至 30%
- **Grounded Perceiver 与通道拼接协同**：两者单独使用效果接近，结合后有明显提升

## 亮点与洞察

- **中间表示的层级实验**：系统对比了点→框→掩码→GT掩码的性能梯度，清晰展示了视觉接地信息密度与操作性能的正相关关系，为后续工作提供了量化参考
- **Grounded Perceiver 的注意力掩码机制**：用物体掩码引导注意力查询只关注目标区域，比简单拼接更有效地利用了空间先验
- **数据生成的可扩展性**：GPT-4 + CLIP + 规则的混合数据生成管线可以低成本地扩展到新场景和新物体

## 局限与展望

- **抓取精度瓶颈**：接触率（89%）与成功率（43%）的巨大差距说明精确抓取姿态仍是难题，可结合 AnyGrasp 等专用抓取模型
- **仅单次掩码提取**：掩码在 episode 开始时提取一次，无法应对动态场景或多步任务中的物体位置变化
- **仅仿真实验**：未在真实机器人上验证，sim-to-real 迁移存在域差距
- **VLM 掩码质量有限**：微调后 mIoU 仅 48.2%，GT 掩码性能上界远高于预测掩码
- **放置区域多样性不足**：数据集中放置目标的多样性被忽视，限制了放置精度

## 相关工作与启发

- **vs GR-1**: GR-1 用纯语言条件，在复杂推理任务上仅 11-16% 成功率。RoboGround 通过掩码接地将其提升到 30-33%，证明视觉接地的必要性
- **vs 端到端 VLA（如 OpenVLA）**: 端到端方法无法提供细粒度的视觉接地，RoboGround 的两阶段设计在保持灵活性的同时提供了更强的语义理解
- **vs SoM/Set-of-Mark**: SoM 用边界框标记物体，RoboGround 用分割掩码提供更精确的形状信息

## 评分
- 新颖性: ⭐⭐⭐⭐ 掩码作为中间表示 + Grounded Perceiver 的组合设计有效且有说服力
- 实验充分度: ⭐⭐⭐⭐⭐ 多种表示对比消融、零样本泛化、数据多样性分析非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据生成流水线描述详尽
- 价值: ⭐⭐⭐⭐ 为 VLM→策略网络的中间表示选择提供了系统性实验证据

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LaDA: Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] INSIGHT Bench: Towards Grounded IN-SItu Guidance for Robotic Manipulation](../../CVPR2026/robotics/insight_bench_towards_grounded_in-situ_guidance_for_robotic_manipulation.md)
- [\[CVPR 2026\] Learning Surgical Robotic Manipulation with 3D Spatial Priors](../../CVPR2026/robotics/learning_surgical_robotic_manipulation_with_3d_spatial_priors.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](../../CVPR2026/robotics/lada_robotic_manipulation.md)
- [\[CVPR 2025\] MoManipVLA: Transferring Vision-Language-Action Models for General Mobile Manipulation](momanipvla_transferring_vision-language-action_models_for_general_mobile_manipul.md)

</div>

<!-- RELATED:END -->
