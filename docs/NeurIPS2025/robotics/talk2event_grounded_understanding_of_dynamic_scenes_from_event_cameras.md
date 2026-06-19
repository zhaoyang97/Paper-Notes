---
title: >-
  [论文解读] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras
description: >-
  [NeurIPS 2025 Spotlight][机器人][事件相机] Talk2Event 提出首个大规模事件相机视觉定位基准（30,690 条标注表达式 + 四种定位属性），并设计 EventRefer 框架通过混合事件-属性专家（MoEE）动态融合外观/状态/观察者关系/物体间关系特征，在纯事件、纯帧和融合三种设置下均超越现有方法。
tags:
  - "NeurIPS 2025 Spotlight"
  - "机器人"
  - "事件相机"
  - "视觉定位"
  - "多模态"
  - "mixture of experts"
  - "自动驾驶"
  - "benchmark"
---

# Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2507.17664](https://arxiv.org/abs/2507.17664)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 事件相机, 视觉定位, multimodal, mixture of experts, 自动驾驶, benchmark  

## 一句话总结
Talk2Event 提出首个大规模事件相机视觉定位基准（30,690 条标注表达式 + 四种定位属性），并设计 EventRefer 框架通过混合事件-属性专家（MoEE）动态融合外观/状态/观察者关系/物体间关系特征，在纯事件、纯帧和融合三种设置下均超越现有方法。

## 背景与动机

1. **事件相机优势未被语言连接**：事件相机具有微秒级延迟和运动模糊鲁棒性，非常适合动态环境感知，但将异步事件流与自然语言连接的研究尚属空白。
2. **视觉定位缺乏事件域数据集**：现有视觉定位基准（RefCOCO、ScanRefer 等）均基于 RGB 帧或点云构建，仅覆盖静态场景，缺少高速/弱光下的时序动态信息。
3. **属性级标注缺失**：已有基准大多只关注外观属性，缺少对物体运动状态、与观察者关系、与其他物体关系的结构化描述，限制了可解释性。
4. **动态场景定位困难**：高速运动物体（行人、骑行者）在传统帧中易模糊，事件相机虽能捕获运动但缺乏纹理信息，需要跨模态融合解决方案。
5. **现有检测方法无语言能力**：事件相机领域的目标检测方法（RVT、LEOD 等）仅做类别预测，不支持自由文本指代定位。
6. **融合策略单一**：已有事件-帧融合方法对多属性信息的利用缺乏自适应机制，无法根据场景动态调整注意力分配。

## 方法详解

### 整体框架

- **功能**：构建 Talk2Event 基准并提出 EventRefer 模型，实现基于自然语言描述从事件流（可选融合 RGB 帧）中定位目标物体。
- **为什么**：事件相机在动态场景中具有独特优势，但缺乏语言定位任务的数据集和方法，需要从基准和模型两个层面填补空白。
- **怎么做**：基于 DSEC 驾驶数据集构建标注流水线，利用时间上下文帧生成丰富的指代表达式，设计属性感知的定位框架，在 DETR 架构上引入混合专家融合。

### 关键设计

#### 四属性标注方案
- **功能**：为每个指代表达式标注四种定位属性——外观（Appearance）、状态（Status）、与观察者关系（Relation-to-Viewer）、与其他物体关系（Relation-to-Others）。
- **为什么**：单一外观描述不足以在动态场景中精确定位，运动状态（"正在转弯"）、自我中心关系（"左前方"）和物体间关系（"公交车旁"）提供关键消歧线索。
- **怎么做**：利用 t₀±200ms 的两帧上下文，用 Qwen2-VL 生成三条不同表达式，每条平均 34.1 词；结合模糊匹配和语言模型进行属性标注，人工验证准确性。

#### 正向词匹配（PWM）
- **功能**：将指代表达式中与各属性相关的文本片段映射为 token 级二元正向图。
- **为什么**：避免手动标注 token 范围（在 4 种属性下不现实），同时让模型在训练时关注属性相关 token。
- **怎么做**：对每个属性的线索短语（如 "moving left" 对应 Status），用模糊匹配器定位表达式中的所有匹配位置，将字符范围投射到 token 索引上，形成 softmax 归一化的正向图 m_i。

#### MoEE 混合事件-属性专家
- **功能**：通过四个属性专家分别提取属性感知特征，然后自适应加权融合为最终表示。
- **为什么**：不同场景下各属性的信息量不同（夜间依赖运动线索，白天外观更有用），静态融合无法适应这种变化。
- **怎么做**：(1) 属性感知遮罩：用二元掩码 m_i^att 对编码器隐状态进行遮罩，保留属性相关 token + 公共上下文；(2) 各属性特征经 FFN 得到专家特征 H_i^exp；(3) 对四个专家描述符做均值池化后拼接，通过可学习投影 W 生成门控权重 λ；(4) 加入高斯噪声防止坍缩，最终 H^fuse = Σλ_i·H_i^exp。

#### 多属性融合训练与推理
- **功能**：将四个属性视为共定位的伪目标进行多目标匹配训练，推理时融合四个属性的评分选择最终预测框。
- **为什么**：利用所有属性的监督信号而不增加解码器复杂度，确保模型在每个属性维度都有密集监督。
- **怎么做**：训练时将 GT box 复制 4 份（每个属性一份），用匈牙利匹配分配查询；推理时对每个查询计算与四个属性图的 softmax 点积得分，取最高分的框作为预测。

## 实验结果

### 实验一：主要对比实验（Val 集 mAcc%）

| 方法 | 模态 | mAcc | Ped | Rider | Car | Bus | Truck | mIoU |
|------|------|------|-----|-------|-----|-----|-------|------|
| BUTD-DETR | Frame | 48.91 | 22.66 | 20.44 | 61.94 | 33.93 | 35.93 | 84.30 |
| GroundingDINO | Frame | 44.50 | 15.62 | 8.62 | 57.70 | 32.52 | 41.20 | 68.67 |
| **EventRefer** | **Frame** | **55.47** | **27.64** | **51.10** | **65.76** | **47.02** | 32.22 | **85.76** |
| EvRT-DETR | Event | 29.34 | 15.45 | 5.50 | 39.24 | 7.74 | 9.26 | 75.66 |
| **EventRefer** | **Event** | **31.96** | 12.09 | **25.00** | **40.83** | **15.48** | **16.30** | **76.46** |
| FlexEvent | Fusion | 59.40 | 30.39 | 33.50 | 71.34 | 47.85 | 38.58 | 86.83 |
| **EventRefer** | **Fusion** | **61.82** | 31.15 | 44.23 | **73.85** | 41.07 | 41.70 | **87.32** |

**发现**：EventRefer 在三种模态设置下均取得最佳 mAcc 和 mIoU。纯帧设置中 Rider 类提升最显著（+24.4%），表明属性级推理对小型动态物体特别有效。纯事件设置性能整体低于帧方法（缺乏纹理），但融合后达到最高 61.82%。

### 实验二：消融实验（Event-only, Val 集 mAcc%）

| 组件配置 | mAcc |
|---------|------|
| Baseline（无 PWM/MAF/MoEE） | 22.07 |
| + PWM | 26.38 (+4.31) |
| + MAF | 27.01 (+4.94) |
| + PWM + MAF | 29.66 (+7.59) |
| + PWM + MAF + MoEE | **31.96 (+9.89)** |

**发现**：三个组件各自贡献独立且互补——PWM 提供 token 级属性监督，MAF 实现独立属性推理，MoEE 自适应融合最终贡献 +2.3%。融合策略对比中，MoEE（31.96%）显著优于注意力融合（29.66%）、加法融合（28.39%）和拼接融合（27.50%）。

### 属性贡献分析

单属性实验显示 Status（28.90%）> Appearance（27.98%）> Viewer（27.03%）> Others（26.97%），但四属性联合使用达到最优 31.96%，证明属性互补性。MoEE 门控权重可视化显示：Rider/Bike 更依赖 Status 线索，Bus/Truck 更依赖 Appearance 和 Viewer Relation，体现了自适应特性。

## 亮点

- 首个事件相机视觉定位基准，5,567 场景 + 30,690 条表达式，每条平均 34.1 词，是语言最丰富的定位数据集之一
- 四属性标注方案提供结构化、可解释的定位维度，覆盖时空和关系推理
- MoEE 的自适应门控机制可根据场景动态/光照条件调整属性权重，兼具性能和可解释性
- 支持纯事件/纯帧/融合三种模态，研究灵活性高

## 局限性

- 数据集仅基于 DSEC（瑞士城市驾驶场景），场景多样性有限，缺少极端天气/夜间/拥挤场景
- 纯事件模式下性能仍明显低于帧方法（31.96% vs 55.47%），事件流的语义信息不足是根本瓶颈
- 标注表达式由 Qwen2-VL 生成后人工验证，可能存在生成偏差
- 未与近期大规模视觉语言模型（如 GPT-4V、Qwen2.5-VL）在此任务上做直接对比

## 相关工作对比

| 维度 | Talk2Event | RefCOCO/RefCOCOg |
|------|-----------|-----------------|
| 传感器 | 事件相机（可选 RGB 帧） | RGB 帧 |
| 场景类型 | 动态驾驶场景 | 静态图像 |
| 属性标注 | 4 种（外观+状态+观察者+物体间） | 仅外观 |
| 表达式长度 | 34.1 词（丰富） | 3.5-8.4 词（简短） |
| 时序信息 | ✓（运动、轨迹） | ✗ |

| 维度 | Talk2Event | ScanRefer |
|------|-----------|-----------|
| 传感器 | 事件相机 + RGB | RGB-D 点云 |
| 场景类型 | 动态室外驾驶 | 静态室内 |
| 属性标注 | 4 种属性 | 仅外观 + 物体间关系 |
| 数据规模 | 30,690 表达式 | 51,583 表达式 |
| 动态支持 | ✓（运动状态、时间推理） | ✗ |

## 评分

- ⭐⭐⭐⭐⭐ 新颖性：首次将语言定位引入事件相机领域，四属性设计原创性强
- ⭐⭐⭐⭐ 技术质量：MoEE 设计合理，消融实验充分，三种模态设置评估全面
- ⭐⭐⭐⭐ 实用价值：对自动驾驶和机器人领域的多模态感知有直接推动作用
- ⭐⭐⭐⭐ 表达清晰度：论文结构清晰，图表丰富，公式推导完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [\[NeurIPS 2025\] To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning](to_distill_or_decide_understanding_the_algorithmic_trade-off_in_partially_observ.md)
- [\[ICLR 2026\] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation](../../ICLR2026/robotics/all-day_multi-scenes_lifelong_vision-and-language_navigation_with_tucker_adaptat.md)
- [\[CVPR 2025\] RoboGround: Robotic Manipulation with Grounded Vision-Language Priors](../../CVPR2025/robotics/roboground_robotic_manipulation_with_grounded_vision-language_priors.md)

</div>

<!-- RELATED:END -->
