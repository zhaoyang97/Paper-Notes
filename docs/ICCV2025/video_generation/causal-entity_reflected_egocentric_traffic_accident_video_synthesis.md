---
title: >-
  [论文解读] Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis
description: >-
  [ICCV 2025][视频生成][交通事故视频合成] 本文提出Causal-VidSyn扩散模型，通过事故原因问答（ArA）模块和驾驶员注视条件的视觉token选择机制实现因果实体定位，并构建了包含154万帧注视数据的Drive-Gaze数据集，在事故视频编辑、正常到事故视频扩散、文本到视频生成三个任务中超越SOTA。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "交通事故视频合成"
  - "因果实体"
  - "驾驶员注视"
  - "扩散模型"
  - "自动驾驶安全"
---

# Causal-Entity Reflected Egocentric Traffic Accident Video Synthesis

**会议**: ICCV 2025  
**arXiv**: [2506.23263](https://arxiv.org/abs/2506.23263)  
**代码**: [项目主页](http://lotvsmmau.net/Causal-VidSyn)  
**领域**: 视频生成  
**关键词**: 交通事故视频合成, 因果实体, 驾驶员注视, 扩散模型, 自动驾驶安全

## 一句话总结
本文提出Causal-VidSyn扩散模型，通过事故原因问答（ArA）模块和驾驶员注视条件的视觉token选择机制实现因果实体定位，并构建了包含154万帧注视数据的Drive-Gaze数据集，在事故视频编辑、正常到事故视频扩散、文本到视频生成三个任务中超越SOTA。

## 研究背景与动机

1. **领域现状**: 视频扩散模型（CogVideoX, StoryDiffusion等）在通用视频生成上取得巨大进展，但主要面向正常场景，对第一人称交通事故视频的生成能力不足。
2. **现有痛点**: SOTA视频扩散模型在生成事故视频时无法准确识别因果实体和其事故相关行为。例如CogVideoX在文本编辑"行人→摩托车碰撞"时虽能生成摩托车但未体现碰撞，Abductive-OAVD则无法生成目标物体。
3. **核心矛盾**: 事故场景中因果实体通常很小、场景变化快速，在自车视角下极难识别目标物体及其细微行为。现有扩散模型缺乏领域知识来理解事故因果关系。
4. **本文目标**: (1) 如何在视频扩散中精确定位因果实体？(2) 如何让扩散模型理解事故因果关系来忠实响应反事实文本编辑？
5. **切入角度**: 融入两个关键信息线索——事故原因-碰撞文本描述（提供参与者和不当行为信息）和驾驶员注视点（提供直接视觉注意力线索）。
6. **核心 idea**: 通过事故原因问答和注视条件的token选择使3D-Unet骨干因果定位化，在扩散过程中精确识别和生成反映因果关系的事故实体。

## 方法详解

### 整体框架
Causal-VidSyn分为两个渐进层级：❶扩散配方层（Reciprocal Prompted Frame Diffusion, RPFD）——对比正向/反向时间顺序帧的扩散过程；❷知识层（CTS + CTG模块）——通过因果token选择和因果token定位将3D-Unet改造为因果感知的骨干网络。

### 关键设计

1. **互反提示帧扩散（RPFD）**:
    - 功能: 增强因果场景学习，通过正反时间顺序的对比干预
    - 核心思路: 前向路径用事故原因+碰撞描述文本$P_f$配合正序帧$V_f$；反向路径用事故预防建议文本$P_r$配合倒序帧$V_r$。对比两个噪声表示的学习: $\mathcal{L}_{ST1} = \mathcal{L}_{MSE}(e_f, \hat{e}_f) + \mathcal{L}_{MSE}(e_r, \hat{e}_r) + \lambda\mathcal{L}_{NS}(\hat{e}_f, \hat{e}_r)$。
    - 设计动机: 反向扩散可视为对正向文本/视觉提示的反事实干预，外生噪声$e$帮助因果场景关联互反的帧和文本提示。不同文本提示应激活主要与事故因果实体相关的不同视觉内容。

2. **因果倾向token选择（CTS）+ 因果token定位（CTG）**:
    - 功能: 在3D-Unet内部层注入CTS选择因果相关的视觉token，在末层通过CTG进行因果token定位
    - 核心思路: CTS利用驾驶员注视图（gaze map）与视觉token做注意力加权选择，过滤非因果区域；CTG设计事故原因问答（ArA）头，从多个候选事故原因中检索正确答案，并将其融入噪声表示学习指导因果定位。训练阶段使用ArA和驾驶员注视，推理阶段只需视频/文本提示。
    - 设计动机: 驾驶员基于驾驶经验能敏锐感知道路危险，其注视点提供直接的事故区域视觉注意力线索，弥合文本描述到视觉表现的模态鸿沟。ArA模块帮助模型检索正确的事故原因。

3. **Drive-Gaze数据集**:
    - 功能: 提供大规模驾驶事故场景下的驾驶员注视数据支持Causal-VidSyn训练
    - 核心思路: 基于MM-AU的11,727个事故视频，收集9,727个视频共计154万帧的驾驶员注视数据。10位被试（4女6男），使用Tobii Pro Fusion眼动追踪仪（250Hz）。将相似事故类型分组为长视频以让被试积累经验，每帧最终注视图由50×50高斯核卷积所有被试注视点得到。
    - 设计动机: 现有驾驶注视数据集（DADA-2000, BDD-A等）或规模小或仅覆盖正常驾驶场景，Drive-Gaze是最大的针对事故场景的驾驶员注视数据集，且包含事故原因和碰撞描述文本标注。

### 损失函数 / 训练策略
三阶段训练：Stage-0直接优化正向时序扩散（Stable Diffusion初始化）；Stage-1引入RPFD对比学习；Stage-2注入CTS和CTG模块进行因果定位化训练。ArA和注视仅参与训练，推理仅需视频/文本输入。

## 实验关键数据

### 主实验

| 任务 | 指标 | Causal-VidSyn | CogVideoX-2B | 提升 |
|--------|------|------|----------|------|
| 事故视频编辑 (DADA-2000) | FID↓ | 最优 | 较差 | 显著提升 |
| 事故视频编辑 | Causal Sensitivity↑ | 最优 | 碰撞不体现 | 大幅提升 |
| 正常→事故扩散 (BDD-A) | 帧质量 | 最优 | — | — |
| 文本到视频生成 | FVD↓ | 最优 | — | — |

CTS/CTG扩展到CogVideoX-2B和Latte后也取得一致显著提升。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full Causal-VidSyn | 最优 | 完整模型 |
| w/o RPFD (Stage-1) | 下降 | 缺少正反对比干预 |
| w/o CTS | 下降 | 没有注视引导的token选择 |
| w/o CTG | 下降 | 没有事故原因定位 |
| w/o Drive-Gaze | 下降 | 没有注视信号 |

### 关键发现
- RPFD通过正反时序对比显著增强了因果实体区域的激活
- 注视信号在训练时提供精准的因果区域先验，推理时无需但模型已内化该能力
- CTS和CTG模块可迁移到Transformer架构（CogVideoX, Latte），表明方法的通用性
- ArA模块对反事实编辑的忠实度至关重要

## 亮点与洞察
- 将因果推理引入视频扩散的独特视角：不是通用的因果发现，而是利用事故领域知识
- Drive-Gaze数据集（154万帧注视）的构建具有持久价值，可服务于多种事故理解任务
- 训练时用注视/推理时不用的设计巧妙：模型内化了因果注意力
- CTS/CTG的可迁移性：可插入不同架构的视频扩散模型

## 局限与展望
- 基于3D-Unet骨干的生成质量受限，未用最新的大规模DiT模型
- 注视数据收集成本高（10被试3个月），可探索自动化替代方案
- 仅关注生成侧的因果感知，未与下游事故理解任务（预测、责任判定）端到端结合
- 未评估生成视频在实际自动驾驶测试中的效用

## 相关工作与启发
- **vs Abductive-OAVD**: 后者仅关注文本到视频生成，不探索视频条件合成中的因果关系
- **vs CogVideoX**: 通用模型在事故场景下碰撞语义不到位，CTS/CTG可作为插件增强
- **vs 驾驶场景视频扩散（DriverDreamer等）**: 这些主要面向无事故场景的多视角一致性

## 评分
- 新颖性: ⭐⭐⭐⭐ 因果实体定位+驾驶注视引入视频扩散的独特视角
- 实验充分度: ⭐⭐⭐⭐ 三个任务+可迁移性验证+消融实验
- 写作质量: ⭐⭐⭐ 内容丰富但结构偏复杂
- 价值: ⭐⭐⭐⭐ Drive-Gaze数据集和因果感知扩散思路对自动驾驶安全有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[ECCV 2024\] DragAnything: Motion Control for Anything using Entity Representation](../../ECCV2024/video_generation/draganything_motion_control_for_anything_using_entity_representation.md)
- [\[ICCV 2025\] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation](fvgen_accelerating_novel-view_synthesis_with_adversarial_video_diffusion_distill.md)
- [\[CVPR 2025\] MIMO: Controllable Character Video Synthesis with Spatial Decomposed Modeling](../../CVPR2025/video_generation/mimo_controllable_character_video_synthesis_with_spatial_decomposed_modeling.md)
- [\[ICML 2025\] Ca2-VDM: Efficient Autoregressive Video Diffusion Model with Causal Generation and Cache Sharing](../../ICML2025/video_generation/ca2-vdm_efficient_autoregressive_video_diffusion_model_with_causal_generation_an.md)

</div>

<!-- RELATED:END -->
