---
title: >-
  [论文解读] Deforming Videos to Masks: Flow Matching for Referring Video Segmentation
description: >-
  [ICLR 2026][语义分割][RVOS] 把指代视频目标分割（RVOS）重新定义为「在文本引导下、把视频隐空间表征连续形变成掩码」的 ODE 流问题，直接微调预训练文生视频（T2V）模型 Wan2.1，用三个聚焦轨迹起点的策略稳住训练，在 MeViS、Ref-YouTube-VOS、Ref-DAVIS17 上全面刷到 SOTA。
tags:
  - "ICLR 2026"
  - "语义分割"
  - "RVOS"
  - "Flow Matching"
  - "文生视频模型"
  - "端到端分割"
  - "时序一致性"
---

# Deforming Videos to Masks: Flow Matching for Referring Video Segmentation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3KaIcArMAB](https://openreview.net/forum?id=3KaIcArMAB)  
**代码**: [https://github.com/xmz111/FlowRVS](https://github.com/xmz111/FlowRVS)  
**领域**: 指代视频目标分割 / 生成式分割 / Flow Matching  
**关键词**: RVOS, Flow Matching, 文生视频模型, 端到端分割, 时序一致性  

## 一句话总结
把指代视频目标分割（RVOS）重新定义为「在文本引导下、把视频隐空间表征连续形变成掩码」的 ODE 流问题，直接微调预训练文生视频（T2V）模型 Wan2.1，用三个聚焦轨迹起点的策略稳住训练，在 MeViS、Ref-YouTube-VOS、Ref-DAVIS17 上全面刷到 SOTA。

## 研究背景与动机
**领域现状**：RVOS 要根据一句自然语言描述，在视频里把对应物体逐帧分割出来，核心难点是把抽象的语言概念锚定到动态、细粒度的像素空间。主流方法走「先定位再分割」（locate-then-segment）的级联管线：先用 query-based（如 ReferFormer）或 VLM-based（如 LISA、VISA、ReferDINO）模型把文本 grounding 成一个粗糙的几何提示（框/点），再交给独立的分割器逐帧出掩码。

**现有痛点**：这种级联设计有两个结构性缺陷。其一是**信息瓶颈**——把丰富的语义压缩成框/点这类粗几何中间表征，整体场景理解被砍掉；其二是**时序割裂**——每帧的分割虽受条件约束，却不源自一个统一的时空形变过程，导致时序一致性差。即便有人把 T2V 模型当作冻结的特征提取器（VD-IT、HCD），也是两阶段设计，生成模型的动态能力和最终任务被解耦，独立 decoder 还得从时序孤立的特征里重建时序关系，白白浪费了 T2V 自带的视频连贯性。

**核心矛盾**：T2V 生成是一个**发散**过程——从简单噪声先验映射到一组可能的视频，探索空间很宽；而 RVOS 恰恰是一个**收敛**任务——必须把一个高熵、复杂的视频映射到唯一、低熵的掩码。这是一次确定性的、受引导的信息收缩，文本 query 在这里不是创意 prompt，而是从丰富视觉输入里精确锁定目标的「选择器」（比如把"小猴子"和"大猴子"区分开）。直接把 T2V 范式照搬过来必然水土不服。

**本文目标**：把整个生成过程本身改造成判别任务，学一条从视频像素到掩码的、语言引导的连续形变流，彻底绕开级联管线的信息瓶颈。

**核心 idea**：**【范式重构】** 不再「从噪声生成掩码」也不「单步直接预测掩码」，而是学一个速度场 $v(z_t,c,t)$，把视频隐表征 $z_0$ 沿 ODE 路径连续形变到掩码隐表征 $z_1$；**【起点强化】** 针对收敛任务"第一步最关键、错了就无法挽回"的非对称性，用三个协同策略专门加固轨迹起点。

## 方法详解

### 整体框架
FlowRVS 建立在 Wan2.1（1.3B 参数 DiT）之上，把 RVOS 形式化为一个文本条件的连续流：训练阶段冻结文本编码器和 VAE 编码器，只微调 DiT 学习速度场，并用 boundary-biased 时间采样稳住训练；推理阶段用 ODE solver 把视频隐表征确定性地形变到目标掩码隐表征，再经过专门微调的 VAE decoder 还原成像素级二值掩码。

```mermaid
flowchart LR
    V[视频帧] -->|VAE Encoder 冻结| Z0[视频隐表征 z0]
    T[文本 query c] -->|Text Encoder 冻结| C[文本 embedding]
    Z0 --> FM[DiT 速度场 v=fθ]
    C --> FM
    Z0 -.->|DVI 通道拼接| FM
    FM -->|ODE Sampling| Z1[掩码隐表征 z1]
    Z1 -->|VAE Decoder 微调| M[二值掩码序列]
```

### 关键设计

**1. RVOS 即收敛流：从噪声/单步预测改成视频到掩码的多步形变。** 传统 RVOS 当作一步判别映射 $M=f_\theta(V,c)$，需要在单次变换里把高维动态视频塌缩成精确掩码，本质病态。本文改为学习一个由 ODE 支配的连续形变：$\frac{dz_t}{dt}=v(z_t,c,t)$，边界条件 $z_0\sim P_{video}$、$z_1\sim P_{mask}$。这样学习目标从"掌握一个复杂的全局函数"降级为"学一个简单的局部速度场"，文本 query $c$ 在每一步充当消歧力。消融里这一改动效果立竿见影：把目标从"预测绝对状态"换成"预测残差速度"，单步速度预测相比单步掩码预测直接 +14.6 J&F，强力佐证了流式重构的合理性。

**2. Boundary-Biased Sampling（BBS）：把梯度火力集中到轨迹起点。** 由于视频到掩码的形变是非对称的——起点高确定性、结构化，终点低确定性、稀疏——均匀采样时间步会浪费在不关键的区域。BBS 是一种课程学习策略，过采样 $t=0$ 附近的时间步，强迫模型先把"基于文本 query 算出的初始推力"学准。这是稳住整条多步流的命门：基础流（均匀采样）只有 47.9 J&F，甚至比单步速度预测还差；加上 $p=0.5$ 的 BBS 直接飙到 57.9（+10.0），证明掌握初始的文本引导速度是成败最关键的因素。

**3. Start-Point Augmentation（SPA）+ Direct Video Injection（DVI）：从两侧加固起点。** SPA 在训练时对初始视频隐表征 $z_0$ 做随机编码与归一化扰动，给模型呈现一个围绕原始隐表征、局部连续的起点分布，作为正则化器，逼模型学一个不仅在流形上、也在其邻域内都鲁棒的速度场。DVI 则把原始视频隐表征 $z_0$ 沿通道维和当前状态 $z_t$ 拼接，使每一步的速度预测从 $v(z_t,t)$ 变成 $v([z_t,z_0],t)$，让全局源上下文在整条收缩轨迹里始终可达，防止轨迹漂移、提升细粒度精度，且几乎零额外开销。两者叠加在 BBS 基础上分别再贡献增益，DVI 单独 +2.0 J&F。

**4. 任务专属 VAE 解码器微调：弥合连续视频隐空间与二值掩码的域差。** 预训练 VAE 面向自然视频，直接拿来重建二值掩码会有域差。本文冻结 VAE 编码器，单独在 MeViS 训练集上微调 VAE 解码器，使其专门擅长从隐空间还原高质量掩码。实验显示冻结解码器已能支撑有竞争力的性能（60.0 J&F），微调解码器进一步提升重建质量并带来 +0.9 J&F。

## 实验关键数据

### 主实验表格
在三大 RVOS 基准上对比「locate-then-segment」方法（J&F，越高越好）：

| 方法 | 范式 | MeViS J&F | Ref-YT-VOS J&F | Ref-DAVIS17 J&F |
|------|------|-----------|----------------|-----------------|
| ReferFormer [CVPR'22] | locate-then-seg | 31.0 | 62.9 | 61.1 |
| VISA [ECCV'24] | VLM-based | 43.5 | 61.5 | 69.4 |
| SAMWISE [CVPR'25] | VLM+SAM | 49.5 | 69.2 | 70.6 |
| ReferDINO [ICCV'25] | grounding-based | 49.3 | 69.3 | 68.9 |
| **FlowRVS (ours)** | **一阶段生成式** | **51.1** | **69.6** | **73.3** |

相比前 SOTA：MeViS +1.6、Ref-DAVIS17（零样本）+2.7；对 VISA-13B 在 MeViS 上领先 7.0 点。

### 消融实验表格
在 MeViS validu 集上逐项消融（J&F）：

| 配置 | BBS(p) | SPA | DVI | WI | J&F |
|------|--------|-----|-----|----|----|
| (a) 多步 噪声→掩码 流 | – | – | ✓ | ✓ | 32.3 |
| (b) 单步 掩码预测 | – | – | – | ✓ | 38.9 |
| (c) 单步 速度预测 | – | – | – | ✓ | 50.8 |
| Base Flow（均匀采样） | 0.0 | – | – | ✓ | 47.9 |
| + BBS | 0.25 | – | – | ✓ | 55.2 |
| + BBS | 0.50 | – | – | ✓ | 57.9 |
| + SPA | 0.50 | ✓ | – | ✓ | 58.6 |
| + DVI（最终默认） | 0.50 | ✓ | ✓ | ✓ | **60.6** |
| − WI（从零训练） | 0.50 | ✓ | ✓ | ✗ | 21.1 |

### 关键发现
- **残差速度 > 绝对状态**：单步速度预测 (50.8) 比单步掩码预测 (38.9) 高 +14.6，先验证了流式目标更稳。
- **多步流必须被稳住**：朴素均匀采样的多步基础流 (47.9) 甚至不如单步速度预测，BBS 是把多步流潜力解锁的关键，$p=0.5$ 时增益最大（+10.0）。
- **预训练权重是命脉**：去掉 Wan 初始化从零训练直接崩到 21.1，说明这套方法是专门为"驾驭并适配生成基础模型先验"设计的，而非通用训练 trick。
- **零样本泛化强**：在 Ref-DAVIS17 上不做任何微调即达 73.3 J&F，超过许多专门在该类高质量数据上训练过的方法。

## 亮点与洞察
- **范式层面的洞察清晰有力**：把"生成是发散、判别是收敛"这一矛盾点破得很透，并据此推出"第一步最关键、错了不可挽回"，所有设计（尤其 BBS）都服务于这条主线，逻辑自洽。
- **三个适配策略不是孤立 trick**：BBS（采样侧）、SPA（起点扰动）、DVI（持续注入）围绕"加固流的起点"这一原则协同，消融能逐项看到贡献。
- **充分用上了 T2V 的三大原生能力**：像素级合成→细粒度控制、文本条件生成→多模态对齐、视频原生架构→时空推理，比把 T2V 当冻结特征提取器的两阶段方案更彻底。
- **对替代范式的分析很扎实**：在同一 Wan2.1、同一 VAE、同一训练设置下公平对比"直接掩码预测 / 噪声→掩码流 / 单步速度预测"三种方案，逐一说明为何失败。

## 局限与展望
- **依赖大规模 T2V 预训练**：去掉权重初始化即崩溃，方法的有效性高度绑定 Wan2.1 这类基础模型的先验，迁移到没有强 T2V 预训练的场景前景不明。
- **推理需多步 ODE 采样**：相比一步预测，多步形变带来推理开销，论文未深入讨论速度/延迟权衡，实时应用需评估。
- **VAE 解码器需按数据集单独微调**：掩码重建解码器在 MeViS 上单独训练，跨数据集是否需重新适配、对二值掩码以外的标注（如实例/多类）是否同样有效仍待验证。
- **正文部分数字略有不一致**（如 MeViS 50.7 vs 表 1 的 51.1），属写作细节，但提示结果汇报需对齐。

## 相关工作与启发
- **生成式分割的延续**：与把 T2V 当冻结特征提取器的 VD-IT、HCD 形成对照，也区别于直接一步预测的并行工作 ReferEverything（REM）——本文强调"微调整个生成过程学形变流"。
- **Flow Matching 的文本化**：DepthFM 等把生成过程改造成视觉到视觉任务（深度估计），但通常 text-agnostic；本文把自然语言 query 作为调制整条 ODE 路径的核心条件力，是关键区别。
- **对 ControlNet 式条件生成的反思**：ControlNet 给"发散的噪声→图像"过程加外部引导，而本文学的是"从视频源本身出发的收敛、判别式变换"。
- **启发**：把"判别任务重构成受文本条件约束的收敛流，并针对起点做非对称强化"这一思路，或可推广到其他需要把高熵输入塌缩成结构化输出的视频理解任务（如视频实例分割、动作定位）。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把 RVOS 重构为文本条件的视频到掩码连续形变流，是对生成式范式用于判别任务的一次有原则的反转，"发散 vs 收敛"的洞察新颖且自洽。
- **实验充分度**: ⭐⭐⭐⭐ 三大基准全面 SOTA、零样本泛化亮眼，消融逐项拆解三策略与替代范式；略欠推理开销/速度分析与跨任务验证。
- **写作质量**: ⭐⭐⭐⭐ 动机与范式分析讲得透彻、图示清晰；个别数字前后不一致是小瑕疵。
- **价值**: ⭐⭐⭐⭐⭐ 为"如何把 T2V 生成基础模型适配到判别式视频理解"给出了可复用的方法论与强 baseline，对生成式分割方向有较大推动。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching](../../CVPR2026/segmentation/flowdis_language-guided_dichotomous_image_segmentation_with_flow_matching.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[CVPR 2025\] LiVOS: Light Video Object Segmentation with Gated Linear Matching](../../CVPR2025/segmentation/livos_light_video_object_segmentation_with_gated_linear_matching.md)
- [\[ICLR 2026\] VINCIE: Unlocking In-context Image Editing from Video](vincie_unlocking_in-context_image_editing_from_video.md)
- [\[ICLR 2026\] Matting Anything 2: Towards Video Matting for Anything](matting_anything_2_towards_video_matting_for_anything.md)

</div>

<!-- RELATED:END -->
