---
title: >-
  [论文解读] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes
description: >-
  [ICCV 2025][HDR成像] 本文提出AdaptiveAE，利用深度强化学习将HDR曝光包围拍摄建模为马尔可夫决策过程（MDP），同时优化ISO和快门速度的组合，在用户定义的时间预算内自适应地为动态场景选择最优曝光参数，在HDRV数据集上达到PSNR 39.70，比之前最好的方法Hasinoff et al. (37.59) 高出2.1 dB。
tags:
  - "ICCV 2025"
  - "HDR成像"
  - "自动曝光"
  - "强化学习"
  - "运动模糊"
  - "曝光融合"
---

# AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes

**会议**: ICCV 2025  
**arXiv**: [2508.13503](https://arxiv.org/abs/2508.13503)  
**代码**: 无（未公开）  
**领域**: 其他  
**关键词**: HDR成像, 自动曝光, 强化学习, 运动模糊, 曝光融合  

## 一句话总结
本文提出AdaptiveAE，利用深度强化学习将HDR曝光包围拍摄建模为马尔可夫决策过程（MDP），同时优化ISO和快门速度的组合，在用户定义的时间预算内自适应地为动态场景选择最优曝光参数，在HDRV数据集上达到PSNR 39.70，比之前最好的方法Hasinoff et al. (37.59) 高出2.1 dB。

## 背景与动机
HDR成像的核心思路是将多张不同曝光的LDR图像融合成一张覆盖宽动态范围的HDR图像。这个过程中，曝光参数的选择至关重要：快门速度太长会引入运动模糊，ISO太高会引入噪声，曝光差异太大会增加对齐失败的风险。然而，现有方法存在几个关键不足：

1. **忽视ISO与快门速度的交互**：大多数方法只调整快门速度（EV），保持ISO固定，无法在噪声和模糊之间做最优权衡
2. **不考虑动态场景中的运动模糊**：Hasinoff et al. 等经典方法针对静态场景设计，只优化SNR而忽略运动导致的画质退化
3. **运动模糊和鬼影被当作后处理问题**：现有pipeline通常在融合后再做去模糊，但论文实验表明后处理去模糊效果很有限——一旦LDR拍糊了，再好的融合方法也无法挽回

## 核心问题
**如何在动态场景中自适应地选择ISO和快门速度的最优组合，使得融合后的HDR图像质量最高？** 这个问题的难点在于：(1) ISO和快门速度构成一个高维离散动作空间（24种ISO × 19种快门速度 = 456种组合）；(2) 最优策略依赖于场景内容——有快速运动物体的区域需要更短快门，暗区域需要更高ISO；(3) 每一帧的选择会影响后续帧的最优策略，具有序列决策的性质。

## 方法详解

### 整体框架
AdaptiveAE的pipeline分为两部分：**训练**和**推理**。

- **输入**：3张预览LDR图像（欠曝、正常曝光、过曝，EV间距{-2, 0, +2}）
- **输出**：每张LDR的最优ISO和快门速度组合

整个过程是一个3阶段的序列精炼过程：
1. **Stage 1**：预测中间帧（0 EV）的最优ISO和快门速度，侧帧（±2 EV）按对称方式自动调整
2. **Stage 2**：精炼欠曝帧的EV偏移为-y，中间帧继承上一轮参数，过曝帧对称调整为+y
3. **Stage 3**：精炼过曝帧的EV偏移为+z，允许打破对称性，最终曝光设置为{-y, 0, +z}

训练时，通过blur-aware数据合成pipeline模拟LDR图像；推理时，直接用相机拍摄。

### 关键设计

1. **Blur-aware数据合成Pipeline**：这是本文最重要的技术贡献之一。现有HDR数据集不包含运动模糊，因此无法训练考虑运动的曝光策略。本文设计了一个两步合成流程：

    - **运动模糊合成**：给定两帧连续HDR ground truth，先用μ-law tone mapping转到LDR空间，再用RIFE插帧到256帧，然后根据快门速度T_j选取对应数量的帧做平均得到模糊HDR（公式2）。关键洞察是模糊必须在噪声之前施加，因为模糊影响的是光子捕获的原始过程。
    - **噪声合成**：采用[Hasinoff 2010]的物理噪声模型，噪声方差由三个独立源组成——光子噪声（与信号强度和曝光时间成正比）、读出噪声和ADC噪声（公式3）。根据指定的ISO和快门速度即可精确合成对应噪声水平。

2. **MDP建模与A3C优化**：将曝光包围选择建模为MDP，状态是当前三张LDR的曝光设置，动作是为下一帧选择的(ISO, 快门速度)对。策略网络（actor）输出动作概率分布，价值网络（critic）估计状态值。使用A3C（Asynchronous Advantage Actor-Critic）进行端到端训练。动作空间离散化为24种ISO ×19种快门速度。

3. **多分支CNN网络架构**：

    - **语义特征分支**：用预训练AlexNet提取中间帧的语义特征（4096维→1024→256），帮助识别场景中重要区域
    - **辐照度特征分支**：对每张LDR分别提取直方图并拼接，通过3层1D卷积（128→256→512, kernel=4）处理曝光信息
    - **阶段编码分支**：编码当前阶段号和总阶段数（2维→32→64），让网络根据剩余曝光预算调整策略
    - **特征融合**：所有分支特征拼接后通过两层全连接（512→256）融合

4. **精心设计的奖励函数**（公式5-7）：

    - **构建奖励 P_construction**：融合HDR与ground truth的L2损失（主要奖励项）
    - **优先区域奖励 P_priority**：通过显著性预测器[SalGAN]生成重要区域掩码，对这些区域额外施加L2约束，确保人脸等关键区域画质最高
    - **鬼影奖励 P_ghost**：用RAFT计算光流，选取运动超过阈值K=0.2的像素区域，对这些高风险区域额外施加L2约束
    - **步长惩罚 P(j)**：超过H=3帧时施加α(j-H)²的惩罚，鼓励用尽量少的帧完成高质量HDR

### 损失函数 / 训练策略
- 训练使用A3C异步优化，融合网络使用DeepHDR（训练时用于计算奖励，推理时不需要）
- 训练数据：Real-HDRV的770个场景（440动态 + 330静态），裁剪为512×512，使用随机翻转、旋转做增强
- RIFE插帧较耗时，blur合成在训练前离线完成
- 推理时RL agent在RTX3080上<5ms/场景，整体流程<250ms

## 实验关键数据

| 数据集 | 指标 | 本文 | Hasinoff et al. | Wang et al. | Pourreza et al. | 提升(vs Hasinoff) |
|--------|------|------|----------|------|------|------|
| HDRV-Test (1 preview) | PSNR-μ | **39.70** | 37.59 | 36.46 | 33.64 | +2.11 |
| HDRV-Test (1 preview) | SSIM-μ | **0.9408** | 0.9052 | 0.8902 | 0.8617 | +0.036 |
| HDRV-Test (1 preview) | HDR-VDP-2 | **59.20** | 57.02 | 56.09 | 54.55 | +2.18 |
| HDRV-Test (1 preview) | PU-PSNR | **34.67** | 32.87 | 32.68 | 30.61 | +1.80 |
| DeepHDRVideo (3 preview) | PSNR-μ | **39.81** | 38.47 | 37.95 | 35.57 | +1.34 |

跨融合方法测试（HDRV-Test, 1 preview）：

| 融合方法 | 本文 PSNR-μ | Hasinoff PSNR-μ | Wang PSNR-μ |
|----------|-------------|-----------------|-------------|
| DeepHDR | **39.70** | 37.59 | 36.46 |
| HDR-GAN | **40.73** | 38.58 | 37.95 |
| HDR-Transformer | **41.37** | 39.11 | 38.89 |

接近最优解：通过高斯采样搜索局部最优（50次/参数/帧），本文方法PSNR 39.70 vs 局部最优39.93，几乎接近理论上限。

### 消融实验要点
- **Base**（仅构建奖励+步长惩罚）：PSNR 38.21 / SSIM 0.9227
- **Base + P_priority**：PSNR 38.57 / SSIM 0.9261（+0.36 dB）
- **Base + P_priority + P_ghost**（完整模型）：PSNR **39.70** / SSIM **0.9408**（+1.49 dB）
- P_ghost贡献最大（+1.13 dB），验证了在曝光阶段考虑运动的重要性
- 后处理去模糊（BANet前处理/后处理/融合阶段）最多只能将Wang et al.从36.46提升到37.33，远低于本文39.70——说明运动问题必须在拍摄阶段解决
- 固定ISO的最优选择（W-optimal）只能小幅提升到37.64，证实了自适应ISO的必要性
- 运动越大优势越明显：在60像素运动级别下，本文方法的PSNR优势更加显著

## 亮点
- **从源头解决运动问题**的思路非常直觉且有效——与其拍糊了再后处理，不如一开始就拍好。这个在拍摄阶段就同时优化ISO和快门速度的思路，在曝光策略领域是首次
- **物理驱动的数据合成pipeline**设计巧妙：先模糊后加噪的顺序符合物理规律，噪声模型基于光子统计，使得训练数据足够真实
- **灵活的帧数**：通过步长惩罚机制，模型能自动决定拍3帧还是4帧，不像传统方法固定3帧
- **实际设备验证**：在Sony Alpha 7C-II上通过手动设置参数做了真实拍摄测试，不只是仿真实验
- **模型极轻量**：RL agent仅7-8M参数，推理<5ms，完整pipeline<250ms，具有实时部署潜力

## 局限与展望
- **光圈固定**：当前假设光圈和焦距不变，论文结尾提到未来要加入可调光圈
- **依赖预训练融合网络**：训练时使用DeepHDR计算奖励，策略的最优性受限于融合网络质量。如果换用更强的融合网络，策略可能需要重新训练
- **离线blur合成**：RIFE插帧在训练前离线完成，这限制了训练的灵活性
- **真实世界验证有限**：实际拍摄测试需要手动设置ISO和快门速度，还未做到端到端的相机集成
- **语义分支使用AlexNet**：特征提取能力有限，可以用更强的视觉基础模型替代
- **动作空间离散化**：24×19=456个离散动作，可能遗漏连续空间中的最优解
- **单一参考帧**：始终用中间帧作为融合参考，对于极端动态场景可能不是最优选择

## 与相关工作的对比
- **vs Hasinoff et al. [2010]**：Hasinoff通过数学优化ISO和快门速度使worst-case SNR最优，但假设静态场景，不考虑运动模糊。本文在动态场景上PSNR高2.1 dB。
- **vs Wang et al. [2020]**：Wang et al.也用RL预测曝光，但①只预测快门速度不改ISO，②不考虑运动模糊。本文通过同时优化ISO和快门速度+运动感知奖励，PSNR高3.24 dB。
- **vs 后处理去模糊pipeline**：将Wang et al.与BANet/DeepHDR-blur结合，最佳也只到37.33 dB（vs 本文39.70），证明后处理无法替代拍摄优化。

## 启发与关联
- **RL用于相机控制**的范式可以推广到其他成像任务：如多光谱成像的波段选择、低光摄影的降噪策略、手机HDR+的帧选择
- **物理感知的数据合成**思路值得借鉴：通过精确建模物理过程（模糊+噪声）来生成训练数据，避免了大规模真实标注的困难
- **序列决策的思路**可以迁移到视频压缩中的码率分配、自动驾驶中的多帧融合策略等场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在曝光策略中同时优化ISO和快门速度并考虑运动模糊，但RL+MDP的框架本身已在Wang et al. 2020中出现
- 实验充分度: ⭐⭐⭐⭐⭐ 多数据集、多融合方法、vs后处理去模糊、vs最优ISO、vs局部最优解、运动量级分析、真实拍摄，消融非常详细
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题动机阐述充分，公式推导完整，但部分符号较多需要反复对照
- 价值: ⭐⭐⭐⭐ 实用价值高——拍摄端优化的思路比后处理更根本，有向手机/相机部署的潜力，但需要硬件集成才能真正落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](../../ACL2025/others/sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[ICCV 2025\] Recovering Parametric Scenes from Very Few Time-of-Flight Pixels](recovering_parametric_scenes_from_very_few_time-of-flight_pixels.md)
- [\[ACL 2025\] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](../../ACL2025/others/inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](../../ACL2025/others/fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)

</div>

<!-- RELATED:END -->
