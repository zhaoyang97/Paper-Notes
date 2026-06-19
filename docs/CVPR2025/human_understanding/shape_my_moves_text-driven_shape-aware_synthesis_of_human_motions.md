---
title: >-
  [论文解读] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions
description: >-
  [CVPR 2025][人体理解][人体动作生成] 本文提出 ShapeMove 框架，通过 Shape-Aware FSQ-VAE 将连续体型信息注入离散量化的动作 token，并利用预训练语言模型同时预测体型参数和动作 token，实现了首个从自然语言描述端到端生成体型感知动作的方法。 领域现状：文本到动作生成已取得显著…
tags:
  - "CVPR 2025"
  - "人体理解"
  - "人体动作生成"
  - "体型感知"
  - "文本驱动"
  - "量化自编码器"
  - "大语言模型"
---

# Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions

**会议**: CVPR 2025  
**arXiv**: [2504.03639](https://arxiv.org/abs/2504.03639)  
**代码**: [https://shape-move.github.io/](https://shape-move.github.io/)  
**领域**: 人体理解  
**关键词**: 人体动作生成, 体型感知, 文本驱动, 量化自编码器, 大语言模型

## 一句话总结

本文提出 ShapeMove 框架，通过 Shape-Aware FSQ-VAE 将连续体型信息注入离散量化的动作 token，并利用预训练语言模型同时预测体型参数和动作 token，实现了首个从自然语言描述端到端生成体型感知动作的方法。

## 研究背景与动机

**领域现状**：文本到动作生成已取得显著进展，主流方法（T2M-GPT、MDM、MotionDiffuse）将动作表示标准化到规范人体模型上——所有不同体型的人执行相同动作时被映射为完全一致的运动序列。

**现有痛点**：现实中不同体型的人执行同一动作存在显著的生理差异——胖人和瘦人跑步的步态、臂展、重心转移都不同。现有方法忽略了这些差异，生成的"一刀切"动作在迁移到不同体型时会产生自穿透、足部滑移、关节不自然弯曲等伪影。最根本的问题是：连续的体型信息和离散的动作量化表示之间存在表示鸠合困难。

**核心矛盾**：动作量化（VQ-VAE）通过离散化大幅提升了生成质量和效率，但离散 codebook 容量不足以编码体型变化带来的细粒度差异。直接用体型感知的动作训练量化器会导致 codebook 急剧膨胀或利用率低下。

**本文目标**：构建一个端到端框架，从自然语言同时预测体型参数和体型感知的动作序列。

**切入角度**：作者提出"内容-风格分离"策略——用标准化动作训练离散量化器（学习运动内容），再用连续体型参数作为条件在解码阶段注入体型"风格"，从而既利用了量化器的高效性，又保留了连续体型信息。

**核心 idea**：在量化网络中将运动内容（离散 token）和个体风格（连续体型参数）解耦，通过 FSQ 量化标准动作 + 条件解码恢复体型感知动作。

## 方法详解

### 整体框架

系统分为两个阶段：Stage 1 是 Shape-Aware FSQ-VAE（SA-VAE），将标准化动作量化为离散 token，并在解码时注入体型参数恢复体型感知动作；Stage 2 是 ShapeMove，利用预训练的 T5 语言模型从文本描述预测体型参数和动作 token 序列。推理时，T5 输出的动作 token 经 FSQ 反量化后与预测的体型参数一起送入 SA-VAE 解码器生成最终动作。

### 关键设计

1. **Shape-Aware FSQ-VAE (SA-VAE)**:

    - 功能：在动作量化过程中引入连续体型信息，实现内容-风格解耦的动作表示
    - 核心思路：编码器 $\mathcal{E}$ 接收标准化动作 $X^N \in \mathbb{R}^{T \times D}$（D=263），编码为特征 $Z \in \mathbb{R}^{\tau \times D}$（$\tau$ 为下采样后的长度）。使用 Finite Scalar Quantization (FSQ) 量化 $Z$ 为离散特征 $\hat{Z}$（codebook 大小 k=1000，维度配置 $\ell=[8,5,5,5]$）。解码器 $\mathcal{D}$ 接收 $\hat{Z}$ 与通过 MLP 投影的体型特征 $\tilde{\beta} = P_{\theta_s}(\beta)$ 在时间维度上的拼接，解码出体型感知动作 $\hat{X}^R = \mathcal{D}(\hat{Z} \oplus \tilde{\beta})$。关键在于编码器只编码标准化动作（无体型信息），保证 codebook 只学运动语义
    - 设计动机：直接量化体型感知动作需要指数级更大的 codebook 才能覆盖所有体型变化；通过编码标准动作+条件解码的分离策略，1000 个 code 即可获得优于 T2M-GPT 的重建质量

2. **ShapeMove（体型-动作 Token 预测器）**:

    - 功能：从自然语言描述同时预测体型参数和动作 token 序列
    - 核心思路：在预训练 T5 模型的词表中新增 k+2 个动作词汇（k 个 codebook code + 起止标记）和 1 个体型标记 [BETA]。模型以文本为输入，自回归预测 {[BETA], $\hat{C}$}。体型参数通过提取 [BETA] 对应的嵌入 $M_\beta$ 并投影获得 $\hat{\beta} = P_{\theta_e}(M_\beta)$，利用了语言模型最后一层嵌入包含连续信息的特性。训练目标为动作 token 的 cross-entropy 损失 + 体型参数的 L1 损失
    - 设计动机：体型参数是固定值而非序列，无法像动作那样用 token 序列表示；通过嵌入空间提取连续信息是一种优雅的"搭便车"策略，无需为体型设计单独的离散化方案

3. **物理约束损失与体型数据增强**:

    - 功能：保证生成动作的物理合理性，并扩展训练数据中体型的多样性
    - 核心思路：SA-VAE 的损失函数包含重建损失 $L_r$、浮空损失 $L_{\text{float}}$（最低关节离地距离）、足部滑移损失 $L_{\text{slide}}$（接地脚的地面速度）和骨骼长度损失 $L_{\text{bone}}$（重建与真实的骨骼长度偏差）。数据增强使用 Shapy 的 A2S 模型从语言属性生成合成体型参数，替换 10% 训练样本的真实体型
    - 设计动机：量化引入的信息损失可能破坏物理合理性（自穿透、足部浮空/滑移），物理损失提供显式约束；现有数据集体型多样性有限，合成增强扩展了模型的泛化范围

### 损失函数 / 训练策略

SA-VAE 训练 300K 迭代（200K lr=2e-4 + 100K lr=1e-5），batch size 256，单卡 A100 约 12 小时。ShapeMove 使用 T5 模型（编码器和解码器各 12 层），先在动作-文本和文本-动作双任务上训练 120K 步（8×A100，1 天），再仅在文本-动作任务上微调 30K 步（10 小时）。动作序列裁剪长度 T=64，下采样后 τ=16。

## 实验关键数据

### 主实验

| 方法 | Shape Input | Penetrate↓(cm) | Float↓(cm) | Skate↓(%) | Bone Length Var↓ | FID↓ | R-Precision Top3↑ |
|------|------------|-----------------|------------|-----------|-----------------|------|-------------------|
| T2M-GPT | ✓ | 0.1789 | 0.5241 | 6.162 | 1.176 | 0.269 | 0.683 |
| MotionGPT | ✓ | 0.6986 | 0.2245 | 7.889 | 2.271 | 1.020 | 0.271 |
| MotionDiffuse | ✓ | 0.2401 | 0.2703 | 7.710 | 0.138 | 0.563 | 0.723 |
| **ShapeMove** | **✓** | **0.0268** | 0.2658 | **6.143** | **0.625** | **0.198** | **0.705** |

### 消融实验

| 配置 | FID↓ | Bone Length Diff↓(mm) | Float↓(cm) | Skate↓(%) |
|------|------|----------------------|-----------|-----------|
| No shape-conditioning | 0.148 | 99.18 | 0.575 | 6.76 |
| + shape-conditioning | 0.105 | 66.41 | 0.567 | 6.60 |
| + $L_{\text{bone}}$ | 0.107 | 45.11 | 0.480 | 7.07 |
| + $L_{\text{bone}} + L_{\text{float}}$ | 0.137 | 45.97 | 0.255 | 7.90 |
| + Full (all losses) | 0.125 | 45.88 | 0.266 | 6.14 |

### 关键发现

- ShapeMove 在自穿透指标（Penetrate）上以 0.0268cm 大幅领先 T2M-GPT 的 0.1789cm（提升 6.7 倍），接近重建上界 SA-VAE 的 0.0289cm，说明体型感知生成几乎消除了地面穿透
- 体型条件注入将骨骼长度误差从 99.18mm 降至 66.41mm，加入 $L_{\text{bone}}$ 后进一步降至 45.11mm（减半），证明了分离式设计和物理损失的有效性
- 量化器重建对比中，SA-VAE 的骨骼长度差异为 45.88mm，是 T2M-GPT (83.42mm) 的约一半，FID 也最低（0.125 vs 0.151），证明体型条件解码的优越性
- 人类感知评估中，ShapeMove 在体型-文本匹配、动作-文本匹配和体型动作合理性三个维度上都接近真实数据水平，领先所有基线 12%-38%
- 体型属性预测误差约 1cm（身高 0.58cm，腰围 1.06cm），说明语言模型嵌入能有效编码连续体型信息

## 亮点与洞察

- **内容-风格解耦的量化策略**非常巧妙——编码器只看标准化动作保证 codebook 高利用率，解码器通过体型条件注入恢复个体差异。这种思路可迁移到任何"共同结构+个体变化"的生成任务（如说话人风格的语音合成）
- **利用语言模型嵌入提取连续信息**是一个优雅的 hack——体型参数无法离散化，但语言模型的嵌入空间天然携带连续信息。这种"从离散模型中挖掘连续信号"的技巧在多模态生成中有广泛潜力
- FSQ 相比 VQ-VAE 省去了 codebook 正则化和码字坍缩问题，是量化方案的更优选择

## 局限与展望

- 仅在 HumanML3D 数据集上验证，该数据集体型多样性有限（449 个被试），合成增强只覆盖 10%
- 依赖 SMPL 模型的 β 参数表示体型，无法捕捉超出 SMPL 表达范围的体型变异
- 物理合理性约束是几何层面的（浮空、滑移、穿透），未引入动力学约束（如惯性、肌肉力）
- 文本描述中的体型描述较为粗粒度（身高、体型描述语），对体型的控制精度受限
- 未探索与真实物理模拟器的结合来进一步提升物理真实性

## 相关工作与启发

- **vs T2M-GPT**: 同样使用量化器 + Transformer 预测 token 序列，但 T2M-GPT 量化标准化动作后无法恢复体型信息；ShapeMove 的 SA-VAE 在解码端注入体型实现了质的飞跃
- **vs MDM / MotionDiffuse**: 扩散方法可以接受连续条件输入（包括体型），但在 Penetrate 和 FID 上都显著劣于 ShapeMove，说明扩散模型在物理约束学习上不如量化+解码方案
- **vs HUMOS**: HUMOS 也做体型感知动作生成但侧重物理仿真层面，ShapeMove 则从学习和量化角度解决问题，两者互补

## 评分

- 新颖性: ⭐⭐⭐⭐ 内容-风格解耦量化和嵌入提取连续信息是有创意的设计，但整体框架是组合式的
- 实验充分度: ⭐⭐⭐⭐⭐ 定量对比、消融、量化器对比、属性预测、人类感知评估全面覆盖
- 写作质量: ⭐⭐⭐⭐ 方法讲解清晰，图示丰富，但 Related Work 部分过长
- 价值: ⭐⭐⭐⭐ 填补了文本到体型感知动作生成的空白，物理合理性提升显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](../../ECCV2024/human_understanding/humos_human_motion_model_conditioned_on_body_shape.md)
- [\[CVPR 2025\] StickMotion: Generating 3D Human Motions by Drawing a Stickman](stickmotion_generating_3d_human_motions_by_drawing_a_stickman.md)
- [\[AAAI 2026\] Generating Attribute-Aware Human Motions from Textual Prompt](../../AAAI2026/human_understanding/generating_attribute-aware_human_motions_from_textual_prompt.md)
- [\[CVPR 2026\] Towards Storytelling Animations: Joint Synthesis of Human and Camera Motions](../../CVPR2026/human_understanding/towards_storytelling_animations_joint_synthesis_of_human_and_camera_motions.md)

</div>

<!-- RELATED:END -->
