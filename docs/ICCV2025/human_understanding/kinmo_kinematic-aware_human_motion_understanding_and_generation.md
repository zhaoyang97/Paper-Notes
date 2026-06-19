---
title: >-
  [论文解读] KinMo: Kinematic-Aware Human Motion Understanding and Generation
description: >-
  [ICCV 2025][人体理解][人体运动生成] 提出 KinMo 框架，将人体运动分解为六大运动学组及其交互的层级可描述表示，通过自动标注管线生成细粒度文本描述，结合层级文本-运动对齐和由粗到细的运动生成策略，显著提升运动理解和细粒度运动生成能力。 当前文本驱动的人体运动生成方法依赖全局动作描述（如"run"）…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "人体运动生成"
  - "文本-运动对齐"
  - "运动学分组"
  - "层级表示"
  - "细粒度控制"
---

# KinMo: Kinematic-Aware Human Motion Understanding and Generation

**会议**: ICCV 2025  
**arXiv**: [2411.15472](https://arxiv.org/abs/2411.15472)  
**代码**: [https://andypinxinliu.github.io/KinMo](https://andypinxinliu.github.io/KinMo)  
**领域**: 人体理解  
**关键词**: 人体运动生成, 文本-运动对齐, 运动学分组, 层级表示, 细粒度控制

## 一句话总结

提出 KinMo 框架，将人体运动分解为六大运动学组及其交互的层级可描述表示，通过自动标注管线生成细粒度文本描述，结合层级文本-运动对齐和由粗到细的运动生成策略，显著提升运动理解和细粒度运动生成能力。

## 研究背景与动机

当前文本驱动的人体运动生成方法依赖全局动作描述（如"run"），存在根本性的**模态鸠**问题：

**多对多映射歧义**：同一运动可用多种文本描述（"pick up an object" vs "bend down to reach something"），同一文本也可对应多种运动变体（"running"可以是快跑/慢跑/抬臂跑等）

**局部运动不可控**：现有模型擅长从全局描述生成连贯全身运动，但无法独立控制特定身体部位。例如无法指定"走路时右手向上举"

**空间细节缺失**：全局描述无法捕获速度、肢体定位、运动学动力学等细节

现有方法（如 LGTM、FG-MDM）虽然尝试用 LLM 生成补充文本，但这些文本是任意的，缺乏基于运动学的系统化formulation。

## 方法详解

### 整体框架

KinMo 框架包含四个部分：(1) 可描述运动表示：将运动分解为六大运动学组及其交互；(2) KinMo 数据集：通过半监督标注管线生成三层文本描述；(3) 层级文本-运动对齐（HTMA）：逐级编码文本并与运动对齐；(4) 由粗到细运动生成：利用层级嵌入条件化生成。

### 关键设计

1. **可描述运动表示**：将人体关节按运动学树组织为六大运动学组 $G = \{\text{Torso, Neck, Left Arm, Right Arm, Left Leg, Right Leg}\}$，为每个组 $g$ 定义：

    - 组位置：$\mathbf{P}_g(t) = \frac{1}{|J_g|} \sum_{j \in J_g} \mathbf{p}_j(t)$
    - 肢体角度：$\Theta_g(t) = \{r_j(t) | j \in J_g\}$
    - 组速度：$\mathbf{V}_g(t) = \frac{1}{|J_g|} \sum_{j \in J_g} \mathbf{v}_j(t)$
   
   组间交互定义为位置差、连接关节角度、相对速度。该表示是现有关节级表示的**线性变换**，可无损转换回原始表示，但天然适合自然语言描述。

2. **半监督标注管线**：三步流程生成细粒度标注：

    - **空间信息**：用 PoseScript 为每帧姿态生成详细文本描述
    - **关键帧选择**：用 sBERT 计算帧间描述的余弦相似度，低于阈值 0.8 的帧标记为关键帧
    - **LLM 推理**：用 GPT-4o-mini 根据关键帧姿态描述推断各运动学组及其交互的运动描述
   
   两名人类评估者迭代优化 prompt，直到 Cohen's Kappa > 0.8。整个标注成本约 23 美元。

3. **层级文本-运动对齐（HTMA）**：核心创新。不同于直接编码所有描述，HTMA 逐级编码并通过 cross-attention 逐步细化：

    $\mathbf{h_c} = E_c(\text{emb}(T_c))$
    $\mathbf{h_g} = E_g(\text{CrossAttn}(\text{emb}(T_g), \mathbf{h_c}))$
    $\mathbf{h_i} = E_i(\text{CrossAttn}(\text{emb}(T_i), \mathbf{h_g}))$

   每级使用共享架构的 VAE-based ACTOR 编码器，cross-attention 使上下级描述建立联系。在每个层级都应用 InfoNCE 对比学习损失进行文本-运动对齐。

4. **由粗到细运动生成**：基于 MoMask 生成架构。生成过程分三步：先以全局描述嵌入 $\mu_c$ 条件生成初始 token，再以组级嵌入 $\mu_g$ 条件重新输入生成器得到中间 token，最后以交互级嵌入 $\mu_i$ 条件生成最终 token。生成器在三个层级共享权重。

5. **Motion Reasoner**：基于 LLaMA-3 微调，给定全局动作描述自动生成组级和交互级描述，使系统在推理时只需用户提供简单文本。

### 损失函数 / 训练策略

- 对齐训练：InfoNCE + KL 散度 + 跨模态嵌入相似度 + 运动重建损失
- 生成训练：MoMask 的掩码重建损失，共享权重用于三级条件
- Motion Reasoner：标准 next-token prediction 损失，条件化于全局描述

## 实验关键数据

### 主实验

**文本-运动检索**（HumanML3D，Protocol (a) - 全量测试集）：

| 方法 | R@1 ↑ | R@3 ↑ | R@10 ↑ | MedR ↓ |
|------|-------|-------|--------|--------|
| TMR | 5.68 | 14.04 | 30.94 | 28.00 |
| KinMo (DistilBERT) | 8.13 | 19.69 | 39.18 | 18.00 |
| **KinMo (RoBERTa)** | **9.05** | **20.47** | **41.60** | **16.00** |

*R@1 提升 59%（5.68→9.05），MedR 降低 43%（28→16）。*

**文本-运动生成**（HumanML3D）：

| 方法 | R-Prec Top3 ↑ | FID ↓ | MM-Dist ↓ |
|------|--------------|-------|----------|
| MoMask | 0.807 | 0.045 | 2.958 |
| FineMoGen | 0.784 | 0.151 | 2.998 |
| ParCo | 0.801 | 0.109 | 2.927 |
| **KinMo (HTMA)** | **0.821** | **0.039** | **2.901** |

*FID 降低 13%（0.045→0.039），Top3 R-Prec 提升 1.7%。*

### 消融实验

| 语义层级 | R@1 ↑ | R@3 ↑ | MedR ↓ |
|---------|-------|-------|--------|
| 仅全局 | 3.67 | 10.32 | 40.00 |
| + 组级 | 7.58 | 16.97 | 22.00 |
| + 组级 + 交互级 | **9.05** | **20.47** | **16.00** |
| - cross-attention | 7.63 | 16.94 | 22.00 |

*每增加一层描述都带来显著提升。无 cross-attention 时交互级描述几乎无额外收益，证明层级编码的必要性。*

| 嵌入器 | 配置 | FID ↓ | R-Prec Top3 ↑ |
|--------|------|-------|--------------|
| CLIP | 仅全局 | 0.115 | 0.499 |
| CLIP | +组+交互 | 0.098 | 0.512 |
| HTMA | 仅全局 | 0.056 | 0.512 |
| HTMA | +组+交互 | **0.044** | **0.527** |

*HTMA 在所有配置下均优于 CLIP，且由粗到细策略在两种嵌入器下均有效。*

### 关键发现

- 用户研究（20 人 × 320 样本）：KinMo 在真实感、文本对齐度、整体印象三项 MOS 均最高
- KinMo 是唯一能执行局部时序编辑的方法（如仅修改右臂动作），同时保持全身运动的自然性
- 运动轨迹控制实验中，KinMo 的平均控制误差最低（0.1657），FID 最低（0.103）
- 6 组运动学分组比 ParCo 的 2 组分解（上/下身）效果显著更好

## 亮点与洞察

- 运动学分组是连接运动和语言的优雅桥梁：6 大组的划分既符合人体运动学，又天然适合自然语言描述
- 半监督标注管线极其高效：整个 HumanML3D 数据集（44,970 条运动）的标注成本仅约 23 美元
- 层级对齐的设计哲学（由粗到细，逐级细化）对其他多粒度文本-X 对齐任务有普遍借鉴意义

## 局限与展望

- 对 HumanML3D 的依赖：数据集规模和动作多样性仍有限
- Motion Reasoner 基于 LLM 生成的描述存在误差传播风险
- 组间交互描述数量随组数平方增长（6组→15对交互），描述文本可能过于冗长

## 相关工作与启发

- 与 TMR 的对比表明：细粒度描述比增强编码器更能实质性提升对齐质量
- LGTM/FG-MDM 使用 LLM 生成随机补充描述，而 KinMo 基于运动学知识系统化生成，效果更好（Table 3）
- PoseScript 从单帧姿态提取的描述可作为从运动到语言的通用中间表示

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](../../CVPR2025/human_understanding/ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)
- [\[ECCV 2024\] Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases](../../ECCV2024/human_understanding/bridging_the_gap_between_human_motion_and_action_semantics_via_kinematic_phrases.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](../../AAAI2026/human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[ICCV 2025\] Contact-Aware Refinement of Human Pose Pseudo-Ground Truth via Bioimpedance Sensing](contact-aware_refinement_of_human_pose_pseudo-ground_truth_via_bioimpedance_sens.md)

</div>

<!-- RELATED:END -->
