---
title: >-
  [论文解读] Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation
description: >-
  [ECCV 2024][预训练][文本到动作生成] 本文提出 PRO-Motion 分治框架，将文本到动作生成分解为三个阶段：LLM 驱动的动作规划（Plan）、基于脚本的姿态扩散生成（Posture）、以及全身平移旋转估计（Go），通过降低各阶段的复杂度实现了开放词汇的高质量动作生成。 领域现状：文本到动作生成（text-…
tags:
  - "ECCV 2024"
  - "预训练"
  - "文本到动作生成"
  - "开放词汇"
  - "LLM规划"
  - "扩散模型"
  - "人体运动"
---

# Plan, Posture and Go: Towards Open-Vocabulary Text-to-Motion Generation

**会议**: ECCV 2024  
**arXiv**: [2312.14828](https://arxiv.org/abs/2312.14828)  
**代码**: [https://moonsliu.github.io/Pro-Motion](https://moonsliu.github.io/Pro-Motion)  
**领域**: 动作生成 / 多模态  
**关键词**: 文本到动作生成, 开放词汇, LLM规划, 扩散模型, 人体运动

## 一句话总结
本文提出 PRO-Motion 分治框架，将文本到动作生成分解为三个阶段：LLM 驱动的动作规划（Plan）、基于脚本的姿态扩散生成（Posture）、以及全身平移旋转估计（Go），通过降低各阶段的复杂度实现了开放词汇的高质量动作生成。

## 研究背景与动机

**领域现状**：文本到动作生成（text-to-motion generation）旨在根据自然语言描述自动生成3D人体运动序列。现有方法通常在有限的文本-动作配对数据集（如 HumanML3D、KIT-ML）上训练，生成能力受限于训练集的文本分布。一些方法尝试用 CLIP 对齐动作空间和文本空间，但生成结果仍局限于简单的原地动作。

**现有痛点**：（1）训练数据中的文本描述覆盖面有限，模型无法理解训练集中未出现过的动作描述；（2）现有方法生成的动作大多是"原地"运动（in-place motions），缺乏全身平移和旋转，动态性不足；（3）文本到动作的直接映射过于复杂——自然语言的表达空间无穷，但可能的人体姿态空间是有规律的。

**核心矛盾**：直接从开放词汇的文本到完整动作的端到端生成太难——需要同时理解任意自然语言、生成合理的姿态序列、预测全局运动轨迹。现有数据和模型都无法支撑这种端到端映射的泛化。

**本文目标** 如何实现真正的开放词汇文本到动作生成，即模型能理解并生成训练集中未出现过的动作描述？

**切入角度**：作者的关键观察是：虽然自然语言描述的空间是无穷的（"跳着舞高兴地原地转圈"），但底层的人体姿态可以用一种结构化的"脚本"模板来描述，而且这种模板覆盖所有可能的姿态。如果能用 LLM 将任意文本翻译成这种标准化的脚本序列，那么脚本到姿态的映射就变得简单很多，因为脚本空间远小于自然语言空间。

**核心 idea**：分治策略——用 LLM 将自然语言规划为结构化姿态脚本序列，再分别用扩散模型生成姿态和全身运动轨迹。

## 方法详解

### 整体框架
PRO-Motion 由三个模块组成，形成一个流水线：（1）**Motion Planner**：输入自然语言动作描述，利用 LLM 生成一系列关键姿态的结构化脚本描述；（2）**Posture-Diffuser**：将每个脚本描述转化为对应的 SMPL 姿态参数；（3）**Go-Diffuser**：在姿态序列基础上，预测运动帧之间的全身平移和旋转，生成完整的动态运动。使用 SMPL 模型作为人体表示，最终输出维度为 $64 \times 135$ 的运动序列。

### 关键设计

1. **Motion Planner（动作规划器）**:

    - 功能：将开放词汇的自然语言动作描述转化为结构化的关键姿态脚本序列
    - 核心思路：利用 LLM（如 GPT-4）作为动作规划器，设计精心的 prompt 引导 LLM 输出标准化的姿态描述脚本。prompt 中定义了五类基本的身体部位规则：弯曲程度、相对距离、相对位置、朝向、地面接触，并指定了适用的身体部位。LLM 根据这些规则将复杂的自然语言指令（如"自由地跳舞"）分解为多个关键帧的结构化姿态描述。关键帧之间的中间姿态由后续扩散模型补全
    - 设计动机：这是整个框架的核心创新。自然语言空间无穷大，但结构化脚本空间是可枚举的——因为它只涉及有限的身体部位和有限的关系类型。LLM 的强大语言理解能力天然适合做这种"翻译"。同时用户还可以手动编辑脚本实现精确控制

2. **Posture-Diffuser（姿态扩散器）**:

    - 功能：将单个姿态脚本描述转化为对应的 SMPL 姿态参数
    - 核心思路：实现为一个条件扩散模型，以姿态脚本的文本特征为条件，通过去噪过程生成对应的 SMPL 姿态（不含全局位移）。使用 PoseScript 数据集进行训练，该数据集提供了大量自动生成的姿态描述-姿态配对。模型采用 3 层 Transformer，潜在维度 512，使用线性噪声调度训练 1000 个 epoch。同时使用预训练的文本-姿态检索模型的编码器提取文本和姿态的语义特征
    - 设计动机：由于脚本描述遵循简单的文本模板（而非复杂的自然语言），Posture-Diffuser 的学习难度大幅降低。脚本的组合性质使得模型可以泛化到训练时未见过的姿态组合

3. **Go-Diffuser（运动扩散器）**:

    - 功能：为姿态序列添加全身平移和旋转，生成完整的动态运动
    - 核心思路：实现为另一个扩散模型，以 Posture-Diffuser 生成的关键姿态序列为条件，预测每帧的全身平移（3维）和根关节旋转（6维，使用6D连续旋转表示）。同时将关键帧之间的中间帧补全到固定长度（64帧）。模型采用 8 层 Transformer、4 头注意力，使用余弦噪声调度，扩散步数 100。平移使用速度表示（相邻帧位移差）来归一化
    - 设计动机：将原地姿态生成和全局运动估计分开，解决了现有方法只能生成原地动作的问题。Go-Diffuser 实际上学习的是"给定一系列姿态，如何合理地在空间中移动"，这个任务本身比端到端生成简单得多

### 损失函数 / 训练策略
两个扩散模型都使用标准的 DDPM 框架进行训练。Posture-Diffuser 在 PoseScript-A 数据集（自动生成的姿态描述）上训练，batch size 512，学习率 1e-4。Go-Diffuser 在 AMASS 数据集上训练，batch size 64，学习率 1e-4，条件掩码概率 0.1（用于 classifier-free guidance）。训练时对帧数进行截取和补全，统一为 64 帧。

## 实验关键数据

### 主实验

| 方法 | R-Precision ↑ | FID ↓ | MM-Dist ↓ | 测试集 |
|------|---------------|-------|-----------|--------|
| MDM | 较低 | 较高 | 较高 | HumanML3D |
| T2M-GPT | 中等 | 中等 | 中等 | HumanML3D |
| MotionDiffuse | 中等 | 中等 | 中等 | HumanML3D |
| PRO-Motion (本文) | **最高** | **最低** | **最低** | HumanML3D |
| PRO-Motion (开放词汇) | 有效 | 合理 | 合理 | IDEA-400 |

### 消融实验

| 配置 | APE-Root ↓ | APE-Mean ↓ | 说明 |
|------|-----------|------------|------|
| 仅 Posture-Diffuser | 无轨迹 | 仅原地 | 无Go-Diffuser，只有原地姿态 |
| w/o Motion Planner | 受限 | 受限 | 直接用自然语言条件化，泛化差 |
| Full PRO-Motion | **最低** | **最低** | 完整三模块 |
| 不同LLM | 变化小 | 变化小 | GPT-4最优但其他LLM也可用 |

### 关键发现
- Motion Planner 是实现开放词汇泛化的关键——没有它，模型退化为传统的闭集方法
- Go-Diffuser 解决了原地动作的问题，使生成结果更具动态性和真实感
- 用户可以直接编辑 Motion Planner 生成的脚本来精确控制特定姿态，实现精细的交互式编辑
- 在 IDEA-400 开放词汇测试集上，PRO-Motion 能够处理复杂的文本指令（如"感受到深深的快乐"），而其他方法完全无法生成合理动作

## 亮点与洞察
- **LLM 作为规划者的巧妙应用**：不直接让 LLM 生成动作，而是让它做"翻译"工作——将自然语言翻译成结构化脚本。这充分利用了 LLM 的语言理解能力，同时避免了让 LLM 处理连续数值的困难。这种 LLM-as-planner 的思路可以迁移到其他生成任务
- **分治降低复杂度**：将一个困难的端到端问题分解为三个简单的子问题，每个子问题都有足够的数据和合适的模型来解决。这种系统设计思路在工程上非常实用
- **脚本的组合泛化**：结构化脚本的组合性质天然支持泛化——即使训练时没见过"左手举过头顶同时右脚向后伸"的组合，但见过各个部位的独立描述就能组合生成

## 局限与展望
- 依赖 LLM（如 GPT-4）进行推理，增加了推理成本和 API 依赖
- 固定 64 帧的动作长度限制了长序列动作的生成能力
- SMPL 模型不包含手指运动表示，无法生成精细的手部动作
- Motion Planner 的质量高度依赖 prompt 设计，不同 LLM 可能需要不同的 prompt 调优
- 缺乏物理约束，生成的运动可能存在穿透或不符合物理规律的情况

## 相关工作与启发
- **vs MDM (Motion Diffusion Model)**: MDM 直接从文本端到端生成动作，受限于训练集的文本分布。PRO-Motion 通过分治策略突破了这一限制
- **vs MotionGPT**: MotionGPT 用 LLM 直接预测动作 token，但仍受限于训练数据。PRO-Motion 让 LLM 只做规划而非生成，更符合 LLM 的能力边界
- **vs TEMOS**: TEMOS 在 SMPL 空间中工作但只估计局部运动。PRO-Motion 的 Go-Diffuser 额外估计全局轨迹，生成更动态的结果

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 LLM 规划 + 结构化脚本 + 双扩散模型组合的分治思路非常巧妙和原创
- 实验充分度: ⭐⭐⭐⭐ 闭集和开放词汇实验、消融、用户控制示例丰富，但定量开放词汇评估较薄弱
- 写作质量: ⭐⭐⭐⭐ 故事讲得清晰流畅，分治动机阐述充分
- 价值: ⭐⭐⭐⭐ 开放词汇动作生成是重要方向，分治思路有广泛的借鉴意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] DragAPart: Learning a Part-Level Motion Prior for Articulated Objects](dragapart_learning_a_part-level_motion_prior_for_articulated_objects.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICML 2026\] If open source is to win, it must go public](../../ICML2026/llm_pretraining/if_open_source_is_to_win_it_must_go_public.md)
- [\[CVPR 2026\] Reconstructing CLIP for Open-Vocabulary Dense Perception](../../CVPR2026/llm_pretraining/reconstructing_clip_for_open-vocabulary_dense_perception.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](../../CVPR2025/llm_pretraining/scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)

</div>

<!-- RELATED:END -->
