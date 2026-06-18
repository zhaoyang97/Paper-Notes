---
title: >-
  [论文解读] PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments
description: >-
  [NeurIPS 2025][多模态VLM][主动视觉推理] 本文提出主动视觉推理（AVR）任务范式，构建了CLEVR-AVR仿真基准和AVR-152k数据集（含丰富CoT标注），训练PhysVLM-AVR模型在部分可观测交互环境中通过感知-推理-动作闭环迭代获取信息并正确回答问题，显著优于现有MLLM。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "主动视觉推理"
  - "部分可观测环境"
  - "多步交互"
  - "Chain-of-Thought"
  - "具身智能"
---

# PhysVLM-AVR: Active Visual Reasoning for Multimodal Large Language Models in Physical Environments

**会议**: NeurIPS 2025  
**arXiv**: [2510.21111](https://arxiv.org/abs/2510.21111)  
**代码**: [GitHub](https://anonymous.4open.science/r/anonymous-je99tt)  
**领域**: 多模态VLM  
**关键词**: 主动视觉推理, 部分可观测环境, 多步交互, Chain-of-Thought, 具身智能

## 一句话总结
本文提出主动视觉推理（AVR）任务范式，构建了CLEVR-AVR仿真基准和AVR-152k数据集（含丰富CoT标注），训练PhysVLM-AVR模型在部分可观测交互环境中通过感知-推理-动作闭环迭代获取信息并正确回答问题，显著优于现有MLLM。

## 研究背景与动机
当前多模态大语言模型（MLLM）在视觉推理方面取得了显著进展，如物体计数、VQA等任务上表现优异。然而，这些模型几乎都依赖于**静态、完全可观测**的视觉输入——给定一张完整图片，模型直接推理出答案。

这种被动推理范式在现实物理世界中存在根本性限制：真实环境中的信息往往是**部分可观测**的，物体可能被遮挡、堆叠或超出视野范围。人类面对这种情况会自然地**主动探索**——移动视角、翻转物体、改变位置——通过"感知→推理→行动"的闭环过程逐步获取信息。

现有研究存在**范式碎片化**问题：传统视觉推理任务（如CLEVR）假设信息完整，不涉及交互；具身问答（如OpenEQA）虽有交互但聚焦于被动观察视频序列；具身探索方法侧重导航成功率而非推理所需的定向信息获取。核心矛盾在于：**没有方法能有效连接推理与策略性的序列信息收集行动**。

受人类主动感知能力启发，本文提出主动视觉推理（AVR）任务，将视觉推理扩展到部分可观测的交互环境，要求智能体：(1) 通过序列物理动作主动获取信息；(2) 整合多步观测进行连贯推理；(3) 基于增量视觉反馈动态调整决策。

## 方法详解

### 整体框架
AVR被形式化为一个**闭环感知-推理-动作**范式。在每个时间步 $t$，智能体接收部分观测 $o_t$，维护观测历史 $h_t = \{o_0, \ldots, o_t\}$，基于问题 $Q$ 和历史 $h_t$ 生成推理轨迹 $\text{Think}_t = f_{\text{reason}}(Q, h_t, A)$。如果信息充足则给出答案，否则选择最优信息收集动作：

$$a_t = \arg\max_{a_t \in A} \mathbb{E}_{o_{t+1} \sim E(h_t, a_t)} [I(Y; y_{t+1} | h_{t+1}, Q)]$$

该公式的核心思想是选择**期望信息增益最大化**的动作，这是一个高阶马尔可夫决策过程（MDP）。

### 关键设计

1. **CLEVR-AVR 仿真基准**:

    - 基于Genesis物理仿真平台，扩展经典CLEVR为交互式具身环境
    - 包含10种遮挡类型、10种堆叠类型、10种复合场景
    - 动作空间包括物体操纵和视角变换
    - 每个问题配有最终答案选项和中间[Action]选项，迫使模型判断当前信息是否充足
    - 设计动机：需要一个控制良好但信息丰富的评估环境来全面衡量主动推理能力

2. **AVR-152k 数据集（三级递进结构）**:

    - **AVR-Caption (100k)**: 基于ScanNet、RT1等场景的密集描述，建立基础视觉感知能力
    - **AVR-Embodied Reasoning (50k)**: 多图序列配合时空推理问答，由Gemini生成描述、DeepSeek-R1生成推理链
    - **AVR-Core (2k)**: 核心组件，用UMI设备在640个真实桌面场景收集，由人类专家标注结构化CoT，包含(i)不确定性识别、(ii)动作条件信息增益预测、(iii)信息最大化动作选择三个关键推理步骤
    - 设计动机：AVR-Core将任务建模为高阶MDP，其CoT标注显式教授模型类人的主动信息寻求推理过程

3. **PhysVLM-AVR 模型**:

    - 架构类似LLaVA：Qwen2.5-3B作为LLM解码器，SigLIP-400M作为视觉编码器
    - 关键修改：视觉编码器输出后加入max pooling层，将视觉token数量减少3倍，支持高效多图推理
    - 设计动机：针对AVR任务需要处理多步观测序列的需求，减少视觉token有助于扩展上下文容量

### 损失函数 / 训练策略
采用四阶段混合数据增量训练：
- **Stage 1**: 对齐阶段，仅训练connector（2xMLP），使用LLaVA-Pretrain数据
- **Stage 2.1**: 单图理解，全参数微调，使用LLaVA-OneVision数据
- **Stage 2.2**: 综合视觉理解，训练M4-Instruct + AVR-Caption
- **Stage 3**: 通用推理与主动推理，微调Reason-RFT-129k + AM-DeepSeek-R1-100k + AVR-Embodied + AVR-Core

## 实验关键数据

### 主实验

| 模型 | $ACC_{ISJ}$ (信息充分性判断) | $IGR$ (信息增益率) | $ACC_{FA}$ (最终答案) |
|------|------|------|------|
| LLaVA-OV-7B | 0 | 0 | 0 |
| Qwen2.5-VL-7B | 4.9 | 3.7 | 2.6 |
| Embodied-Reasoner-7B | 20.2 | 10.9 | 1.6 |
| GPT-4o | **88.4** | **50.8** | **45.7** |
| AVR-Qwen2.5-VL-7B | 89.3 | 34.7 | 38.1 |
| PhysVLM-AVR-3B | **90.5** | 29.9 | 39.7 |

### 消融实验

| 配置 | $ACC_{ISJ}$ | $IGR$ | $ACC_{FA}$ | 说明 |
|------|---------|------|------|------|
| Full Model | 90.5 | 29.9 | 39.7 | 完整模型 |
| w/o CoT | 47.6 | 18.0 | 16.9 | CoT标注对推理步骤监督至关重要 |
| w/o AVR-Core | 16.4 | 11.2 | 2.3 | AVR-Core是主动推理能力的基础 |

### 关键发现
- 现有开源MLLM和被动推理模型在CLEVR-AVR上近乎零分，说明被动能力无法迁移到主动推理
- Embodied-Reasoner-7B能检测信息不完整（20.2% ISJ），但几乎无法正确行动和推理（1.6% FA），揭示了当前具身模型的根本短板
- PhysVLM-AVR-3B仅3B参数就在信息充分性判断上超越GPT-4o（90.5% vs 88.4%）
- 高ISJ与相对低FA之间的差距表明：识别"需要行动"已学会，但"选择最优动作并整合多步信息"仍是核心挑战

## 亮点与洞察
- **任务定义新颖**：首次将视觉推理从静态完全可观测扩展到动态部分可观测的交互环境，填补了被动推理与主动具身理解之间的空白
- **CoT设计精巧**：AVR-Core的三步CoT（不确定性评估→信息增益预测→策略决策）精准模拟人类主动信息搜寻的认知过程，是小模型超越GPT-4o的关键
- **实验设计严谨**：CLEVR-AVR仿真数据与训练数据无重叠，三个互补评估指标全面衡量主动推理各环节能力

## 局限与展望
- 最终答案准确率（39.7%）仍远低于GPT-4o（45.7%），多步信息整合能力有待提升
- AVR-Core仅2k样本且限于桌面场景，向复杂开放场景的扩展性未验证
- 当前评估限于仿真环境，真实物理世界的主动推理部署存在sim-to-real gap

## 相关工作与启发
- **vs Embodied Reasoner**: PhysVLM-AVR强调信息增益驱动的策略性行动选择，而非简单的导航任务完成
- **vs OpenEQA/RoboVQA**: AVR将推理与动作的闭环关系置于核心，要求模型的行动必须由推理需求驱动

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次定义AVR任务，高阶MDP形式化新颖，开创性地将主动信息获取与视觉推理结合
- 实验充分度: ⭐⭐⭐⭐ 基准设计完善，消融充分，但仅限仿真评估，缺少真实世界部署
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，从人类认知到AI范式的逻辑链条完整
- 价值: ⭐⭐⭐⭐ 揭示了MLLM在主动推理上的根本能力缺失，为后续研究奠定基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[ACL 2025\] Progressive Multimodal Reasoning via Active Retrieval](../../ACL2025/multimodal_vlm/progressive_multimodal_reasoning_via_active_retrieval.md)
- [\[NeurIPS 2025\] FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models](flexac_towards_flexible_control_of_associative_reasoning_in_multimodal_large_lan.md)
- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](../../ICML2026/multimodal_vlm/the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)

</div>

<!-- RELATED:END -->
