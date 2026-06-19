---
title: >-
  [论文解读] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning
description: >-
  [NeurIPS 2025][机器人][VLA推理] 提出ThinkAct双系统框架，通过动作对齐的视觉奖励对MLLM进行强化学习微调以激发具身推理能力，并将推理计划压缩为视觉潜在表示来指导下游动作模型，实现"先思考再行动"的VLA推理范式。 当前VLA模型的核心困境在于：端到端地从视觉和文本输入直接映射到低层动作…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "VLA推理"
  - "强化学习"
  - "视觉潜在规划"
  - "具身推理"
  - "双系统架构"
---

# ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning

**会议**: NeurIPS 2025  
**arXiv**: [2507.16815](https://arxiv.org/abs/2507.16815)  
**代码**: [项目页面](https://jasper0314-huang.github.io/thinkact-vla/)  
**领域**: 机器人  
**关键词**: VLA推理, 强化学习, 视觉潜在规划, 具身推理, 双系统架构

## 一句话总结

提出ThinkAct双系统框架，通过动作对齐的视觉奖励对MLLM进行强化学习微调以激发具身推理能力，并将推理计划压缩为视觉潜在表示来指导下游动作模型，实现"先思考再行动"的VLA推理范式。

## 研究背景与动机

当前VLA模型的核心困境在于：端到端地从视觉和文本输入直接映射到低层动作，缺乏显式的推理和规划能力。具体来说存在三方面不足：

**缺少长程规划能力**：现有VLA模型如OpenVLA、TraceVLA虽在短程技能上表现良好，但面对需要多步推理的长程操作任务时力不从心，因为它们没有中间推理步骤来分解复杂目标。

**CoT监督数据昂贵且易过拟合**：ECoT和RAD等方法尝试引入链式思维(CoT)推理，但依赖离线MLLM生成的CoT标注数据进行有监督微调。高质量推理轨迹的生成成本高昂，且模型容易过拟合到特定的视觉场景或推理模式。

**现有RL推理缺乏动作对齐**：Video-R1等工作将RL应用于VLM推理，但使用的是QA式准确率奖励，这种奖励信号无法支撑长程规划，也难以在推理和真实动作执行之间建立联系。

ThinkAct的核心洞察是：应当利用**动作对齐的视觉反馈**（而非QA式奖励）来引导MLLM学习具身推理，并通过将推理结果压缩为紧凑的视觉潜在表示，架起高层规划与低层控制之间的桥梁。

## 方法详解

### 整体框架

ThinkAct采用双系统架构：推理MLLM $\mathcal{F}_\theta$ 负责高层规划（System 2慢思考），DiT动作模型 $\pi_\phi$ 负责低层控制（System 1快控制）。两者通过**视觉计划潜在表示** $c_t$ 连接。在每个推理步，MLLM接收当前观测 $o_t$ 和指令 $l$，生成推理文本和视觉计划 $c_t$；动作模型基于 $c_t$ 预测后续 $N$ 步可执行动作。两个模块可以异步运行——MLLM低频推理，动作模型高频控制。

### 关键设计

1. **动作对齐的视觉奖励设计**：这是ThinkAct最核心的创新。作者将高层规划表示为预测的2D夹爪轨迹 $\tau = [p_k]_{k=1}^K$（$K=8$个关键点），然后设计两种奖励信号：

    - **目标奖励** $r_{\text{goal}}$：比较预测轨迹的起止点与真实轨迹起止点的接近程度，$r_{\text{goal}} = \frac{1}{2}(f(p_1, \hat{p}_1) + f(p_K, \hat{p}_K))$，其中 $f(p, p') = \max(0, 1 - \|p - p'\|_2^2)$。这鼓励模型预见任务目标的达成。
    - **轨迹奖励** $r_{\text{traj}}$：使用动态时间规整(DTW)距离来衡量预测轨迹与真实轨迹的分布匹配程度，$r_{\text{traj}} = \max(0, 1 - d(\tau, \hat{\tau}))$。这确保预测轨迹对应物理上合理的夹爪运动。
    - 最终奖励为 $r = 0.9 r_{\text{visual}} + 0.1 r_{\text{format}}$，视觉奖励占主导。

2. **GRPO强化微调**：使用Group Relative Policy Optimization对MLLM进行强化微调。给定输入 $(o_t, l)$，先从旧策略采样 $M$ 组不同响应，计算各自奖励后用组内相对优势 $A_i$ 指导优化。与标准SFT相比，RL允许模型自由探索推理路径而非死记标注数据，且奖励中的视觉反馈提供了具身场景的锚定。同时融入QA数据（RoboVQA、失败检测等）以增强通用推理能力。

3. **视觉潜在规划桥接推理与执行**：MLLM生成的推理嵌入 $v_t$ 和视觉计划嵌入 $c_t$ 在模型内部产生。$c_t$ 通过Q-Former潜在投影器（32个queries）映射到动作模型的输入空间，作为条件信息指导DiT扩散策略预测动作。关键在于 $c_t$ 浓缩了长程时空规划意图，使动作模型能利用高层推理来提升低层控制的鲁棒性。

### 损失函数 / 训练策略

采用多阶段训练：
- **SFT冷启动**：用OXE轨迹数据、RoboVQA、EgoPlan-IT、Video-R1-CoT数据微调MLLM 20K步，学习正确输出格式和基本推理能力。
- **GRPO强化微调**：用OXE+Something-Something V2的视觉轨迹数据和QA数据进行6K步RL训练。
- **推理增强动作适配**：冻结MLLM，在目标环境（如LIBERO）上用模仿学习训练动作模型，损失为 $\mathcal{L}_{\text{IL}}(\phi) = \mathbb{E}[\ell(\pi_\phi(c_t, o_i, l), a_i)]$。

## 实验关键数据

### 主实验

**机器人操作 - SimplerEnv & LIBERO**

| 基准 | 指标 | ThinkAct | DiT-Policy | CoT-VLA | Magma | 提升(vs DiT) |
|------|------|----------|-----------|---------|-------|-------------|
| SimplerEnv-Google-VM | 成功率 | **71.5%** | 56.0% | – | 68.4% | +15.5% |
| SimplerEnv-Google-VA | 成功率 | **65.1%** | 48.2% | – | 62.6% | +16.9% |
| SimplerEnv-Bridge-VM | 成功率 | **43.8%** | 32.4% | – | 35.4% | +11.4% |
| LIBERO 总体 | 成功率 | **84.4%** | 76.8% | 83.9% | – | +7.6% |
| LIBERO-Long | 成功率 | **70.9%** | 57.6% | 69.0% | – | +13.3% |

**具身推理任务**

| 基准 | 指标 | ThinkAct | Qwen2.5-VL* | InternVL3 | 提升 |
|------|------|----------|-------------|-----------|------|
| EgoPlan-Bench2 | 准确率 | **48.2%** | 45.7% | 36.2% | +2.5% |
| RoboVQA | BLEU均值 | **59.8%** | 55.7% | 35.3% | +4.1 |
| OpenEQA | LLM评分 | **56.2%** | 52.0% | 55.5% | +0.7% |

### 消融实验

| 配置 | SimplerEnv | EgoPlan | RoboVQA | 说明 |
|------|-----------|---------|---------|------|
| ThinkAct（完整） | **60.1** | **48.2** | **59.8** | 两种奖励均使用 |
| w/o $r_{\text{traj}}$ | 59.2 | 47.9 | 58.5 | 去轨迹奖励，规划连贯性下降 |
| w/o $r_{\text{goal}}$ | 59.1 | 47.6 | 58.9 | 去目标奖励，长程推理削弱 |
| w/o 两种视觉奖励 | 56.9 | 47.2 | 58.3 | 仅QA奖励，几乎回到SFT水平 |
| SFT冷启动 | 56.4 | 46.4 | 57.9 | 无RL，性能最低 |

### 关键发现

- **RL增强的推理显著优于SFT**：RL微调后模型能进行更细致的环境分析和多步推理，而非仅关注当前状态。
- **少样本适配能力突出**：在LIBERO 10-shot设置中，ThinkAct在所有任务上均超越SOTA方法，在LIBERO-Goal上比Magma高7.3%，LIBERO-Spatial上高9.5%。
- **自纠错行为涌现**：当感知到执行失败时（如物体掉落），推理MLLM能生成"Let's reconsider"并修正规划，引导夹爪回到掉落位置重新抓取。

## 亮点与洞察

- **视觉奖励的巧妙设计**：将抽象的"推理质量"转化为可量化的2D轨迹匹配问题，解决了具身推理中奖励信号难以定义的难题。
- **异步双系统架构**：MLLM慢思考+动作模型快控制的设计非常优雅，每个推理步对应N步动作执行，兼顾了推理深度和控制频率。
- **从SFT到RL的训练范式迁移**：证明了在具身AI领域，RL也能像在语言模型推理中一样，激发出超越监督数据的推理能力。

## 局限与展望

- 继承了预训练MLLM的幻觉问题，可能生成引用错误物体属性或空间关系的计划。
- 推理开销增加（比OpenVLA慢约17%），虽然性能提升显著但实时性受限。
- 2D轨迹作为规划表示的表达能力有限，无法编码深度信息和复杂3D交互。
- 奖励信号依赖现成的夹爪检测器，其精度直接影响训练质量。

## 相关工作与启发

- 与CoT-VLA（用视觉子目标帧替代语言CoT）相比，ThinkAct用RL替代SFT生成推理，更具扩展性。
- 与RAD（用无动作人类视频学推理）互补，ThinkAct进一步将推理与动作执行对齐。
- 启发：未来可探索将视觉奖励信号扩展到3D空间，或引入在线RL让MLLM在模拟器中直接交互学习。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 动作对齐视觉奖励+视觉潜在规划桥接推理与执行，想法原创且完整
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖操作和推理两类benchmark共5个数据集，消融详尽，含少样本/自纠错分析
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图示精美，但部分符号较密集
- **价值**: ⭐⭐⭐⭐⭐ 为VLA模型引入了可扩展的推理能力，双系统架构和RL训练范式具有广泛影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning](../../CVPR2026/robotics/fast-thinkact_efficient_vision-language-action_reasoning_via_verbalizable_latent.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[NeurIPS 2025\] AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning](autovla_a_vision-language-action_model_for_end-to-end_autonomous_driving_with_ad.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)
- [\[CVPR 2025\] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models](../../CVPR2025/robotics/cot-vla_visual_chain-of-thought_reasoning_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->
