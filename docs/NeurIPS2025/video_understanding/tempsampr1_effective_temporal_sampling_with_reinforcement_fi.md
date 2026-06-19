---
title: >-
  [论文解读] TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs
description: >-
  [NeurIPS 2025][视频理解][temporal grounding] 提出TempSamp-R1强化微调框架，针对GRPO在视频时序定位中因搜索空间巨大而on-policy采样低效的问题，通过引入GT作为off-policy监督信号+非线性软优势估计+混合CoT训练范式，在Charades-STA/ActivityNet/QVHighlights三个基准上达到新SOTA。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "temporal grounding"
  - "GRPO"
  - "off-policy"
  - "soft advantage"
  - "hybrid CoT"
  - "video LLM"
---

# TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2509.18056](https://arxiv.org/abs/2509.18056)  
**代码**: [github.com/HVision-NKU/TempSamp-R1](https://github.com/HVision-NKU/TempSamp-R1)  
**领域**: 视频时序理解 / 强化微调  
**关键词**: temporal grounding, GRPO, off-policy, soft advantage, hybrid CoT, video LLM

## 一句话总结
提出TempSamp-R1强化微调框架，针对GRPO在视频时序定位中因搜索空间巨大而on-policy采样低效的问题，通过引入GT作为off-policy监督信号+非线性软优势估计+混合CoT训练范式，在Charades-STA/ActivityNet/QVHighlights三个基准上达到新SOTA。

## 研究背景与动机
**领域现状**：MLLM在通用视频问答中表现出色，但在需要精确时序理解的任务（temporal grounding, highlight detection）上仍然困难。SFT方法容易过拟合确定性时间戳标注，缺乏时序推理能力。GRPO（DeepSeek-R1风格）在数学推理中有效，但在视频时序定位中效果受限。

**现有痛点**：(1) 视频时序定位的搜索空间巨大——需要在连续时间轴上搜索(起始, 结束)对，比离散数学答案难得多；(2) GRPO纯on-policy采样在大搜索空间中难以命中高IoU解，导致奖励稀疏且不稳定（ActivityNet上top-1 IoU奖励持续低且震荡）；(3) 引入off-policy高奖励解（如GT）会使优势估计偏倚——GT的高奖励拉高组均值，导致所有on-policy解的优势全变负。

**核心矛盾**：如何在大搜索空间中有效引导策略学习精确的时序定位，同时避免off-policy引入的分布偏移？

**切入角度**：将GT标注作为off-policy解混入GRPO采样组，但通过非线性奖励整形消除分布偏移对优势估计的负面影响。

## 方法详解

### 整体框架
TempSamp-R1基于GRPO框架，对每个查询采样$G$个解（$G-1$个on-policy + 1个off-policy GT），计算IoU奖励后通过软优势估计模块转换为标准化优势值进行策略优化。训练分两阶段：先学直接输出，再引入format reward鼓励CoT推理。推理时单一模型支持CoT和non-CoT两种模式。

### 关键设计
1. **混合策略采样（Mix-Policy Sampling）**:
    - 功能：将GT标注作为off-policy解混入GRPO采样组，为时序定位提供精确的正例信号
    - 核心思路：对每个查询$q$，从当前策略$\pi_\theta$采样$G-1$个解$\{o_1,...,o_{G-1}\}$，加入一个外部off-policy解$o_G$（来自GT标注），用联合分布计算归一化优势 $A_i = \frac{r_i - \text{mean}(\{r_1,...,r_{G-1}\} \cup \{r_G\})}{\text{std}(\{r_1,...,r_{G-1}\} \cup \{r_G\})}$。同时提出优势锚定策略 $A_G = \lambda_{\text{off}} \cdot \max\{A_i | i \in \{1,...,G-1\}\}$（$\lambda_{\text{off}}=1.2$）解耦off-policy与on-policy的优势
    - 设计动机：GRPO纯on-policy在大搜索空间中几乎无法采到高IoU解→奖励稀疏、学习信号弱。GT提供精确的时序锚点，补偿on-policy的探索不足；但GT的高奖励会拉偏组均值，需配合软优势消除偏倚

2. **非线性软优势估计（Non-Linear Soft Advantage Estimation）**:
    - 功能：对奖励进行非对称非线性变换，压缩高奖励区域、放大低奖励区域的差异
    - 核心思路：定义分段函数 $\tilde{r}_i = \begin{cases}\tau + \alpha_1 \cdot \ln((r_i - \tau) + 1), & r_i \geq \tau \\ \tau - \frac{e^{\alpha_2(\tau - r_i)} - 1}{e^{\alpha_2} - 1}, & r_i < \tau\end{cases}$，其中$\tau=0.8$为阈值，$\alpha_1=0.01$控制对数压缩，$\alpha_2=1$控制指数放大。对数分支抑制GT等最优解的梯度尖峰，指数分支放大次优解之间的区分度
    - 设计动机：标准GRPO中off-policy高奖励解使所有on-policy解优势变负→高质量on-policy解被错误惩罚。非线性整形后高奖励区域被压缩、低奖励区域被放大，使梯度更有信息量、优化更稳定

3. **混合CoT训练范式（Hybrid Chain-of-Thought Training）**:
    - 功能：训练单一模型同时支持CoT和non-CoT推理，推理时按查询复杂度选择模式
    - 核心思路：两阶段训练——初始化阶段优化模型生成准确最终答案（non-CoT模式），随后引入format reward鼓励在`<Think>...</Think>`中生成推理步骤、在`<Answer>...</Answer>`中输出最终答案。format reward = 1（格式正确）或 0（格式不符）。推理时Mixed CoT取两种模式的最佳结果
    - 设计动机：不同查询复杂度不同——简单查询直接输出即可，复杂查询需要推理。CoT和non-CoT互补，Mixed模式在所有指标上均优于单一模式

### 损失函数 / 训练策略
使用标准GRPO目标函数 $\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}[\min(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i) - \beta\text{KL}(\pi_\theta||\pi_{ref})]$。采用$\pi_{\theta_{old}} = \pi_\theta$简化计算。任务奖励：时序定位用IoU奖励$R_{\text{IoU}}$，高光检测用时间戳匹配奖励$R_{\text{ts}} = \lambda_{\text{rec}} \cdot F2 + \lambda_{\text{score}} \cdot \frac{1}{1+\text{WMSE}}$。基础模型Qwen2.5-VL-7B-Instruct，4×A100 GPU，视频2 FPS采样。

## 实验关键数据

### 主实验：时序理解基准SOTA对比

| 方法 | 类型 | Charades R1@0.7 | ActivityNet R1@0.5 | QVHighlights mAP |
|---|---|---|---|---|
| TimeChat | SFT | 23.7 | — | 21.7 |
| iMOVE | SFT | 45.3 | 50.7 | — |
| VideoChat-R1 | RL | 50.2 | — | — |
| TimeZero | RL | 47.9 | 47.3 | — |
| **TempSamp-R1 (no-CoT)** | RL | 52.2 | 55.4 | **30.0** |
| **TempSamp-R1 (CoT)** | RL | **52.9** | **56.0** | 28.3 |
| **TempSamp-R1 Mixed CoT** | RL | **56.3** | **58.7** | 29.3 |

### 消融实验：各组件贡献（Charades-STA）

| 配置 | R1@0.5 | R1@0.7 | mIoU |
|---|---|---|---|
| GRPO baseline | 71.7 | 50.2 | 60.8 |
| + off-policy (reward scaling) | 72.5 | 51.1 | 61.0 |
| + off-policy (advantage anchor) | 73.0 | 51.7 | 61.3 |
| + off-policy (non-linear shaping) | 73.6 | 52.2 | 61.7 |
| + hybrid CoT (Mixed) | **76.0** | **56.3** | **64.2** |

### 关键发现
- 纯on-policy GRPO在ActivityNet上top-1 IoU奖励持续低于0.3且不稳定，off-policy引导使奖励快速稳定在0.6+
- 三种off-policy整合策略中，非线性奖励整形>优势锚定>奖励缩放
- Mixed CoT在所有指标上超越单独CoT和non-CoT模式，mIoU提升2.1-2.5个点
- Few-shot能力：仅用10%训练数据仍能达到GRPO全量数据的90%+性能

## 亮点与洞察
- 精确诊断了GRPO在时序定位中失效的根本原因——搜索空间大导致on-policy采样奖励稀疏
- 非线性软优势的分段设计巧妙：对数压缩高奖励区的梯度尖峰+指数放大低奖励区的区分度
- Mixed CoT是一个简单但有效的设计——让同一模型自适应选择推理深度
- 将RL fine-tuning从数学推理推广到视频时序理解，验证了R1范式的跨领域潜力

## 局限与展望
- Off-policy依赖GT标注——推理时无GT可用，训练时的探索策略与推理时不一致
- 主要在时序定位任务验证，对通用视频QA的效果未知
- 非线性变换的超参（$\tau, \alpha_1, \alpha_2$）可能需要任务特定调整
- 仅在7B模型上验证，更大模型是否仍需off-policy引导？

## 相关工作与启发
- **vs TimeZero/VideoChat-R1**：这些GRPO方法仅用on-policy采样，TempSamp-R1引入off-policy信号解决稀疏奖励问题，ActivityNet上R1@0.5提升8.7个点
- **vs SFT方法（iMOVE等）**：SFT过拟合确定性时间戳，RL微调学到更灵活的时序推理。TempSamp-R1 Mixed CoT在Charades R1@0.7上超过iMOVE 11个点
- **启发**：在大搜索空间RL任务中，适度引入off-policy expert信号可能是普遍有效的策略；非线性奖励整形的思路可推广到其他RL fine-tuning场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 将R1-style RL推广到视频时序定位有价值，off-policy + 软优势组合设计新颖
- 实验充分度: ⭐⭐⭐⭐ 3个benchmark SOTA + 详细消融 + few-shot评估
- 写作质量: ⭐⭐⭐⭐ 问题分析到位，动机-方案逻辑链清晰
- 价值: ⭐⭐⭐⭐ 为视频时序理解提供了实用的RL微调框架，Mixed CoT设计可复用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning to Refuse: Refusal-Aware Reinforcement Fine-Tuning for Hard-Irrelevant Queries in Video Temporal Grounding](../../CVPR2026/video_understanding/learning_to_refuse_refusal-aware_reinforcement_fine-tuning_for_hard-irrelevant_q.md)
- [\[NeurIPS 2025\] Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders](enhancing_temporal_understanding_in_videollms_through_stacke.md)
- [\[ACL 2025\] RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning](../../ACL2025/video_understanding/raven_robust_advertisement_video_violation_temporal_grounding_via_reinforcement_.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](../../AAAI2026/video_understanding/r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)
- [\[NeurIPS 2025\] PixFoundation 2.0: Do Video Multi-Modal LLMs Use Motion in Visual Grounding?](pixfoundation_20_do_video_multi-modal_llms_use_motion_in_visual_grounding.md)

</div>

<!-- RELATED:END -->
