---
title: >-
  [论文解读] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis
description: >-
  [AAAI 2026][医学图像][医学VQA] 提出 MedEyes，一个混合策略强化学习框架，通过注视引导推理导航器（GRN）模拟临床医生"扫描-钻探"的诊断视觉搜索模式，结合置信度值采样器（CVS）和双流 GRPO 优化，实现动态视觉聚焦的医学渐进式诊断推理，在五个医学 VQA 基准上平均提升 8.5pp。
tags:
  - "AAAI 2026"
  - "医学图像"
  - "医学VQA"
  - "强化学习"
  - "视觉思维链"
  - "动态注意力"
  - "GRPO"
---

# MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis

**会议**: AAAI 2026  
**arXiv**: [2511.22018](https://arxiv.org/abs/2511.22018)  
**代码**: [GitHub](https://github.com/zhcz328/MedEyes)  
**领域**: 医学图像  
**关键词**: 医学VQA, 强化学习, 视觉思维链, 动态注意力, GRPO

## 一句话总结
提出 MedEyes，一个混合策略强化学习框架，通过注视引导推理导航器（GRN）模拟临床医生"扫描-钻探"的诊断视觉搜索模式，结合置信度值采样器（CVS）和双流 GRPO 优化，实现动态视觉聚焦的医学渐进式诊断推理，在五个医学 VQA 基准上平均提升 8.5pp。

## 研究背景与动机

近年来医学视觉语言模型（VLM）在问答和报告生成上取得进展，但在需要渐进式视觉推理的诊断场景中仍有明显不足。现有方法的痛点：

**SFT 过拟合**：监督微调捕获任务知识但容易过拟合到记忆的推理轨迹，在未见临床场景中泛化差、产生模糊响应

**纯文本 CoT 缺乏视觉锚定**：文本推理步骤与视觉证据之间缺乏显式关联，导致信息丢失和视觉幻觉

**纯 on-policy RL 的"优势坍塌"**：模型初始能力有限时，自主探索容易陷入局部最优，产生表面连贯但临床错误的推理路径——即"认知陷阱"

**朴素行为克隆的局限**：简单模仿专家轨迹只能复制动作序列，无法捕获底层推理逻辑

核心问题：**如何让模型习得专家医生那种渐进式视觉聚焦和迭代诊断细化的能力？**

本文的 insight：引入结构化的 off-policy 专家轨迹作为"认知锚点"，与 on-policy 自主探索结合，在模仿专家行为与自主发现之间取得平衡。关键是让推理步骤显式锚定到视觉区域，建立"精确观察"与"结构化推理"的一致映射。

## 方法详解

### 整体框架

MedEyes 是一个混合策略 RL 框架，包含两个协同流：
- **On-policy 探索流**：策略模型 $\pi_\theta$ 自主采样诊断轨迹
- **Off-policy 引导流**：通过 GRN 和 CVS 构建专家轨迹作为认知锚点

医学视觉推理被形式化为马尔可夫决策过程：给定医学图像 $I$ 和查询 $q$，策略生成诊断轨迹 $\tau = [n_1, n_2, \ldots, n_T, a]$，每步 $n_t = \langle s_t, \mathcal{G}_t \rangle$ 包含文本认知 $s_t$ 和视觉锚定 $\mathcal{G}_t$（边界框坐标），最终输出诊断答案 $a$。

### 关键设计

1. **注视引导推理导航器（GRN）**:

    - 功能：模拟临床医生的视觉搜索模式，通过双模式探索生成结构化专家轨迹
    - 核心思路：维护三元注意力状态 $\psi_t = (\mathcal{R}_t, \mathcal{C}_t, \mathcal{F}_t)$，分别为候选区域集、置信度分布和探索模式。通过大规模多模态专家模型（MedPLIB）进行区域级 VQA 查询生成候选区域。状态转移 $\psi_{t+1} = \mathcal{T}(\psi_t, a_t, o_t)$ 由两种互补模式驱动：
        - **扫描模式**（$\mathcal{F}_t = \text{global}$）：提示专家模型定位图像中所有异常区域，生成全局候选
        - **钻探模式**（$\mathcal{F}_t = \text{local}$）：对特定候选区域做定向分析，生成精细化置信度
    - 模式切换规则：计算置信度变化率 $\Delta c = \frac{c_{t+1}(r_i) - c_t(r_i)}{c_t(r_i) + \epsilon}$。若 $\Delta c \geq \delta$ 则继续钻探（信息增益充分），否则切回扫描（当前区域信息饱和）
    - 设计动机：直接受放射科医生眼动追踪研究启发——专家诊断先系统扫描定位可疑区域，再深入分析特定区域

2. **置信度值采样器（CVS）**:

    - 功能：从 GRN 的多轮轨迹中生成多样但可信的探索路径
    - 核心思路：在每个决策步应用 nucleus sampling，从 top-$p_0$ 置信度区域中采样：$\mathcal{P}_{\text{nucleus}} = \{a_i : \sum_{j=1}^i P(a_j|\psi_t) \leq p_0\}$，生成 $N_\text{expert}=6$ 条变长轨迹。终止条件：局部置信度超过阈值 $\xi=0.85$ 或达到最大长度 $T_\max=4$
    - 设计动机：单一专家轨迹不足以覆盖诊断策略的多样性，nucleus sampling 在保持可信度的同时引入多样性。自适应长度反映了不同病例的诊断复杂度差异

3. **双流 GRPO 优化**:

    - 功能：解耦 on-policy 和 off-policy 学习信号，避免奖励同化和熵坍塌
    - 核心思路：
        - **源自适应重要性比**：on-policy 轨迹用 $\rho_i^\theta = \pi_\theta(\tau_i|I,q) / \pi_{\theta_\text{old}}(\tau_i|I,q)$，off-policy 轨迹用 $\rho_i^\theta = \pi_\theta(\tau_i|I,q) / \pi_\text{expert}(\tau_i|I,q)$（其中 $\pi_\text{expert}=1$）
        - **优势解耦**：分别在 on-policy 和 off-policy 数据上独立计算归一化统计量：$A_i = \frac{R(\tau_i) - \mu^{s(i)}}{\sigma^{s(i)} + \varepsilon}$
    - 设计动机：统一归一化会让高奖励的专家轨迹压制 on-policy 的自主学习信号，导致梯度主导。解耦后两个流保持独立学习速率，既能从专家学习又不牺牲对新场景的适应性

4. **多组件可验证奖励函数**:

    - **准确性奖励** $r_\text{acc}$：答案是否正确（indicator function），$\lambda_\text{acc}=0.7$
    - **语法奖励** $r_\text{grammar}$：推理-动作-感知的标签结构是否正确（二值），$\lambda_\text{grammar}=0.2$
    - **多样性奖励** $r_\text{div}$：探索的唯一区域数 + 区域间 IoU 低于阈值的比例，$\lambda_\text{div}=0.1$

### 损失函数 / 训练策略

使用 PPO clip 风格的目标函数：

$$\mathcal{J}(\theta) = \frac{1}{N}\sum_{i=1}^N \frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|} \min(\rho_{i,t}^\theta A_i, \text{clip}(\rho_{i,t}^\theta, 1-\epsilon, 1+\epsilon)A_i)$$

基于 Qwen2.5-VL-3B，AdamW 优化器，学习率 $1 \times 10^{-6}$，训练 3 个 epoch，rollout batch size 98，每个 prompt 生成 8 个 rollout。6 张 RTX 3090 训练。

## 实验关键数据

### 主实验

在 5 个医学 VQA 基准上评估：VQA-RAD、SLAKE、PathVQA、PMC-VQA、MMMU (Health)。

| 数据集 | MedEyes | GMAI-VL (医学SOTA) | MedVLM-R1 (RL SOTA) | GPT-4o |
|--------|---------|---------------------|----------------------|--------|
| VQA-RAD | **70.7** | 64.6 | 61.4 | 54.2 |
| SLAKE | **79.1** | 71.9 | 65.9 | 50.1 |
| PathVQA | **64.8** | 47.2 | 55.2 | 59.2 |
| PMC-VQA | **55.3** | 52.3 | 44.8 | 40.8 |
| MMMU* | **59.7** | 51.2 | 35.5 | - |
| 平均 | **65.9** | 57.4 (-8.5) | 52.5 (-13.4) | 51.1 (-14.8) |

### 消融实验

| 配置 | VQA-RAD | SLAKE | PathVQA | 平均 | 说明 |
|------|---------|-------|---------|------|------|
| MedEyes (完整) | 70.7 | 79.1 | 64.8 | 71.5 | 基线 |
| 去除 GRN | 62.4 | 69.8 | 56.2 | 62.8 | 降 8.7pp |
| 去除 CVS | 65.3 | 73.5 | 59.1 | 66.0 | 降 5.5pp |
| 去除 Off-policy | 61.2 | 67.4 | 54.3 | 61.0 | 降 10.5pp，影响最大 |
| 仅扫描模式 | 66.8 | 74.2 | 58.7 | 66.6 | 细粒度任务受损 |
| 仅钻探模式 | 64.5 | 71.9 | 60.3 | 65.6 | 缺乏系统性探索 |

### 关键发现

- **Off-policy 专家轨迹是最关键组件**：去除后平均下降 10.5pp，证明纯 on-policy 学习在医学推理中力不从心
- **扫描+钻探双模式缺一不可**：单模式分别在不同类型任务上表现不佳，两者互补才能覆盖完整诊断流程
- **训练动态呈现"探索-效率"转变**：轨迹长度先从 2.1 步增加到 3.0 步（学习何时需要视觉锚定），再缩短至 2.6 步（学会更高效推理）
- **6 条专家轨迹、3 步推理长度为最优平衡点**：更多/更长均带来边际递减甚至性能下降
- **轨迹质量至关重要**：GRN+CVS 轨迹远优于随机采样（+12.8pp）和 DeepSeek-R1 生成的轨迹（+7.6pp）

## 亮点与洞察

- **临床对齐的设计**：扫描-钻探双模式直接对应放射科医生眼动研究中的视觉搜索模式，是"人类认知启发 AI 设计"的优秀范例
- **混合策略训练范式新颖**：不是简单的 off-policy pre-training + on-policy fine-tuning，而是双流同步训练并解耦优势归一化，技术上有深度
- **可解释性强**：每步推理显式绑定到图像区域，注意力热图直观展示渐进聚焦过程
- **"认知锚点"概念有启发性**：off-policy 轨迹作为认知锚点帮助初始能力弱的模型跳出局部最优，这个思路可推广到其他 RL 场景

## 局限与展望

- **定量测量问题**：在肿瘤大小测量等需要像素-厘米校准的任务上，基于比例估计的方法误差大
- **细粒度概念区分**：偶尔混淆形态相似的病理亚型（如动脉瘤 vs 夹层动脉瘤）
- **依赖外部专家模型**：GRN 使用 MedPLIB 作为视觉专家，轨迹质量受限于外部模型能力
- **基础模型规模较小**：仅在 3B 参数模型上验证，扩展到更大模型的效果待观察
- **计算成本**：每个 prompt 需 8 个 rollout + 6 条专家轨迹，训练资源需求不低

## 相关工作与启发

- 与 GRIT、DeepEyes 等视觉 CoT 方法相比，MedEyes 通过 off-policy 引导解决了初始模型能力不足导致的训练困难
- 双流 GRPO 的优势解耦机制可推广到任何混合 on/off-policy RL 场景
- 眼动追踪数据驱动 AI 设计的思路，在医学影像分析中有更大应用空间
- 未来方向：更丰富的工具调用（测量、对比增强）、多轮对话式诊断、与实际临床工作流的集成

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] X-PCR: A Benchmark for Cross-modality Progressive Clinical Reasoning in Ophthalmic Diagnosis](../../CVPR2026/medical_imaging/x-pcr_a_benchmark_for_cross-modality_progressive_clinical_reasoning_in_ophthalmi.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[AAAI 2026\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[AAAI 2026\] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering](q-fsru_quantum-augmented_frequency-spectral_fusion_for_medical_visual_question_a.md)

</div>

<!-- RELATED:END -->
