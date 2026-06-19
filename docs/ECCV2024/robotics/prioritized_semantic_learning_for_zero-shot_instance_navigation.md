---
title: >-
  [论文解读] Prioritized Semantic Learning for Zero-shot Instance Navigation
description: >-
  [ECCV 2024][机器人][零样本导航] 提出Prioritized Semantic Learning (PSL)方法，通过语义增强的Agent架构、优先语义训练策略和语义扩展推理方案，显著提升零样本目标/实例导航中Agent的语义感知能力，在ObjectNav和新提出的InstanceNav任务上实现SOTA。
tags:
  - "ECCV 2024"
  - "机器人"
  - "零样本导航"
  - "实例导航"
  - "语义学习"
  - "CLIP"
  - "具身智能"
---

# Prioritized Semantic Learning for Zero-shot Instance Navigation

**会议**: ECCV 2024  
**arXiv**: [2403.11650](https://arxiv.org/abs/2403.11650)  
**代码**: [https://github.com/XinyuSun/PSL-InstanceNav](https://github.com/XinyuSun/PSL-InstanceNav)  
**领域**: 机器人（具身导航）  
**关键词**: 零样本导航, 实例导航, 语义学习, CLIP, 具身智能

## 一句话总结

提出Prioritized Semantic Learning (PSL)方法，通过语义增强的Agent架构、优先语义训练策略和语义扩展推理方案，显著提升零样本目标/实例导航中Agent的语义感知能力，在ObjectNav和新提出的InstanceNav任务上实现SOTA。

## 研究背景与动机

零样本目标导航（ZSON）要求Agent在训练时不使用任何场景物体标注，仅通过图像目标导航（ImageNav）任务预训练，然后借助CLIP等视觉-语言模型零样本迁移到目标导航任务。这是构建通用具身Agent的重要路径。但现有ZSON任务仅要求Agent找到给定类别的**任意一个**物体，距离实际应用中需要识别**特定实例**还有很大差距。

**关键发现——语义忽视问题**：论文通过精心设计的先导实验揭示了一个被忽视的重要问题：

- **Semantic-Non-dominant (SN) Agent**：使用Canny算子（破坏语义信息）和可学习ResNet50，在ImageNav任务上竟然获得了与ZSON可比的成功率
- **Semantic-Dominant (SD) Agent**：使用两个冻结的CLIP编码器获取语义信息，反而效果最差

这一反直觉的结果说明：**ImageNav预训练任务并不要求Agent学习语义信息**，仅靠布局/轮廓信息做视图匹配就能获得高成功率。因此ZSON Agent的语义感知能力实际上很弱，限制了其在依赖语义线索的导航任务上的表现。

**核心矛盾**：ImageNav预训练目标与下游语义导航需求之间的错配——训练时不需要语义就能完成任务，但零样本迁移时必须依赖语义。

**切入角度**：从Agent架构、训练策略和推理方案三个层面同时加强语义学习，确保Agent在预训练阶段就建立强大的语义感知能力。

**核心Idea**：通过选择语义清晰的目标图像、放松精确视图匹配的奖励约束、增加语义感知模块、以及用图像特征扩展文本查询，全方位提升Agent的语义理解能力。

## 方法详解

### 整体框架

PSL方法由三部分组成：(1) PSL Agent架构——增加CLIP语义观测编码器和语义感知模块(SPM)；(2) 优先语义训练策略——熵最小化目标视图选择+视角奖励放松；(3) 语义扩展推理方案——用图像特征检索丰富文本查询。三者协同工作，分别从模型能力、训练信号和推理粒度三个维度提升语义理解。

### 关键设计

1. **Semantic Perception Module (SPM)**：

   功能：编码目标图像和观测之间的语义对应关系。

   核心思路：在ZSON基础上增加一个冻结的CLIP编码器提取语义级观测 $\mathbf{z}_S$，然后用一个MLP瓶颈层将目标嵌入 $\mathbf{z}_G$ 和语义观测 $\mathbf{z}_S$ 压缩为低维语义感知嵌入 $\mathbf{z}_{SP} \in \mathbb{R}^{C_2}$（$C_2 < 2 \times C_1$）。策略网络基于语义感知和观测嵌入做决策：

    $\mathbf{s}_t, \mathbf{h}_t = \pi_\theta(\mathbf{z}_{SP} \oplus \mathbf{z}_O \oplus \mathbf{a}_{t-1} | \mathbf{h}_{t-1})$

   使用PPO训练actor-critic网络，预测6种动作（前进、左转、右转、停止、抬头、低头）。

   设计动机：原始ZSON仅靠可学习的观测编码器可能无法有效学习语义信息，通过显式加入语义感知通道并用瓶颈压缩，强制Agent关注关键语义对应关系。

2. **优先语义训练策略（Prioritized Semantic Training）**：

   功能：解决ImageNav训练数据中目标图像语义模糊的问题。

   **熵最小化目标视图选择**：对每个目标点旋转 $\Omega$ 次渲染不同视角图像，用CLIP对6个物体类别计算分类熵，选择熵最小的视图（即有明确主体物体的图像）作为目标：

    $\omega^* = \arg\min_{\omega \in \Omega} -\frac{1}{\log(|\mathcal{C}|)}\sum_{c \in \mathcal{C}} \mathbf{p}_c \log \mathbf{p}_c$

   其中 $\mathbf{p}_c = \text{softmax}(\tau \cdot \frac{\mathbf{v}_\omega^T \mathbf{q}_c}{\|\mathbf{v}_\omega\|_2\|\mathbf{q}_c\|_2})$。

   **视角奖励放松**：在多个俯仰角和偏航角渲染额外图像后选择，并修改PPO奖励函数，仅鼓励Agent朝向目标的x-z平面方向，忽略pitch角度匹配：

    $R_t^{PSL} = \underbrace{\gamma^{suc}\mathbb{1}\{d_t < \epsilon^d\}}_{\text{到达位置}} + \underbrace{\gamma^{suc}\mathbb{1}\{d_t<\epsilon^d\}\mathbb{1}\{\text{extract}_Y(\mathbf{a}_t)<\epsilon^a\}}_{\text{朝向匹配}} + r_d + r_a - \gamma^{delay}$

   设计动机：原始数据中大量目标图像是墙壁、空房间等无意义场景，这些歧义目标加剧了语义忽视问题。放松奖励使Agent关注语义对应而非精确几何匹配。

3. **语义扩展推理（Semantic Expansion Inference）**：

   功能：缓解训练用图像嵌入和测试用文本嵌入之间的粒度差异。

   核心思路：训练时维护一个支持集 $\mathcal{V}$（约0.1M个多样化图像嵌入，两两相似度低于阈值 $\lambda=0.8$）。推理时，给定文本描述生成 $\mathbf{z}_T$，通过加权检索生成目标嵌入：

    $\mathbf{z}_R = \sum_{\mathbf{v}_i \in \mathcal{V}} \frac{\exp(g(\mathbf{z}_T, \mathbf{v}_i))}{\sum_{\mathbf{v}_j \in \mathcal{V}} \exp(g(\mathbf{z}_T, \mathbf{v}_j))} \ast \mathbf{v}_i$

   设计动机：文本嵌入和图像嵌入存在模态间隙和粒度差异。用图像特征扩展文本查询，使推理时的目标嵌入粒度与训练时一致。

### 损失函数 / 训练策略

使用PPO强化学习在HM3D ImageNav数据集上预训练，7.2M个episode。每个episode随机采样4个熵最小的目标图像（从10个候选中选取）。训练完成后直接零样本迁移到ObjectNav和InstanceNav任务。

## 实验关键数据

### 主实验

**ObjectNav任务（HM3D）：**

| 方法 | 是否依赖LLM | 是否需要地图 | SR(%) | SPL(%) |
|------|-----------|------------|-------|--------|
| L3MVN | ✔ | ✔ | 35.2 | 16.5 |
| PixelNav (GPT-4) | ✔ | ✘ | 37.9 | 20.5 |
| ESC (GPT-3.5) | ✔ | ✔ | 39.2 | 22.3 |
| ZSON | ✘ | ✘ | 25.5 | 12.6 |
| **PSL (Ours)** | **✘** | **✘** | **42.4** | **19.2** |

**InstanceNav任务（Text-goal）：**

| 方法 | SR(%) | SPL(%) | 说明 |
|------|-------|--------|------|
| CoW | 1.8 | 1.1 | 需要Depth+GPS |
| ESC (GPT-3.5) | 6.5 | 3.7 | 需要Depth+GPS |
| ZSON | 10.6 | 4.9 | 端到端 |
| **PSL (Ours)** | **16.5** | **7.5** | 端到端,无额外传感器 |

**InstanceNav任务（Image-goal）：**

| 方法 | SR(%) | SPL(%) |
|------|-------|--------|
| FGPrompt | 9.9 | 2.8 |
| ZSON | 14.6 | 7.3 |
| OVRL-V2 (有监督) | 24.8 | 11.8 |
| **PSL (Ours, 无监督)** | **23.0** | **11.4** |

### 消融实验

| SPM | GVS | PRR | ZSIN-image SR | ZSIN-text SR | ZSON SR | 说明 |
|-----|-----|-----|---------------|-------------|---------|------|
| ✘ | ✘ | ✘ | 12.7 | 10.6 | 25.5 | ZSON基线 |
| ✔ | ✘ | ✘ | 19.5 | 13.0 | 33.7 | +语义感知模块 |
| ✘ | ✔ | ✘ | 14.8 | 11.8 | 30.4 | +目标视图选择 |
| ✔ | ✔ | ✘ | 16.5 | 12.3 | 35.0 | +SPM+GVS |
| ✔ | ✔ | ✔ | 22.0 | 16.5 | 42.4 | 全部组件 |

**语义扩展推理消融（Text-goal InstanceNav）：**

| PSL Agent | 语义扩展 | 支持集 | SR(%) | 说明 |
|-----------|---------|--------|-------|------|
| ✘ | ✔ | IIN 3.5K | 11.1 | 有限类别 |
| ✘ | ✔ | ImageNav 0.1M | 12.4 | 多样性更好 |
| ✔ | ✘ | - | 6.6 | 直接用文本嵌入 |
| ✔ | ✔ | ImageNav 0.1M | 16.5 | 完整方案 |

### 关键发现

1. **PSL首次在不使用LLM的情况下超越LLM-based方法**：ObjectNav SR 42.4% vs ESC 39.2%，且不需要额外传感器（Depth, GPS）
2. **SPM是最重要的单一组件**：单独加入SPM，ZSIN-image SR从12.7%提升到19.5%（+6.8%）
3. **视角放松需要与PSL Agent配合**：在原始Agent上加PRR反而降低性能（12.7%→10.8%），但在PSL Agent上效果显著（16.5%→22.0%）
4. **语义扩展推理解决了模态间隙问题**：直接用文本嵌入SR仅6.6%，通过图像检索扩展后提升到16.5%（+9.9%）
5. **PSL在Image-goal InstanceNav上接近有监督方法**：SR 23.0% vs OVRL-V2有监督的24.8%，SPL几乎持平

## 亮点与洞察

- **先导实验设计精妙**：Canny/Sobel + ResNet50的Layout-Only Agent成功率媲美ZSON，有力证明了语义忽视问题的存在
- **三位一体的解决方案**：从模型（SPM）、数据（训练策略）、推理（语义扩展）三个维度系统性地解决同一问题
- **InstanceNav任务的提出有前瞻性**：相比ObjectNav（找任意一把椅子），InstanceNav（找"米色竹框双人床"）更接近真实应用需求
- **无LLM、无地图、无额外传感器仍SOTA**：方法简洁高效，适合实际机器人部署

## 局限与展望

- 仅在模拟环境HM3D中验证，未在真实机器人上验证sim-to-real迁移效果
- InstanceNav的文本描述由CogVLM自动生成，可能存在标注噪声
- 支持集大小（0.1M）和阈值（λ=0.8）需要手动调节，可考虑自适应策略
- SPM目前只是简单的MLP瓶颈，可以尝试更复杂的语义推理模块
- ResNet-50 backbone较老，换用更大的ViT可能进一步提升

## 相关工作与启发

- ZSON [Majumdar et al., 2022] 是本文的直接基线，奠定了ImageNav预训练→零样本ObjectNav迁移的范式
- ESC [Zhou et al.] 和PixelNav利用LLM进行导航，但计算成本高且不适合实时场景
- 语义扩展推理的思想（用图像特征丰富文本查询）可推广到其他零样本视觉-语言任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 先导实验揭示语义忽视问题，解决方案三位一体且巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ ObjectNav+InstanceNav两个任务，text/image-goal两个设定，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机由先导实验自然引出，逻辑链完整
- 价值: ⭐⭐⭐⭐ 无LLM超越LLM方法，对具身导航领域有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation](../../CVPR2026/robotics/trajrag_retrieving_geometric-semantic_experience_for_zero-shot_object_navigation.md)
- [\[ECCV 2024\] SemGrasp: Semantic Grasp Generation via Language Aligned Discretization](semgrasp_semantic_grasp_generation_via_language_aligned_discretization.md)
- [\[NeurIPS 2025\] Zero-Shot Context Generalization in Reinforcement Learning from Few Training Contexts](../../NeurIPS2025/robotics/zero-shot_context_generalization_in_reinforcement_learning_from_few_training_con.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](../../AAAI2026/robotics/panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/robotics/semantic_audio-visual_navigation_in_continuous_environments.md)

</div>

<!-- RELATED:END -->
