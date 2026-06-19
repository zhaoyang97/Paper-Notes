---
title: >-
  [论文解读] GRIM: Task-Oriented Grasping with Conditioning on Generative Examples
description: >-
  [AAAI 2026][机器人][任务导向抓取] 本文提出 GRIM（Grasp Re-alignment via Iterative Matching），一种**免训练**的任务导向抓取（TOG）框架，通过 **retrieve–align–transfer** 流水线结合视频生成模型和多源记忆库，利用基于 DINO 特征的语义 3D 对齐实现跨物体的功能性抓取迁移，仅用 210 个记忆实例即超越了在 379K 样本上训练的 GraspMolmo。
tags:
  - "AAAI 2026"
  - "机器人"
  - "任务导向抓取"
  - "免训练"
  - "视频生成模型"
  - "语义对齐"
  - "抓取迁移"
---

# GRIM: Task-Oriented Grasping with Conditioning on Generative Examples

**会议**: AAAI 2026  
**arXiv**: [2506.15607](https://arxiv.org/abs/2506.15607)  
**代码**: [项目主页](https://grim-tog.github.io/)  
**领域**: 机器人  
**关键词**: 任务导向抓取, 免训练, 视频生成模型, 语义对齐, 抓取迁移

## 一句话总结

本文提出 GRIM（Grasp Re-alignment via Iterative Matching），一种**免训练**的任务导向抓取（TOG）框架，通过 **retrieve–align–transfer** 流水线结合视频生成模型和多源记忆库，利用基于 DINO 特征的语义 3D 对齐实现跨物体的功能性抓取迁移，仅用 210 个记忆实例即超越了在 379K 样本上训练的 GraspMolmo。

## 研究背景与动机

### 从几何抓取到功能性抓取

传统抓取合成主要关注几何稳定性——"能不能拿起来"。但真正的操作智能在于选择**功能合适**的抓取方式——"为了完成任务 X，应该怎么拿"。例如锤子必须握住把手才能用来锤钉，而不是握锤头。这就是任务导向抓取（Task-Oriented Grasping, TOG）需要解决的问题。

### 数据瓶颈

TOG 的核心瓶颈是**数据稀缺**：
- **监督学习方法**（如 TaskGrasp, GraspGPT）依赖大规模手工标注的数据集，标注"哪些抓取适合哪些任务"
- **知识图谱方法**需要大量工程构建和维护
- 即使利用 LLM/VLM 的开放世界知识（如 GraspGPT, GraspMolmo），仍需在预定义的任务-抓取数据集上训练

### GRIM 的核心思路

完全免训练！通过以下方式绕过数据瓶颈：
1. 从多种低成本来源（AI 生成视频、网络图片、人工示范）构建小型记忆库
2. 利用语义特征（而非几何形状）进行跨物体对齐
3. 将迁移的抓取姿态与几何稳定的候选抓取融合

## 方法详解

### 整体框架

GRIM 遵循 **Retrieve → Align → Transfer** 的三阶段流水线：

1. **Retrieve**：查询记忆库找到最相关的先验经验（基于 DINO 视觉相似性 + CLIP 任务语义相似性）
2. **Align**：将检索到的记忆物体与场景物体进行 3D 语义对齐
3. **Transfer & Refine**：将任务抓取姿态迁移到场景物体，并与几何稳定的候选抓取融合

### 关键设计

#### 1. **多源记忆构建流水线**

每个记忆实例是一个四元组 $(F_M, G_t, T, O)$：特征网格、6D 任务抓取姿态、任务描述、物体名称。

记忆来源包括：
- **AI 生成视频**：用 VLM（Gemini Pro）生成文本描述 → VGM（VEO2）生成视频 → 采样帧提取抓取。可大规模、低成本生成
- **网络图片**：从网上爬取展示抓取动作的图片，用 VLM 生成任务描述
- **专家示范**：机器人失败时，人提供单张示范图片，无缝加入记忆库

从图像/视频帧中提取抓取的方法：使用手-物体重建模型提取物体网格和手部网格 → 从手部网格推导 6D 平行钳式抓取姿态（利用拇指、食指中指质心和手掌质心确定夹持方向和接近方向）。

特征网格 $F_M$ 的构建：在物体网格表面采样点并计算密集 DINOv2 特征向量，形成语义描述符场。

#### 2. **基于联合相似度的记忆检索**

给定场景物体的点云 DINO 特征 $\bar{F}_{SO}^D$ 和任务 CLIP 嵌入 $E_{T_S}$，检索分数为：

$$S_{\text{joint}}(i) = \alpha \cdot \text{sim}_{\cos}(\bar{F}_{SO}^D, \bar{F}_{MO,i}^D) + (1-\alpha) \cdot \text{sim}_{\cos}(E_{T_S}, E_{T_{M,i}})$$

其中 $\alpha=0.5$ 平衡视觉和任务语义相似性。这种设计允许系统在"看起来像"和"任务匹配"之间取得平衡。

#### 3. **语义 3D 对齐（核心创新）**

传统 ICP 仅基于几何进行对齐——当物体形状不同（如金属锅铲 vs 塑料锅铲）时会失败。GRIM 提出粗到精的语义导向对齐：

**粗对齐**：
- 用 PCA 将 DINO 特征降至 4 维
- 在三个欧拉角各采样 8 步（$8^3=512$ 个候选旋转）
- 对每个候选旋转，计算联合特征-几何代价（$w_f=100, w_g=10$，重度偏向语义特征）
- 选择代价最低的 10 个候选

**精对齐**：
- 用最佳粗对齐结果初始化标准 ICP 算法进行几何精炼
- 输出最终变换 $T_{\text{final}}$

**设计动机**：语义引导初始猜测 + 几何精炼 = 即使物体"语义相似但几何不同"也能鲁棒对齐。

#### 4. **抓取迁移与精炼**

迁移：$G_S = T_{\text{final}} \cdot G_M$

用 AnyGrasp 在场景物体上生成 $N$ 个几何稳定的候选抓取 $\{G_{A,i}\}$。对每个候选计算任务兼容性分数：

$$S_{\text{task},i} = \underbrace{(\mathbf{v}_{\text{target}} \cdot \mathbf{v}_{A,i})}_{\text{方向相似}} + \underbrace{\exp(-\frac{\|\mathbf{t}_{A,i} - \mathbf{t}_S\|^2}{2\sigma^2})}_{\text{位置相似}}$$

最终分数为加权和：$S_i = w_{\text{task}} S_{\text{task},i} + w_{\text{geo}} S_{\text{geo},i}$，其中 $w_{\text{task}}=0.95, w_{\text{geo}}=0.05$（重度偏向任务兼容性，因为 AnyGrasp 已确保几何质量）。

## 实验关键数据

### 主实验：TaskGrasp 数据集上的 mAP

| 方法 | All Data | Held-out Objects | Held-out Tasks |
|------|----------|-----------------|----------------|
| Random | 0.49 | 0.41 | 0.43 |
| RTAGrasp（免训练SOTA） | 0.58 | 0.52 | 0.51 |
| GraspMolmo（379K训练） | 0.62 | 0.57 | 0.55 |
| **GRIM（免训练，210实例）** | **0.67** | **0.65** | **0.64** |

- GRIM 在全数据集上超越 GraspMolmo 5 个点——后者使用了 379K 标注样本训练
- 在 held-out 泛化场景中优势更明显：GRIM 仅下降约 3%，RTAGrasp 下降超过 10%

### 消融实验

| 配置 | mAP (All Data) | 说明 |
|------|----------------|------|
| GRIM w/o Semantic Alignment | 0.50 | 接近随机，证明语义对齐是最关键组件 |
| GRIM w/o Grasp Refinement | 0.59 | 还行但不够好，精炼步骤将功能意图转化为物理可行 |
| **GRIM (Full Model)** | **0.67** | 两个组件缺一不可 |

### 真机验证

在 Kinova Gen3 Lite 上用两个 RGB-D 相机测试 5 种新物体 × 10 次试验：

| 结果 | 数值 | 说明 |
|------|------|------|
| 成功率 | 39/50 (78%) | 失败源于点云噪声和标定误差，非抓取选择错误 |
| 推理时间 | ~10 秒 | 在线推理高效 |
| 记忆构建 | ~7 分钟/实例 | 一次性离线成本 |

### 关键发现

1. **语义对齐是核心**：移除后性能近乎随机。DINO 特征使得系统能处理"功能相似但形状不同"的物体
2. **免训练优于大规模训练**：210 个多源记忆实例 > 379K 标注样本，说明检索-对齐范式对抓取迁移非常有效
3. **泛化鲁棒性强**：Held-out 场景下性能几乎不下降，说明语义对齐具有强泛化能力
4. **失败模式明确**：依赖于输入点云质量——严重传感器噪声或极度稀疏时语义对应建立失败

## 亮点与洞察

- **数据效率极其出色**：用 210 个未经精选的记忆实例打败了在 379K 精心标注数据上训练的模型
- **生成式 AI 在机器人学中的优雅应用**：不是用 VGM 直接做决策，而是用它来"想象"抓取场景、创造训练数据，非常聪明
- **语义优先、几何精炼**的设计理念：在 3D 对齐中把语义权重设为几何的 10 倍，在最终评分中把任务兼容性权重设为几何质量的 19 倍——这种强先验非常合理
- **终身学习能力**：记忆库可在运行时通过人工示范动态扩展

## 局限与展望

- 依赖多个上游预训练模型（Gemini Pro, VEO2, SAM, 手物重建等），可能继承幻觉和偏差
- 离线记忆构建每个实例需 7 分钟，大规模记忆库的可扩展性受限
- 真机实验中失败主要源于点云质量——需要更好的感知前端
- 未考虑非刚性物体或可变形物体的抓取

## 相关工作与启发

- 与 RTAGrasp 的关键区别在于 GRIM 使用 3D 语义特征对齐而非 2D 特征匹配，对视角变化更鲁棒
- 与 GraspGPT 的关键区别在于完全免训练——避免了数据获取瓶颈
- 启发：在机器人操作中，**检索+对齐+精炼** 的范式可能是一种普适的高效学习方式

## 评分

- 新颖性: ⭐⭐⭐⭐ — 语义 3D 对齐 + 生成式记忆构建是核心创新
- 实验充分度: ⭐⭐⭐⭐ — 仿真基准+消融+真机全面覆盖
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，公式规范
- 价值: ⭐⭐⭐⭐⭐ — 免训练超越大规模训练方法具有很强示范意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors](towards_affordance-aware_robotic_dexterous_grasping_with_human-like_priors.md)
- [\[ICML 2025\] Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples](../../ICML2025/robotics/flow_of_reasoning_training_llms_for_divergent_reasoning_with_minimal_examples.md)
- [\[CVPR 2026\] Obstruction Reasoning for Robotic Grasping](../../CVPR2026/robotics/obstruction_reasoning_for_robotic_grasping.md)
- [\[CVPR 2026\] Humanoid Generative Pre-Training for Zero-Shot Motion Tracking](../../CVPR2026/robotics/humanoid_generative_pre-training_for_zero-shot_motion_tracking.md)
- [\[AAAI 2026\] Human-Centric Open-Future Task Discovery: Formulation, Benchmark, and Scalable Tree-Based Search](human-centric_open-future_task_discovery_formulation_benchmark_and_scalable_tree.md)

</div>

<!-- RELATED:END -->
