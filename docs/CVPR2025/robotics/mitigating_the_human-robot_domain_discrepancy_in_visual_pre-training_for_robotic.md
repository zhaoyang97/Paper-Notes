---
title: >-
  [论文解读] Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation
description: >-
  [CVPR 2025][机器人][robotic manipulation] 提出 HR-Align 适配范式，利用配对人-机器人视频数据和对比对齐损失，以参数高效的方式弥合人类数据预训练模型与机器人域之间的语义差距，在 20 个仿真任务和 5 个真实任务上平均成功率提升 7%+。 领域现状： 机器人操作领域的视觉表示学习面…
tags:
  - "CVPR 2025"
  - "机器人"
  - "robotic manipulation"
  - "visual pre-training"
  - "human-robot domain gap"
  - "contrastive alignment"
  - "parameter-efficient adapter"
---

# Mitigating the Human-Robot Domain Discrepancy in Visual Pre-training for Robotic Manipulation

**会议**: CVPR 2025  
**arXiv**: [2406.14235](https://arxiv.org/abs/2406.14235)  
**代码**: [项目主页](https://jiaming-zhou.github.io/projects/HumanRobotAlign)  
**领域**: 机器人  
**关键词**: robotic manipulation, visual pre-training, human-robot domain gap, contrastive alignment, parameter-efficient adapter

## 一句话总结

提出 HR-Align 适配范式，利用配对人-机器人视频数据和对比对齐损失，以参数高效的方式弥合人类数据预训练模型与机器人域之间的语义差距，在 20 个仿真任务和 5 个真实任务上平均成功率提升 7%+。

## 研究背景与动机

**领域现状**: 机器人操作领域的视觉表示学习面临数据稀缺问题。现有方案利用大规模人类活动数据集（如 Ego4D、Kinetics）预训练视觉模型，然后将冻结模型作为机器人策略学习的视觉骨干。

**核心矛盾**: 人类和机器人之间存在显著的形态学差异（morphological differences），导致「人-机器人域差距」问题——预训练模型在人类数据上学到的表示难以有效迁移到机器人域。

**现有方案的不足**:

1. **操作导向代理任务方案**（如手部检测）：在人类数据上定义代理任务来间接适配预训练模型，但缺少对机器人数据的显式接触，无法直接缓解域差距
2. **下游微调方案**（在每个下游环境中微调）：需要为每个不同环境定制预训练模型，牺牲了模型的通用性

**本文切入点**: 提出一种新的「适配范式」，利用社区中已有的配对人-机器人演示数据（如 RH20T 数据集）作为桥梁，在保持模型通用性的同时缓解域差距。核心洞察是：配对数据中人类和机器人演示的动态语义是对齐的，可以利用这种对齐关系来指导适配。

## 方法详解

### 整体框架

HR-Align（Human-Robot Semantic Alignment）采用三流架构：

- **冻结人类流**：冻结预训练模型 $\mathcal{F}$ 提取人类视频特征 $h^f$
- **冻结机器人流**：同一冻结模型提取机器人视频特征 $r^f$（未适配，作为负样本参考）
- **适配机器人流**：在预训练模型中注入可学习 Adapter 模块，提取适配后的机器人视频特征 $r^t$

三流特征经任务感知注意力聚合后，通过对比对齐损失训练 Adapter 参数。

### 关键设计

#### 1. 参数高效 Adapter 模块

在预训练模型的中间层插入轻量级 Adapter，采用残差结构实现特征适配：

$$r^{t,next} = r^{f,inter} + \text{Conv}_{up}(g(\text{Conv}_{down}(r^{f,inter})))$$

- $\text{Conv}_{down}$：通道降维卷积
- $g$：激活函数
- $\text{Conv}_{up}$：通道升维卷积
- 仅训练 Adapter 参数，保持预训练骨干冻结，实现参数高效适配

#### 2. 任务感知特征建模

引入任务描述文本作为 query，通过注意力机制从视频 spatiotemporal 特征中提取与当前任务相关的语义：

- 使用冻结 DistilBert 编码任务描述 $L$，得到 query $l$
- 对每个流的视频特征计算注意力权重 $\mathcal{A}^r = \text{softmax}(r^t \cdot l)$
- 加权聚合得到任务感知特征 $\bar{r}^t = (r^t)^T \cdot \mathcal{A}^t$

#### 3. 人-机器人对比对齐损失

设计双向对比损失来约束适配过程，包含两个核心原则：

**原则一**：对于配对的人-机器人视频，适配后的机器人特征 $\bar{r}_i^t$ 应比未适配的 $\bar{r}_i^f$ 与人类特征 $\bar{h}_i^f$ 更一致

**原则二**：配对的人-机器人特征在批次中应比不配对的更相似（标准对比学习思想）

$$\mathcal{L} = \frac{1}{2M}\sum_{i=1}^{M} -\log\frac{\mathcal{S}(\bar{h}_i^f, \bar{r}_i^t)}{\mathcal{S}(\bar{h}_i^f, \bar{r}_i^t) + \mathcal{S}(\bar{h}_i^f, \bar{r}_i^f) + \sum_{j \neq i}\mathcal{S}(\bar{h}_i^f, \bar{r}_j^t)} + \text{对称项}$$

其中 $\mathcal{S}(x,y) = \exp(x^T y / \tau)$，$\tau=0.1$。损失的独特之处在于将「未适配机器人特征」$\bar{r}_i^f$ 也纳入分母作为负样本，直接惩罚域差距。

### 损失函数

总损失仅包含上述人-机器人对比对齐损失，训练时优化 Adapter 参数和 task-aware 线性层参数。

**训练配置**: Adam 优化器，lr=$1 \times 10^{-4}$，batch size=200，约 8k 步，4×NVIDIA A6000。

## 实验关键数据

### 主实验表

| 设置 | 模型 | 基线 | +HR-Align | 提升 |
|------|------|------|-----------|------|
| Adroit 单任务 (2 tasks) | D4R | 63.0% | 65.0% | +2.0% |
| Adroit 单任务 (2 tasks) | R3M | 74.0% | 81.3% | **+7.3%** |
| RLBench 多任务 (18 tasks) | D4R | 55.3% | 59.9% | +4.6% |
| RLBench 多任务 (18 tasks) | R3M | 50.3% | 59.2% | **+8.9%** |
| 真实环境 (5 tasks) | D4R | — | — | **+13%** |
| 真实环境 (5 tasks) | R3M | — | — | **+11%** |

### 消融表

| 方法 | learned params | pen | relocate | Avg |
|------|---------------|-----|----------|-----|
| R3M（冻结） | 0M | 78.0 | 70.0 | 74.0 |
| R3M-PreT（继续人类预训练） | 25M | 78.0 | 77.3 | 77.7 |
| R3M-ClS（动作分类微调） | 25M | — | — | 较差 |
| R3M-Align（本文） | 少量 | 81.3 | 81.3 | **81.3** |

### 关键发现

1. **跨模型一致有效**: 在 R3M 和 D4R 两个完全不同预训练方法的模型上都显著提升，验证方法的通用性
2. **多任务设置提升更大**: R3M 在 RLBench 18 个多任务上提升 8.9%，说明适配后的模型在多样化任务上泛化更好
3. **真实环境大幅提升**: 真实任务上提升 11-13%，远超仿真环境，表明域差距在真实场景中更严重
4. **参数高效**: 仅训练少量 Adapter 参数即可获得显著提升，无需全参数微调

## 亮点与洞察

1. **范式创新**: 首次提出「配对数据桥接」的适配范式，介于冻结使用和下游微调之间，兼顾通用性和域适应
2. **负样本设计精巧**: 将未适配的机器人特征作为对比学习的额外负样本，直接量化和惩罚域差距
3. **低成本高回报**: 利用社区已有数据集（RH20T）的 56k 配对视频即可完成适配，无需额外收集数据
4. **对下游无侵入**: 适配后的模型作为通用视觉骨干使用，无需为每个下游环境定制

## 局限性

1. **依赖配对演示数据**: 需要人类和机器人执行相同任务的配对视频，数据获取虽已有公开数据集但仍有限
2. **适配与下游环境的视觉差异**: 适配阶段的机器人演示数据与下游机器人环境视觉外观不同，仅靠「同构机器人结构」来缩小差距
3. **仅验证了 Adapter 位置在最后一层**: 未充分探索不同层级插入 Adapter 的效果
4. **图像分辨率限制**: 实验中采用的分辨率和帧数较低（5帧），可能限制了时间建模能力

## 相关工作与启发

- **R3M, MVP, data4robotics**: 人类数据预训练三大基线，本文在它们之上做域适应
- **RH20T 数据集**: 提供了高质量配对人-机器人演示数据，使本文方法成为可能
- **参数高效微调（PEFT）**: Adapter 设计借鉴了 NLP/CV 领域的 PEFT 方法论
- **启发**: 「域差距」是 embodied AI 中被忽视但关键的问题，配对数据作为桥梁是一个通用思路，可推广到不同 embodiment 之间的迁移

## 评分 ⭐

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Robotic Visual Instruction](robotic_visual_instruction.md)
- [\[CVPR 2025\] A Data-Centric Revisit of Pre-Trained Vision Models for Robot Learning](a_data-centric_revisit_of_pre-trained_vision_models_for_robot_learning.md)
- [\[CVPR 2025\] 3D-MVP: 3D Multiview Pretraining for Robotic Manipulation](3d-mvp_3d_multiview_pretraining_for_manipulation.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[NeurIPS 2025\] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data](../../NeurIPS2025/robotics/egobridge_domain_adaptation_for_generalizable_imitation_from_egocentric_human_da.md)

</div>

<!-- RELATED:END -->
