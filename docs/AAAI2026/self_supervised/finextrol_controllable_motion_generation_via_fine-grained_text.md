---
title: >-
  [论文解读] FineXtrol: Controllable Motion Generation via Fine-Grained Text
description: >-
  [AAAI 2026][自监督学习][motion generation] 提出 FineXtrol 框架，利用带时间标注的细粒度身体部位文本描述作为控制信号，通过双分支 ControlNet 架构和层级对比学习增强文本编码器的区分能力，实现高效、用户友好且精确的可控人体动作生成，在 HumanML3D 上多身体部位控制性能显著优于现有方法。
tags:
  - "AAAI 2026"
  - "自监督学习"
  - "motion generation"
  - "controllable generation"
  - "fine-grained text"
  - "对比学习"
  - "扩散模型"
---

# FineXtrol: Controllable Motion Generation via Fine-Grained Text

**会议**: AAAI 2026  
**arXiv**: [2511.18927](https://arxiv.org/abs/2511.18927)  
**代码**: 无  
**领域**: 自监督  
**关键词**: motion generation, controllable generation, fine-grained text, contrastive learning, diffusion model

## 一句话总结

提出 FineXtrol 框架，利用带时间标注的细粒度身体部位文本描述作为控制信号，通过双分支 ControlNet 架构和层级对比学习增强文本编码器的区分能力，实现高效、用户友好且精确的可控人体动作生成，在 HumanML3D 上多身体部位控制性能显著优于现有方法。

## 研究背景与动机

文本驱动的人体动作生成在动画和数字人领域应用广泛。现有可控动作生成方法主要有两类：一类利用 LLM 扩展文本描述来增强生成精度，但扩展描述与 ground-truth 动作常常不对齐，且缺乏显式的时间线索（如何时抬手）；另一类使用全局 3D 坐标序列作为控制信号，虽然精确但计算代价高昂（需要坐标系转换），且用户难以提供合理的坐标序列。核心矛盾在于：既要精确可控，又要用户友好且计算高效。本文的切入点是：使用 FineMotion 数据集中与 ground-truth 动作严格对齐的、带时间标注的细粒度文本描述（如"在1.0-1.5s内将左手移向左大腿"）作为控制信号，结合 ControlNet 范式和层级对比学习来实现高效可控生成。

## 方法详解

### 整体框架

FineXtrol 采用双分支框架（类似 ControlNet），输入为粗粒度文本 $\boldsymbol{p}$、细粒度文本控制信号 $\boldsymbol{c}$ 和噪声动作序列 $\boldsymbol{x_t}$，输出去噪后的干净动作序列 $\boldsymbol{x_0}$。下分支复用预训练的 MDM (Motion Diffusion Model) 保持稳定的粗粒度文本条件生成能力；上分支是 MDM 的可训练副本，通过条件特征自适应接收细粒度控制信号的调制。两分支通过零初始化线性层连接。

### 关键设计

**1. 基于文本的细粒度控制信号注入机制**

与直接将细粒度文本和粗粒度文本拼接为单一输入（"Direct"方式）不同，FineXtrol 将控制信号作为残差引导来调制动作特征。具体地，上分支先构建与下分支相同的文本-动作嵌入 $\boldsymbol{e}'$，然后将控制信号 $\boldsymbol{c}$ 经文本编码器提取嵌入 $\boldsymbol{e}_c$，通过线性层对齐后加到 $\boldsymbol{e}'$ 上：

$$\boldsymbol{h}_0^{\text{ctrl}} = \boldsymbol{e}' + \text{Linear}(\boldsymbol{e}_c)$$

上下分支在第 $l$ 层的交互通过零初始化线性层 $\mathcal{P}_l$ 实现：

$$\boldsymbol{h}_l^{\text{out}} = \boldsymbol{h}_l^{\text{ori}} + \mathcal{P}_l(\boldsymbol{h}_l^{\text{ctrl}})$$

训练时对控制信号施加随机掩码（将随机时间区间替换为 `<Mask>`），增强模型对部分控制的鲁棒性。

**2. 层级对比学习增强文本编码器**

CLIP 和 T5 等预训练文本编码器在细粒度动作描述上缺乏区分性。论文分析控制信号的三层结构（句子级→片段级→序列级），设计了层级对比学习模块，以 T5 为基础编码器，逐级训练：

- **句子级**：构建身体部位运动句子语料库，使用 DeepSeek-V2 重写生成正样本对，采用 InfoNCE 损失
- **片段级**：对单个时间区间内的句子进行随机替换和打乱顺序，生成正样本对
- **序列级**：在保持时间顺序的前提下，对各时间区间分别应用片段级增强

每一级的训练从上一级学到的权重初始化，对比学习损失为：

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(\boldsymbol{z}_i, \boldsymbol{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\boldsymbol{z}_i, \boldsymbol{z}_k)/\tau)}$$

**3. 高效推理设计**

由于使用文本而非坐标作为控制信号，FineXtrol 无需进行姿态表示之间的坐标转换，推理速度更快，可训练参数更少。

### 损失函数 / 训练策略

训练目标继承 MDM 的简单重建损失 $\mathcal{L}_\theta = \|\epsilon_\theta(\boldsymbol{x}_t, t, \boldsymbol{p}; \theta) - \boldsymbol{\hat{x}_0}\|_2^2$。先进行对比学习训练文本编码器（三级渐进式），然后冻结编码器后训练 FineXtrol 框架。所有实验在单张 A100 40G GPU 上完成。

## 实验关键数据

### 主实验

在 HumanML3D 测试集上与现有可控动作生成方法的对比：

| 方法 | 控制信号 | 用户友好 | FID ↓ | R-Top3 ↑ | Diversity | MM-Dist ↓ |
|------|---------|---------|-------|----------|-----------|-----------|
| Real | - | - | 0.002 | 0.796 | 9.503 | 2.965 |
| MDM | - | - | 0.544 | 0.611 | 9.559 | 5.432 |
| OmniControl | 坐标 | ✗ | 0.255 | 0.680 | 9.735 | 5.054 |
| InterControl | 坐标 | ✗ | 0.209 | 0.684 | 9.301 | 5.164 |
| CoMo | 文本 | ✓ | 0.347 | 0.625 | 9.568 | 5.588 |
| **FineXtrol** | **文本** | **✓** | **0.245** | **0.685** | **9.492** | **5.087** |

多身体部位交叉控制（更难的设定）：

| 方法 | FID ↓ | R-Top3 ↑ | MM-Dist ↓ |
|------|-------|----------|-----------|
| OmniControl | 0.624 | 0.601 | 5.252 |
| CoMo | 0.606 | 0.611 | 5.638 |
| **FineXtrol** | **0.351** | **0.676** | **5.146** |

推理效率对比：

| 方法 | 推理时间(s) ↓ | 可训练参数 |
|------|-------------|----------|
| OmniControl | 168.51 | 48.79M |
| InterControl | 159.72 | 42.00M |
| GMD | 153.25 | 238.63M |
| **FineXtrol** | **128.57** | **23.39M** |

### 消融实验

| 消融项 | FID ↓ | R-Top3 ↑ |
|--------|-------|----------|
| Direct 拼接控制范式 | 1.383 | 0.601 |
| **Ours (残差引导)** | **0.245** | **0.685** |

| 文本编码器 | FID ↓ | R-Top3 ↑ | MM-Dist ↓ |
|-----------|-------|----------|-----------|
| CLIP | 0.579 | 0.603 | 5.927 |
| T5 | 0.374 | 0.659 | 5.483 |
| **Ours (层级对比)** | **0.245** | **0.685** | **5.087** |

### 关键发现

- FineXtrol 在多部位交叉控制场景中性能仅轻微下降，而 OmniControl 和 CoMo 性能显著恶化
- 用户研究中 33 名受试者 78.79%（无控制信号时）和 74.24%（有控制信号时）偏好 FineXtrol
- 直接拼接文本的 Direct 范式性能远不如残差引导范式，说明单分支难以处理密集信息

## 亮点与洞察

- 用细粒度文本替代坐标序列作为控制信号的思路很新颖，既保留了精确控制能力，又大幅降低了计算成本和用户使用门槛
- 层级对比学习模块针对控制信号的三层结构设计了对应的数据增强策略，有效解决了预训练编码器对细粒度语义区分不足的问题
- 零初始化连接确保训练初期不注入噪声，渐进式学习控制信号语义

## 局限与展望

- 依赖 FineMotion 数据集提供与 ground-truth 对齐的细粒度标注，泛化到开放域文本控制的能力有待验证
- 当前仅在 HumanML3D 数据集上验证，缺乏对更多动作数据集的评估
- 控制信号的精度依赖文本描述的质量，用户手动编写细粒度描述仍有一定门槛

## 相关工作与启发

- **vs OmniControl / InterControl**: 这两种方法使用 3D 坐标序列作为控制信号，精确但需要坐标转换且对用户不友好；FineXtrol 用文本替代坐标，推理速度快 30s+，参数量仅为 OmniControl 的一半
- **vs CoMo**: CoMo 使用 LLM 扩展的文本但缺乏时间标注且不与 ground-truth 对齐，FineXtrol 使用 FineMotion 的对齐描述，在 R-Top3 上高出 0.060

## 评分

- 新颖性: ⭐⭐⭐⭐ 用细粒度文本替代坐标序列的控制范式思路新颖，层级对比学习设计合理
- 实验充分度: ⭐⭐⭐⭐ 定量对比、消融、用户研究、可视化均有覆盖，实验设计系统
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机阐述充分，图表丰富
- 价值: ⭐⭐⭐⭐ 为可控动作生成提供了一种实用高效的新范式，具有落地潜力
---
title: >-
  [论文解读] FineXtrol: Controllable Motion Generation via Fine-Grained Text
description: >-
  [AAAI2026][人体理解][运动生成] 提出 FineXtrol 框架，使用带时间标注的细粒度文本描述作为控制信号，结合层次化对比学习增强 text encoder 的判别力，实现对特定身体部位在指定时间区间内的精确动作生成控制。
tags:
  - AAAI2026
  - 人体理解
  - 运动生成
  - fine-grained control
  - 对比学习
  - 扩散模型
  - ControlNet
---


<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TAR: Token-Aware Refinement for Fine-grained Generalized Category Discovery](../../CVPR2026/self_supervised/tar_token-aware_refinement_for_fine-grained_generalized_category_discovery.md)
- [\[CVPR 2026\] Nonparametric Deep Fine-grained Clustering with Low-Rank Guided Vision-Language Model](../../CVPR2026/self_supervised/nonparametric_deep_fine-grained_clustering_with_low-rank_guided_vision-language_.md)
- [\[CVPR 2026\] From Few-way to Many-way: Rethinking Few-shot Fine-grained Image Classification](../../CVPR2026/self_supervised/from_few-way_to_many-way_rethinking_few-shot_fine-grained_image_classification.md)
- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](../../ICCV2025/self_supervised/generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)
- [\[NeurIPS 2025\] CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing](../../NeurIPS2025/self_supervised/cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)

</div>

<!-- RELATED:END -->
