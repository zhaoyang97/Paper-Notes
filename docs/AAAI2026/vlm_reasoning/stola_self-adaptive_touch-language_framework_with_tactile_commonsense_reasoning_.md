---
title: >-
  [论文解读] SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios
description: >-
  [AAAI 2026][多模态VLM][触觉感知] SToLa 提出首个基于混合专家（MoE）的触觉-语言框架，通过动态路由机制管理触觉和语言两种模态的差异，并构建了覆盖8种物理属性、4种交互特征的开放式触觉常识推理数据集 TactileBench，在 PhysiCLeAR 基准上以 7B 参数量超越 13B 的 Octopi 取得 SOTA。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "触觉感知"
  - "多模态推理"
  - "混合专家模型"
  - "触觉-语言模型"
  - "常识推理"
---

# SToLa: Self-Adaptive Touch-Language Framework with Tactile Commonsense Reasoning in Open-Ended Scenarios

**会议**: AAAI 2026  
**arXiv**: [2505.04201](https://arxiv.org/abs/2505.04201)  
**代码**: [项目页面](https://cocacola-lab.github.io/SToLa-Page/)  
**领域**: 多模态VLM  
**关键词**: 触觉感知, 多模态推理, 混合专家模型, 触觉-语言模型, 常识推理

## 一句话总结

SToLa 提出首个基于混合专家（MoE）的触觉-语言框架，通过动态路由机制管理触觉和语言两种模态的差异，并构建了覆盖8种物理属性、4种交互特征的开放式触觉常识推理数据集 TactileBench，在 PhysiCLeAR 基准上以 7B 参数量超越 13B 的 Octopi 取得 SOTA。

## 研究背景与动机

### 领域现状
触觉是人类与物理世界交互的基础感知方式，在视觉遮挡等场景中尤为重要。在机器人和人工智能领域，触觉传感已被广泛认为是机器人与环境交互的关键模态。近年来，研究者开始尝试将触觉信号与大语言模型结合，利用 LLM 的推理能力实现触觉常识推理。

### 现有痛点

**模态差异被忽视**: 现有触觉-语言模型（如 Octopi、Touch-LLM）将触觉简单视为语言的"子模态"——用触觉编码器将触觉数据映射到与文本相似的表示空间，然后强制共享同一个 Transformer 架构。这种做法忽略了一个关键事实：即使被映射到共享空间，触觉和语言的表征在语义上仍然是截然不同的（类比于大脑有专门的触觉处理神经通路）

**开放式触觉数据稀缺**: 现有数据集（如 PhysiCLeAR）范围有限——仅覆盖硬度、粗糙度、凹凸度3种物理属性，且采用模板化问答格式。这与真实场景中问答形式不可预测的特点严重脱节，限制了模型的泛化能力

### 核心矛盾
如何在一个统一框架中既有效地融合触觉和语言两种语义差异显著的模态，又能处理现实世界中多样、开放式的触觉推理问题。

### 本文核心 idea
利用 MoE 的动态路由特性为不同模态的 token 分配不同的专家网络——让模型**自适应地**区分和管理触觉 token 与文本 token，而非简单地共享所有参数。同时构建涵盖更广维度的开放式触觉常识推理数据集。

## 方法详解

### 整体框架
SToLa 由三个核心组件构成：**触觉编码器**（处理原始触觉数据）、**触觉-语言适配器**（桥接模态差距）、**带 MoE 层的 LLM**（动态管理多模态 token）。采用两阶段渐进训练策略。

### 关键设计

1. **统一输入处理（Input Unification）**:

    - **触觉信号统一化**: 将静态触觉图像和动态时序数据统一处理——单张图像作为单帧视频处理
    - **多传感器支持**: 兼容 GelSight 和 GelSight Mini 两种传感器配置，使用相同处理流程
    - **时序聚合**: 触觉视频输入 $X_{touch} \in \mathbb{R}^{N \times H \times W \times C}$ 经编码器独立处理 N 帧，生成帧级 token 序列后通过平均池化得到视频级表征 $\mathcal{Z}' \in \mathbb{R}^{P \times C}$
    - **触觉-文本拼接**: 适配器将触觉 token 映射到 LLM 维度后与文本 token 拼接输入 LLM
    - **设计动机**: 受 ViFi-CLIP 启发，平均池化隐式融入时序模式，实现对不同形式触觉输入的统一处理

2. **MoE 模块（核心创新）**:

    - **替换 FFN**: 在 LLM 内部的 Transformer 块中，将标准前馈网络替换为 MoE 层
    - **共享自注意力 + 专家路由**: 每个 MoE 块保留共享的自注意力层（适用于两种模态），加上路由器和基于 FFN 的专家网络
    - **路由公式**: $\mathcal{P}(\mathbf{x}) = Softmax(Top\text{-}k(x \cdot W_r, k))$，其中 $W_r \in \mathbb{R}^{D \times K}$ 为路由器权重
    - **MoE 输出**: $\text{MoE}(x) = \sum_{i=1}^{K} \mathcal{P}_i(x) \cdot E_i(x)$，加权求和激活专家的输出
    - **设计动机**: 不同专家发展出对触觉/文本 token 的不同偏好，实现模态感知的动态知识分配

3. **两阶段渐进训练**:

    - **Stage I（触觉 token 适配）**:
        - 仅训练触觉-语言适配器，冻结触觉编码器和 LLM
        - 使用 Touch100k 的触觉-语言对训练
        - 此阶段**不使用** MoE 层
        - 目标: 让 LLM 学会理解触觉输入的内容
        - 损失: 交叉熵损失 $\mathcal{L}_{ce} = -\mathbb{E}\left[\log\pi_\theta(\mathcal{Y}_i|\mathcal{V},\mathcal{T}_{<i})\right]$
    - **Stage II（MoE 端到端微调）**:
        - 冻结触觉编码器和词嵌入层，微调适配器和 LLM
        - LLM 自注意力层使用 LoRA 微调，FFN 层通过**稀疏上行扩展（sparse upcycling）**从稠密转为稀疏 MoE
        - 关键: FFN 权重从 Stage I 复制多份作为 K 个专家的初始化
        - 使用 PhysiCLeAR 数据集和自构建的触觉指令数据

### 损失函数 / 训练策略
- Stage II 的总损失: $\mathcal{L}_{total} = \mathcal{L}_{ce} + \mathcal{L}_{aux}$
- 辅助负载均衡损失: $\mathcal{L}_{aux} = \alpha \cdot K \cdot \sum_{i=1}^{K} \mathcal{F}_i \cdot \mathcal{G}_i$
    - $\mathcal{F}_i$: 分配给专家 $E_i$ 的 token 比例
    - $\mathcal{G}_i$: 路由器分配给专家 $E_i$ 的概率
    - 防止所有 token 都路由到少数专家，保持专家间的均衡利用
- 基础模型: Vicuna-7B v1.5
- 触觉编码器: TLV-Link（已与语言模态良好对齐）
- 训练硬件: 1 × Nvidia A100-80G GPU，batch size 16

## 实验关键数据

### 主实验

| 模型 | PhysiCLeAR CIDEr | PhysiCLeAR METEOR | TactileBench METEOR | TactileBench GPT-4 | TactileBench DeepSeek-R1 |
|------|------------------|-------------------|---------------------|--------------------|-----------------------|
| Touch-LLM (7B) | - | - | 17.92 | 6.88 | 7.06 |
| Octopi-7B | 138.60 | 77.63 | 21.47 | 6.91 | 7.17 |
| Octopi-13B | 141.20 | 77.79 | 28.83 | 7.85 | 7.97 |
| **SToLa (7B, Ours)** | **195.03** | **82.58** | **30.27** | **8.02** | **8.12** |

### 消融实验

| 配置 | PhysiCLeAR CIDEr | PhysiCLeAR METEOR | TactileBench GPT-4 | 说明 |
|------|------------------|-------------------|--------------------|----- |
| SToLa (完整) | 195.03 | 82.58 | 8.02 | 所有组件齐全 |
| w/o MoE | 176.79 | 81.55 | 7.44 | 去除 MoE，CIDEr 下降 18.24 |
| w/o LoRA | 166.71 | 80.39 | 7.95 | 去除 LoRA，CIDEr 下降 28.32 |
| w/o Stage I | 172.52 | 80.55 | 7.72 | 跳过第一阶段，性能显著下降 |

### PhysiCLeAR 子任务细粒度结果

| 模型 | 属性比较 | 属性最高选择 | 属性-对象匹配 | 属性场景推理 | 对象属性描述(Combined) |
|------|---------|------------|-------------|------------|----------------------|
| Octopi-7B | 48.10 | 74.67 | 44.39 | 69.57 | 47.37 |
| Octopi-13B | 55.06 | 84.00 | 60.43 | 67.39 | 55.26 |
| **SToLa** | **62.28** | 74.86 | 57.32 | **69.80** | 48.72 |

### 关键发现

1. **7B 超越 13B**: SToLa (7B) 在 PhysiCLeAR 的 CIDEr 指标上比 Octopi-13B 高出 53.83 点（195.03 vs 141.20），以更小参数量取得全面领先
2. **MoE 是核心贡献**: 去除 MoE 模块后 CIDEr 下降 18.24 点，证明动态专家路由对多模态管理的关键作用
3. **渐进训练不可或缺**: 跳过 Stage I 导致模型性能显著下降，说明先让 LLM 适应触觉输入再引入 MoE 是必要的
4. **专家发展出模态偏好**: 路由分布可视化显示不同专家对触觉和文本 token 发展出明确的选择偏好，触觉 token 倾向于专家 3 和 4，文本 token 在浅层倾向于专家 2
5. **开放式推理优势**: 在自由形式问答的 TactileBench 上，SToLa 相比 Octopi-13B 在 GPT-4 评分上提升 0.17 分（8.02 vs 7.85），展现更强的开放场景泛化能力

## 亮点与洞察

1. **MoE 用于触觉-语言模型的首次探索**: 开创性地将 MoE 引入触觉领域，证明动态专家路由比简单共享参数更有效
2. **稀疏上行扩展的巧妙设计**: 从 Stage I 的稠密 FFN 复制初始化多个专家，既保留了已学到的知识又引入了多样性
3. **TactileBench 数据集填补空白**: 涵盖 8 种物理属性（硬度、粗糙度、重量、纹理等）、4 种交互特征（可抓性、弯曲性等），采用自由形式问答
4. **认知层次设计**: TactileBench 按基础属性理解(50%)→触觉交互感知(30%)→常识驱动推理(20%)的层次结构分配数据
5. **路由可视化分析深入**: 通过 PCA 提取的 token 路径可视化清晰展示了模型如何动态管理不同模态

## 局限与展望

1. **计算资源限制**: 受限于资源仅使用 7B LLM，未能扩展到 13B，导致在某些 PhysiCLeAR 子任务上不如 Octopi-13B
2. **MoE 仅从模态级别设计**: 未从任务角度或模态+任务组合角度分配专家，可能存在更优设计
3. **触觉编码器冻结**: 未探索端到端训练触觉编码器的效果
4. **传感器覆盖有限**: 仅支持 GelSight 系列，未验证其他触觉传感器（如 BioTac、DIGIT）
5. **TactileBench 基于 GPT-4 生成**: 数据集构建中使用 GPT-4 生成问答对，可能引入偏差
6. **对象种类少**: TactileBench 仅覆盖 14 种对象，多样性有待扩展

## 相关工作与启发

- **Octopi**: 基于 Vicuna 的触觉-语言模型，引入 PhysiCLeAR 基准，是本文主要对比对象
- **Touch-LLM**: 通过对比学习将触觉嵌入与图像嵌入对齐，但不支持触觉时序信号的交错处理
- **Switch Transformer / GLaM**: MoE 在语言模型中的经典应用，证明了稀疏激活的效率优势
- **MoE-LLaVA / Uni-MoE**: MoE 在视觉-语言模型中的应用，本文将此思路扩展到触觉-语言领域
- **启发**: MoE 的动态路由特性天然适合处理多模态输入的异质性——不同语义空间的 token 应由不同的专家处理，这一思路可推广到更多模态组合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Seeing Through Touch: Tactile-Driven Visual Localization of Material Regions](../../CVPR2026/multimodal_vlm/seeing_through_touch_tactile_localization.md)
- [\[AAAI 2026\] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model](o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)
- [\[AAAI 2026\] Exploring LLMs for Scientific Information Extraction using the SciEx Framework](exploring_llms_for_scientific_information_extraction_using_the_sciex_framework.md)
- [\[ECCV 2024\] Towards Open-ended Visual Quality Comparison](../../ECCV2024/multimodal_vlm/towards_open-ended_visual_quality_comparison.md)
- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)

</div>

<!-- RELATED:END -->
