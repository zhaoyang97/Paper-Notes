---
title: >-
  [论文解读] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs
description: >-
  [ICCV 2025][多模态VLM][Composed Pose Retrieval] 本文提出AutoComPose，首个利用多模态大语言模型（MLLM）自动生成人体姿态转换描述的框架，通过身体部位级描述生成、多样化增强和循环一致性损失，在取代昂贵的人工标注的同时实现了更优的组合姿态检索性能。 组合姿态检索（CPR）允许…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "Composed Pose Retrieval"
  - "MLLM"
  - "Pose Transition"
  - "Cyclic Consistency"
  - "Data Annotation"
---

# AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs

**会议**: ICCV 2025  
**arXiv**: [2503.22884](https://arxiv.org/abs/2503.22884)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: Composed Pose Retrieval, MLLM, Pose Transition, Cyclic Consistency, Data Annotation

## 一句话总结
本文提出AutoComPose，首个利用多模态大语言模型（MLLM）自动生成人体姿态转换描述的框架，通过身体部位级描述生成、多样化增强和循环一致性损失，在取代昂贵的人工标注的同时实现了更优的组合姿态检索性能。

## 研究背景与动机
组合姿态检索（CPR）允许用户通过指定一个参考姿态和一段转换描述来搜索目标姿态，是组合图像检索（CIR）在人体姿态领域的特化应用。CPR面临的核心瓶颈在于**标注困难**：

**人工标注成本高**：姿态转换涉及多个关节部位的精细运动描述，标注者可能遗漏细微变化、使用不一致的措辞或引入主观语言

**基于规则的生成受限**：如PoseFix的方法依赖预定义的聚合规则和模板化句子，受限于固定的"paircode"描述符（基于绝对3D关键点位置），表达能力和泛化性不足

**数据稀缺**：与FashionIQ、CIRR等CIR数据集不同，CPR缺乏大规模标注数据，且姿态变化是连续的、缺少纹理线索

核心思路：利用MLLM的姿态理解能力自动生成富有表达力、结构化且多样的姿态转换描述，同时通过循环一致性约束缓解MLLM可能的错误输出。

## 方法详解

### 整体框架
AutoComPose分为两个主要步骤：（1）自动生成姿态转换描述（三阶段pipeline）；（2）带循环约束的检索模型训练。

### 关键设计

1. **阶段一：身体部位级描述生成**:

    - 功能：利用MLLM逐个分析和比较两个姿态之间各身体部位的变化
    - 核心思路：将人体分解为头部、颈部、肩膀、手臂、肘部、手腕、手部、躯干、臀部、腿部、膝盖、足踝和脚部等关键解剖标志点。MLLM为每个发生变化的部位生成简洁的运动描述
    - 设计动机：直接生成整体描述容易遗漏细微但关键的关节运动（如手腕旋转、膝盖屈曲），逐部位分析确保精细覆盖

2. **阶段二：整合与多样化**:

    - 功能：将身体部位级描述整合为自然流畅的完整句子，并生成多个释义变体
    - 核心思路：提示MLLM将结构化的逐部位描述合成为连贯叙述，鼓励使用类比表达增强直觉性。默认为每对姿态生成3个释义版本
    - 设计动机：用户实际查询是自然语言句子而非结构化列表，多释义覆盖语言变异性

3. **阶段三：交换与镜像增强**:

    - 功能：通过交换（时间反转）和镜像（左右翻转）输入图像对自动生成更多转换描述
    - 核心思路：对每对姿态应用交换、镜像、以及二者组合三种变换，每种变换下MLLM生成对应的转换描述。默认情况下，每对姿态最终产生 $3 \times 4 = 12$ 条描述
    - 设计动机：在CPR中直接对图像做数据增强需要同步修改描述，传统方法需要额外标注。AutoComPose可自动生成一致的描述

4. **循环一致性约束**:

    - 功能：训练时通过正向和反向转换描述构建循环约束，缓解MLLM生成错误
    - 核心思路：基于假设——若组合特征（参考图+正向描述）正确，其应能通过反向描述映射回参考图特征。总训练损失为：$L_{total} = \omega \cdot L_{bbc} + (1-\omega) \cdot L_{cycle}$，其中 $L_{bbc}$ 是标准batch分类损失，$L_{cycle}$ 约束组合特征经反向转换后应匹配参考图像。$\omega = 0.5$
    - 设计动机：MLLM可能生成错误描述（误认身体部位、幻觉不存在的运动），循环约束提供自验证机制而无需检测和修正错误

### 训练细节
基于CLIP的四种backbone（RN50、RN101、ViT-B/32、ViT-B/16），分两步训练：先微调文本编码器50个epoch，再在冻结编码器的情况下训练Combiner模块100个epoch。MLLM使用GPT-4o。

## 实验关键数据

### 主实验（CLIP-RN50 + Combiner）

| 数据集 | 描述来源 | R@1 | R@5 | R@10 | R@50 |
|--------|---------|-----|-----|------|------|
| FIXMYPOSE | 人工标注 | 3.14 | 13.14 | 20.98 | 44.12 |
| FIXMYPOSE | **AutoComPose** | **9.41** | **31.76** | **43.92** | **75.49** |
| PoseFixCPR | 人工标注 | 67.93 | 82.32 | 86.36 | 94.11 |
| PoseFixCPR | **AutoComPose** | **81.40** | **92.68** | **94.95** | **98.15** |
| PoseFixCPR | 基于规则 | 73.15 | 86.62 | 90.07 | 96.46 |
| PoseFixCPR | **AutoComPose** | **81.40** | **92.68** | **94.95** | **98.15** |

### 消融实验（CLIP-RN50）

| 配置 | FIXMYPOSE R@1 | FIXMYPOSE R@50 | PoseFixCPR R@1 | 说明 |
|------|-------------|--------------|---------------|------|
| AutoComPose (完整) | 8.24 | 63.53 | 61.36 | 全部组件 |
| (-) Cyclic | 5.88 | 55.10 | 56.48 | 移除循环损失 |
| (-) SW & MI | 2.35 | 34.51 | 48.15 | 移除交换和镜像 |
| 1个释义 | 5.88 | 54.31 | — | 无多样化 |
| 3个释义 (默认) | 8.24 | 63.53 | — | 适度多样化 |
| 5个释义 | 9.02 | 64.71 | — | 更多多样化 |

### 关键发现
- AutoComPose在**所有配置和数据集**上一致超越人工标注，从未落后
- 交换与镜像增强贡献最大（移除后R@50在FIXMYPOSE上从63.53降至34.51，近乎腰斩）
- 循环一致性损失提供稳定提升（+2~5个百分点），零推理成本
- 使用更小的GPT-4o mini生成描述虽然性能略降，但仍大幅超越人工标注
- 循环训练策略对基于规则的描述也有效（PoseFixCPR从42.00提升至55.05）

## 亮点与洞察
- 首次证明自动生成的姿态转换描述可以**全面超越**人工标注，彻底改变了"标注越精细效果越好"的传统认知
- 身体部位级的中间表示是关键创新——将复杂的全身运动分解为可管理的子问题
- 循环一致性约束巧妙利用了交换图像对自动获得的反向描述，无需额外标注成本
- 两个新基准（AIST-CPR和PoseFixCPR）填补了CPR评估标准的空白

## 局限与展望
- 依赖GPT-4o的API调用成本（虽然远低于人工标注，但仍有经济成本）
- MLLM偶尔生成偏离指引的回复（约2.5%），目前简单丢弃
- FIXMYPOSE的gallery较大且姿态多样性低，导致检索模糊性高，整体R@1仍较低
- 未探索端到端训练MLLM来直接优化描述质量的可能性
- AIST-CPR数据集中没有训练集的人工标注，因此无法与人工标注做直接对比

## 相关工作与启发
- **vs PoseFix规则方法**: PoseFix依赖3D关键点和模板句子，表达力受限。AutoComPose的自由文本生成带来11+个百分点的R@1提升
- **vs 人工标注**: 人工标注不仅昂贵且质量不稳定（遗漏细节、主观用语），AutoComPose在所有场景下均更优
- **vs 通用CIR**: CPR对描述精度要求更高（连续关节运动 vs 离散属性变化），AutoComPose的部位级分析专门解决这一挑战

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将MLLM用于CPR自动标注，简洁有效的框架设计
- 实验充分度: ⭐⭐⭐⭐ 三个数据集、四种backbone、多种对比基线和全面消融
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，阶段划分合理，图表说明充分
- 价值: ⭐⭐⭐⭐ 为CPR研究提供了可扩展的标注方案和新基准，开创了自动标注的新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models](capellm_support-free_category-agnostic_pose_estimation_with_multimodal_large_lan.md)
- [\[CVPR 2026\] CLEP: Contrastive Language-Pose Pretraining](../../CVPR2026/multimodal_vlm/clep_contrastive_language-pose_pretraining.md)
- [\[CVPR 2026\] Adapting In-context Generation for Enhanced Composed Image Retrieval](../../CVPR2026/multimodal_vlm/adapting_in-context_generation_for_enhanced_composed_image_retrieval.md)
- [\[CVPR 2026\] STiTch: Semantic Transition and Transportation in Collaboration for Training-Free Zero-Shot Composed Image Retrieval](../../CVPR2026/multimodal_vlm/stitch_semantic_transition_and_transportation_in_collaboration_for_training-free.md)
- [\[CVPR 2026\] From 3D Pose to Prose: Biomechanics-Grounded Vision-Language Coaching](../../CVPR2026/multimodal_vlm/from_3d_pose_to_prose_biomechanics-grounded_vision-language_coaching.md)

</div>

<!-- RELATED:END -->
