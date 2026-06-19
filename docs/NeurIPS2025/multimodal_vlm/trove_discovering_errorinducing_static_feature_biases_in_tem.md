---
title: >-
  [论文解读] TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][时序VLM] TRoVe提出自动化方法发现时序VLM中导致系统性预测错误的静态特征偏差，通过结合"错误贡献分数"和"静态偏差分数"的双评分机制，在101个合成模型上以28.6%优势超越基线，并成功应用于7个真实VLM揭示新偏差。 领域现状：时序VLM（用于理解多图像序列中的视觉变…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "时序VLM"
  - "静态特征偏差"
  - "模型鲁棒性"
  - "偏差发现"
  - "视频理解"
---

# TRoVe: Discovering Error-Inducing Static Feature Biases in Temporal Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2512.01048](https://arxiv.org/abs/2512.01048)  
**代码**: [GitHub](https://github.com/Stanford-AIMI/TRoVe)  
**领域**: 多模态VLM  
**关键词**: 时序VLM, 静态特征偏差, 模型鲁棒性, 偏差发现, 视频理解

## 一句话总结
TRoVe提出自动化方法发现时序VLM中导致系统性预测错误的静态特征偏差，通过结合"错误贡献分数"和"静态偏差分数"的双评分机制，在101个合成模型上以28.6%优势超越基线，并成功应用于7个真实VLM揭示新偏差。

## 研究背景与动机

**领域现状** 时序VLM（用于理解多图像序列中的视觉变化）在行为识别、疾病进展分类等任务上性能强劲，但可能依赖静态特征捷径而非动态变化来做决策。

**现有痛点** (1) 现有自动化错误发现方法（Domino, George等）面向单图像设定，无法处理多图像序列中的静态偏差；(2) 真实模型的偏差真值未知，定量评估困难；(3) 静态特征可能只出现在序列中部分图像中，增加了检测难度。

**核心矛盾** 时序任务本应依赖动态变化来分类，但模型可能走捷径——仅凭背景中的树就预测"爬树"。如何自动发现这些导致错误的静态偏差？

**本文目标** 自动发现已训练时序VLM中学到的、会导致系统性预测错误的静态特征偏差。

**切入角度** 将问题分解为两个可量化的子问题：(1) 某特征的存在是否导致特定类别错误增加？(2) 模型是否在该特征上学到了偏差？

**核心 idea** 将多图像序列分解为单帧→聚类提取候选特征→用ECS+SBS双评分定位导致错误的静态偏差。

## 方法详解

### 整体框架
TRoVe的工作流程：(1) 将验证集所有序列的图像分解为单帧，用VLM视觉编码器提取嵌入并聚类，每个聚类代表一个候选静态特征；(2) 对每个聚类计算双评分——错误贡献分数和静态偏差分数；(3) 输出高分的(特征, 类别)对作为发现的偏差。

### 关键设计

1. **候选特征提取**:
    - 功能：从数据集中识别反复出现的静态特征
    - 核心思路：对每张图像创建重复序列（去除时序变化），通过VLM视觉编码器提取嵌入，使用球面K-means聚类。聚类数通过遍历候选值+轮廓系数自动选择
    - 设计动机：将多图像序列降为单帧能隔离静态特征信息，聚类能发现跨序列的共性模式

2. **错误贡献分数（ECS）**:
    - 功能：评估静态特征是否与某类别错误相关
    - 核心思路：$ECS_C^y = acc_{\neg C}^y - acc_C^y$，即包含特征C的序列和不包含的序列在类别y上的准确率差。差值大代表该特征的存在显著增加了该类错误
    - 设计动机：直接衡量特征对分类性能的因果影响

3. **静态偏差分数（SBS）**:
    - 功能：评估模型是否对该静态特征学到了偏差
    - 核心思路：对错误分类的序列中属于聚类C的图像，创建纯静态序列（单帧重复$n_i$次），用VLM分类并取预测置信度的平均：$SBS_C^y = \frac{1}{|C_{wrong}|}\sum_{I_i \in C_{wrong}} \text{softmax}(F([I_i,...,I_i]))_{\hat{y}_i}$。高置信度说明模型仅凭静态特征就做出了（错误的）决策
    - 设计动机：时序任务需要动态变化才能正确解决——如果模型对纯静态输入仍高置信度预测，说明它学到了静态捷径

### 最终评分
TRoVe分数 = $ECS_C^y + SBS_C^y$，同时衡量"特征是否导致错误"和"模型是否学到了该偏差"。

## 实验关键数据

### 主实验——合成模型评估（P@10/P@25/P@100/R-Prec）

| 方法 | 背景 P@10 | 物体 P@10 | 属性 P@10 |
|------|----------|----------|----------|
| Random | 20.7 | 14.3 | 16.2 |
| Domino | 48.6 | 52.2 | 63.1 |
| Dist. Failures | 62.9 | 60.4 | 69.2 |
| Confidence | 61.0 | 97.8 | 58.5 |
| **TRoVe** | **100.0** | **97.8** | **100.0** |

### 真实VLM测试时性能提升（Kinetics400 Acc@5）

| 模型 | 偏差类标签y~ | 整体 |
|------|-------------|------|
| VideoCLIP-XL | 51.7 | 82.2 |
| + TRoVe | **94.4** (+82.6%) | **86.7** |
| ViCLIP-L | 71.4 | 77.1 |
| + TRoVe | **96.9** (+35.7%) | **80.7** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅ECS | 下降 | 缺少偏差确认，混入巧合性错误 |
| 仅SBS | 下降 | 缺少错误关联，报告非有害偏差 |
| ECS+SBS | 最优 | 两者互补 |

### 关键发现
- TRoVe在三类偏差上均近乎完美（P@10接近100%），且跨类一致性远高于基线
- 非时序VLM（如CLIP）比时序VLM（如VideoCLIP-XL）有更多静态偏差（平均134.5 vs 84.5）
- 利用发现的偏差做prompt修正，偏差类上性能可提升至111%

## 亮点与洞察
- 101个合成模型+真值标注的评估框架本身是重要贡献，支持严格定量评估
- ECS+SBS双评分机制巧妙解决了"偏差发现"既需要因果关联又需要模型内省的双重需求
- 发现医学影像VLM也存在静态特征偏差（BioViL-T对严重肺炎的X光静态特征有偏差），具有重要临床意义

## 局限与展望
- 当前仅支持分类任务，字幕生成、VQA等生成式任务需新的评分方式
- SBS依赖温度校准的softmax置信度，某些模型的校准可能不佳
- 聚类粒度影响发现质量，过粗可能混合不同特征

## 相关工作与启发
- **vs Domino/George**: 面向单图像设定，无法感知序列中特定帧的静态特征
- **vs Li et al. (Confidence)**: 仅用最大置信度排序，不考虑错误因果关系
- **vs RAVL**: 利用区域级信息但仅用于单图像，TRoVe将帧级分析扩展到时序设定

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个面向时序VLM的自动化偏差发现方法，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 101个合成模型+7个真实VLM+2个任务，极其全面
- 写作质量: ⭐⭐⭐⭐ 问题motivate清晰，方法描述系统化
- 价值: ⭐⭐⭐⭐ 对时序VLM部署前的安全审查有重要实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](../../CVPR2025/multimodal_vlm/revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](../../CVPR2026/multimodal_vlm/pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [\[NeurIPS 2025\] Adapting Vision-Language Models for Evaluating World Models](adapting_visionlanguage_models_for_evaluating_world_models.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)

</div>

<!-- RELATED:END -->
