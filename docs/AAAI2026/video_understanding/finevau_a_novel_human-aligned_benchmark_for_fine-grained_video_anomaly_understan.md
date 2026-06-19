---
title: >-
  [论文解读] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding
description: >-
  [AAAI 2026][视频理解][视频异常理解] 本文提出FineVAU基准，将视频异常理解 (VAU) 分解为事件(What)、实体(Who)、地点(Where)三个维度，设计了与人类感知高度对齐的FV-Score评估指标，并通过全自动LVLM辅助管线构建了FineW³数据集，实验揭示当前LVLM在细粒度异常事件感知上的关键短板。
tags:
  - "AAAI 2026"
  - "视频理解"
  - "视频异常理解"
  - "评估基准"
  - "LLM评判"
  - "细粒度评估"
  - "人类对齐"
---

# FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding

**会议**: AAAI 2026  
**arXiv**: [2601.17258](https://arxiv.org/abs/2601.17258)  
**代码**: [https://finevau.github.io](https://finevau.github.io)  
**领域**: 可解释性  
**关键词**: 视频异常理解, 评估基准, LLM评判, 细粒度评估, 人类对齐

## 一句话总结
本文提出FineVAU基准，将视频异常理解 (VAU) 分解为事件(What)、实体(Who)、地点(Where)三个维度，设计了与人类感知高度对齐的FV-Score评估指标，并通过全自动LVLM辅助管线构建了FineW³数据集，实验揭示当前LVLM在细粒度异常事件感知上的关键短板。

## 研究背景与动机

### 领域现状
视频异常理解 (Video Anomaly Understanding, VAU) 是视频监控中的核心任务。随着大型视觉语言模型 (LVLM) 的兴起，VAU 从简单的二分类（正常/异常）发展到更丰富的任务形态，包括密集描述、视频问答和推理链分析等。

### 现有痛点
现有 VAU 评估方法存在两大类致命问题：

**N-gram类指标**（如BLEU、ROUGE-L）仅测量词汇重叠，无法捕捉自由形式回答的语义等价性。一个事实正确但用词不同的描述会被错误惩罚。

**LLM-based指标**（如AnomEVAL、VAU-EVAL）侧重评估语言流畅性和推理一致性，缺乏对异常视频特定元素的细粒度检测。这些指标给出模糊、主观的分数，与人类对异常的感知严重不对齐。

### 核心矛盾
评估指标与人类关注点之间存在鸿沟——人类在判断异常描述质量时，主要关注三个核心问题："发生了什么事件"、"谁参与了"、"在哪里发生"，而非文本的流畅度或词汇匹配度。

### 切入角度
作者将 VAU 界定为一个三维度结构化问题，通过检测 LVLM 输出中是否涵盖关键视觉元素来评估质量，而非依赖主观打分。

## 方法详解

### 整体框架
FineVAU 包含三个核心贡献：
1. **问题定义**：将 VAU 形式化为 What/Who/Where 三维评估问题
2. **FV-Score 指标**：提出基于 LLM 的结构化评估指标
3. **FineW³ 数据集**：通过自动化管线构建精细标注数据集

### 关键设计

#### 1. **三维结构化评估框架**
- **What (事件维度)**：捕捉关键动作（如"放火"）、交互（如"打架"）和孤立状态变化（如"爆炸"），使用三级评分（0=缺失/错误，0.5=部分正确，1=完全准确）
- **Who (实体维度)**：描述参与异常的实体及其视觉属性（服装、年龄、性别等），使用二级评分（0/1）
- **Where (地点维度)**：涵盖物理环境、时间、光照条件、人群密度等，使用二级评分（0/1）
- **设计动机**：人类感知异常时本能地关注这三个维度。不同于现有指标的复杂评分标准，简化为二/三级评分可以降低 LLM 评判的难度并提高可解释性

#### 2. **FV-Score 与 FineVAU-Judge**
- 定义结构化评分函数：$\mathcal{S}(R) = \lambda_{what} \cdot \mathcal{J}_{what}(R) + \lambda_{who} \cdot \mathcal{J}_{who}(R) + \lambda_{where} \cdot \mathcal{J}_{where}(R)$
- 使用 Gemini-2.5-Flash 作为 LLM 评判器，对每个GT元素判断语义成员性
- 核心思路：将评估转化为"多部分检测问题"——检查报告中是否覆盖了 GT 中的关键元素
- 权重消融实验表明 $\lambda_{who}=2.0$ 时与人类判断最对齐，说明人类高度重视实体识别

#### 3. **FineW³ 数据集构建**
- **两阶段全自动标注管线**，基于高质量人工标注 (UCA数据集) 进行增强：
    - **Stage 1（事件分解与实体链接）**：LVLM 将复杂事件描述分解为因果链式原子事件，补充遗漏事件，识别并链接参与实体
    - **Stage 2（实体定位与场景描述）**：为每个实体补充细粒度物理属性，描述场景特征
- 使用 Gemini-2.5-Pro，以 1fps 采样帧 + 原始 UCA 标注作为输入
- 最终数据集：1544 个视频，17813 个事件（13393正常+4420异常），59392 个实体，74593 个属性，7669 个地点属性

### 人类对齐验证
- 60 个视频，8 名人类专家，180 个排名判断
- 使用 4 种相关性度量（PCC、1-R²、Kendall τ、Spearman τ）

## 实验关键数据

### 主实验

| 模型 | 总体 | 地点 | 事件 | 实体 | 属性 |
|------|------|------|------|------|------|
| InternVL3-9B | 40.5 | 71.8 | 18.0 | 51.2 | 25.5 |
| LLaVA-VID-7B | 35.0 | 65.7 | 14.4 | 44.0 | 21.0 |
| LLaVA-OV-7B | 32.2 | 58.3 | 13.0 | 41.1 | 19.9 |
| Qwen2.5-VL-7B | 32.9 | 70.8 | 9.1 | 38.3 | 20.3 |
| VideoLLaMA3-7B | 19.3 | 40.3 | 6.5 | 24.3 | 10.2 |
| **平均** | **32.0** | **61.3** | **12.2** | **39.8** | **19.4** |

### 消融实验（指标人类对齐度）

| 指标 | PCC ρ↑ | 1-R²↓ | Kendall τ↑ | Spearman τ↑ |
|------|--------|-------|------------|-------------|
| FV-Score (本文) | **0.61** | **0.63** | **0.56** | **0.56** |
| VAU-EVAL | 0.53 | 0.72 | 0.49 | 0.47 |
| ROUGE-L | 0.47 | 0.78 | 0.43 | 0.44 |
| AnomEVAL | 0.42 | 0.82 | 0.39 | 0.37 |
| BLEU | 0.19 | 0.96 | 0.17 | 0.17 |
| CIDEr | -0.63 | 0.60 | -0.59 | -0.58 |

### FV-Score 权重消融

| λ_what | λ_who | λ_where | PCC ρ↑ | Kendall τ↑ |
|--------|-------|---------|--------|------------|
| 1.0 | **2.0** | 1.0 | **0.61** | **0.56** |
| 2.0 | 1.0 | 1.0 | 0.56 | 0.50 |
| 1.0 | 1.0 | 1.0 | 0.51 | 0.46 |
| 1.0 | 1.0 | 2.0 | 0.47 | 0.42 |

### 关键发现
1. **LVLM擅长静态粗粒度信息**：地点维度平均61.3%，远高于事件维度12.2%
2. **事件理解极其薄弱**：仅12.2%的平均准确率，尤其是缺乏强视觉线索的异常（如入店行窃）
3. **LVLM存在"正常偏向"**：模型倾向于将异常事件描述为正常行为（如把打架描述为对话）
4. **实体识别比事件更容易**：39.8% vs 12.2%，但仍有很大提升空间
5. **InternVL3全面领先**：在所有维度上均获得最佳性能

## 亮点与洞察
- **评估范式创新**：从主观打分转向结构化元素检测，大幅提升可解释性
- **Who维度权重最高**：消融实验揭示人类评估异常描述时最看重参与实体的准确识别，这一发现反直觉但有据可查
- **揭示LVLM盲点**：当前模型在理解细粒度时空事件上存在根本性缺陷，这不是通过简单scaling就能解决的
- **全自动标注管线**：可扩展到更多数据集，为VAU数据构建提供了可复用方案

## 局限与展望
- 数据集来源于CCTV监控视频，场景多样性有限
- 评估依赖单一LLM (Gemini-2.5-Flash)，可能存在评判偏差
- 目前仅评估开源7-9B模型，未涵盖更大规模或闭源模型
- 三维度框架可能遗漏"Why"（异常原因）这一重要维度
- 事件维度的三级评分标准中，0.5分的判定可能仍有主观性

## 相关工作与启发
- UCA (Yuan et al., CVPR 2024) 首创密集人工标注描述，但采用N-gram评估
- HAWK (Tang et al., NeurIPS 2024) 提出合成描述，但评估标准仍以语言质量为主
- Holmes-VAU (Zhang et al., CVPR 2025) 多粒度描述，但缺乏实体和场景信息
- 评估方法启发：结构化分解+元素检测的思路可推广到其他视觉理解任务的评估

## 评分
- 新颖性: ⭐⭐⭐⭐ （评估范式创新，但本质上是一个benchmark工作）
- 实验充分度: ⭐⭐⭐⭐⭐ （人类对齐验证完善，多维度消融充分）
- 写作质量: ⭐⭐⭐⭐⭐ （逻辑清晰，图表丰富）
- 价值: ⭐⭐⭐⭐ （为VAU评估提供了更好的标准，但应用范围较窄）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning](../../CVPR2026/video_understanding/fine-vad_towards_fine-grained_video_anomaly_detection_via_progressive_cross-gran.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](../../CVPR2026/video_understanding/frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](../../CVPR2026/video_understanding/ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[AAAI 2026\] R-AVST: Empowering Video-LLMs with Fine-Grained Spatio-Temporal Reasoning in Complex Audio-Visual Scenarios](r-avst_empowering_video-llms_with_fine-grained_spatio-temporal_reasoning_in_comp.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)

</div>

<!-- RELATED:END -->
