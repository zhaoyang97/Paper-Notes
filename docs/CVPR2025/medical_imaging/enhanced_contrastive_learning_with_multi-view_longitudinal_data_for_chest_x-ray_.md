---
title: >-
  [论文解读] Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation
description: >-
  [CVPR 2025][医学图像][放射报告生成] 提出 MLRG 两阶段框架，通过多视角纵向对比学习融合当前多视角图像的空间信息和历史纵向数据的时间信息进行视觉-文本预训练，并用 tokenized absence encoding 灵活处理缺失的患者先验知识，在 MIMIC-CXR 上 BLEU-4 提升 2.3%，MIMIC-ABN 上 F1 提升 5.5%。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "放射报告生成"
  - "多视角纵向数据"
  - "对比学习"
  - "缺失数据处理"
  - "胸部X光"
---

# Enhanced Contrastive Learning with Multi-view Longitudinal Data for Chest X-ray Report Generation

**会议**: CVPR 2025  
**arXiv**: [2502.20056](https://arxiv.org/abs/2502.20056)  
**代码**: [https://github.com/mk-runner/MLRG](https://github.com/mk-runner/MLRG)  
**领域**: 医学图像  
**关键词**: 放射报告生成, 多视角纵向数据, 对比学习, 缺失数据处理, 胸部X光

## 一句话总结
提出 MLRG 两阶段框架，通过多视角纵向对比学习融合当前多视角图像的空间信息和历史纵向数据的时间信息进行视觉-文本预训练，并用 tokenized absence encoding 灵活处理缺失的患者先验知识，在 MIMIC-CXR 上 BLEU-4 提升 2.3%，MIMIC-ABN 上 F1 提升 5.5%。

## 研究背景与动机

**领域现状**：放射报告自动生成（RRG）通过视觉编码器+文本生成器从 X 光图像自动生成临床报告，可有效减轻放射科医生工作负担。现有方法大多只用单张图像或固定双视角生成报告。

**现有痛点**：临床实践中放射科医生通常综合多视角图像（PA、AP、lateral 等不同视角有不同几何特性，如 AP 可能放大心影）、患者历史影像和先前报告来做诊断。但现有方法存在三个主要问题：(1) 只用单张图像无法区分不同视角的差异；(2) 忽略疾病进展的纵向信息，可能导致模型幻觉（生成无依据的疾病描述）；(3) 部分患者缺失"INDICATION"、"previous report"或"previous image"（首次就诊或数据存储问题），现有方法难以灵活处理这种缺失。

**核心矛盾**：多源异构信息（多视角×多时间点×可选文本先验）的灵活融合，以及不同患者数据可用性不一致的问题。

**本文目标** (1) 如何灵活融合数量可变的当前多视角图像和可能缺失的历史影像？(2) 如何利用报告中固有的时空信息指导视觉-文本预训练？(3) 如何处理患者特定先验知识（INDICATION、previous report）的缺失？

**切入角度**：将放射科医生的诊断流程建模为两阶段：先用多视角纵向对比学习预训练视觉表示（利用报告的时空语义作为监督信号），再用 tokenized absence encoding 在生成阶段灵活适配数据可用性。

**核心 idea**：用多正样本对比学习+跨模态对齐预训练多视角纵向视觉表示，用 tokenized absence encoding 处理先验知识缺失，实现灵活且准确的报告生成。

## 方法详解

### 整体框架
两阶段架构。Stage 1（预训练）：RAD-DINO 视觉编码器提取多视角图像特征，加入可学习视角位置嵌入和时间位置嵌入，通过多视角纵向融合网络（MLF）融合空间-时间视觉特征，与 CXR-BERT 提取的文本特征做多粒度对比学习。Stage 2（报告生成）：用 tokenized absence encoding 处理缺失的 INDICATION 和 previous report，与预训练的视觉特征一起输入多模态融合网络，由 DistilGPT2 生成报告。

### 关键设计

1. **Multi-Positive Contrastive Learning（多正样本对比学习）**:

    - 功能：增强同一次就诊多视角图像之间视觉特征的一致性
    - 核心思路：将每张当前图像作为 anchor，同次就诊的其他视角图像为正样本，不同就诊的图像为负样本。加入可学习的视角位置嵌入 $E_v$ 区分 PA/AP/lateral 等视角差异。用 multi-positive 交叉熵损失 $\mathcal{L}_{MPC}$ 最大化同次就诊图像的相似度。跳过只有单张图像的就诊记录（无正样本对）。
    - 设计动机：不同视角提供互补信息（如 PA 和 lateral 分别看到不同解剖结构），但现有方法只简单区分"正面"和"侧面"，忽略了 PA vs AP 等细粒度差异。可学习视角嵌入让模型自动捕捉视角间的几何差异。

2. **Multi-view Longitudinal Fusion Network (MLF) + Instance-wise Cross-modal Alignment**:

    - 功能：灵活融合当前多视角图像和历史影像的时空特征，并与文本报告对齐
    - 核心思路：MLF 使用交叉注意力机制，以 anchor scan 的特征为 query，辅助视角和历史影像的特征为 key/value。由于每次就诊视角数量和历史可用性不同，逐样本处理保证灵活性。融合后的时空视觉特征 $V^{st}$ 与报告文本特征做 instance-wise 跨模态对齐（CLIP-style），用交叉熵损失 $\mathcal{L}_G$ 对齐全局视觉和文本表示。关键：报告中天然包含时间比较信息（如"compared to prior study..."），所以用多视角纵向融合特征去对齐报告比只用当前图像更合理。
    - 设计动机：放射报告不仅描述当前状态还可能比较历史变化，仅用当前图像做跨模态对齐会导致学习不到时序变化信息甚至产生幻觉。MLF 的交叉注意力天然支持可变数量的 key/value 输入。

3. **Tokenized Absence Encoding（标记化缺失编码）**:

    - 功能：在生成阶段灵活处理缺失的患者特定先验知识（INDICATION 和 previous report）
    - 核心思路：当某项先验知识缺失时，不是简单丢弃或用零填充，而是用专门的可学习 absence token 替代。文本生成器（DistilGPT2）接收多模态融合网络的输出，该网络会将视觉特征与可用/缺失的先验知识灵活融合。absence token 让模型学会区分"这个信息不存在"和"这个信息为空"。
    - 设计动机：实际临床数据中，约 35-50% 的患者缺失 INDICATION 或 previous report/image（见表1统计）。简单忽略会降低模型利用可用信息的能力，而用 absence token 让模型在有/无先验知识时都能最优化报告生成。

### 损失函数 / 训练策略
- Stage 1 预训练：$\mathcal{L}_{pretrain} = \mathcal{L}_{MPC} + \mathcal{L}_G$
- Stage 2 微调：标准 auto-regressive 交叉熵损失训练 DistilGPT2 生成器
- 视觉编码器：RAD-DINO (ViT, 冻结)；文本编码器：CXR-BERT；文本生成器：DistilGPT2

## 实验关键数据

### 主实验

| 数据集 | 指标 | MLRG | 之前最优 | 提升 |
|--------|------|------|---------|------|
| MIMIC-CXR | BLEU-4 ↑ | **0.158** | 0.135 (SEI) | +2.3% |
| MIMIC-CXR | F1 (CE) ↑ | **0.505** | 0.473 (B-LLM) | +3.2% |
| MIMIC-CXR | RadGraph ↑ | **0.291** | 0.249 (SEI) | +4.2% |
| MIMIC-CXR | ROUGE-L ↑ | **0.320** | 0.304 (CoFE) | +1.6% |
| MIMIC-ABN | F1 (CE) ↑ | **0.515** | 0.460 (CMN) | +5.5% |
| Two-view CXR | F1 RadGraph ↑ | **0.254** | 0.227 (CXRMate) | +2.7% |

### 消融实验

| 配置 | BLEU-4 | F1 | RadGraph | 说明 |
|------|--------|-----|----------|------|
| Full MLRG | **0.158** | **0.505** | **0.291** | 完整模型 |
| w/o MPC | 下降 | 下降 | 下降 | 去掉多正样本对比学习 |
| w/o MLF | 下降 | 下降 | 下降 | 去掉纵向融合网络 |
| w/o Absence Encoding | 下降 | 下降 | 下降 | 去掉缺失编码 |
| Single-view only | 下降明显 | 下降明显 | 下降明显 | 仅用单张图像 |

### 关键发现
- 多视角纵向数据的联合使用（MVL 输入）在所有数据集上全面超越单图像（SI）、双视角（MVD）、仅纵向（Long）等方案
- 在 MIMIC-CXR 上相比基于 LLM 的方法（如 B-LLM 用 MiniGPT-4），MLRG 用更小的 DistilGPT2 就取得更好成绩，证明表示学习比模型规模更重要
- 跨模态对齐使用纵向融合后的时空特征比只用当前图像特征效果显著更好
- 在临床准确性指标（F1、RadGraph）上的提升比 NLG 指标（BLEU）更大，表明方法在医学实质内容上的改进最明显

## 亮点与洞察
- **利用报告的时空语义做监督**：放射报告天然包含历史比较信息（"unchanged..."、"new opacity..."），用多视角纵向融合特征去对齐报告文本，能学到传统单图像对齐学不到的时序变化信息。这个洞察值得在其他纵向医学任务中借鉴。
- **tokenized absence encoding 的通用性**：用专门的 absence token 处理缺失输入，比零填充或 dropout 更语义化。这个技巧可迁移到任何多源输入且部分源可能缺失的生成任务。
- **交叉注意力天然支持可变输入**：MLF 用 query-KV 的交叉注意力处理不同数量的视角和历史图像，无需 padding 或固定长度，设计简洁优雅。

## 局限与展望
- 仅使用最近一次历史就诊，更长的纵向序列（多次历史记录）可能提供更丰富的疾病进展信息
- 报告生成器 DistilGPT2 规模较小，替换为更大的 LLM（如 LLaMA）可能进一步提升生成质量
- 视觉编码器 RAD-DINO 冻结不微调，可能限制了视觉特征对特定任务的适应性
- 未在多中心数据集上验证泛化性
- 仅针对胸部 X 光，扩展到 CT、MRI 等其他模态需要调整

## 相关工作与启发
- **vs CXRMate**: CXRMate 也用多视角纵向数据但通过合成历史报告引入噪声，且不区分视角差异；MLRG 直接用真实数据、区分视角，效果更好。
- **vs SEI**: SEI 只用单图像+INDICATION，MLRG 使用更丰富的多视角纵向输入，在所有指标上超越。
- **vs BioViL-T**: BioViL-T 用纵向数据做视觉表示预训练但未用于报告生成；MLRG 将纵向预训练和报告生成统一到同一框架。

## 评分
- 新颖性: ⭐⭐⭐⭐ 多视角纵向对比学习和 absence encoding 组合有创新，但单个组件并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、多维度指标、丰富对比和消融
- 写作质量: ⭐⭐⭐⭐ 问题分析清晰，方法描述系统
- 价值: ⭐⭐⭐⭐ 贴近临床实际需求的实用方法，多视角纵向融合方向值得进一步探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] PriorRG: Prior-Guided Contrastive Pre-training and Coarse-to-Fine Decoding for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/priorrg_prior-guided_contrastive_pre-training_and_coarse-to-fine_decoding_for_ch.md)
- [\[CVPR 2026\] Phrase-grounded APO for Improving Chest X-ray Report Generation](../../CVPR2026/medical_imaging/phrase-grounded_apo_for_improving_chest_x-ray_report_generation.md)
- [\[CVPR 2025\] GIIM: Graph-based Learning of Inter- and Intra-view Dependencies for Multi-view Medical Image Diagnosis](giim_graph-based_learning_of_inter-_and_intra-view_dependencies_for_multi-view_m.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)

</div>

<!-- RELATED:END -->
