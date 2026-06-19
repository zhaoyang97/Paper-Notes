---
title: >-
  [论文解读] Enrich and Detect: Video Temporal Grounding with Multimodal LLMs
description: >-
  [ICCV 2025][多模态VLM][视频时序定位] 提出 ED-VTG，将视频时序定位分为"先丰富查询、再预测时间区间"两阶段，利用多模态 LLM 的描述能力增补查询细节，配合轻量区间解码器和多实例学习框架，在多个基准上首次让 LLM 方法全面追平甚至超越专用模型。 视频时序定位（Video Temporal Groun…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视频时序定位"
  - "多模态LLM"
  - "查询丰富化"
  - "多实例学习"
  - "时序检测"
---

# Enrich and Detect: Video Temporal Grounding with Multimodal LLMs

**会议**: ICCV 2025  
**arXiv**: [2510.17023](https://arxiv.org/abs/2510.17023)  
**代码**: [项目页面](https://shramanpramanick.github.io/ED-VTG/)  
**领域**: 多模态VLM  
**关键词**: 视频时序定位, 多模态LLM, 查询丰富化, 多实例学习, 时序检测

## 一句话总结

提出 ED-VTG，将视频时序定位分为"先丰富查询、再预测时间区间"两阶段，利用多模态 LLM 的描述能力增补查询细节，配合轻量区间解码器和多实例学习框架，在多个基准上首次让 LLM 方法全面追平甚至超越专用模型。

## 研究背景与动机

视频时序定位（Video Temporal Grounding, VTG）要求根据自然语言查询定位视频中的时间区间。现有方法面临两大挑战：

**查询质量问题**：数据集中的查询往往粗糙、不完整甚至模糊（如"Man starts surfing"缺少外观细节），直接用于定位精度有限。同时描述（captioning）和定位（grounding）是对偶任务——**描述的输出正是定位的输入**，但这种协同关系很少被利用。

**LLM 方法的精度瓶颈**：现有基于 LLM 的方法要么用文本 token 表示时间戳（受 token 化精度限制），要么添加特殊 token 表示帧编号（扩展词表开销大）。它们无法使用针对检测任务设计的 L1/gIoU 等损失函数，且泛化能力虽强但精度不及专用模型。

核心假设：**更详细的查询能带来更精确的定位**。例如将"Man starts surfing"丰富为"The man with a yellow surfboard slowly runs to start surfing"后，定位边界会更准确。

## 方法详解

### 整体框架

ED-VTG 是一个两阶段级联框架：
- **Enrich 阶段**：多模态 LLM 根据视频内容将输入查询丰富为更详细的描述。
- **Detect 阶段**：LLM 输出特殊 token `<INT>`，其隐状态通过轻量区间解码器预测精确时间边界 $(c, w)$。

关键链条：$(V, Q) \to \hat{Q}^{enr}$，然后 $(V, \hat{Q}^{enr}) \to \hat{I}$。丰富查询的质量直接影响区间预测精度。

### 关键设计

1. **查询丰富化（Enrich）**  
   LLM 逐 token 自回归生成丰富查询 $\hat{Q}^{enr}$，生成完毕后发出 `<INT>` token 触发区间预测。生成格式为："The query $\hat{Q}^{enr}$ occurs at `<INT>`"。训练时的丰富查询伪标签由外部强描述模型（LLaVA OneVision 72B）生成——给定原始查询和对应视频片段，提示模型在保留原意的基础上增补细节。

2. **轻量区间解码器（Detect）**  
   取 `<INT>` token 的隐状态 $\mathbf{h}_{int}$，经线性投影后与视频 token $\mathbf{T}_V$ 拼接，送入 **两层 Transformer + MLP**，输出区间中心 $\hat{c}$ 和宽度 $\hat{w}$。
   
   设计动机：(a) 将精确的时间回归任务从 LLM 的 token 预测中解耦，让 LLM 专注其擅长的语言生成；(b) 可直接应用 L1 + gIoU 等检测领域成熟的损失函数。区间以 $(c, w)$ 参数化，解耦位置和尺度，借鉴目标检测的经验。

3. **多实例学习（MIL）框架**  
   外部描述模型生成的丰富查询可能含幻觉，不一定比原始查询更易定位。为此引入 MIL：训练时对每个样本做两次前向传播，分别使用原始查询和丰富查询（teacher-forcing），得到两组预测 $\hat{I}^{dir}$ 和 $\hat{I}^{enr}$。选择 **grounding loss 更小的那个** 进行反向传播：

    $\mathcal{L} = \begin{cases} \lambda_{LM}\mathcal{L}_{LM}^{dir} + \lambda_{grnd}\mathcal{L}_{grnd}^{dir} & \text{if } \mathcal{L}_{grnd}^{dir} < \mathcal{L}_{grnd}^{enr} \\ \lambda_{LM}\mathcal{L}_{LM}^{enr} + \lambda_{grnd}\mathcal{L}_{grnd}^{enr} & \text{otherwise} \end{cases}$

   这使模型在推理时能自主判断是否需要丰富查询，对已足够详细的查询直接"按原样通过"。

### 损失函数 / 训练策略

- 语言建模损失 $\mathcal{L}_{LM}$: 标准交叉熵，监督丰富查询生成
- 时序定位损失 $\mathcal{L}_{grnd}$: $\lambda_{L1}(|\hat{c}-c| + |\hat{w}-w|) + \lambda_{gIoU} \cdot \text{gIoU}((\hat{c},\hat{w}), (c,w))$
- 预训练：8 个公开数据集共 136K 样本，40 epochs，16 节点 V100
- 微调：在各下游数据集上进一步训练
- 视觉编码器使用 EVA-CLIP ViT-G/14（冻结），LLM 使用 Video-LLaMA-7B + LoRA (rank 32)

## 实验关键数据

### 主实验（表格）

**Zero-Shot 单查询时序定位（STG）**

| 方法 | 是否通用模型 | 训练样本量 | Charades R@0.5 | ANet R@0.5 | TACoS mIoU |
|------|-----------|----------|---------------|-----------|-----------|
| HawkEye | ✓ | 715K | 31.4 | 29.3 | - |
| ChatVTG | ✓ | 100K | 33.0 | 22.5 | 5.5 |
| Momenter | ✓ | 10M | 26.6 | 23.0 | - |
| **ED-VTG** | **✓** | **136K** | **39.3** | **33.1** | **12.7** |
| Δ vs HawkEye | | | **+7.9** | **+3.8** | - |

ED-VTG 在所有三个 zero-shot 基准上大幅领先全部已有方法，包括使用 10M 数据的 Momenter（mIoU +11.7 on Charades）。

**Fine-tuned 视频段落定位（VPG）— Charades-CD-OOD**

| 方法 | 类型 | R@0.3 | R@0.5 | mIoU |
|------|------|-------|-------|------|
| SiamGTR（专用模型）| 非通用 | 59.1 | 35.5 | 38.9 |
| TimeChat（LLM）| 通用 | 60.5 | 36.1 | 38.3 |
| **ED-VTG** | **通用** | **70.7** | **47.3** | **45.0** |
| Δ vs TimeChat | | **+10.2** | **+11.2** | **+6.7** |

首个在 VPG 任务上报告结果的 LLM 方法，且超越所有专用模型。

### 消融实验（表格）

**查询丰富化 + MIL 的效果（Charades-STA, Zero-Shot）**

| 训练范式 | R@0.3 | R@0.5 | mIoU |
|---------|-------|-------|------|
| Detect（直接定位）| 48.1 | 30.6 | 31.0 |
| Enrich & Detect | 58.1 | 37.3 | 37.7 |
| Enrich & Detect + MIL | **59.5** | **39.3** | **40.2** |

查询丰富化带来 +6.7 mIoU 的巨大提升，MIL 进一步贡献 +2.5 mIoU。

**离线丰富 vs 在线丰富（FT w/o PT 设置）**

| 策略 | Charades mIoU | ANet mIoU |
|------|-------------|----------|
| 直接定位 | 33.2 | 34.0 |
| 离线丰富后定位 | 33.4 | 33.7 |
| 在线丰富并定位（ours）| **38.4** | **37.8** |

离线丰富（预处理训练集但推理用原始查询）几乎无效，验证了"推理时也需要丰富"的核心论点。

### 关键发现

- 在 NeXT-GQA 问题定位（QG）的 zero-shot 评测上也取得 SOTA（IoU@0.3: 39.5），展示了强泛化能力。
- 在 HT-Step 文章定位（AG）任务上首次实现 LLM 方法报告结果，且在 unseen split 上超越所有专用模型。
- 区间解码器使用 L1+gIoU 联合训练最优，仅用 LM 损失（无解码器）性能大幅下降。
- 预训练数据仅 136K，远少于 HawkEye (715K) 和 Momenter (10M)，但效果最好。

## 亮点与洞察

- **Enrich-and-Detect 的核心创意**：巧妙利用了描述和定位的对偶性——不直接定位模糊查询，而是先"翻译"为可定位的详细描述，再精确定位。
- **MIL 框架处理伪标签噪声**：不依赖复杂的置信度估计，简单比较两个前向传播的 loss 即可自动选择最优查询，优雅高效。
- **区间解码器的设计哲学**：让 LLM 做它擅长的事（语言生成），让专用模块做精确回归，通过 `<INT>` token 的隐状态传递上下文信息。
- 首个在 STG、VPG、QG、AG 四类任务上全面评测的 LLM 方法，且跨任务一致性强。

## 局限与展望

- 依赖 72B 级别的外部描述模型生成伪标签，训练数据准备成本较高。
- 视频编码仍受固定帧数限制，超长视频中的细粒度事件定位可能不够精确。
- MIL 仅在两个选项（原始 vs 丰富查询）间选择，扩展到多个候选可能进一步提升。
- 在长视频中对小且被遮挡目标的定位是失败案例的主要来源。
- 未探索更复杂的丰富策略（如将抽象概念分解为多个具体可定位的子查询）。

## 相关工作与启发

- TimeChat [2024]、VTimeLLM [2024] 是核心对比方法，均使用 LLM 做时序定位但不做查询丰富化。
- Moment-DETR [NeurIPS 2021] 提出了端到端时序定位的 DETR 范式，ED-VTG 的区间解码器可视为其轻量变体。
- LaViLa 在视频-文本对齐中用改述器增强训练数据，但仅做离线增强；ED-VTG 证明推理时在线丰富更关键。
- 该方法可扩展至视频摘要、视频编辑等需要精确时间理解的下游任务。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （Enrich-and-Detect 范式新颖，MIL 选择机制巧妙）
- 实验充分度: ⭐⭐⭐⭐⭐ （4 类任务、6+ 数据集、ZS 和 FT 两种协议、详尽消融）
- 写作质量: ⭐⭐⭐⭐ （思路清晰，图表直观）
- 价值: ⭐⭐⭐⭐⭐ （首次让 LLM 方法在时序定位上全面追平专用模型，影响力大）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](../../CVPR2026/multimodal_vlm/timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)
- [\[ICCV 2025\] Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images](visual_chronicles_using_multimodal_llms_to_analyze_massive_collections_of_images.md)

</div>

<!-- RELATED:END -->
