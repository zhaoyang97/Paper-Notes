---
title: >-
  [论文解读] Aligning Moments in Time using Video Queries
description: >-
  [ICCV 2025][视频生成][视频时刻检索] 本文提出MATR（Moment Alignment TRansformer），通过双阶段序列对齐（soft-DTW）将目标视频表示条件化于查询视频特征，实现视频到视频的时刻检索（Vid2VidMR），并设计自监督预训练策略，在ActivityNet-VRL上R@1提升13.1%、mIoU提升8.1%。
tags:
  - "ICCV 2025"
  - "视频生成"
  - "视频时刻检索"
  - "视频查询"
  - "序列对齐"
  - "自监督预训练"
  - "Transformer"
---

# Aligning Moments in Time using Video Queries

**会议**: ICCV 2025  
**arXiv**: [2508.15439](https://arxiv.org/abs/2508.15439)  
**代码**: [GitHub](https://github.com/vl2g/MATR)  
**领域**: 视频生成  
**关键词**: 视频时刻检索, 视频查询, 序列对齐, 自监督预训练, Transformer

## 一句话总结
本文提出MATR（Moment Alignment TRansformer），通过双阶段序列对齐（soft-DTW）将目标视频表示条件化于查询视频特征，实现视频到视频的时刻检索（Vid2VidMR），并设计自监督预训练策略，在ActivityNet-VRL上R@1提升13.1%、mIoU提升8.1%。

## 研究背景与动机

1. **领域现状**: 视频时刻检索（VMR）主要研究文本查询的场景，如Moment-DETR、QD-DETR等。视频查询的VMR（Vid2VidMR）相对新兴，由Feng等人正式提出。
2. **现有痛点**: 文本查询对复杂动作（如倒钩射门）往往无法精确描述，用户更难以自然语言传达所见。现有Vid2VidMR方法缺乏显式的语义帧级对齐和建模查询-目标复杂依赖关系的能力。
3. **核心矛盾**: 视频查询和目标视频长度、速度、上下文差异大，需要同时捕获高层语义关系和细粒度帧级依赖。现有方法要么直接用C3D特征替代文本编码器，要么缺少精确的时序对齐。
4. **本文目标**: (1) 如何精确对齐查询和目标视频的语义序列？(2) 如何增强模型在无标注数据上的泛化能力？
5. **切入角度**: 利用可微的soft-DTW在Transformer编码器前后执行双阶段序列对齐，配合自监督预训练（从视频中随机采样片段作为查询进行自定位）。
6. **核心 idea**: 通过Transformer框架内的显式双阶段序列对齐将目标视频转化为查询对齐表示，实现精确的视频时刻定位。

## 方法详解

### 整体框架
输入目标视频$V_t$（$M$帧）和查询视频$V_q$（$N$帧），通过CLIP ViT-B/32编码并线性投影为$d$维嵌入，拼接后送入Transformer Encoder进行联合理解，输出的查询对齐表示用于前景分类和边界预测两个头。

### 关键设计

1. **双阶段序列对齐（Dual-stage Sequence Alignment）**:
    - 功能: 在Encoder前后两次执行序列对齐，捕获全局语义和细粒度时序依赖
    - 核心思路: 使用soft-DTW（可微动态时间规整）对齐。**预融合对齐**：在Encoder之前，对目标$E_t$和查询$E_q$计算基于余弦相似度的对齐代价矩阵$C_{i,j} = 1 - \frac{\langle e_i^t, e_j^q \rangle}{\|e_i^t\| \|e_j^q\|}$，最小化$\mathcal{L}_{\text{align}}^{\text{pre}} = \text{soft-DTW}_\gamma(A^{\text{pre}}, C^{\text{pre}})$。**后融合对齐**：在Encoder输出的融合表示$E_t^g, E_q^g$上再次执行soft-DTW对齐。后者确定的对齐子序列$E_t^g[s:e]$送入Decoder进一步精化。
    - 设计动机: 预融合对齐增强语义表示，后融合对齐在融合后的特征上进行更精细的匹配，双阶段互补提升定位精度。soft-DTW允许不同长度序列的非线性对齐，可微分且对速度变化鲁棒。

2. **自监督预训练策略**:
    - 功能: 无需标注数据即可初始化模型的时刻定位能力
    - 核心思路: 从目标视频$V_t$中随机采样一个片段作为查询$V_q$，训练模型在$V_t$中定位该片段的起止时间。对查询片段施加随机增强（反转帧、加高斯噪声、加速/减速），使预训练样本翻倍。预训练损失与主训练损失结构相同（前景分类+边界预测+双对齐）。
    - 设计动机: 该预训练目标与Vid2VidMR任务高度一致，利用Kinetics700等大规模无标签视频数据学习时序定位技能，增强泛化能力。

3. **预测头（Classification + Boundary Prediction）**:
    - 功能: 从查询对齐表示中输出前景标签和时刻边界
    - 核心思路: 最终表示$E_f = [E_t^g; E_t^l]$结合Encoder全局语义和Decoder细粒度特征。分类头使用3层1×3卷积+Sigmoid输出前景概率$\hat{f}_i$，训练用二元交叉熵$\mathcal{L}_{fg}$。边界头共享初始结构但输出左右偏移$(d_i^L, d_i^R)$，用smooth L1 + gIoU损失$\mathcal{L}_{seg}$在前景位置训练。推理时用1D NMS（阈值0.7）过滤重叠边界。
    - 设计动机: 卷积层沿时序轴操作保持时序连续性，前景/边界分离设计简化了密集预测中的定位问题。

### 损失函数 / 训练策略
总损失: $\mathcal{L} = \frac{1}{S}\sum(\lambda_{fg}\mathcal{L}_{fg} + \lambda_{seg}\mathcal{L}_{seg} + \lambda_{\text{align}}^{\text{pre}}\mathcal{L}_{\text{align}}^{\text{pre}} + \lambda_{\text{align}}^{\text{post}}\mathcal{L}_{\text{align}}^{\text{post}})$，所有$\lambda$均设为1。AdamW优化器，学习率1e-4，权重衰减1e-4。ActivityNet-VRL上训练200 epoch，batch size 1200。

## 实验关键数据

### 主实验

| 方法 | ActivityNet-VRL mIoU | ActivityNet-VRL R@1 | SportsMoments mIoU | SportsMoments R@1 |
|--------|------|------|----------|------|
| FFI+SRM (之前SOTA) | 48.7 | 40.6 | — | — |
| MATR (本文) | **61.8** | **53.7** | **最优** | **最优** |
| 提升 | +13.1 | +13.1 | +14.4 | +14.7 |

与VLM方法对比:

| 方法 | mIoU | R@1 |
|------|------|------|
| TimeChat | 26.4 | 23.8 |
| Video-LLaMA2 | 17.6 | 15.2 |
| MATR | **61.8** | **53.7** |

### 消融实验

| 配置 | R@1 | 说明 |
|------|---------|------|
| Full MATR | 最优 | 完整模型 |
| w/o Pre-fusion alignment | 下降明显 | 去掉预融合对齐 |
| w/o Post-fusion alignment | 下降 | 去掉后融合对齐 |
| w/o Self-supervised pretraining | 下降 | 去掉自监督预训练 |
| w/o Dual-stage alignment | 下降最多 | 两个对齐都去掉 |

### 关键发现
- 双阶段对齐是最关键的设计，去掉任一阶段性能显著下降
- 自监督预训练在SportsMoments（小数据集）上带来更大提升
- MATR在"重现"场景（角色在多个镜头间重复出现）中表现尤佳
- 视频查询显著优于文本查询，尤其在复杂动作描述场景

## 亮点与洞察
- 构建了SportsMoments数据集（770K对，176.6小时完整比赛视频），填补了体育领域细粒度Vid2VidMR的空白
- 自监督预训练的关键洞察：从自身随机裁剪片段进行自定位，与下游任务目标高度一致
- soft-DTW的灵活性使得不同长度/速度的视频对齐成为可能
- 将文本VMR方法（Moment-DETR等）适配到视频查询的对比实验设计很全面

## 局限与展望
- 目前基于CLIP ViT-B/32的特征可能限制了细粒度动态理解
- 计算复杂度随视频长度增长：soft-DTW为$O(MN)$，全注意力编码器为$O((M+N)^2)$
- 未探索多查询视频同时检索的场景
- SportsMoments仅覆盖足球和板球，可扩展到更多运动

## 相关工作与启发
- **vs GDP/SRL**: 这些早期Vid2VidMR方法缺乏显式序列对齐和Transformer架构
- **vs Moment-DETR/QD-DETR**: 原设计为文本查询，适配到视频查询后性能远不如MATR
- **vs VLMs (Video-LLaVA等)**: 零样本VLMs在Vid2VidMR上表现差，说明该任务需要专用架构

## 评分
- 新颖性: ⭐⭐⭐⭐ 双阶段序列对齐+自监督预训练的组合设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖4类基线（全监督/VLM/文本VMR/图像VMR），消融详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 推动Vid2VidMR领域发展，新数据集有持续影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[CVPR 2025\] One-Minute Video Generation with Test-Time Training](../../CVPR2025/video_generation/one-minute_video_generation_with_test-time_training.md)
- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](../../CVPR2025/video_generation/mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2026\] When Numbers Speak: Aligning Textual Numerals and Visual Instances in Text-to-Video Diffusion Models](../../CVPR2026/video_generation/when_numbers_speak_aligning_textual_numerals_and_visual_instances_in_text-to-vid.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)

</div>

<!-- RELATED:END -->
