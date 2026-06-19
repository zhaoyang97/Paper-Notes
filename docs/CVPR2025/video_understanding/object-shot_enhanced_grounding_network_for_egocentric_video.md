---
title: >-
  [论文解读] Object-Shot Enhanced Grounding Network for Egocentric Video
description: >-
  [CVPR 2025][视频理解][Ego4D NLQ] OSGNet 针对第一人称视频自然语言查询 (NLQ) 的两大短板——视觉特征缺细粒度物体信息、忽略头戴相机运动暗含的注意力切换——提出"object branch (Co-DETR + CLIP 文本编码) + shot branch (按头转分镜 + 镜头级对比)"双分支架构，在 Ego4D-NLQ / Goal-Step / TACoS 上刷新 SOTA。
tags:
  - "CVPR 2025"
  - "视频理解"
  - "Ego4D NLQ"
  - "Object-aware grounding"
  - "shot 对比学习"
  - "Mamba"
---

# Object-Shot Enhanced Grounding Network for Egocentric Video

**会议**: CVPR 2025  
**arXiv**: [2505.04270](https://arxiv.org/abs/2505.04270)  
**代码**: [https://github.com/Yisen-Feng/OSGNet](https://github.com/Yisen-Feng/OSGNet)  
**领域**: 视频理解 / 第一人称 (egocentric) / 视频时间定位  
**关键词**: Ego4D NLQ、Object-aware grounding、shot 对比学习、Mamba

## 一句话总结
OSGNet 针对第一人称视频自然语言查询 (NLQ) 的两大短板——视觉特征缺细粒度物体信息、忽略头戴相机运动暗含的注意力切换——提出"object branch (Co-DETR + CLIP 文本编码) + shot branch (按头转分镜 + 镜头级对比)"双分支架构，在 Ego4D-NLQ / Goal-Step / TACoS 上刷新 SOTA。

## 研究背景与动机
1. **领域现状**：Ego4D NLQ 任务要求在长第一人称视频中根据问句 (例如 "How many drill bits did I remove from the drill before I moved the yellow carton?") 定位答案出现的时间区间，是具身智能 / 智能助手的核心能力。
2. **现有痛点**：
    - 第一人称视频的预训练特征 (clip-narration 对比) 学到的是"动作"，丢失了 query 关心的"背景小物体" (如 measuring tape) 的细粒度信息；
    - 第三人称视频定位方法 (如 Moment-DETR、SnAG) 直接迁移过来效果差，因为忽略了头戴相机的"高频镜头切换"。
3. **核心矛盾**：通用 video-text backbone 训练目标 (动作对齐) 与 NLQ 任务目标 (背景对象的细粒度记忆检索) 不一致。
4. **本文目标**：
    - 弥补"细粒度物体信息缺失"；
    - 利用"佩戴者头部运动 = 注意力切换"这条隐藏信号；
5. **切入角度**：用现成检测器 + CLIP 文本化把物体类别信息显式注入 video token；同时把视频按"头转点"切成 shot 段，做 shot-query 对比学习。
6. **核心 idea**：双分支——main branch 用并行 cross-attention 融合 video / query / object，shot branch 用对比损失对齐 shot-query。

## 方法详解

### 整体框架
- **特征提取**：(a) Object Extraction：Co-DETR (LVIS 预训练) 检测每帧物体，按 query 名词过滤 + 置信度阈值 $\theta$，类别名经 CLIP ViT-B/32 编码得 $\mathbf{O}_{clip} \in \mathbb{R}^{T \times N_o \times D_o}$；(b) 视频 backbone (EgoVLP/InternVideo) 得 $\mathbf{V}_{clip} \in \mathbb{R}^{T \times D}$；(c) 文本 encoder 得 $\mathbf{Q}_F \in \mathbb{R}^{L \times D}$。
- **Object Encoder**：把物体作为 query、文本 query 作为 key/value 做 cross-attention，得到与 query 对齐的物体特征 $\mathbf{O}_F$。
- **Main Branch**：堆叠多层 [BiMamba video → CA(video, query) ∥ CA(video, object) → gating fusion]，得到融合特征；再过多尺度 transformer 生成 feature pyramid → classification head + regression head 出时间区间。
- **Shot Branch**：单独用视频特征，按头部运动分镜，做镜头级对比学习。
- **推理**：取 main branch 的 top-K confidence 区间。

### 关键设计

1. **Object 注入与并行 Cross-Attention**

    - 功能：把 query 关心的细粒度物体作为额外 modality，弥补 video backbone 的信息缺失。
    - 核心思路：视觉特征和 query / object 分别做 CA：$\mathbf{V}_Q^{(i)} = \hat{\mathbf{V}}^{(i)} + CA(\hat{\mathbf{V}}^{(i)}, \mathbf{Q}_F, \mathbf{Q}_F)$、$\mathbf{V}_O^{(i)} = \hat{\mathbf{V}}^{(i)} + CA(\hat{\mathbf{V}}^{(i)}, \mathbf{O}_F, \mathbf{O}_F)$，再用门控 $\mathbf{A} = \sigma(\text{MLP}(\hat{\mathbf{V}}_Q \| \hat{\mathbf{V}}_O))$ 融合：$\mathbf{V}^{(i+1)} = \mathbf{A}\cdot\hat{\mathbf{V}}_Q + (1-\mathbf{A})\cdot\hat{\mathbf{V}}_O$。
    - 设计动机：query 关键词 ("drill bit", "yellow carton") 不一定在视频特征里被捕获；显式过 detector 把物体显形，且并行 CA 避免 query 信息掩盖 object 信息。

2. **BiMamba 长视频建模**

    - 功能：替换传统 self-attention 处理长 (Ego4D 视频可达数小时) 序列。
    - 核心思路：在 fusion block 内用双向 Mamba，复杂度线性，比 transformer 节省显存。
    - 设计动机：NLQ 视频长度远超普通 moment retrieval，self-attention 成本爆炸。

3. **Shot 分支：头转分镜 + 对比学习**

    - 功能：利用第一人称视频独有的"头戴相机随佩戴者注意力切换"信号，把视频自动切成语义相对独立的 shot。
    - 核心思路：从相机运动 (光流 / 视角变化) 估计头部转动幅度，按峰值切 shot；每个 shot 提取一个表示，与 query 文本特征做对比损失，让相关 shot 与 query 在嵌入空间靠近。
    - 设计动机：NLQ 中有大量"先做 A、再做 B、最后回到 C"的时序结构，head turn 对应了佩戴者注意力 / 任务边界，是天然的弱监督分段信号。

### 损失函数 / 训练策略
- 定位损失 $\mathcal{L}_{ML} = \mathcal{L}_{cls} + \mathcal{L}_{reg}$，分类用 focal loss，回归用 Distance-IoU loss (仅正样本计入)。
- Shot 分支对比损失 $\mathcal{L}_{shot}$ (InfoNCE)。
- 总损失 = $\mathcal{L}_{ML} + \lambda \mathcal{L}_{shot}$。

## 实验关键数据

### 主实验
**Ego4D-NLQ v1 Test (R@1, IoU=0.5)**：

| 方法 | Feature | R@1@0.5 |
|------|---------|---------|
| InternVideo | E+I | 10.06 |
| CONE | E | 7.84 |
| SnAG | E | 10.29 |
| RGNet | E | 10.61 |
| RGNet† (NaQ pretrain) | E | 11.69 |
| **OSGNet** | E | 10.71 |
| **OSGNet†** | E | **15.46** |

OSGNet† 比 RGNet† 提升 +3.77 点 R@1@0.5，提升 +6.74 点 R@5@0.5。

**Ego4D-Goal-Step**：R@1@0.3 比 BayesianVSLNet +3.65。
**TACoS** (第三人称对照)：R@1@0.5 比 SnAG +3.32，证明在普通视频也有效。
**vs GroundVQA on NLQ**：R@1@0.5 +2.15。

### 消融实验

| 配置 | R@1@0.5 (Ego4D-NLQ) |
|------|--------------------|
| Baseline (no object, no shot) | ~13 |
| + Object branch | ~14 |
| + Shot branch | ~14.3 |
| Full (object + shot) | 15.46 |
| 用 self-attn 替换 BiMamba | 显存 OOM 或下降 |

### 关键发现
- Object 信息对"问背景物体"的 query (如 "where is the screwdriver") 提升最大 (~+5 R@1)。
- Shot 分支对长视频 (>5min) 提升明显，短视频几乎无增益。
- 物体过滤阈值 $\theta$ 不能太高 (会漏小物)，也不能太低 (引入噪声)，存在 sweet spot。

## 亮点与洞察
- **第一次显式建模"头戴相机运动 = 注意力信号"**：这是第一人称视频独有的、之前被忽略的隐藏 supervision，可推广到 ego-centric action recognition、anticipation。
- **Object-as-text 的轻量注入**：用 detector 出 label 而非 region feature，再用 CLIP 文本编码，避免引入巨大额外参数。这种"把检测结果文本化"的做法可被任何"视频+QA"工作借用。
- **并行 CA + gating** 比串行 CA 更稳，因为防止 modality 顺序偏差，是个值得复用的 multimodal fusion 小技巧。
- **BiMamba 替换 transformer** 验证了 SSM 在长视频建模的实用性。

## 局限与展望
- 物体分支严重依赖 Co-DETR 在 LVIS 词表上的检测能力，对长尾物体仍会漏检。
- Shot 分镜依赖低层光流 / 运动估计，遇到剧烈光照变化或颠簸时会误切。
- Object 文本化丢失了几何信息 (位置/大小)，可能在涉及空间关系的 query 上不够；后续可加入 box 坐标作为附加 token。
- 未在多模态 (音频) 上扩展，第一人称视频常含语音指令可作为额外 cue。
- 改进方向：把 shot 段当作"chapter"输入到 LLM 做检索式阅读，可能进一步提升长视频性能。

## 相关工作与启发
- **vs SnAG (CVPR 24)**：单分支无 object 信息；OSGNet 加 object + shot。
- **vs NaQ (CVPR 23)**：纯数据增广方案，OSGNet 与 NaQ 预训练正交可叠加。
- **vs GroundVQA**：GroundVQA 用 LLM 接 video grounding，本文走专用模型路线，效率更高。
- 启发：任何模态特征"丢失关键细粒度"的任务都可借鉴"用专家模型生成文本 token 注入"的策略。

## 评分
- 新颖性: ⭐⭐⭐ Object 注入 + shot 对比都不算全新，但组合很贴合 NLQ 任务需求
- 实验充分度: ⭐⭐⭐⭐ 三大数据集 + 详细消融 + NaQ pretrain 对照
- 写作质量: ⭐⭐⭐⭐ 双分支动机讲得很清楚
- 价值: ⭐⭐⭐⭐ Ego-centric NLQ 上明显提升，可作 SOTA baseline；shot 分镜思路对长视频处理有迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HierarQ: Task-Aware Hierarchical Q-Former for Enhanced Video Understanding](hierarq_task-aware_hierarchical_q-former_for_enhanced_video_understanding.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2025\] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding](dynfocus_dynamic_cooperative_network_empowers_llms_with_video_understanding.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2025\] EgoLife: Towards Egocentric Life Assistant](egolife_towards_egocentric_life_assistant.md)

</div>

<!-- RELATED:END -->
