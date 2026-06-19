---
title: >-
  [论文解读] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations
description: >-
  [ICCV 2025][语义分割][指代视频目标分割] 提出ReferDINO，通过将GroundingDINO视觉定位基础模型端到端适配到指代视频目标分割（RVOS）任务，设计定位引导可变形掩码解码器、目标一致性时序增强器和置信度查询剪枝策略，在五个基准上显著超越SOTA（Ref-YouTube-VOS上+3.9% $\mathcal{J}\&\mathcal{F}$），并实现51 FPS实时推理。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "指代视频目标分割"
  - "视觉定位"
  - "GroundingDINO"
  - "可变形注意力"
  - "查询剪枝"
---

# ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations

**会议**: ICCV 2025  
**arXiv**: [2501.14607](https://arxiv.org/abs/2501.14607)  
**代码**: [项目页面](https://isee-laboratory.github.io/ReferDINO)  
**领域**: 图像分割  
**关键词**: 指代视频目标分割, 视觉定位, GroundingDINO, 可变形注意力, 查询剪枝

## 一句话总结

提出ReferDINO，通过将GroundingDINO视觉定位基础模型端到端适配到指代视频目标分割（RVOS）任务，设计定位引导可变形掩码解码器、目标一致性时序增强器和置信度查询剪枝策略，在五个基准上显著超越SOTA（Ref-YouTube-VOS上+3.9% $\mathcal{J}\&\mathcal{F}$），并实现51 FPS实时推理。

## 研究背景与动机

指代视频目标分割（RVOS）根据文本描述在视频中分割目标物体，需要深度视觉-语言理解、像素级密集预测和时空推理三方面能力。现有方法面临以下核心问题：

**视觉-语言能力不足**：现有RVOS模型在处理复杂属性描述（如"形状+颜色"组合）时常混淆相似物体，根源在于RVOS数据规模有限

**视觉定位基础模型的局限**：GroundingDINO虽有强大的目标级VL理解，但(a)只会画框不会做掩码（缺乏像素级密集预测），(b)无法理解动态属性（如"摇尾巴的猫"）

**效率瓶颈**：GroundingDINO使用900个查询，逐帧处理视频计算量巨大

**非端到端集成问题**：已有方法（如Grounded-SAM-2）将GroundingDINO和SAM2串联使用，但非端到端、不可微分，无法进一步优化

**核心动机**：需要一种端到端方法，将GroundingDINO的开放世界定位知识（区域级VL对齐）与像素级分割和时空推理能力无缝结合。

## 方法详解

### 整体框架

ReferDINO在GroundingDINO基础上新增三个模块：
1. 置信度查询剪枝 → 减少每帧计算
2. 目标一致性时序增强器 → 跨帧交互
3. 定位引导可变形掩码解码器 → 生成高质量掩码

工作流：视频逐帧送入GroundingDINO → 查询剪枝 → 收集各帧目标特征 → 时序增强 → 掩码解码 → 输出掩码序列。

### 关键设计

1. **定位引导可变形掩码解码器**：

    - **核心设计**：将框预测和掩码预测级联为 grounding-deformation-segmentation 流水线，而非简单并行
    - 用预测框的中心 $\{b_x, b_y\}$ 直接作为可变形注意力的参考点，将定位先验注入掩码预测
    - 目标特征 $\tilde{\boldsymbol{o}}$ 作为Query，FPN高分辨率特征 $\boldsymbol{F}_{\text{seg}}$ 作为Memory
    - 采样过程通过双线性插值实现，**端到端可微分**，分割梯度可以回传优化框预测
    - 再通过跨模态注意力（Query=目标特征, K/V=文本特征）过滤背景噪声
    - **对比Dynamic Mask Head**：后者为每个目标存储独立的高分辨率特征图，内存开销巨大；ReferDINO在共享特征图上通过位置引导采样，零额外内存
    - 解码器由 $L_m$ 个Block组成，每个Block = 可变形交叉注意力 + 跨模态注意力

2. **目标一致性时序增强器**：

    - 由记忆增强追踪器 + 跨模态时序解码器组成
    - **记忆增强追踪器**：用匈牙利算法基于余弦相似度对齐相邻帧目标，动量更新记忆 $\mathcal{M}^t = (1 - \alpha \cdot \boldsymbol{c}^t) \cdot \mathcal{M}^{t-1} + \alpha \cdot \boldsymbol{c}^t \cdot \hat{\mathcal{O}}^t$，其中 $\boldsymbol{c}^t$ 为目标与文本的相似度，防止目标不可见的帧污染长期记忆
    - **跨模态时序解码器**：注入**时变文本特征** $\{\boldsymbol{f}_{\text{cls}}^t\}$ 作为帧代理进行帧间交互，而非使用静态文本嵌入；通过时序自注意力 + 跨注意力（文本为Query，目标为K/V）提取动态信息
    - 设计动机：(a)GroundingDINO是逐帧独立的，缺乏时序一致性；(b)之前方法用静态文本嵌入，无法捕捉动态变化（如动作描述）

3. **置信度查询剪枝**：

    - 在跨模态解码器的每层逐步裁减低置信度查询
    - 置信度由两项组成：$s_j = \frac{1}{N_l-1}\sum_{i \neq j} \boldsymbol{A}^s_{ij} + \max_k \boldsymbol{A}^c_{kj}$
        - 第一项：j-th查询被其他查询关注的平均程度（不可替代性）
        - 第二项：j-th目标在文本中被提及的最大概率
    - 每层保留 $1/k$ 高置信度查询，最终 $N_s \ll N_q=900$
    - 将解码器时间复杂度从 $O(L \cdot N^2 d)$ 降为 $O(\frac{k^2}{k^2-1} N^2 d)$，独立于深度 $L$
    - $k=2$ 时解码器计算量降至24.7%

### 损失函数 / 训练策略

- 匈牙利匹配选择最低cost的预测序列作为正样本
- 总损失 $\mathcal{L}_{\text{total}} = \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{box}}\mathcal{L}_{\text{box}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}$
- $\mathcal{L}_{\text{cls}}$: focal loss; $\mathcal{L}_{\text{box}}$: L1 + GIoU; $\mathcal{L}_{\text{mask}}$: DICE + binary focal + projection loss
- 冻结骨干网络，使用LoRA（rank=32）微调跨模态Transformer
- 先在RefCOCO/+/g图像数据上预训练，再在RVOS数据上微调

## 实验关键数据

### 主实验

| 数据集 | 指标 | ReferDINO (Swin-T) | DsHmp (CVPR'24) | SOC (NeurIPS'23) | 提升 |
|---|---|---|---|---|---|
| Ref-YouTube-VOS | $\mathcal{J}\&\mathcal{F}$ | **67.5** | 63.6 | 62.4 | +3.9 |
| MeViS | $\mathcal{J}\&\mathcal{F}$ | **48.0** | 46.4 | - | +1.6 |
| Ref-DAVIS17 | $\mathcal{J}\&\mathcal{F}$ | **66.7** | 64.0 | 63.5 | +2.7 |
| Ref-YouTube-VOS (Swin-B) | $\mathcal{J}\&\mathcal{F}$ | **69.3** | 67.1 | 66.0 | +2.2 |
| MeViS (Swin-B) | $\mathcal{J}\&\mathcal{F}$ | **49.3** | - | - | - |

### 消融实验

| 配置 | $\mathcal{J}\&\mathcal{F}$ (MeViS) | 说明 |
|---|---|---|
| ReferDINO | 48.0 | 完整模型 |
| w/o CMA（去掉跨模态注意力） | 47.6 (-0.4) | 跨模态过滤有辅助作用 |
| w/o DCA（去掉可变形交叉注意力） | 45.3 (-2.7) | 定位引导采样是核心 |
| w/o Tracker | 47.6 (-0.4) | 追踪器提供时序一致性 |
| w/o Temporal Decoder | 45.8 (-2.2) | 时序解码器是关键 |
| 50% 查询剪枝 | 67.5 (vs 67.6全量) | 性能无损，FLOPs降40.6% |
| Random 50% 剪枝 | 38.3 (-29.3) | 随机剪枝灾难性下降 |

### 关键发现

- Swin-T版ReferDINO就超越了使用Swin-B骨干的GroundingDINO baseline（67.5 vs 66.7 $\mathcal{J}\&\mathcal{F}$）
- 查询剪枝策略实现了10倍加速（4.9→51 FPS），同时性能几乎无损
- 定位引导可变形注意力（DCA）贡献最大（-2.7%），验证了框先验对掩码质量的重要性
- 在G-DINO+SH/DH两个baseline上的对比表明，简单适配无法充分释放基础模型潜力

## 亮点与洞察

- **端到端适配范式**：首个将GroundingDINO端到端适配到RVOS的工作，框和掩码可微分联合优化
- **定位→掩码级联设计**：巧妙利用预训练的框预测作为空间先验引导掩码生成，优于并行设计
- **时变文本嵌入**：利用GroundingDINO的跨模态编码器产生逐帧不同的文本表征，比静态文本更好捕捉时序动态
- **查询剪枝的巧妙设计**：复用解码器自注意力权重计算置信度，零额外计算开销
- **实时性能**：51 FPS的推理速度使其可应用于实时视频场景

## 局限与展望

- 骨干网络冻结 + LoRA微调可能限制了对特定RVOS场景的适配深度
- 记忆增强追踪器使用动量更新，长视频中可能存在漂移问题
- 仅支持单目标RVOS（MeViS用阈值选择多目标），多目标场景扩展性未充分验证
- 未探讨与SAM2等分割基础模型的结合可能性

## 相关工作与启发

- 与Video-GroundingDINO（仅加时序自注意力）相比，ReferDINO的适配更彻底（掩码解码器+时序增强+剪枝）
- 可变形注意力的参考点设定（用框中心替代MLP生成）是一个简洁有效的构思
- 查询剪枝思想可推广到其他DETR-like模型中加速推理
- 端到端基础模型适配的范式对其他下游任务（如视频定位、时序检测）有启发价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ 端到端适配基础模型的范式有新意，定位→掩码级联设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 五个数据集+两种骨干+详细消融+效率分析，非常全面
- **写作质量**: ⭐⭐⭐⭐⭐ 结构清晰，motivation和method对应紧密，图表优质
- **价值**: ⭐⭐⭐⭐⭐ 在RVOS上建立新SOTA同时实现实时推理，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](../../NeurIPS2025/segmentation/robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)

</div>

<!-- RELATED:END -->
