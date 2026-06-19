---
title: >-
  [论文解读] DisenQ: Disentangling Q-Former for Activity-Biometrics
description: >-
  [ICCV 2025][多模态VLM][活动生物特征识别] 提出 DisenQ（Disentangling Q-Former），通过结构化语言引导将视频特征解纠缠为生物特征、动作和非生物特征三个独立空间，无需额外视觉模态即可实现活动感知的行人识别 SOTA。 活动生物特征识别（Activity-Biometrics）：是在人…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "活动生物特征识别"
  - "Q-Former"
  - "特征解纠缠"
  - "多模态学习"
  - "视频行人重识别"
---

# DisenQ: Disentangling Q-Former for Activity-Biometrics

**会议**: ICCV 2025  
**arXiv**: [2507.07262](https://arxiv.org/abs/2507.07262)  
**代码**: 无（有 Project Page）  
**领域**: 多模态VLM  
**关键词**: 活动生物特征识别, Q-Former, 特征解纠缠, 多模态学习, 视频行人重识别

## 一句话总结

提出 DisenQ（Disentangling Q-Former），通过结构化语言引导将视频特征解纠缠为生物特征、动作和非生物特征三个独立空间，无需额外视觉模态即可实现活动感知的行人识别 SOTA。

## 研究背景与动机

**活动生物特征识别（Activity-Biometrics）** 是在人执行日常活动（不仅是行走/站立）时识别身份的新任务，比传统行人重识别更具挑战性：

**身份线索与运动/外观纠缠**：行走、跳跃等不同活动引入了大量运动变化，与身份特征混合使识别困难

**现有方法依赖额外视觉模态**：ABNet 等方法需要轮廓图（silhouette），但提取精度受限于环境

**CLIP-based 方法的局限**：只做全局 image-text 对齐，缺乏 identity-specific 特征分离，无法维护时序一致性

核心动机：**能否用语言监督替代额外的视觉模态**，通过结构化文本描述来引导特征解纠缠，使生物特征不受外观和动作变化影响？

## 方法详解

### 整体框架

输入 RGB 视频 → ViT 视觉编码器提取帧特征 → 时序注意力池化得到视频特征 $F$ → DisenQ 用三组可学习 query 分别提取生物特征/动作/非生物特征 → 身份分类头进行识别。训练时使用冻结 VLM 生成结构化文本描述作为引导；推理时不需要文本。

### 关键设计

1. **DisenQ（Disentangling Querying Transformer）**：

    - 基于 BLIP-2 的 Q-Former 架构改造，引入**三组独立的可学习 query**：$z_b$（生物特征）、$z_m$（动作）、$z_{\hat{b}}$（非生物特征）
    - 三组 query 共享 self-attention 和 cross-attention 层，但彼此不交互，各自保持独立
    - 每组 query 与对应的视觉特征和文本特征做 cross-attention：
    $Q_b = Wz_b, \quad K_b = W[F, T_b], \quad V_b = W[F, T_b]$
    - 动作 query $z_m$ 和非生物特征 query $z_{\hat{b}}$ 同理，分别使用 $T_m$ 和 $T_{\hat{b}}$
    - 设计动机：共享层减少参数（消融显示三个独立 Q-Former 参数量×3 但只提升 0.23% R@1），独立 query 保证特征分离

2. **结构化文本生成与编码**：

    - 使用冻结的 LLaVA 1.5 7B 从关键帧生成三类描述：
        - **生物特征描述 $P_b$**：体型、姿态、显著身体特征；每个身份只生成一次，后续用 running average 更新
        - **动作描述 $P_m$**：动作标签和运动方式
        - **非生物特征描述 $P_{\hat{b}}$**：服装、配饰等
    - 文本经冻结 BERT 编码为 $T_b, T_m, T_{\hat{b}}$
    - 推理时不需要 VLM 和文本——DisenQ 已从训练中学会了解纠缠

3. **自适应身份相似度计算**：

    - query 嵌入经 mean pooling 得到 $F_b$, $F_m$, $F_{\hat{b}}$，仅 $F_b$ 和 $F_m$ 用于最终识别
    - 用轻量 MLP 动态计算生物特征和动作特征的权重 $\alpha_1, \alpha_2$：
    $Sim(A,B) = \alpha_1 Sim_b(A,B) + \alpha_2 Sim_m(A,B)$
    - 动态权重让模型在运动信息有意义时利用动作线索，否则依赖生物特征

### 损失函数 / 训练策略

$$\mathcal{L} = \lambda_1 \mathcal{L}_{ID} + \lambda_2 \mathcal{L}_{Tri} + \lambda_3 \mathcal{L}_{Orth} + \lambda_4 \mathcal{L}_{Act}$$

- $\mathcal{L}_{ID}$：交叉熵分类损失（用于 $F_b$），$\lambda_1=0.01$
- $\mathcal{L}_{Tri}$：三元组损失（margin $m=0.3$），确保同一身份特征聚合
- $\mathcal{L}_{Orth} = \|F_b^T F_{\hat{b}}\|$：正交约束，强制生物特征与非生物特征独立
- $\mathcal{L}_{Act}$：交叉熵用于 $F_m$ 的动作分类，确保动作特征保留运动信息
- 训练：EVA-CLIP ViT-G/14 视觉编码器，DisenQ 用 InstructBLIP 权重初始化，AdamW, lr=1e-4, 60 epochs, batch=32（8人×4 clip）

## 实验关键数据

### 主实验（活动生物特征识别基准）

| 方法 | NTU Same R@1 | NTU Cross R@1 | PKU Same R@1 | PKU Cross R@1 | Charades Same R@1 | Charades Cross R@1 |
|------|------|------|------|------|------|------|
| ABNet (CVPR24) | 78.8 | 77.0 | 86.8 | 81.4 | 45.8 | 44.8 |
| CLIP-ReID (AAAI23) | 77.1 | 75.2 | 82.3 | 81.2 | 44.2 | 42.1 |
| Instruct-ReID (CVPR24) | 78.2 | 75.9 | 84.3 | 81.7 | 44.8 | 40.1 |
| **DisenQ (Ours)** | **82.2** | **80.9** | **89.2** | **84.1** | **49.9** | **48.4** |

三个数据集上平均 R@1 提升 3.7%、2.4%、3.9%。在传统 MEVID 基准上也达到 60.7% R@1（SOTA）。

### 消融实验

| 配置 | NTU R@1 | NTU mAP | Charades R@1 | Charades mAP |
|------|---------|---------|-------------|-------------|
| Vision encoder only | 73.2 | 36.2 | 40.1 | 29.2 |
| + Text encoder | 77.7 | 40.6 | 46.5 | 31.8 |
| + DisenQ（完整） | **82.2** | **43.8** | **49.9** | **34.8** |

特征解纠缠类型消融：

| 解纠缠方式 | NTU R@1 | Charades R@1 |
|----------|---------|-------------|
| 无解纠缠 | 74.2 | 42.3 |
| $F_b$ + $F_{\hat{b}}$（生物+非生物） | 76.6 | 44.7 |
| $F_b$ + $F_m$（生物+动作） | 79.2 | 48.2 |
| $F_b$ + $F_{\hat{b}}$ + $F_m$（全部） | **82.2** | **49.9** |

单特征性能：非生物特征单独使用只有 3.8% R@1，说明解纠缠成功剥离了身份信息。

### 关键发现

- **DisenQ 贡献最大**：从 text encoder (+4.5%) 到 DisenQ (+4.5%) 的提升最为显著
- **三路解纠缠效果互补**：生物+动作的组合（79.2%）优于生物+非生物（76.6%），说明动作信息对身份识别更关键
- **非生物特征确实被"清洗"**：单独使用时 R@1 仅 3.8%，几乎不含身份信息
- **VLM 选择不敏感**：LLaVA、InstructBLIP、GPT-4V 差异 < 0.2%，说明方法对文本质量鲁棒
- **自适应权重优于固定权重**：对低运动活动（手势等），固定权重导致动作特征干扰，自适应权重能抑制
- 用随机服装描述替代真实描述导致 R@1 下降 9.2%，验证了精确文本引导的必要性

## 亮点与洞察

- **首次将 Q-Former 用于特征解纠缠**：巧妙改造了 BLIP-2 的 Q-Former，从单一 query 集扩展为多组独立 query，每组对应不同信息维度
- **推理时不需要 VLM/文本**：语言引导仅在训练时使用，推理时 query 已经学会了解纠缠模式，零额外推理开销
- **跨场景泛化强**：在活动生物特征和传统行人重识别基准上都取得 SOTA/竞争力表现
- **生物特征描述的 running average 更新**：优雅地处理了同一身份在不同视频中描述不一致的问题

## 局限与展望

- 依赖 VLM 生成的文本质量，虽然消融显示对 VLM 选择不敏感，但在更极端场景（遮挡严重、低分辨率）下质量可能下降
- ViT-G/14 作为视觉编码器参数量大（1.8B），在资源受限场景不实用
- 未探索无监督或半监督设置（需要 ground truth 身份和动作标签）
- 动作标签作为 ground truth 提供，在真实部署中需要先做动作识别

## 相关工作与启发

- 与 ABNet 的关键区别：ABNet 依赖轮廓图等额外视觉模态，DisenQ 用语言替代
- Q-Former 的解纠缠应用可推广到其他需要多维度特征分离的任务（如情感分析中分离内容与情绪）
- 自适应权重机制可推广到任何需要动态融合多种特征的 retrieval 任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ Q-Former 解纠缠是创新点，语言引导替代额外视觉模态思路好
- **实验充分度**: ⭐⭐⭐⭐⭐ 4个数据集、详细消融、特征空间可视化、VLM/编码器选择分析
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、方法描述完整，图表质量高
- **价值**: ⭐⭐⭐⭐ 为活动感知行人识别提供了有效方案，有监控/智能环境应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ECCV 2024\] X-Former: Unifying Contrastive and Reconstruction Learning for MLLMs](../../ECCV2024/multimodal_vlm/x-former_unifying_contrastive_and_reconstruction_learning_for_mllms.md)
- [\[ACL 2026\] When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models](../../ACL2026/multimodal_vlm/when_seeing_overrides_knowing_disentangling_knowledge_conflicts_in_vision-langua.md)
- [\[ACL 2026\] Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs](../../ACL2026/multimodal_vlm/long_story_short_disentangling_compositionality_and_long-caption_understanding_i.md)
- [\[ICCV 2025\] OpenVision: A Fully-Open, Cost-Effective Family of Advanced Vision Encoders for Multimodal Learning](openvision_a_fully-open_cost-effective_family_of_advanced_vision_encoders_for_mu.md)

</div>

<!-- RELATED:END -->
