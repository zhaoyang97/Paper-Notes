---
title: >-
  [论文解读] Controllable and Expressive One-Shot Video Head Swapping
description: >-
  [ICCV 2025][人体理解][头部替换] 本文提出一个基于扩散模型的多条件可控视频头部替换框架（SwapAnyHead），通过形状无关掩码策略、发型增强策略和表情感知的3DMM驱动landmark重定向模块，实现了高保真的身份保持、无缝背景融合和精确的跨身份表情迁移与编辑。 视频头部替换（将源图像的头部无缝替换到目标视…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "头部替换"
  - "扩散模型"
  - "表情迁移"
  - "身份保持"
  - "视频生成"
---

# Controllable and Expressive One-Shot Video Head Swapping

**会议**: ICCV 2025  
**arXiv**: [2506.16852](https://arxiv.org/abs/2506.16852)  
**代码**: [https://humanaigc.github.io/SwapAnyHead/](https://humanaigc.github.io/SwapAnyHead/)  
**领域**: 人体理解 / 人脸生成  
**关键词**: 头部替换, 扩散模型, 表情迁移, 身份保持, 视频生成

## 一句话总结

本文提出一个基于扩散模型的多条件可控视频头部替换框架（SwapAnyHead），通过形状无关掩码策略、发型增强策略和表情感知的3DMM驱动landmark重定向模块，实现了高保真的身份保持、无缝背景融合和精确的跨身份表情迁移与编辑。

## 研究背景与动机

视频头部替换（将源图像的头部无缝替换到目标视频中）在影视制作、虚拟现实和广告合成中有巨大应用潜力。然而现有方法面临两大核心挑战：

- **身份保持不完整**：人脸替换方法（如SimSwap、DiffSwap）仅替换局部面部区域，忽略头型和发型；头部替换方法（如HeSer）在发型多样性和复杂背景处理上表现不佳
- **表情迁移不精确**：IPAdapter/DreamBooth等方法缺乏表情控制能力；AniPortrait依赖3DMM模板表现力有限；Follow-your-emoji虽用3D landmarks但会受源图像原始表情影响，且跨身份面部比例差异导致表情失真

核心矛盾是：需要在保持源头部完整身份（头型+发型）的同时实现精确的表情迁移，还需无缝融合目标视频的背景和身体。本文的解决思路是将头部替换重构为**条件修复任务**，通过身份特征、背景线索和3D landmarks作为条件，在统一的潜在扩散范式下解决上述问题。

## 方法详解

### 整体框架

基于Latent Diffusion Model (LDM)，扩展了背景和表情条件。框架包含4个模块：
1. **AppearanceNet**：提取源图像身份信息，通过self-attention注入到去噪网络
2. **PoseGuider**：提取多尺度运动特征，建立控制信号与生成图像的空间对应
3. **MotionModule**：维持帧间时序一致性
4. **Image Encoder**：替代文本编码器，通过cross-attention提供参考图像全局信息

训练目标为标准扩散损失加上身份损失：$\mathcal{L} = \mathcal{L}_{LDM} + \mathcal{L}_{id}$

### 关键设计

1. **Shape-Agnostic Mask策略（形状无关掩码）**:

    - 功能：消除训练数据中因头部分割边界导致的形状泄漏问题
    - 核心思路：将前景掩码 $M_f$ 膨胀后划分为 $(k_h \times k_w)$ 的非重叠块，每块统一设为0或1（取决于是否有前景像素），破坏精确的分割边界。训练时还随机缩放前景并与修复背景重组：$M_f^{new}(i,j) = \begin{cases} 1, & \text{if any}(M_f(i,j))>0 \\ 0, & \text{if every}(M_f(i,j))=0 \end{cases}$
    - 设计动机：训练数据中参考图和目标图是同一视频的不同帧（同一身份），修复的背景会隐含头部形状信息（SD可识别但人眼不可见），导致生成结果被约束在原始头部区域，限制发型多样性

2. **Hair Enhancement策略（发型增强）**:

    - 功能：消除肩部区域的发型先验偏差，支持多样化发型生成
    - 核心思路：收集长发视频集合，训练时随机选取长发图像获取发型掩码 $M_{hair}$，并用肩部关键点检测生成矩形掩码 $M_{rect}$，腐蚀原始身体区域：$M_{cloth}^{new} = M_{cloth} \odot (1-M_{hair}) \odot (1-M_{rect})$
    - 设计动机：长发搭肩时衣物分割会在肩部留下反映发型的空洞，训练时会引入目标图像的发型先验，推理时即使源图是短发也会在空洞处生成长发

3. **Expression-Aware Landmark Retargeting（表情感知landmark重定向）**:

    - 功能：实现精确的跨身份表情迁移，消除源图像原始表情影响
    - 核心思路：分两步 —— (a) **Neu-Exp模块**：用3DMM拟合源图像，将表情系数置零获得中性表情模板，计算中性landmarks：$L_{ref}^{neu} = L_{ref} - L_{ref}^t + L_{ref}^{t,exp=0}$；(b) **Scale-aware Retargeting**：根据源和驱动图像面部特征（眼/嘴）的比例差计算缩放因子 $s_{eye}, s_{mouth}$，对表情增量进行自适应调整
    - 设计动机：3DMM模板可有效解耦身份/表情/姿态，但表现力受限；Mediapipe 3D landmarks表现力强但无法去除源表情；两者结合取长补短。比例自适应解决了面部比例差异导致的夸张/不足问题

### 损失函数 / 训练策略

- **扩散损失** $\mathcal{L}_{LDM}$：标准的去噪目标
- **身份损失** $\mathcal{L}_{id}$：从预测噪声恢复去噪图像，计算头部区域L2像素损失 + ArcFace身份嵌入余弦相似度损失
- 训练数据：HDTF、VoxCeleb、VFHQ、TalkingHead1KH，共30K视频片段、约20K身份
- 图像分辨率：512×512，使用Mediapipe提取3D landmarks，lama进行背景修复

## 实验关键数据

### 主实验

| 方法 | 类型 | ID Similarity ↑ | Pose Error ↓ | Expression Error ↓ |
|------|------|-----------------|-------------|-------------------|
| AniPortrait | 肖像动画 | 0.892 | 6.55 | 0.025 |
| Follow-Your-Emoji | 肖像动画 | 0.906 | 16.8 | 0.026 |
| SimSwap | 换脸 | 0.592 | 0.702 | 0.005 |
| DiffSwap | 换脸 | 0.397 | 1.144 | 0.008 |
| HeSer | 换头 | 0.319 | 6.840 | 0.022 |
| **SwapAnyHead (Ours)** | **换头** | **0.895** | **9.83** | **0.014** |

评估使用VFHQ 200个视频+200张肖像图（不同身份），每视频48帧。本方法在头部替换方法中ID相似度远超HeSer（0.895 vs 0.319），表情误差也优于所有同类方法。

### 消融实验

**表情感知Landmark重定向消融：**

| 配置 | Expression Error ↓ | 说明 |
|------|-------------------|------|
| w/o Neu-Exp Module | 0.025 | 源表情叠加驱动表情，如笑容残留 |
| w/o Scale-aware Retargeting | 0.015 | 面部比例不匹配导致闭眼/夸张嘴型 |
| 完整方法 | 0.014 | 精确表情迁移 |

**形状无关掩码与发型增强定性消融：**
- 无Shape-Agnostic Mask：生成的头部受驱动视频头型约束，头部大小不匹配
- 无Hair Enhancement：肩部出现错误的发型伪影，短发参考图也会生成长发

### 关键发现
- 换脸方法（SimSwap/DiffSwap）的Pose和Expression误差最小是因为它们只做微调，不重新生成头部
- 肖像动画方法因为不需要融合新背景/身体，天然有更高的ID保持
- 形状泄漏问题比预期严重——SD网络能感知到人眼不可见的分割边界信息

## 亮点与洞察
- **问题定义精准**：明确区分了"换脸"和"换头"的差异，指出了现有方法在头型/发型保持上的本质缺陷
- **Shape-Agnostic Mask策略巧妙**：通过块化破坏分割边界来防止形状泄漏，思路简单但非常有效
- **3DMM+Mediapipe结合**：取两者之长——3DMM的解耦能力+Mediapipe的表现力——解决了单一方案的局限
- 支持表情编辑（不仅迁移），为实际影视后期制作提供了更大灵活性

## 局限与展望
- 定量评估指标有限，缺少FID/LPIPS等图像质量指标
- 肩部以下的身体-头发交互在极端姿态下可能仍有伪影
- 依赖Mediapipe的稳定性，极端角度或遮挡下可能失效
- 训练数据多为正面/半侧面，大角度侧面的泛化能力未验证
- 视频时序一致性的定量评估（如帧间闪烁度量）缺失

## 相关工作与启发
- **SimSwap / FaceShifter**：换脸方法，仅替换面部区域，不改变头型发型
- **HeSer**：两阶段换头方法，但发型多样性和背景修复能力不足
- **Follow-Your-Emoji**：用Mediapipe 3D landmarks驱动，表现力强但有源表情残留
- **AniPortrait**：3DMM驱动的肖像动画，解耦好但表现力受限于3DMM
- **HS-Diffusion**：语义混合扩散模型换头，但无法迁移表情

## 评分
- 新颖性: ⭐⭐⭐⭐ 形状无关掩码和发型增强策略是新颖独到的训练数据增强思路
- 实验充分度: ⭐⭐⭐ 定量指标种类偏少，但定性结果和消融都很充分
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，方法动机清晰，图示丰富
- 价值: ⭐⭐⭐⭐ 视频换头是实际需求强烈的任务，方法实用性高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[CVPR 2026\] SyncDreamer: Controllable and Expressive Avatar Generation Beyond the Talking Head](../../CVPR2026/human_understanding/syncdreamer_controllable_and_expressive_avatar_generation_beyond_the_talking_hea.md)
- [\[ICCV 2025\] DreamActor-M1: Holistic, Expressive and Robust Human Image Animation with Hybrid Guidance](dreamactor-m1_holistic_expressive_and_robust_human_image_animation_with_hybrid_g.md)
- [\[ICCV 2025\] ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling](imhead_a_large-scale_implicit_morphable_model_for_localized_head_modeling.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)

</div>

<!-- RELATED:END -->
