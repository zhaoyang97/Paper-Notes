---
title: >-
  [论文解读] Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition
description: >-
  [ICCV 2025][视频理解][动作识别] 提出Trokens框架，通过语义感知的轨迹点采样：和关系运动建模：（包含轨迹内HoD和轨迹间相对位移描述子），将点轨迹转化为语义感知的关系token，与外观特征融合后在6个few-shot动作识别基准上取得SOTA。 视频理解的核心在于运动：与外观：信息的协同建模…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "动作识别"
  - "点轨迹追踪"
  - "语义采样"
  - "运动建模"
  - "HoD"
---

# Trokens: Semantic-Aware Relational Trajectory Tokens for Few-Shot Action Recognition

**会议**: ICCV 2025  
**arXiv**: [2508.03695](https://arxiv.org/abs/2508.03695)  
**代码**: [项目页面](https://pulkitkumar95.github.io/trokens)  
**领域**: 视频理解 / 小样本动作识别  
**关键词**: Few-Shot Action Recognition, 点轨迹追踪, 语义采样, 运动建模, HoD

## 一句话总结

提出Trokens框架，通过**语义感知的轨迹点采样**和**关系运动建模**（包含轨迹内HoD和轨迹间相对位移描述子），将点轨迹转化为语义感知的关系token，与外观特征融合后在6个few-shot动作识别基准上取得SOTA。

## 研究背景与动机

视频理解的核心在于**运动**与**外观**信息的协同建模。在few-shot动作识别中，由于训练数据极少，这一协同尤为关键。现有方法存在两个根本性挑战：

### 挑战1：如何选择有信息量的追踪点？

- **密集采样**：覆盖全面但计算昂贵
- **均匀网格采样**（如TATs）：简单高效，但**无法适应物体尺度**——小而关键的物体（如刀、按钮）容易被遗漏，而大面积背景被冗余采样
- 例如：在"用刀涂黄油"的动作中，均匀采样可能完全漏掉刀的运动

### 挑战2：如何有效建模轨迹运动模式？

- 现有Transformer方法（如TATs）将轨迹**仅作为特征采样锚点**，隐式依赖self-attention学习运动
- 但position embedding主要编码静态位置，不直接捕获**时间位移**或**跨轨迹关系**
- 光流方法（如optical flow）仅限于相邻帧分析，在遮挡下退化

Trokens的动机是：利用**语义先验**指导点采样实现自适应覆盖，同时**显式建模**轨迹内和轨迹间的运动动态。

## 方法详解

### 整体框架

Trokens由四个组件串联：

1. **外观特征提取**：DINOv2-base提取视频外观token $\mathcal{F}^{\text{RGB}} \in \mathbb{R}^{H \times W \times T \times C}$
2. **语义感知点采样**：基于DINO特征聚类进行自适应采样
3. **关系运动建模**：轨迹内HoD + 轨迹间相对位移
4. **解耦时空Transformer**：融合运动和外观token进行分类

### 语义感知点采样

核心思想：利用DINO patch token的**自然语义聚类特性**——同一物体的token在特征空间中自然聚集。

具体步骤：
1. 从DINOv2提取patch token特征
2. 将特征聚类为 $L$ 个语义组
3. 每个组均匀采样 $q = M/L$ 个点，其中 $M=256$ 为总轨迹数
4. 从新语义组首次出现的帧开始采样
5. 用预训练的CoTracker追踪这些点，得到语义感知轨迹 $\mathcal{P} \in \mathbb{R}^{M \times T \times 2}$

**关键优势**：小物体（如刀）自成一个语义组，获得与大物体同等密度的采样点，确保不遗漏关键运动信息。

### 关系运动建模

#### 轨迹内运动模块 (Intra-motion)

借鉴HoG (Histogram of Oriented Gradients) 的思想，采用**HoD (Histogram of Oriented Displacements)** 编码每条轨迹内的运动方向和幅度：

对轨迹 $\mathcal{P}^m$ 在时刻 $t$，计算位移：

$$\Delta x_t = x_t - x_{t-\delta}, \quad \Delta y_t = y_t - y_{t-\delta}$$

位移幅度：$\Delta d_t = \sqrt{\Delta x_t^2 + \Delta y_t^2}$

位移方向：$\theta_t = \arctan2(\Delta y_t, \Delta x_t)$

将 $\theta_t$ 量化为 $B=32$ 个方向bin，位移幅度按最近两个bin的距离加权分配，得到每时刻的HoD描述子 $\mathbf{H}_{\text{HoD}} \in \mathbb{R}^{T \times B}$，通过FC层投影到 $C$ 维特征空间：

$$\mathcal{F}_{\text{intra}}^{\text{motion}} = \text{FC}(f_{\text{HoD}}(\mathcal{P})) \in \mathbb{R}^{M \times T \times C}$$

与原始HoD的差异：(1) 逐时刻计算保持时序顺序；(2) 可学习投影增强表达力；(3) 从人体骨骼关键点泛化到任意轨迹。

#### 轨迹间运动模块 (Inter-motion)

捕获不同轨迹之间的**协调运动**（如刀与面包的相对运动区分"涂黄油"和"切菜"）。

对每条轨迹 $\mathcal{P}^m$ 在时刻 $t$，计算其与所有其他轨迹的相对位移：

$$\mathbf{d}_t^m = [(x_t^m - x_t^{m'}, y_t^m - y_t^{m'})]_{m'=1}^{M} \in \mathbb{R}^{2M}$$

完整描述子 $\mathbf{d} \in \mathbb{R}^{M \times T \times 2M}$，通过FC层投影：

$$\mathcal{F}_{\text{inter}}^{\text{motion}} = \text{FC}(\mathbf{d}) \in \mathbb{R}^{M \times T \times C}$$

### 运动感知时空Transformer

1. **轨迹对齐**：根据轨迹坐标从外观特征中采样得到轨迹对齐的外观token $\mathcal{F}_{\text{traj}}^{\text{RGB}}$
2. **特征融合**：通过逐元素相加融合三类特征：

$$\mathcal{F}^{\text{fuse}} = \mathcal{F}_{\text{traj}}^{\text{RGB}} + \mathcal{F}_{\text{intra}}^{\text{motion}} + \mathcal{F}_{\text{inter}}^{\text{motion}}$$

3. **解耦注意力**：在轨迹内（时间维度）和轨迹间（空间维度）分别做self-attention，结果相加
4. **分类输出**：可学习CLS token通过cross-attention聚合最终特征

### 损失函数

标准few-shot双损失：

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(p_{\text{cls}}^Q, y) + \mathcal{L}_{\text{Contrastive}}(\mathcal{F}_{\text{final}}^Q, \mathcal{F}_{\text{final}}^S)$$

## 实验

### 主实验：SSV2 Full (5-way K-shot)

| 方法 | 1-shot | 2-shot | 3-shot | 5-shot |
|------|--------|--------|--------|--------|
| MoLo (CVPR'23) | 56.6 | 62.3 | 67.0 | 70.6 |
| TATs (ECCV'24) | 57.7 | 67.1 | 70.0 | 74.6 |
| **Trokens** | **61.5** | **69.9** | **73.8** | **76.7** |

在SSV2 Full上，1-shot提升 **+3.8%**（相对TATs），5-shot提升 **+2.1%**。SSV2是运动密集型数据集，验证了运动建模的重要性。

### 跨数据集泛化

| 数据集 | 方法 | 1-shot | 3-shot | 5-shot |
|--------|------|--------|--------|--------|
| SSV2 Small | TATs | 47.9 | 60.0 | 64.4 |
| SSV2 Small | **Trokens** | **53.4** | **65.3** | **68.9** |
| HMDB-51 | TATs | 60.0 | 71.8 | 77.0 |
| HMDB-51 | **Trokens** | **69.8** | **80.0** | **82.3** |
| UCF-101 | TATs | 92.0 | 96.8 | 95.5 |
| UCF-101 | **Trokens** | **94.0** | **97.3** | **97.9** |

HMDB-51上1-shot提升**+9.8%**，SSV2 Small上1-shot提升**+5.5%**，改善非常显著。

### 消融分析

论文通过类别级性能分析（图3）展示了改善来源：
- **语义采样优势**：涉及小物体的动作类（如"Unfolding something"、"Twisting something"）提升明显
- **运动建模优势**：需要精细时序动态的类别（如"Pulling something from left to right"）显著改善
- **局限性暴露**：快速运动导致模糊（如"Rolling something on flat surface"）和大幅相机运动（如"Picking something up"）时，点追踪变得困难

## 亮点与洞察

1. **语义感知采样是"被忽视的关键"**：用DINO聚类指导采样的想法简单但效果显著，尤其对小物体动作
2. **经典方法的现代复兴**：HoD本是十年前的手工特征，Trokens将其改造为可学习、逐时刻的版本，在深度学习框架中焕发新生
3. **显式运动建模 > 隐式学习**：Transformer的self-attention虽然理论上可以捕获运动，但在few-shot低数据场景下，显式先验更有效
4. 元素级相加融合虽简单，但在motion+appearance融合中表现出色

## 局限性

1. 依赖CoTracker等外部点追踪模型，引入额外计算开销和依赖
2. 轨迹间模块的 $\mathbb{R}^{2M}$ 描述子随轨迹数平方增长，扩展性受限
3. 在Kinetics等外观偏向数据集上增益有限（1-shot仅+1.0%），运动建模的价值取决于数据集特性
4. 仅在vision-only设置下评估，未与多模态（加语言）方法对比

## 相关工作

- **Few-shot AR**: OTAM, TRX, STRM, MoLo, HYRSM, TATs
- **点追踪**: CoTracker, PIPs, TAPIR, TATs
- **运动特征**: HoG, HoD, 光流方法

## 评分

- 创新性：⭐⭐⭐⭐ — 语义采样+HoD现代化+显式关系运动建模的组合新颖有效
- 实用性：⭐⭐⭐⭐ — 端到端可训练，6个基准全面SOTA
- 实验充分度：⭐⭐⭐⭐⭐ — 6个数据集、多种shot/way设置、类级分析
- 写作质量：⭐⭐⭐⭐ — 动机图清晰，方法推导完整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2025\] TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition](../../CVPR2025/video_understanding/tamt_temporal-aware_model_tuning_for_cross-domain_few-shot_action_recognition.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[CVPR 2025\] Temporal Alignment-Free Video Matching for Few-Shot Action Recognition](../../CVPR2025/video_understanding/temporal_alignment-free_video_matching_for_few-shot_action_recognition.md)

</div>

<!-- RELATED:END -->
