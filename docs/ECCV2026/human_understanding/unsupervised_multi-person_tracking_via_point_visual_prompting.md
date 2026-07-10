---
title: >-
  [论文解读] PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking
description: >-
  [ECCV 2026][人体理解][多目标追踪] PS-Track 提出数据-模型-损失三层递进的点监督多目标追踪框架，通过时间反馈提示、点激发小波注意力和不确定性高斯学习，仅用逐帧单像素标注即接近全监督方法的追踪性能，并在密集/极端运动场景下反超全监督基线。 多目标追踪（MOT）是自动驾驶、人群分析、体育感知等领域的核心前…
tags:
  - "ECCV 2026"
  - "人体理解"
  - "多目标追踪"
  - "点监督"
  - "弱监督学习"
  - "小波变换"
  - "不确定性学习"
---

# PS-MOT: Cultivating Instance Awareness from Point Seeds for Multi-Object Tracking

**会议**: ECCV 2026  
**arXiv**: [2606.30476](https://arxiv.org/abs/2606.30476)  
**代码**: 有（开源）  
**领域**: 人体理解  
**关键词**: 多目标追踪, 点监督, 弱监督学习, 小波变换, 不确定性学习

## 一句话总结

PS-Track 提出数据-模型-损失三层递进的点监督多目标追踪框架，通过时间反馈提示、点激发小波注意力和不确定性高斯学习，仅用逐帧单像素标注即接近全监督方法的追踪性能，并在密集/极端运动场景下反超全监督基线。

## 研究背景与动机

多目标追踪（MOT）是自动驾驶、人群分析、体育感知等领域的核心前置任务。当前最优方法依赖逐帧边界框标注进行训练——标注一个框需 7-10 秒，成本高昂且难以大规模扩展。点标注（每个目标仅标一个像素坐标）是极具吸引力的低成本替代：标注一个点只需 0.7-0.9 秒，成本降低约 64%。然而点标注仅提供目标的拓扑中心位置，完全缺失几何形状和尺度信息，这导致两个深层问题：位置歧义——多个邻近目标的点标注相互干扰，追踪 ID 在密集场景中频繁漂移；边界缺失——解码器无法从单点推断目标完整轮廓，模型被迫在"盲"状态下拟合时空关联。现有方法大多采用"点标注→SAM 膨胀为框→接入标准追踪器"的朴素两阶段级联，但实验表明这种级联严重失败（DanceTrack 上 HOTA 仅约 10.8），因为分割与追踪之间缺乏时序协同，逐帧独立的膨胀结果不满足运动连续性。

失败的核心矛盾在于：从点信号到实例感知之间存在根本性的信息鸿沟——点指示"目标存在"，但不描述"目标长什么样"和"往哪运动"。填补这一鸿沟需要同时在三个层面发力：数据层面需要将孤立点演化为时序一致的实例表达，模型层面需要从点信号"幻觉"出目标几何边界，损失层面则需要容忍伪标签自身的不确定性。已有方法或只处理单个层面（如仅做点→框膨胀），或依赖规则无法端到端优化，都无法系统性弥合这一鸿沟。

本文的切入角度是：将点监督 MOT 视为一个从种子到实例的渐进培育过程，在数据、模型、损失三个层面设计针对性子模块，让它们协同递进。**核心 idea：提出 PS-Track 三层递进框架——Temporal-Feedback Prompting（TFP）用负空间线索和 Kalman 运动先验将点标注演化为时序一致的伪标签，Point-Excited Wavelet Attention（PEWA）利用 Haar 小波变换从频域选择性激发目标高频边界特征，Uncertainty-Guided Gaussian Learning（UGL）将伪标签建模为高斯分布、用联合质量评分自适应加权回归损失，推理阶段完全脱离点输入且零额外开销，在 DanceTrack 上达到 50.3 HOTA，在 JRDB 和 EmboTrack 上反超全监督方法。**

## 方法详解

### 整体框架

PS-Track 的核心思路是将"从点信号培育实例感知"分解为递进的三层。在数据层，输入视频帧和逐帧点标注，Temporal-Feedback Prompting（TFP）模块——结合来自相邻目标的负空间线索和来自上一帧的 Kalman 运动先验——引导 SAM 分割模型生成跨帧一致的伪边界框，同时输出每个伪标签的联合质量评分（融合 SAM 置信度与运动一致性）。在模型层，Point-Excited Wavelet Attention（PEWA）模块在训练阶段插入 backbone 与编码器之间：将点标注构造成高斯热图作为频域激励信号，在对特征图做 Haar 小波分解后选择性地增强高频子带系数（对应边缘和纹理信息），再通过逆变换重建出边界感知增强的特征图。在损失层，Uncertainty-Guided Gaussian Learning（UGL）将每个伪标签视为高斯观测，用 TFP 的质量评分决定其置信方差，使回归损失随伪标签质量动态加权。推理阶段，PS-Track 脱离点标注和 PEWA，退化为标准 DETR 追踪流水线，零额外计算开销。整体架构基于 Deformable DETR + MOTIP（ResNet-50 backbone，6 层编码器 6 层解码器，300 个目标查询），训练完成后 PEWA 完全旁路。

%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：视频帧<br/>+ 逐帧点标注"] --> B["TFP 数据层<br/>负语义线索 + Kalman运动先验<br/>引导SAM生成伪标签"]
    B --> C["伪标签 + 联合质量评分"]
    C --> D["训练阶段"]
    D --> E["PEWA 模型层<br/>Haar DWT→点引导<br/>高频激励→IDWT重建"]
    E --> F["DETR解码 + UGL损失层<br/>高斯似然损失<br/>不确定性动态加权"]
    F -->|训练完成| G["推理：无点依赖<br/>标准DETR追踪流水线"]

### 关键设计

**1. TFP：负空间线索与运动先验驱动的时序伪标签生成**

点是静态的、孤立的，而追踪需要连续、完整的实例表达。TFP 的核心洞察是：直接从单点引导 SAM 做分割会遭遇两类典型失败——身份融合，近距离多目标之间的点标注相互干扰，SAM 将多个目标合并为一个区域；语义碎片化，遮挡导致 SAM 只分割出目标的部分肢体。针对身份融合，TFP 引入交互式负线索机制：对每个目标 i，从其空间邻域内其他目标的点中采样负提示 $\mathcal{P}^-_i = \{p^+_j \mid j \in \mathcal{N}(i), \|p^+_i - p^+_j\|_2 < \tau_{dist}\}$，作为 SAM 的"语义防火墙"强制排除相邻身份的响应。针对语义碎片化，TFP 利用 Kalman 滤波器预测当前帧的目标位置和尺度，得到运动先验框 $B_{t|t-1}$ 作为 SAM 的 box prompt 约束分割搜索空间——完整输入变为 $\mathcal{M}_t = \text{SAM}(I_t, \text{point}=\{p^+_i, \mathcal{P}^-_i\}, \text{box}=B_{t|t-1})$，有效抑制了离群区域的激活。TFP 还为每个伪标签计算联合质量评分 $S_{joint} = S_{SAM} \cdot (\alpha \cdot \text{IoU}(B_{pseudo}, B_{t|t-1}) + \beta)$，融合 SAM 的语义置信度与运动一致性，传递给损失层。这一设计的关键巧妙在于：它不是简单把 SAM 当黑箱调用，而是用追踪知识（运动先验和邻域负线索）主动引导 SAM 的注意力时空场，使逐帧分割结果在时序上自洽——这是朴素级联完全缺失的。

**2. PEWA：点引导的高频激励感知目标边界**

点标注告诉模型目标在哪，但不告诉目标长什么样——这是点监督的根本性信息缺失。PEWA 从频域视角处理这个缺失：将点标注视为一个频率激发器，选择性地增强目标边界处的高频成分。具体而言，Backbone 输出的特征图 $X \in \mathbb{R}^{C \times H \times W}$ 先经过 Haar 离散小波变换（DWT）分解为低频分量 $X_{LF}$ 和三个高频子带 $X_{HF}^k$（LH、HL、HH）。低频承载全局语义，高频编码边缘和纹理。核心激励机制是：以标注点为中心构造二维高斯热图 $G$，下采样到小波分辨率后通过轻量调制器 $\phi$ 生成频域激励掩码 $M_{exc} = \sigma(\phi(G_{\downarrow})) \in \mathbb{R}^{C \times H/2 \times W/2}$。该掩码只作用于高频子带 $\hat{X}_{HF}^k = X_{HF}^k \odot M_{exc}$，低频分量保持不变以避免破坏全局身份一致性，最后通过逆 DWT 和残差连接重建增强特征 $\hat{X} = \text{IDWT}(X_{LF}, \{\hat{X}_{HF}^k\}) + X$。PEWA 只在训练阶段参与前向传播，推理时被完全绕过，不引入推理延迟。从效果上看，这个设计模拟了人类视觉的 top-down 注意力机制——知道目标在哪后有方向地"看"它的轮廓细节——而且用频域操作而非空间域卷积，参数效率更高。

**3. UGL：不确定性引导的高斯似然学习**

TFP 生成的伪标签不可避免带有噪声——某些帧的框偏大、偏小或偏移。传统方法将这些伪标签视为确定性的回归目标（直接算 L1/L2 损失），让模型被迫拟合错误信号。UGL 换了一个视角：每个伪标签不是确定值，而是对真实边界框的一个有噪观测。它将伪标签建模为高斯分布 $B_{pseudo} \sim \mathcal{N}(B_{gt}, \sigma^2)$，方差由联合质量评分映射得 $\sigma_i = 1 / (S_{joint,i} + \varepsilon)$——高质量标签方差小、回归约束强，低质量标签方差大、贡献自然衰减。UGL 据此定义高斯似然损失：

$$\mathcal{L}_{reg} = \frac{1}{N} \sum_i \left[ \frac{1}{2\sigma_i^2} \|B_{pred,i} - B_{pseudo,i}\|_2^2 + \frac{1}{2} \log \sigma_i^2 \right]$$

这个损失的双项结构设计精妙：第一项通过 $\sigma_i^2$ 自适应重加权每个样本——高置信度样本施加强约束、低置信度样本衰减影响力，无需硬阈值过滤；第二项 $\frac{1}{2} \log \sigma_i^2$ 充当概率校准项，防止模型将所有方差推高以逃避回归。相比二值化分类伪标签质量（保留/丢弃），UGL 提供了平滑、可微、端到端的不确定性处理机制。

### 一个完整示例

以 DanceTrack 双人舞蹈交叉场景为例。第 1 帧在两舞者头部各标一个点，TFP 的 SAM 分割——两人间距充足，负线索未触发——直接产出两个干净的伪框。到第 15 帧，两舞者近距离交叉（间距 < 20 像素，$\tau_{dist}=30$）：目标 A 的邻域内出现目标 B 的点，TFP 将其采集为 A 的负提示，同时将 A 的负提示送入 B 的 SAM 调用，两人各自收到"关注自己、拒绝对方"的引导，正确分割出各自的躯干；Kalman 先验框进一步将分割搜索约束在合理范围，防止遮挡导致的碎片化。到第 30 帧大幅旋转，点标注落在颈部虽然精确但已无法框定全身——Kalman 预测框通过运动连续性提供粗略范围，负线索确保 SAM 不吞噬相邻目标，联合产出完整伪框但置信度降至 0.7。UGL 据此将 $\sigma$ 提至约 1.43，该样本的回归约束在损失中自动弱化。最终经过 10 个 epoch 训练，模型学会从视觉特征推断舞者的完整轮廓，推理阶段即使无任何点输入也能稳定追踪双人穿过交叉和遮挡。

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \lambda_{reg} \mathcal{L}_{reg} + \lambda_{cls} \mathcal{L}_{cls} + \lambda_{id} \mathcal{L}_{id}$，其中 $\mathcal{L}_{cls}$ 为分类损失、$\mathcal{L}_{id}$ 为身份预测损失（沿用 MOTIP），$\lambda_{reg}=3$（消融最优）。训练使用 AdamW，骨干学习率 $1\times10^{-5}$、其余 $1\times10^{-4}$，权重衰减 $5\times10^{-4}$，预热 1 epoch，梯度裁剪 0.1。随机翻转、裁剪、尺度抖动（短边 480-800px，长边 max 1440px）和色彩抖动。单卡 RTX 5090，30 帧/batch，10 epoch，epoch 6 和 9 退火。推理：检测阈值 0.3，新生轨迹阈值 0.6，丢失容忍 30 帧，无数据集特定调参。

## 实验关键数据

### 主实验

| 数据集 | 指标 | PS-Track（点监督） | 最佳全监督方法 | 差距 |
|--------|------|------|----------|------|
| DanceTrack | HOTA | 52.3 | MOTIP 69.6 | -17.3 |
| DanceTrack | MOTA | 71.7 | MOTIP 80.0 | -8.3 |
| DanceTrack | IDF1 | 53.4 | MOTIP 65.4 | -12.0 |
| SportsMOT | HOTA | 45.2 | MOTIP 66.7 | -21.5 |
| JRDB | HOTA | 20.72 | TrackFormer 19.16 | **+1.56** |
| EmboTrack QuadTrack | HOTA | 33.9 | ByteTrack 20.7 | **+13.2** |
| EmboTrack BipTrack | HOTA | 33.9 | ByteTrack - | - |

PS-Track 以点监督在 DanceTrack 和 SportsMOT 上虽与全监督方法存在差距，但在 JRDB（密集人群、透视畸变严重）和 EmboTrack（极端 ego-motion、移动平台剧烈震动）上显著超越全监督基线——在 QuadTrack 上超过 ByteTrack 达 13.2 HOTA，在 JRDB 上超过 TrackFormer 和 DiffMOT。

### 消融实验

| 配置 | HOTA | DetA | AssA | 说明 |
|------|------|------|------|------|
| Baseline（无TFP/PEWA/UGL） | 30.3 | 45.4 | 20.4 | 仅点标注直连 DETR |
| + TFP | 49.0 | 65.2 | 37.1 | +18.7，数据层贡献最大 |
| + TFP + UGL | 49.4 | 64.7 | 38.0 | 损失层贡献 +0.4 |
| **Full（TFP+PEWA+UGL）** | **50.3** | **65.3** | **39.1** | 模型层贡献 +0.9 |

TFP 是最大性能来源（+18.7 HOTA），说明伪标签质量是点监督 MOT 的决定性因子。PEWA 贡献 +0.9，主要增益来自边界感知带来的检测精度提升。UGL 贡献 +0.4。Naive TBD 基线（点→框膨胀→追踪）仅 10.8 HOTA，PS-Track 将同配置提升至 42.9（+32.1 HOTA），验证了系统性三层设计的必要性。

### 关键发现
- TFP 是性能提升最大来源：从 30.3 到 49.0 HOTA（+62%），点监督的核心瓶颈确实在数据层而非模型架构
- PEWA 的随机激活（p < 1.0）反而导致性能下降——下游编码器无法学到稳定时序嵌入，p=1.0 的一致激活才是最优
- 对标注噪声鲁棒：16 像素扰动内 HOTA 几乎不变（44.2→44.1），48 像素才降至 39.8
- 推理 FPS（29.48）与全监督 MOTIP（29.82）几乎持平，零额外开销
- 在 JRDB 和 EmboTrack 上反超全监督，暗示点监督在极端场景下可能去除了全监督框标注带来的非必要上下文噪声

## 亮点与洞察
- **TFP 用追踪知识引导 SAM**：将 Kalman 运动先验和邻域负线索注入 SAM 的分割过程，是"任务知识与基础模型协同"的优雅范例，可迁移到任何需要 SAM 做时序一致分割的场景
- **PEWA 训练赋能推理旁路**：用小波变换将边界重建转为频域调制，训练时开启边界感知、推理时零开销——这种"训练增强模块"模式可推广到其他弱监督视觉任务
- **UGL 概率化损失处理伪标签噪声**：把平方误差除以方差等价于自适应样本加权，加上 $\log \sigma^2$ 校准项防止方差坍缩——用两个项完成平滑的不确定性建模，比硬阈值过滤实用得多
- **反全监督直觉的发现**：在 JRDB 和 EmboTrack 上点监督反超全监督，提示框标注可能引入了不必要的视觉琐碎噪声，点标注的稀疏性反而提供了正则化效应

## 局限与展望
- 严重遮挡下（如密集舞蹈编队、摔跤纠缠），TFP 的负线索仍可能失效——多个目标的点落在不可分辨区域时，无法有效分离
- 运动模糊帧上 PEWA 的高频激励策略退化：高频边缘特征已被物理抹除，频域重建的信息基础不足
- 当前每帧都需要点标注（节省约 60% 成本而非 100%），未来可扩展为每 5 帧稀疏标注 + 无监督运动传播填补中间帧
- 点标注从真实框中心合成以控制变量，未测量真实人工点击环境下的性能——实际点击偏差可能更大，需接续地研究
- 目前限于 2D MOT，扩展到 LiDAR/BEV 下的 3D MOT 是自然方向

## 相关工作与启发
- **vs MOTIP（全监督基线）**：共享同一 DETR 架构，PS-Track 以点监督达 52.3 vs 69.6 HOTA（-17.3）但标注成本降至 1/3，在成本敏感场景下有实用价值
- **vs Point2RBox + 追踪器（Naive TBD）**：仅 10.8 HOTA，PS-Track 提升 +32.1，证明系统化多级设计的必要性远大于简单组合现成模块
- **vs SAM 直接分割后追踪**：SAM-v1 20.7 HOTA → SAM-v2 39.0 → SAM-v3+TFP 50.3，说明了"如何引导 SAM"比"用哪个版本的 SAM"更重要

## 评分
- 新颖性: ⭐⭐⭐⭐ 将点监督 MOT 从朴素级联升级为三层递进框架，TFP/PEWA/UGL 各自设计有特色，整体思路的系统性较强
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个数据集（含两个极端场景）、3 种 MOT 范式、5 项深度消融（含噪声鲁棒性、SAM 变体、PEWA 激活概率），覆盖面很广
- 写作质量: ⭐⭐⭐⭐ 方法逻辑清晰、组件动机明确、消融有深度；但"PS-Track"（框架名）与"PS-MOT"（论文名）的命名一致性可优化
- 价值: ⭐⭐⭐⭐ 点监督 MOT 在标注成本敏感场景下有直接应用价值，TFP/PEWA/UGL 的设计思路有广泛的可迁移性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] LAMP: Localization Aware Multi-camera People Tracking in Metric 3D World](../../CVPR2026/human_understanding/lamp_localization_aware_multi-camera_people_tracking_in_metric_3d_world.md)
- [\[ECCV 2026\] Multi-scale Object-Aware Gaze Estimation via Geometric Reasoning](multi-scale_object-aware_gaze_estimation_via_geometric_reasoning.md)
- [\[ICLR 2026\] LINK: Learning Instance-level Knowledge from Vision-Language Models for Human-Object Interaction Detection](../../ICLR2026/human_understanding/link_learning_instance-level_knowledge_from_vision-language_models_for_human-obj.md)
- [\[CVPR 2026\] OSMO: Open-vocabulary Self-eMOtion Tracking](../../CVPR2026/human_understanding/osmo_open-vocabulary_self-emotion_tracking.md)
- [\[CVPR 2026\] Stake the Points: Structure-Faithful Instance Unlearning](../../CVPR2026/human_understanding/stake_the_points_structure-faithful_instance_unlearning.md)

</div>

<!-- RELATED:END -->
