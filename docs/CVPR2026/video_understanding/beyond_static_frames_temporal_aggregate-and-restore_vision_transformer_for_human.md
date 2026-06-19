---
title: >-
  [论文解读] Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation
description: >-
  [CVPR 2026][视频理解][视频姿态估计] 在不改动 ViTPose 朴素 ViT 主干和轻量解码器的前提下，TAR-ViTPose 用「关节为中心的时序聚合（JTA）+ 全局恢复注意力（GRA）」即插即用地把相邻帧的关节特征对齐聚合并注回当前帧，使视频 2D 姿态估计在 PoseTrack2017 上比单帧 ViTPose 提升 +2.3 mAP，同时跑得更快（ViT-S 达 413 fps）。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "视频姿态估计"
  - "Transformer"
  - "时序聚合"
  - "关节级注意力"
  - "ViTPose"
---

# Beyond Static Frames: Temporal Aggregate-and-Restore Vision Transformer for Human Pose Estimation

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Fang_Beyond_Static_Frames_Temporal_Aggregate-and-Restore_Vision_Transformer_for_Human_Pose_CVPR_2026_paper.html)  
**代码**: https://github.com/zgspose/TARViTPose  
**领域**: 视频理解 / 人体姿态估计  
**关键词**: 视频姿态估计, Vision Transformer, 时序聚合, 关节级注意力, ViTPose  

## 一句话总结
在不改动 ViTPose 朴素 ViT 主干和轻量解码器的前提下，TAR-ViTPose 用「关节为中心的时序聚合（JTA）+ 全局恢复注意力（GRA）」即插即用地把相邻帧的关节特征对齐聚合并注回当前帧，使视频 2D 姿态估计在 PoseTrack2017 上比单帧 ViTPose 提升 +2.3 mAP，同时跑得更快（ViT-S 达 413 fps）。

## 研究背景与动机
**领域现状**：2D 人体姿态估计（HPE）虽然主要部署在视频场景，但当前主流做法仍是逐帧独立处理的单帧方法。近期 Vision Transformer（尤其是 ViTPose）凭借朴素 ViT 主干 + 轻量解码器的简洁设计，在静态图像上把单帧 HPE 推到了 SOTA。

**现有痛点**：单帧 ViTPose 把每一帧当成孤立图像处理，完全忽略视频里天然存在的时序连贯性。一旦遇到运动模糊、遮挡、失焦这类视频特有的退化，单帧预测就会变得不稳定、抖动甚至失败——这些恰恰是真实视频里最常见的困难场景。

**核心矛盾**：已有「想借用 ViTPose 做视频」的方法走的是「外挂」路线——把预训练 ViTPose 仅当作单帧特征提取器，再额外堆一套专门设计的时序融合模块（Transformer 类的 DSTA / CM-Pose / MTPose，或 Mamba 类的 GLSMamba）和一个配套解码器去对齐适配。这样做既让 pipeline 变复杂、推理变贵，又背离了朴素 ViT 「简洁」的初衷。即便像 Poseidon 复用了 ViTPose 的轻量解码器，它也只是用简单 cross-attention 笼统融合多帧，难以把跨帧时序对应的关节特征精确对齐。

**本文目标**：能不能把时序建模**直接嵌进** ViTPose 框架内部，而不是外挂——既保留朴素 ViT 设计和轻量解码 pipeline，又把多帧时序线索用上？

**切入角度**：作者的关键观察是——不同关节在运动中往往有**相对独立的时序轨迹**。比如跑步时手腕来回摆动，而头部基本保持朝前。所以「把所有特征 token 一视同仁地做全局注意力」并不合适，必须**按关节**分别建模时序依赖，才能保证跨帧对应的关键点特征被准确对齐。

**核心 idea**：给每个关节分配一个可学习的 query token，用「mask 感知注意力」只从相邻帧里**它自己对应的区域**聚合时序特征（aggregate），再用一次 cross-attention 把聚合到的时序信息**注回**当前帧的 token 序列（restore），最后复用 ViTPose 原解码器回归热图——这就是 Temporal **A**ggregate-and-**R**estore。

## 方法详解

### 整体框架
TAR-ViTPose 沿用「先检测、后估计」的两阶段 top-down 范式。第一阶段用人体检测器在当前帧 $X(t)$ 定位每个人，把检测框向外扩 25% 后，在以当前帧为中心、跨度为 $T$ 的连续帧序列 $S=\langle X(t{-}T),\dots,X(t),\dots,X(t{+}T)\rangle$ 上裁出该人的专属视频片段 $S_i$（论文取 $T=2$，即前后各 2 帧、共 4 张辅助帧）。第二阶段才是本文重点：对片段里每一帧用**同一个共享的 ViT encoder** 提特征 $F^{out}_i(\tau)$，然后把时序建模模块**接在 encoder 之后、解码器之前**，把相邻帧的关节特征聚合进当前帧，再喂回 ViTPose 原版的轻量解码器回归热图 $H_i(t)$。

整条链路只新增两个轻量模块：**JTA** 负责「关节级地把跨帧特征对齐并聚合」，**GRA** 负责「把聚合结果注回当前帧」。其余（ViT encoder、解码器）全部继承 ViTPose 不动，因此是即插即用的——这也是它能在保持朴素 ViT 简洁性的同时拿到时序增益的根本原因。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["人体片段 Si<br/>当前帧+前后各T帧"] --> B["共享 ViT encoder<br/>逐帧提特征 F_out(τ)"]
    B --> C["轻量解码器(共享)<br/>逐帧热图 H(τ)"]
    C -->|阈值 φ 二值化| D["关节级时序聚合 JTA<br/>每关节 query + mask 感知注意力<br/>跨帧对齐聚合 → eQ"]
    B --> D
    D --> E["全局恢复注意力 GRA<br/>eQ 注回当前帧 token"]
    B -->|当前帧 F_out(t) 作 query| E
    E --> F["增强特征 → 原解码器<br/>回归当前帧热图 Hi(t)"]
```

### 关键设计

**1. 关节为中心的时序聚合 JTA：每个关节只盯自己的轨迹**

JTA 针对的是「跨帧特征怎么聚合才不串味」这个痛点。最直接的做法是对所有帧特征做 self-attention，或拿当前帧当 query、相邻帧当 key/value 做 cross-attention（即 Poseidon 那一类），但这会把所有 token 一视同仁、忽略了各关节轨迹彼此独立的事实。JTA 的做法是给 $N$ 个关节各分配一个可学习 query token $Q\in\mathbb{R}^{N\times C}$（PoseTrack 里 $N=15$），让这些 query 与所有帧的特征 token 做 cross-attention，从而**按关节**聚合时序信息：

$$\tilde{Q} = \mathrm{JTA}\big(Q,\ \{F^{out}_i(\tau)\}_{\tau=t-T}^{t+T}\big)$$

具体堆叠 6 层相同结构：每层先做「特征→关节」的 cross-attention（关节 query 去相邻帧捞自己对应的特征），再做一次「关节→关节」的 self-attention（让各关节 query 之间交互），从 $Q_0=Q$ 迭代到 $\tilde{Q}=Q_6$。这样输出的每个 query token 都聚合了「同一关节在多帧里的时序特征」，把独立轨迹的关节分开建模，比一锅炖的全局注意力更准也更省。

**2. Mask 感知的特征→关节注意力：用热图当空间门控**

光有关节 query 还不够——怎么保证「右手腕的 query」真的只去看相邻帧里右手腕那块区域，而不被背景或别的关节干扰？这正是 JTA 内部的关键机制。作者先把每帧 ViT 特征 $F^{out}_i(\tau)$ 喂进**共享的轻量解码器**得到每帧热图 $H(\tau)$，再据此为每个关节、每帧构造一张二值空间 mask：

$$M^{j}_{x,y}(\tau)=\begin{cases}0 & H^{j}_{x,y}(\tau)\ge \phi\\ -\infty & \text{otherwise}\end{cases}$$

其中阈值 $\phi$ 默认 0.2。热图响应高于阈值的位置记 0（允许注意），否则记 $-\infty$（softmax 后注意力被压成 0）。mask resize 到 encoder 输出分辨率后，加进注意力 logits 里做带残差的 masked attention，替代标准 cross-attention：

$$Q_l=\mathrm{softmax}\big(f_q(Q_{l-1})f_k(F^{out}_i)^\top+M\big)f_v(F^{out}_i)+Q_{l-1}$$

这里 $F^{out}_i$ 是所有帧 token 拼接后的 $\frac{H}{d}\frac{W}{d}(2T{+}1)\times C$ 矩阵，$M$ 存的就是上式算出的关节级空间 mask。效果上，每个关节 query 被强制「只在自己关键点附近聚焦」，从而把时序对应的同名关节特征精准对齐——可视化也证实，不加 mask 时注意力四散甚至跑到背景，加了 mask 后注意力紧紧裹在对应关键点周围。

**3. 全局恢复注意力 GRA：把时序线索注回、且不丢全局上下文**

JTA 产出的 $\tilde{Q}$ 虽然富含跨帧时序信息，但它只是 $N$ 个关节 token，缺少关键点之间的全局空间上下文——如果像 DSTA 那样直接从这些 query token 回归热图，会因为丢失全局上下文而严重掉点（消融里掉到 70.3 mAP，比单帧 baseline 还低 11.4 点）。GRA 解决的就是「怎么把时序信息用回去」：它做一次 cross-attention，用当前帧特征 token $F^{out}_i(t)$ 当 query、用 $\tilde{Q}$ 同时当 key 和 value，把聚合到的时序语义注回当前帧的空间特征：

$$\hat{F}^{out}_i(t)=\mathrm{CrossAttn}\big(F^{out}_i(t),\ \tilde{Q},\ \tilde{Q}\big)$$

得到时空增强特征 $\hat{F}^{out}_i(t)$ 后，直接喂进 ViTPose **原版** 轻量解码器出当前帧热图。这一步的妙处在于：增强发生在「当前帧完整 token 序列」这一层，既补进了时序线索，又**完整保留**了当前帧自带的全局上下文，所以定位精度不受损——「aggregate 到关节 token、再 restore 回完整序列」正是 TAR 名字的由来。

### 损失函数 / 训练策略
整个模型（ViT encoder、JTA、GRA、解码器）端到端训练，损失就是预测热图与 GT 热图的 MSE：

$$L=\sum_i\sum_{j=1}^{N}\big\lVert H^{j}_i(t)-G^{j}_i(t)\big\rVert_2^2$$

其中 encoder/decoder 用 ViTPose 在 COCO 上的预训练权重初始化，时序模块（JTA/GRA）随机初始化从头训。单卡 RTX A6000、30 epochs、$\phi=0.2$。推理时用简单的 IoU 跨帧跟踪，使每个裁好的人像只过一次 backbone，特征在当前/前/后帧间复用以提速。

## 实验关键数据

### 主实验
在 PoseTrack2017 val 上，TAR-ViTPose 在 ViT-S/B/L/H 四种规模主干上都稳定优于同主干的单帧 ViTPose，且对手腕、脚踝这类困难关节增益尤其明显：

| 主干 | 方法 | Wrist | Ankle | Mean mAP |
|------|------|-------|-------|----------|
| ViT-S | ViTPose | 75.0 | 70.4 | 80.1 |
| ViT-S | **TAR-ViTPose** | 77.8 | 74.2 (+3.8) | **81.9 (+1.8)** |
| ViT-B | ViTPose | 77.7 | 73.9 | 81.7 |
| ViT-B | **TAR-ViTPose** | 80.3 | 77.3 (+3.4) | **84.0 (+2.3)** |
| ViT-H | ViTPose | 81.6 | 77.8 | 84.7 |
| ViT-H | **TAR-ViTPose** | 83.8 | 80.2 | **86.8 (+2.1)** |

与 SOTA 视频方法比（PoseTrack2017 val，检测框来自 Faster R-CNN）：ViT-H 版达 86.8 mAP，超过同主干的 DSTA（85.6）1.2 点；即便最小的 ViT-S 版（81.9）也比 HRNet-W48（77.3）高 4.6 点。若改用 GT 框（表中标 *），ViT-H 版冲到 90.3 mAP，比同样用 GT 框的 Poseidon（88.9）再高 1.4 点。在 PoseTrack2018 / PoseTrack21 上也分别拿到 84.2 / 84.1 mAP（GT 框下 89.8 / 91.0）的新 SOTA。

速度上（A6000，2 辅助帧，batch=16）：

| 方法 | 主干 | #Params | FPS | mAP |
|------|------|---------|-----|-----|
| PoseWarper | HRNet-W48 | 71.1M | 52 | 81.0 |
| DSTA | ViT-H | 422.2M | 25 | 84.3 |
| **TAR-ViTPose** | ViT-S | 35.6M | **413** | 81.5 |
| **TAR-ViTPose** | ViT-H | 672.5M | 28 | **86.3** |

ViT-S 版以 413 fps 远超 PoseWarper/DCPose 还精度更高；ViT-H 版即便参数量大，吞吐（28 fps）仍高于 DSTA-ViT-H（25 fps）而精度更好。

### 消融实验
均在 PoseTrack2017 val、ViT-B、$T=2$ 下进行：

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| (a) 全帧 self-attention | 82.2 mAP / 38.22 GFLOPs | 所有帧 token 一锅炖 |
| (b) 当前↔辅助 cross-attention | 82.6 mAP / 12.74 GFLOPs | 当前帧作 query |
| (c) 辅助帧 self-attn + (b) | 82.8 mAP / 16.99 GFLOPs | Poseidon 式 |
| (d) **本文关节级建模** | **84.0 mAP / 3.89 GFLOPs** | JTA+GRA，又准又省 |
| JTA only（无 GRA） | 70.3 mAP | 直接从关节 query 回归，丢全局上下文 |
| JTA + GRA（完整） | 84.0 mAP | 比单帧 baseline 81.7 高 2.3 |
| 无 mask | 82.6 mAP | 注意力四散到背景 |
| 有 mask（完整） | 84.0 mAP | mask 带来 +1.4 mAP |

### 关键发现
- **GRA 是命脉**：去掉 GRA、像 DSTA 那样直接从关节 query token 回归，掉到 70.3 mAP，比单帧 ViTPose（81.7）还低 11.4 点——证明关节 token 缺乏关键点间的全局上下文，必须 restore 回当前帧完整序列才能精确定位。
- **关节级建模又准又省**：相比全帧 self-attention（82.2 mAP / 38.22 GFLOPs），本文方案（84.0 mAP / 3.89 GFLOPs）精度高 1.8 点、计算量却不到十分之一，说明「按关节对齐」比「无差别全局聚合」更对路。
- **mask 感知注意力贡献 +1.4 mAP**：可视化显示，无 mask 时关节 query 注意力四散甚至漂到背景；有 mask 时紧贴对应关键点，跨帧对齐更干净。
- **困难关节收益最大**：脚踝、手腕这类运动剧烈、易遮挡模糊的关节提升最显著（ViT-S 脚踝 +3.8），正好印证「时序线索对补救退化帧有用」。

## 亮点与洞察
- **「即插即用」的真·轻量**：JTA + GRA 合计仅约 16.6M 参数、推理开销极小，却不动 ViTPose 主干和解码器，等于给任意预训练 ViTPose「免费」加上视频能力——这种「不重训主干、只外接对齐模块」的思路可迁移到其他基于朴素 ViT 的视频任务（如视频分割、视频检测）。
- **用解码热图当注意力 mask**：把轻量解码器在辅助帧上的热图直接二值化成空间门控，等于「免费」拿到了关节的空间先验来约束 cross-attention，省去了显式光流/对齐网络，是个很省事的对齐 trick。
- **aggregate-restore 的解耦很关键**：先把时序信息浓缩到少量关节 token（聚合维度低、好对齐），再注回完整空间序列（保留全局上下文），把「时序对齐」和「全局定位」两件事拆开各自做好，避免了二者互相拖累——这个「降维聚合 + 升维恢复」的模式很有启发性。

## 局限与展望
- 作者明确声明本工作**不处理时序姿态跟踪**（tracking），只做当前帧的姿态估计，跟踪靠简单 IoU。
- ViT-H 版参数量高达 672.5M，虽然吞吐尚可，但显存/部署成本不低；小主干（ViT-S）才是真正速度友好的甜点。
- ⚠️ mask 来自解码热图，若某些极端退化帧（连续严重遮挡/失焦）本身热图就崩了，mask 可能把正确区域也屏蔽掉，时序聚合的可靠性依赖单帧热图质量——论文未深入讨论这种级联失败。
- 时序跨度 $T=2$（共 4 辅助帧）是固定的，对快慢不一的动作是否需要自适应跨度，值得进一步探索。

## 相关工作与启发
- **vs ViTPose（单帧 baseline）**：ViTPose 逐帧独立、无时序感知；本文在其之后接 JTA+GRA 即插即用地引入时序，主干解码器全部复用，PoseTrack2017 上 +2.3 mAP，思路是「补时序」而非「换架构」。
- **vs DSTA / CM-Pose / MTPose / GLSMamba（外挂式时序融合）**：它们把 ViTPose 仅当特征提取器，再额外堆 Transformer/Mamba 融合模块 + 专用解码器，pipeline 复杂、推理贵；本文反其道，把时序嵌进 ViTPose 内部、复用原解码器，更简洁也更快（ViT-H 28 vs DSTA 25 fps）。
- **vs Poseidon**：Poseidon 同样复用 ViTPose 轻量解码器，但只用简单 cross-attention 笼统融合多帧，难以精确对齐跨帧关节；本文用「关节 query + mask 感知注意力」做关节级精确对齐，消融里关节级建模（84.0）明显优于 Poseidon 式 (c)（82.8）。

## 评分
- 新颖性: ⭐⭐⭐⭐ 「关节级 query + 热图 mask 对齐 + aggregate-restore」的组合切口清晰，但模块本身仍是 cross/self-attention 的工程化组合。
- 实验充分度: ⭐⭐⭐⭐⭐ 三个 PoseTrack 数据集 + 四种主干 + 检测框/GT 框两套 + 速度对比 + 4 组消融，相当扎实。
- 写作质量: ⭐⭐⭐⭐ 动机和方法叙述清楚，公式与 pipeline 图配合到位。
- 价值: ⭐⭐⭐⭐ 即插即用、又快又准，对实际视频 HPE 部署有直接价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](../../AAAI2026/video_understanding/lifelong_domain_adaptive_3d_human_pose_estimation.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)
- [\[CVPR 2026\] Polyphony: Diffusion-based Dual-Hand Action Segmentation with Alternating Vision Transformer and Semantic Conditioning](polyphony_diffusion-based_dual-hand_action_segmentation_with_alternating_vision_.md)
- [\[CVPR 2026\] TAPFormer: Robust Arbitrary Point Tracking via Transient Asynchronous Fusion of Frames and Events](ttapformer_robust_arbitrary_point_tracking_via_transient_asynchronous_fusion_of_.md)
- [\[CVPR 2026\] Spatio-Temporal Conditional Denoising Transformer for Modality-Missing RGBT Tracking](spatio-temporal_conditional_denoising_transformer_for_modality-missing_rgbt_trac.md)

</div>

<!-- RELATED:END -->
