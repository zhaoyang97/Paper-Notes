---
title: >-
  [论文解读] FastAvatar: Towards Unified and Fast 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers
description: >-
  [ICLR 2026][3D视觉][3D Avatar] 用一个统一的前馈 Transformer（LGRT）把任意数量（1~16 帧）的人脸观测——单图、多视角、单目视频——在几秒内重建成可驱动的高质量 3DGS 头像，并首次实现"观测越多质量越好"的增量式重建。 领域现状：3D 头像重建从 NeRF 走到 3DGS…
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "3D Avatar"
  - "3D Gaussian Splatting"
  - "前馈重建"
  - "变长输入"
  - "增量重建"
  - "VGGT"
---

# FastAvatar: Towards Unified and Fast 3D Avatar Reconstruction with Large Gaussian Reconstruction Transformers

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=P7zBSCs4Xt](https://openreview.net/forum?id=P7zBSCs4Xt)  
**代码**: [https://github.com/TyrionWuYue/FastAvatar](https://github.com/TyrionWuYue/FastAvatar)  
**领域**: 3D 视觉 / 数字人头像重建  
**关键词**: 3D Avatar, 3D Gaussian Splatting, 前馈重建, 变长输入, 增量重建, VGGT  

## 一句话总结
用一个统一的前馈 Transformer（LGRT）把任意数量（1~16 帧）的人脸观测——单图、多视角、单目视频——在几秒内重建成可驱动的高质量 3DGS 头像，并首次实现"观测越多质量越好"的增量式重建。

## 研究背景与动机
**领域现状**：3D 头像重建从 NeRF 走到 3DGS，画质和渲染速度都大幅提升，但主流方法仍是 per-scene 优化，需要单人几十秒的多视角视频反复迭代训练。近期出现的前馈方法（LAM 单图、Avat3r 4 图）能秒级出结果，代表了新范式。

**现有痛点**：作者归纳出三个卡点。其一是**无法利用先验**——优化式方法所有信息只来自当前输入观测，缺数据就补不出来，对完整 3D 观测高度依赖；其二是**对齐精度低**——方法普遍完全依赖 FLAME/3DMM 这类参数化代理模型做粗对齐，而代理模型受表达能力、光照、数据质量限制，不经精修就用会很不鲁棒，难以统一适配光场/手机/单反等不同来源；其三是**处理不了变长数据**——优化式方法要求最少 30 秒输入，少了就建模失败，而前馈方法为训练方便又固定了输入长度（LAM 只吃 1 帧，Avat3r 固定 4 帧），现实数据是任意帧数，固定长度等于浪费观测。

**核心矛盾**：单图前馈能"幻想"出合理细节但无法严格利用更多观测，多帧优化能精确拟合观测但慢且吃不了稀疏输入——**画质、速度、数据利用率三者难以兼得**，且没有一个统一模型能同时吃单图/多视角/视频。

**本文目标**：做一个数据高效、高质量、快速的统一前馈框架，能吃任意长度输入，并随观测增加持续提升质量。

**核心 idea**：把通用 3D 重建大模型 VGGT 的成功架构搬到头像任务上，**改造成专门预测规范空间 3DGS 的 LGRT**——用交替的全局/帧内注意力做变长对齐与配准、用多粒度位置编码（相机/表情/头姿）消除动态人脸的错位、再用 landmark tracking loss 与 sliced fusion loss 让多帧 3DGS 能平滑增量融合。

## 方法详解

### 整体框架
FastAvatar 是一个前馈函数 $A \leftarrow G(I, \pi, z^{exp}, z^{pose})$：输入任意 $N$（1~16）张无序 RGB 观测及其相机参数 $\pi$、表情系数 $z^{exp}$、头姿 $z^{pose}$（由多视角 FLAME tracking 粗估），输出一个可被任意表情/视角驱动的规范空间 3DGS 头像。核心是 **Large Gaussian Reconstruction Transformer (LGRT)**，分 6 个阶段串起来：人脸特征提取 → 人脸编码 → 聚合与配准 → 3DGS 属性生成 → 规范 3DGS 融合 → 光栅化。

```mermaid
flowchart LR
    A[N 张人脸观测<br/>1~16 帧] --> B[DINOv2 提 token x_i]
    B --> C[多粒度编码<br/>拼相机/表情/头姿 → h_i]
    C --> D[L 层交替<br/>帧内注意力 + 全局注意力]
    D --> E[GS Head MLP<br/>预测每帧 3DGS g_i]
    E --> F[Canonical 3DGS Fusion<br/>g_f = U·g_1...g_N]
    F --> G[Gumbel-Softmax 剪枝<br/>去 50% 冗余点]
    G --> H[可驱动 3DGS 头像]
    E -.Lmk Tracking + Sliced Fusion.-> D
```

### 关键设计

**1. 把 VGGT 改造成 3DGS 头像专用的 LGRT：交替注意力做变长配准。** 头像任务比 SLAM 式环境重建要求更细粒度，且人脸是动态的（有表情、有姿态），不能像静态场景那样无差别聚合。LGRT 用 DINOv2 把每张观测编成 token，再走 $L$ 对交替的**帧内注意力**与**全局注意力**：帧内注意力（dual-stream DiT 块）抽取单帧内特征并融入 3D 位置 prompt 加速收敛，全局注意力则把不同帧的人脸 token 对齐到同一规范 3D 空间完成配准融合。相比 VGGT，作者把不稳定的 DPT 头换成直接预测规范 3DGS 的 MLP，并用 FLAME 网格顶点作为输出的 3D 位置 prompt 给初始高斯位置——这是全文最关键的组件，消融里去掉全局注意力 16 视角 PSNR 从 22.04 暴跌到 20.06、Identity 误差从 0.118 恶化到 0.210。

**2. 多粒度位置编码消除动态人脸错位。** 仅靠单一相机位姿编码不足以对齐动态人脸，无差别聚合会导致过平滑与混叠。FastAvatar 把相机位姿 $\pi_i$、表情系数 $z^{exp}_i$、头姿 $z^{pose}_i$ 一起经轻量 MLP 对齐后，沿维度拼接进 token：$h_i = U\big(x_i, \text{MLP}([\pi_i, z^{exp}_i, z^{pose}_i])\big)$。这三类标签把"同一张脸不同表情/角度"的 token 区分开，让跨帧聚合更精准，使模型能处理变长且无序的输入。

**3. Sliced Fusion Loss + Landmark Tracking Loss 实现增量融合。** 每帧各自预测的 3DGS $g_i$ 直接朴素拼成 $g_f = U(g_1, \dots, g_N)$ 会有点云错位、鬼影、颜色不一致。为此训练时随机采单帧得 $g_i$、再随机选 $N_{sliced}$ 帧融成 $g_{sliced}$，两者都用全部输入/目标帧的相机表情头姿渲染并监督：$\hat I_i = \Psi(g_i, \pi_i, z^{exp}_i, z^{pose}_i)$、$\hat I_{sliced} = \Psi(g_{sliced}, \cdots)$，这迫使任意帧组合渲染都一致，从而支持任意帧数融合。同时 landmark tracking loss 直接在输入帧上监督人脸关键点定位 $L_{track} = \sum_{j}\sum_{i}\|y_{j,i} - \hat y_{j,i}\|$，强化配准精度。总损失 $L = \lambda_1 L_{rgb} + \lambda_2 L_{ssim} + \lambda_3 L_{lpips} + \lambda_4 L_{track} + \lambda_5 L_{mask}$。这套设计让模型首次做到"加帧即提质"，sliced fusion loss 是 16 视角的命门，去掉后 PSNR 从 22.04 掉到 20.62。

**4. Gumbel-Softmax 高斯剪枝控制增量场景的点数膨胀。** 增量重建下高斯点数随输入帧数线性增长，既费显存又拖慢渲染。借鉴 LP-3DGS 与 MaskGaussian，用 Gumbel-Softmax 采样可微的 0/1 掩码 $M_i$ 直接嵌进光栅化，把高斯的"是否存在"与不透明度/形状解耦，并对掩码加 L1 正则 $L_{mask} = \frac{1}{N}\sum_i |m|$。该机制能剪掉 50%+ 的高斯基元而几乎不掉质量，16 视角点数从 346.8K 降到 138.9K，显著提速。

## 实验关键数据

训练数据来自 NeRSemble（多人多机位多表情视频），含单目与多视角子集，每对采 16 帧再随机取 1~16 帧作输入；测试在未见主体上做单图/单目视频/多视角重建。所有对比在同一块 48GB RTX 4090 上完成。

### 主实验表格（不同输入帧数的重建质量与速度）

| 输入 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Identity↓ | FPS↑ | 建模时间(s)↓ |
|------|------|-------|-------|--------|-----------|------|--------------|
| 1 View | LAM | 17.30 | 0.773 | 0.149 | 0.135 | 125 | 0.31 |
| 1 View | GaussianAvatars | 16.35 | 0.740 | 0.332 | 0.299 | <10 | >100 |
| 1 View | **FastAvatar** | **20.08** | **0.860** | **0.143** | **0.116** | **240.17** | 1.33 |
| 8 View | LAM* | 16.59 | 0.718 | 0.235 | 0.206 | 24 | 0.43 |
| 8 View | GaussianAvatars | 20.35 | 0.820 | 0.320 | 0.252 | <10 | >100 |
| 8 View | **FastAvatar** | **22.19** | **0.880** | **0.093** | **0.097** | 52.28 | 8.56 |
| 16 View | LAM* | 16.49 | 0.697 | 0.265 | 0.238 | 13 | 0.69 |
| 16 View | GaussianAvatars | 21.48 | 0.873 | 0.281 | 0.185 | <10 | >100 |
| 16 View | **FastAvatar** | **22.29** | **0.881** | **0.092** | **0.095** | 17.65 | 26.06 |

### 消融实验表格（16 View）

| 变体 | PSNR↑ | SSIM↑ | LPIPS↓ | Identity↓ | #GS(K)↓ |
|------|-------|-------|--------|-----------|---------|
| w/o sliced fusion loss | 20.62 | 0.839 | 0.159 | 0.180 | 330.5 |
| w/o tracking loss | 21.66 | 0.865 | 0.123 | 0.129 | 164.2 |
| w/o global attention | 20.06 | 0.830 | 0.167 | 0.210 | 155.7 |
| w/o GS fusion | 16.25 | 0.811 | 0.196 | 0.259 | 12.4 |
| w/o GS pruning | 21.61 | 0.867 | 0.110 | 0.136 | 346.8 |
| **Ours full** | **22.04** | **0.876** | **0.102** | **0.118** | **138.9** |

### 关键发现
- **质量随帧数单调上升**：FastAvatar 从 1→16 帧 PSNR 由 20.08 升到 22.29，验证了"增量重建"假设；优化式方法虽也随帧数提升但建模慢（>100s）；而 LAM 反而随视角增多质量下降（缺配准，单图生成偏置引入姿态/表情伪影）。
- **速度全面领先**：单图建模仅 1.33s、渲染 240 FPS，远快于优化式方法的 >100s。
- **GS Fusion 与 Global Attention 是骨架**：去掉 GS fusion 16 视角 PSNR 崩到 16.25，去掉全局注意力崩到 20.06。
- **剪枝几乎无损提速**：GS pruning 把点数从 346.8K 砍到 138.9K（剪掉 60%），PSNR 反而从 21.61 微升到 22.04。
- **可扩展到长序列**：借鉴 FramePack，把 16 帧稀疏输入外的全部帧压成聚合 token，单次前馈可处理数百帧（图 5 中 512 视角进一步补出口腔等细节），避免均匀采样丢信息或全喂 OOM。

## 亮点与洞察
- **统一模型吃任意输入**是真正的卖点：单图、多视角、单目视频共用一个前馈模型，且 1~16 帧任意无序输入都能用，解决了"前馈方法固定长度浪费数据"的结构性缺陷。
- **"增量重建"范式有想象空间**：把头像重建从"一次性拟合固定数据"变成"边来观测边精修"，更贴合日常随手拍的真实采集场景。
- **巧妙复用大模型先验**：直接拿 VGGT 架构 + LAM 块权重初始化帧内注意力，把通用 3D 重建大模型的能力迁移到头像，省去从零训练。
- **质量-速度可调范式**：数据少时快速给可用结果、数据多时给高质量模型，让用户按算力与质量需求自由权衡。

## 局限与展望
- 单帧 PSNR 20.08 虽超基线，但绝对画质相对优化式多帧仍有上限，前馈"幻想"与严格拟合之间的平衡仍是开放问题。
- 全局注意力的显存/算力随帧数增长，超长序列只能靠 FramePack 式压缩 token 近似，全帧直喂会 OOM。
- 依赖 FLAME tracking 提供的相机/表情/头姿粗估作为位置编码，tracking 质量差时对齐可能受影响（虽然论文强调对代理模型的依赖比前作弱）。
- 训练仅用 NeRSemble 单一来源，对极端配饰、夸张表情、跨域采集设备的泛化还需更多验证。
- 与 Avat3r 的对比因其未开源只能引用原论文数字，公平性受限。

## 相关工作与启发
- **前馈头像**：LAM（单图）、Avat3r（固定 4 图）是直接对标，FastAvatar 的核心增量是"变长 + 增量"。
- **通用前馈 3D 重建**：DUSt3R、VGGT 代表无需标定的大模型重建范式，本文是把 VGGT 迁到头像并改输出头为 3DGS。
- **优化式 3DGS 头像**：GaussianAvatars、MonoGaussianAvatar 用 FLAME 代理 + per-scene 优化，画质高但慢、吃不了稀疏输入。
- **高斯剪枝**：LP-3DGS、MaskGaussian 的 Gumbel-Softmax 掩码被借来控制增量点数膨胀。
- **启发**：把"通用大重建模型 + 任务专用输出头 + 任务专用位置编码 + 任务专用融合损失"作为模板，可推广到手、全身等其他可驱动数字资产的统一前馈重建。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首次实现变长输入下的增量式前馈 3DGS 头像重建，统一模型吃单图/多视角/视频，范式上有清晰增量。
- **实验充分度**: ⭐⭐⭐⭐ 1/4/8/16 帧全梯度对比 + 五项消融 + 长序列(512 帧)扩展 + 跨数据集(Ava256)对比 Avat3r，覆盖到位；唯多数据源训练与开源对比稍欠。
- **写作质量**: ⭐⭐⭐⭐ 痛点—三因素—设计逻辑链清晰，公式与图配合好，可读性强。
- **价值**: ⭐⭐⭐⭐ 秒级、统一、增量的可用性导向方案对低成本数字人落地有实际意义，代码开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SkyEvents: A Large-Scale Event-Enhanced UAV Dataset for Robust 3D Scene Reconstruction](skyevents_a_large-scale_event-enhanced_uav_dataset_for_robust_3d_scene_reconstru.md)
- [\[ICLR 2026\] Signal Structure-Aware Gaussian Splatting for Large-Scale Scene Reconstruction](signal_structure-aware_gaussian_splatting_for_large-scale_scene_reconstruction.md)
- [\[ICLR 2026\] UP2You: Fast Reconstruction of Yourself from Unconstrained Photo Collections](up2you_fast_reconstruction_of_yourself_from_unconstrained_photo_collections.md)
- [\[CVPR 2026\] FHAvatar: Fast and High-Fidelity Reconstruction of Face-and-Hair Composable 3D Head Avatar from Few Casual Captures](../../CVPR2026/3d_vision/fhavatar_fast_and_high-fidelity_reconstruction_of_face-and-hair_composable_3d_he.md)
- [\[ICLR 2026\] Implicit 4D Gaussian Splatting for Fast Motion with Large Inter-Frame Displacements](implicit_4d_gaussian_splatting_for_fast_motion_with_large_inter-frame_displaceme.md)

</div>

<!-- RELATED:END -->
