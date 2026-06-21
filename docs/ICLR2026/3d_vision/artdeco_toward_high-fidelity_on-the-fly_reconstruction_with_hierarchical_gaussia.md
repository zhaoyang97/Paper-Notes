---
title: >-
  [论文解读] ARTDECO：用分层高斯结构 + 前馈先验做高保真在线 3D 重建
description: >-
  [ICLR 2026][3D视觉][On-the-fly Reconstruction] ARTDECO 把前馈 3D 基础模型（MASt3R / π³）当作模块化的位姿与点云先验，接上一个能从多尺度特征解码出结构化高斯的 Gaussian decoder，再配上带 LoD 的分层半隐式高斯表示，从单目视频流里同时拿到 SLAM 级速度、前馈级鲁棒性和接近逐场景优化的渲染质量。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "On-the-fly Reconstruction"
  - "Monocular SLAM"
  - "3D Gaussian Splatting"
  - "Feed-forward Foundation Model"
  - "Level-of-Detail"
---

# ARTDECO：用分层高斯结构 + 前馈先验做高保真在线 3D 重建

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QxvDyJP7g9](https://openreview.net/forum?id=QxvDyJP7g9)  
**代码**: [https://city-super.github.io/artdeco/](https://city-super.github.io/artdeco/)（项目主页，承诺录用后开源）  
**领域**: 3D 视觉 / 在线 3D 重建 / SLAM / 3D Gaussian Splatting  
**关键词**: On-the-fly Reconstruction, Monocular SLAM, 3D Gaussian Splatting, Feed-forward Foundation Model, Level-of-Detail  

## 一句话总结
ARTDECO 把前馈 3D 基础模型（MASt3R / π³）当作模块化的位姿与点云先验，接上一个能从多尺度特征解码出结构化高斯的 Gaussian decoder，再配上带 LoD 的分层半隐式高斯表示，从单目视频流里同时拿到 SLAM 级速度、前馈级鲁棒性和接近逐场景优化的渲染质量。

## 研究背景与动机
**领域现状**：从单目图像序列做在线（on-the-fly）3D 重建是 real-to-sim、AR/VR、机器人等应用的刚需。3D Gaussian Splatting（3DGS）凭借显式表示和高效光栅化成为主流场景表示，但单目设置下缺乏可靠几何线索（尺度模糊、视差有限、运动模糊、重叠不足），很难同时兼顾精度、速度和鲁棒性。

**现有痛点**：当前 3DGS 重建分两条路线，各有硬伤。**逐场景优化派**（MonoGS、On-the-fly-NVS 等）依赖 SfM/SLAM 估计的位姿，精度高但计算昂贵，且鲁棒性受限于这些管线的脆弱性；**前馈派**（用大规模数据学单目先验、直接回归位姿和高斯基元）推理快、跨场景鲁棒，但渲染保真度低、全局一致性弱。此外 3DGS 对场景尺度极其敏感——场景一大，所需高斯基元数量暴涨，效率骤降；已有的事后锚点剪枝会引入边界伪影、增加显存，训练期加多尺度高斯又缺乏显式结构组织。

**核心矛盾**：效率与精度不可兼得，叠加缺乏一个原则化的 level-of-detail（LoD）机制来应对大尺度可漫游场景。

**本文目标**：造一个统一系统，把前馈模型的效率与逐场景优化的可靠性合到一起，做到精确、鲁棒、实时的在线重建。

**核心 idea**：**「前馈先验当模块 + 分层 LoD 高斯当表示」**。一方面把前馈基础模型拆成位姿估计、闭环检测、稠密点云预测三个即插即用模块塞进 SLAM 式管线，用它们压住单目歧义又保住交互速度；另一方面设计一个挂在稀疏空间网格上的分层半隐式高斯结构，用 LoD-aware 致密化在保真度和渲染效率之间做原则化权衡。

## 方法详解

### 整体框架
ARTDECO（名字取自 **A**ccurate localization + **R**obust reconstruction + **De**coder-based rendering，也呼应 Art Deco 风格对结构与几何的强调）以流式 SLAM 风格处理单目序列，分三个模块串联。

```mermaid
flowchart LR
    A[单目 RGB 帧流] --> B[前端 Frontend]
    B -->|MASt3R 匹配<br/>估相对位姿| C{帧分类}
    C -->|关键帧/建图帧| D[后端 Backend]
    C -->|普通帧| F[建图模块]
    D -->|π³ 闭环检测<br/>全局 BA| E[一致位姿 + 点云置信度]
    E --> F[建图模块 Mapping]
    F -->|LoG 选点初始化<br/>分层半隐式高斯| G[结构化 3D 高斯场]
    G -->|LoD-aware 光栅化| H[新视角渲染]
```

前端给每帧估计相对最新关键帧的位姿，并把帧分成普通帧 / 建图帧 / 关键帧三类；后端通过闭环和全局 BA 精修关键帧位姿，并估计点云置信度；建图模块用所有类型的帧把逐图点云初始化成高斯并增量优化。三类帧各司其职是这套设计的关键——不像传统 3DGS-SLAM 只用关键帧。

### 关键设计

**1. 前馈基础模型作即插即用先验：用 MASt3R 跟踪、用 π³ 闭环。** 前端把 MASt3R 当作两视图重建与匹配先验，拿到逐帧点云、置信度和当前帧与最新关键帧之间的像素对应，再把当前帧 3D 点投影到关键帧像平面、用 Gauss–Newton 最小化重投影残差求出 $T_{KC} \in \mathrm{SIM}(3)$ 相对位姿；焦距未知时与位姿联合优化。由于 MASt3R 在物体边界处预测不稳，作者对每个点从半径 $\delta$ 邻域估一个局部协方差 $\Sigma_c$ 来加权残差、过滤掉不可靠的重投影。后端则在 ASMK 粗筛闭环候选后，再用 3D 基础模型 **π³** 对当前帧和 top-$N_a$ 候选生成点云、按角度误差挑出三个几何最一致的关键帧连进因子图，比单纯 ASMK 更抗弱对应和噪声。值得注意的是消融显示把前端骨干从 MASt3R（成对推理）换成 π³（多图推理）反而更差——π³ 缺乏 metric-scale 能力、在视角变化下保不住物体比例。

**2. 三类帧分流 + 重投影置信度。** 关键帧在与最新关键帧的有效对应数低于阈值 $\tau_k$ 时创建，送后端精修位姿、送建图重建；建图帧在能提供足够视差时选出（当前帧与最新关键帧像素位移的第 70 百分位超过 $\tau_m$），用来初始化新高斯；普通帧两个条件都不满足，只参与已有细节的梯度精修、不引入新结构。置信度不直接信 MASt3R 的预测值，而是用重投影误差：把点云投到 ASMK 分最高的 $N_c$ 个先前关键帧上算平均重投影误差 $\bar e$，定义 $C=1$（当 $\bar e \le \varepsilon_c$）否则 $C=\frac{1}{\bar e - \varepsilon_c + 1}$，给出更可靠的跨帧几何一致性度量，后续用来下调低置信区域高斯的初始不透明度。

**3. LoG 引导的概率化高斯插入 + 半隐式区域-个体特征。** 为避免在每个像素都铺高斯，作者只在需要精修的区域插入：用高斯拉普拉斯（LoG）算子在多分辨率图上算插入概率 $P_a(u,v)=\max\big(\min(\|\nabla^2(G_\sigma)*I\|,1)-\min(\|\nabla^2(G_\sigma)*\tilde I\|,1),\,0\big)$（$I$、$\tilde I$ 为真值与渲染图），优先填高频和重建差的区域，超过阈值 $\tau_a$ 才加。每个高斯参数化为中心 $\mu$、球谐 SH、不透明度 $\alpha$、基础尺度 $S_b$、个体特征 $f_l$ 和体素索引 $v_{id}$；基础尺度由图像空间尺度 $s'$ 与深度推得 $S_b=\frac{d_i s'}{f}$，再用两个 MLP 从区域特征 $f_r$ 与个体特征 $f_l$ 联合细化尺度与旋转：$S=S_b\cdot\mathrm{MLP}_s(f_r\oplus f_l)$，$R=\mathrm{MLP}_r(f_r\oplus f_l)$。其中 $f_r$ 编码体素局部上下文（空间按 $\epsilon$ 体素化、每体素特征初始化为零），这种「区域共享 + 个体独有」的半隐式设计兼顾全局一致与局部区分度。

**4. 距离感知的分层 LoD 高斯。** 把高斯按 level $l<L$ 组织（level 0 最细、$L{-}1$ 最粗），初始化时 level-$l$ 的高斯对应原图 $2^{2l}$ 像素的 patch，逐级下采样输入帧并从各分辨率初始化。除基础尺度按 $2^{2l}$ 加权外，每个高斯还带一个距离参数 $d_{max}=D\cdot 2^{2l}$（$D$ 为高斯到相机距离）。渲染时按观察距离 $d_r$ 决定取舍：$d_r\le d_{max}$ 纳入、$d_r>2d_{max}$ 剔除、中间区间按 $\alpha'=\alpha\cdot(2d_{max}-d_r)/d_{max}$ 平滑淡出。这种距离感知 LoD 抑制了闪烁、在不同尺度下保持稳定渲染质量同时维持效率。训练上采用分阶段流式策略：建图帧/关键帧到来时初始化新高斯并优化 $K$ 次迭代，普通帧只触发 $K/2$ 次且不加新高斯，训练帧以 0.2 概率采当前帧、0.8 采历史帧防局部过拟合；序列流式处理完后再对所有帧做一遍全局优化，并把位置/旋转的梯度传给相机位姿做联合优化。

## 实验关键数据

### 主实验：渲染质量（八大室内外基准，节选）

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练时间↓ |
|---|---|---|---|---|---|
| ScanNet++ | LongSplat | 24.94 | 0.827 | 0.260 | 442.96 min |
| ScanNet++ | OnTheFly-NVS | 18.01 | 0.761 | 0.386 | 2.29 min |
| ScanNet++ | **Ours** | **29.12** | **0.918** | **0.167** | 5.33 min |
| TUM | LongSplat | 25.09 | 0.804 | 0.272 | — |
| TUM | **Ours** | **26.18** | **0.850** | **0.224** | 5.33 min |
| Fast-LIVO2 | LongSplat | 26.37 | 0.792 | 0.276 | 313.60 min |
| Fast-LIVO2 | **Ours** | **29.54** | **0.894** | **0.158** | 6.58 min |
| Waymo | S3PO-GS | 27.28 | 0.865 | 0.352 | 34.89 min |
| Waymo | **Ours** | **28.75** | **0.880** | **0.276** | 6.58 min |

ARTDECO 在室内外全部数据集上质量最优，尤其在 TUM、ScanNet 这类带结构复杂度、运动模糊和噪声的挑战集上领先明显；训练时间仅次于 OnTheFly-NVS，但比 LongSplat（动辄数百分钟）快了一两个量级。

### 主实验：跟踪精度（ATE RMSE，越低越好）

| 数据集 | MonoGS | S3PO-GS | MASt3R-SLAM | OnTheFly-NVS | **Ours** |
|---|---|---|---|---|---|
| ScanNet++ | 1.217 | 0.632 | 0.025 | 0.891 | **0.018** |
| TUM | 0.244 | 0.117 | 0.031 | — | **0.025** |
| Waymo | 7.370 | 1.236 | — | 3.118 | **1.213** |

在 TUM fr1 上与纯 SLAM 系统比，ARTDECO（0.028）也优于 DROID-SLAM（0.038）、Go-SLAM（0.035）、MASt3R-SLAM（0.030）。

### 消融实验（ScanNet++）

| 组件 | 配置 | 指标变化 |
|---|---|---|
| 前/后端 | Full（ATE 0.018）| 换 π³ 做骨干 → 0.374；π³→vggt 闭环 → 0.096；去掉 loop → 0.057；密集关键帧 → 0.094 |
| 建图 | Full（PSNR 29.12 / SSIM 0.918 / LPIPS 0.167）| 去 LoD → 28.13；去半隐式结构 → 28.54；去全局特征 → 28.89；去建图帧 → 26.38；去普通帧 → 27.20 |

### 关键发现
- **MASt3R > π³ 当前端骨干**：成对推理保 metric-scale，多图推理虽数据更丰富但视角变化下比例失真，ATE 从 0.018 暴涨到 0.374。
- **闭环不可或缺**：去掉后 ATE 翻三倍（0.018→0.057）。
- **建图帧贡献最大**：去掉建图帧 PSNR 掉 2.74 dB（29.12→26.38），多视图约束最关键；普通帧也贡献约 1.9 dB。
- **不是帧越密越好**：用建图帧+关键帧一起做跟踪推理反而降精度——3D 基础模型在小视差稠密输入下会产生 ghosting 和模糊，污染点云与对应。

## 亮点与洞察
- **模块化前馈先验的工程美学**：不重训大模型，而是把 MASt3R、π³ 当作可替换的「位姿/闭环/点云」组件嵌进经典 SLAM 因子图，既吃到大规模预训练先验，又保留了 SLAM 的全局优化与闭环能力，可解释、可调、可换。
- **重投影置信度替代模型自报置信度**：MASt3R 边界预测不稳，作者用跨帧重投影误差重新定义置信度并下调低置信高斯不透明度，是个朴素但有效的可靠性补丁。
- **距离感知 LoD 的连续淡出**：用 $d_{max}=D\cdot2^{2l}$ 把 level 与观察距离绑定，并在 $(d_{max},2d_{max}]$ 区间线性插值不透明度，避免了 LoD 切换时的硬跳变/闪烁。
- **三类帧分流**：把「该不该建新结构、该不该精修」量化成对应比例阈值 $\tau_k$ 和视差百分位阈值 $\tau_m$，让普通帧也物尽其用做梯度精修，是质量提升的重要来源。

## 局限与展望
- **强依赖前馈基础模型**：correspondence 与几何部分靠前馈 3D 基础模型，在噪声、模糊、光照变化或输入落在训练分布外时鲁棒性下降。
- **场景假设较强**：默认场景静态刚性、光照一致、视差充足；低纹理表面、重复结构或近退化轨迹会导致漂移或伪影。
- **作者展望**：引入不确定性估计、自适应模型选择和更强先验，提升真实场景下的泛化与可靠性。

## 相关工作与启发
- **逐场景优化 3DGS-SLAM**（MonoGS、S3PO-GS、SEGS-SLAM、On-the-fly-NVS）：精度高但慢/脆，是本文要在速度和鲁棒性上超越的对象。
- **前馈 3D 基础模型**（MASt3R、π³、VGGT、pose-free 重建）：本文不与之竞争，而是把它们当模块化先验复用——这是「前馈 vs 优化」二选一困境的一种务实折中范式。
- **LoD / 大尺度 3DGS**（锚点剪枝、多尺度高斯）：ARTDECO 用挂在稀疏网格上的分层半隐式高斯 + 距离感知淡出，给出了比事后剪枝更原则化的 LoD 方案，对做大场景可漫游 3DGS 的工作有借鉴价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把前馈基础模型模块化嵌入经典 SLAM 因子图，并配距离感知分层半隐式高斯 LoD，组合新颖且工程化扎实，单点创新偏整合型。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 八大室内外基准、渲染+跟踪双指标、对前后端骨干/闭环/帧分类/LoD/半隐式结构全面消融，附录还有大量逐场景结果。
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰、动机与设计对应明确，公式与图示到位；部分符号（如 $f_r$ 来源、staged training 细节）需翻附录补全。
- **价值**: ⭐⭐⭐⭐ — 直击在线单目重建的速度-精度-鲁棒三难，给出可落地的 real-to-sim/AR-VR/机器人路径，承诺开源，实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)
- [\[ICLR 2026\] NGS-Marker: Robust Native Watermarking for 3D Gaussian Splatting](ngs-marker_robust_native_watermarking_for_3d_gaussian_splatting.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)

</div>

<!-- RELATED:END -->
