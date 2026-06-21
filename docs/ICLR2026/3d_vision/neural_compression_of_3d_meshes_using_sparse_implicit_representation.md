---
title: >-
  [论文解读] Neural Compression of 3D Meshes using Sparse Implicit Representation
description: >-
  [ICLR 2026][3D视觉][3D 网格压缩] 把网格转成「只在表面附近存 SDF」的稀疏隐式张量（SIR），再用一个 0.42 MB 的稀疏卷积自编码器（SNC）端到端做率失真压缩，在多类网格上以近实时速度比 Draco / V-DMC / G-PCC / NeCGS 节省 30%–90% 码率。
tags:
  - "ICLR 2026"
  - "3D视觉"
  - "3D 网格压缩"
  - "稀疏隐式表示"
  - "有符号距离场"
  - "稀疏卷积自编码器"
  - "率失真优化"
---

# Neural Compression of 3D Meshes using Sparse Implicit Representation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=AzsN1qdLwv](https://openreview.net/forum?id=AzsN1qdLwv)  
**代码**: [https://github.com/yydlmzyz1/SIR-SNC](https://github.com/yydlmzyz1/SIR-SNC)  
**领域**: 3D 视觉 / 几何压缩  
**关键词**: 3D 网格压缩, 稀疏隐式表示, 有符号距离场, 稀疏卷积自编码器, 率失真优化  

## 一句话总结
把网格转成「只在表面附近存 SDF」的稀疏隐式张量（SIR），再用一个 0.42 MB 的稀疏卷积自编码器（SNC）端到端做率失真压缩，在多类网格上以近实时速度比 Draco / V-DMC / G-PCC / NeCGS 节省 30%–90% 码率。

## 研究背景与动机
- **领域现状**：3D 网格需求暴涨（VR、机器人、自动驾驶），但带宽受限使得高效网格压缩变得紧迫。主流方法分三派——Draco / V-DMC 直接压顶点+连接性；G-PCC / SparsePCGC 先转点云再压；以及把网格转成 SDF 等隐式场再压（如 NeCGS）。
- **现有痛点**：直接压显式网格在低码率下顶点量化与拓扑简化会造成严重几何畸变、拓扑断裂；点云压缩因点的离散无序本质，转换与泊松重建过程引入大量失真；而稠密 SDF 表示随分辨率立方级膨胀，内存爆炸，被迫停留在 $64^3$ / $128^3$ 低分辨率，细节捕捉受限。
- **核心矛盾**：隐式距离场有连续几何精度的优势，但稠密采样数据冗余巨大；显式离散点高效却不连续。**两者的优点无法兼得，正是压缩效率上不去的根因。**
- **本文目标**：构造一种既有隐式场连续精度、又有稀疏表示数据效率的网格表示，并配套一个轻量、端到端率失真优化的神经压缩器。
- **核心 idea**：**只在表面附近的网格单元上记录 SDF**——观察到 Marching Cubes 只关心穿过表面的立方体，远处单元毫无信息量，于是把稠密 SDF 张量「掏空」成稀疏 SDF 张量（SIR），在同分辨率下数据量降一个数量级却保持等同精度，再用稀疏卷积自编码器（SNC）压成码流。

## 方法详解

### 整体框架
SIR-SNC 分两阶段：**几何表示**（SIR）把不规则网格转成规则的稀疏 SDF 张量；**数据压缩**（SNC）用稀疏卷积自编码器把张量嵌入到低分辨率紧凑隐特征，量化后熵编码成码流，解码端对称重建并提取表面。

```mermaid
flowchart LR
    A[原始网格 S] -->|表面附近采样 SDF| B[稀疏 SDF 张量 V]
    B -->|编码器 E_φ 下采样×2| C[紧凑隐特征 Y]
    C -->|坐标 C_Y: G-PCC<br/>属性 F_Y: 因子化熵模型| D[码流 Bitstream]
    D -->|解码器 D_θ 上采样×2 + 剪枝| E[重建稀疏 SDF V̂]
    E -->|Sparse Marching Cubes| F[解压网格]
```

### 关键设计
**1. 稀疏 SDF 张量（SIR）：只存表面邻域的距离场**。给定网格 $S$，先在空间均匀采样得到稠密网格点 $V_{dense}\in\mathbb{R}^{K\times K\times K\times 3}$，然后只保留那些到表面距离小于阈值 $\tau$ 的点（默认 $\tau$ 取一个体素对角线长度），算出它们的 SDF 值，构成稀疏集合 $V=\{(v,s(v))\mid v\in V_{dense},\ d(v)<\tau\}$。SDF 的正负号（内/外）用射线投射法判定——从 $v$ 发射一条射线，奇数次相交记 $-1$、偶数次记 $+1$。这样在高分辨率（如 $256$、$512$）下数据量仍只与表面面积成正比，比稠密表示少一个数量级，本质上是「在离散点位置之上额外携带了表面距离信息」，既有显式点的效率又有隐式场的连续精度。当 $K$ 很大导致直接算 SDF 太贵时，作者用一个 coarse-to-fine 采样策略加速保留点的 SDF 计算。

**2. Sparse Marching Cubes（SMC）与非水密网格支持**。表面恢复时不用全空间扫描，而是只对「八个顶点都在稀疏张量 $V$ 中」的立方体执行标准 Marching Cubes 三角化，因为只在表面附近做提取，速度远快于传统 MC（实测仅 0.02 s）。更关键的是，由于 SIR 只聚焦表面邻域并显式存储其空间位置，这种**局部化**特性让它能稳健表示非水密网格（开放表面）而不会表示崩塌——尽管开放表面缺乏全局内外一致性，但局部有符号距离仍编码了有意义的几何关系，足以引导 MC 提取零等值面。这点是稠密 SDF / 占据场（只能表示水密网格）做不到的。

**3. 稀疏神经压缩（SNC）：端到端率失真优化的轻量自编码器**。编码器 $\mathcal{E}_\varphi$ 用两个下采样块（每块 5 个带残差连接的稀疏卷积 SConv + 一次 strided 卷积把分辨率减半）把稀疏 SDF 张量逐级嵌成低分辨率隐特征 $Y$；$Y$ 的属性量化成整数走因子化熵模型编码，坐标则单独用 G-PCC 编码。解码器 $\mathcal{D}_\theta$ 对称地用转置稀疏卷积上采样，并在每次上采样后由一个占据预测层估计体素占据概率 $p$、**剪掉低于阈值的冗余体素**，全程维持稀疏性以省算力。训练用 MAE 重建损失 $L_{MAE}=\|V-\hat V\|_1$ 配合占据的二元交叉熵 $L_{BCE}$（$L_{Rec}=L_{MAE}+\alpha L_{BCE}$，$\alpha=0.01$），量化用加性噪声近似以保持可微，码率项 $L_{Rate}=-\sum_i\log_2(p_i)$，总损失 $L=L_{Rec}+\lambda L_{Rate}$ 由 $\lambda$ 控制率失真权衡。整个网络仅 16 通道、参数 8-bit 量化，存储约 0.42 MB。

**4. 可变码率与动态序列扩展**。推理时利用模型的**分辨率无关**特性，对同一个训练好的模型喂入不同分辨率（$\{192,256,384,512\}$）得到粗粒度码率档，再在每档内选不同 $\lambda$ 的模型做细调，实现免重训的可变码率。对动态网格序列，则借鉴隐式神经表示思路——把解码器参数 overfit 到整段序列、连同每帧隐特征一起量化进码流，从而隐式利用帧间相关性、避开复杂的网格运动估计。

## 实验关键数据

### 主实验（BD-BR 增益，以 CD 衡量，负值=码率节省）

| 测试集 | 顶点/面 | vs G-PCC | vs S.PCGC | vs Draco | vs V-DMC | vs NeCGS |
|---|---|---|---|---|---|---|
| Mixed | 11k/21k | -55.8% | -74.8% | -57.4% | - | -39.2% |
| ShapeNet | 82k/162k | -58.0% | -91.0% | -92.0% | - | -50.8% |
| MPEG | 24k/37k | -61.3% | -31.7% | -93.8% | -30.5% | -46.7% |

### 计算复杂度与模型大小

| 指标 | G-PCC | S.PCGC | Draco | V-DMC | NeCGS | Ours |
|---|---|---|---|---|---|---|
| 预编码 (s) | 0.53 | 0.53 | - | - | 20 | 0.40 |
| 编码 (s) | 1.64 | 0.40 | 0.22 | 3.42 | 60 | **0.10** |
| 解码 (s) | 0.26 | 0.68 | 0.19 | 0.42 | 0.11 | **0.10** |
| 后解码 (s) | 5.54 | 5.22 | - | - | 0.15 | **0.02** |
| 模型大小 (MB) | - | 6.63 | - | - | 0.77 | **0.42** |

### 消融实验

| 消融项 | 设置 | 结论 |
|---|---|---|
| 稀疏阈值 $\tau$ | 0.75 / 1.0 / 1.25 体素对角线 | 点数随 $\tau$ 线性增长；失真在 $\tau{>}0.75$ 后达最小，默认取 1.0 留缓冲 |
| 残差块数 | 1/3/5/7 (100K–970K 参数) | BD-BR 从 anchor → -18%/-25%/-28%，权衡后取 5 块 |
| 特征通道 | 16 → 32 | 复杂度大涨但增益不显著，保持 16 |
| 占据损失 $L_{BCE}$ | 有/无 | 去掉会保留全部体素、产生碎片伪影；加入使重建质量约提升 47% |

### 关键发现
- **压缩增益与网格复杂度正相关**：在高保真、海量顶点的 MPEG/ShapeNet 上对 Draco 增益最大（最高 -92%~-93.8%），因显式法在复杂结构上畸变更严重。
- 核心编/解码各约 0.1 s，整条管线解码+重建仅 0.12 s，近实时；mesh→SDF 转换 0.4 s（目前 CPU，可迁 GPU 加速）。
- 动态序列模式「Ours (dynamic)」比前馈自编码器再省 >28% 码率。

## 亮点与洞察
- **「掏空稠密 SDF」是简单而有力的观察**：Marching Cubes 只看表面附近立方体，远处全是浪费——一个表征层面的稀疏化就同时解决了内存膨胀与分辨率受限两大痛点。
- **打通显式与隐式的优点**：稀疏 SDF = 离散点位置 + 表面距离，兼得效率与连续精度，且天然支持非水密网格，这是稠密 SDF / 占据场做不到的。
- **轻量到可部署**：0.42 MB 模型、8-bit 量化、0.1 s 编解码，把神经几何压缩从「重网络优化」（NeCGS 编码要 1 分钟）拉回工程实用区间。
- **可变码率几乎零成本**：靠分辨率无关性免重训实现多档码率，工程上很讨喜。

## 局限与展望
- **预处理仍在 CPU**：mesh→SDF 约 0.4 s 是总编码时间的主要来源，未迁 GPU，真正实时还差一步工程优化。
- **开放表面边界伪影**：非水密网格整体重建良好，但开放边界处有轻微伪影，作者提出未来用 UDF 做进一步细化。
- **动态序列编码慢**：overfit 解码器需训练，编码耗时长，只适合离线编码场景。
- **未覆盖纹理与超大网格**：当前只压几何，未来计划纳入纹理压缩并面向高保真大体量网格。

## 相关工作与启发
- **隐式几何表示**：BOF/SDF（只能表示水密网格）、UDF（可表示一般网格但难提面）——本文用「稀疏化 + 显式存位置」绕过了水密限制。
- **网格压缩**：Draco（Morton 码 + EdgeBreaker）、V-DMC（base mesh + 位移向量）、几何图像类（转 2D 用视频编码器）——都受显式拓扑量化畸变之苦。
- **点云压缩**：G-PCC（八叉树）、SparsePCGC（学习式稀疏体素）——转换+泊松重建引入失真。
- **隐式场压缩**：Tang et al. 分块压 TSDF、NeCGS 学形变场——稠密表示困于低分辨率，本文用稀疏表示直接突破到 512 分辨率。
- **启发**：「按信息密度稀疏化表征」的思路可推广到其他需要规则张量但本质稀疏的几何/科学数据压缩任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 稀疏 SDF 张量这一表示设计简洁且切中稠密 SDF 的根本痛点，端到端率失真 + 稀疏卷积的组合在网格压缩上是清晰的增量创新。
- **实验充分度**: ⭐⭐⭐⭐ — 覆盖 Mixed/ShapeNet/MPEG/ScanNet/MGN 多源数据、对比 5 个代表方法、含 BD-BR / 复杂度 / 可视化 / 动态序列 / 非水密 / 多项消融，相当扎实。
- **写作质量**: ⭐⭐⭐⭐ — 动机递进清晰、图示到位（pipeline 与 SMC 直观），方法与实验叙述连贯。
- **价值**: ⭐⭐⭐⭐ — 0.42 MB 模型 + 近实时 + 30%–90% 码率节省，对 VR/机器人等带宽敏感场景有直接实用价值，且开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](../../ICCV2025/3d_vision/sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[AAAI 2026\] Split-Layer: Enhancing Implicit Neural Representation by Maximizing the Dimensionality of Feature Space](../../AAAI2026/3d_vision/split-layer_enhancing_implicit_neural_representation_by_maximizing_the_dimension.md)
- [\[ICLR 2026\] Path Matters: Unveiling Geometric Implicit Bias via Curvature-Aware Sparse View Optimization](path_matters_unveiling_geometric_implicit_bias_via_curvature-aware_sparse_view_o.md)
- [\[ICCV 2025\] LINR-PCGC: Lossless Implicit Neural Representations for Point Cloud Geometry Compression](../../ICCV2025/3d_vision/linr-pcgc_lossless_implicit_neural_representations_for_point_cloud_geometry_comp.md)
- [\[CVPR 2026\] NTK-Guided Implicit Neural Teaching](../../CVPR2026/3d_vision/ntk-guided_implicit_neural_teaching.md)

</div>

<!-- RELATED:END -->
