# Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos

**会议**: ICLR 2026  
**arXiv**: [2510.18489](https://arxiv.org/abs/2510.18489)  
**代码**: https://liujf1226.github.io/Mono4DGS-HDR  
**领域**: 3D视觉 / HDR重建  
**关键词**: 4D Gaussian Splatting, HDR, monocular video, alternating exposure, dynamic scene

## 一句话总结
首次解决从无位姿交替曝光单目视频重建可渲染 4D HDR 场景的问题，通过两阶段优化（正交视频空间 → 世界空间）、Video-to-World 高斯变换策略和时间亮度正则化，在合成数据上达到 37.64 dB HDR PSNR、161 FPS，全面超越现有方法。

## 研究背景与动机

1. **领域现状**：4D 动态场景重建（尤其基于 3DGS）取得进展，HDR 重建也有方法（GaussHDR, HDR-HexPlane），但两者结合——从单目交替曝光视频做 4D HDR 重建——尚无方法。

2. **现有痛点**：(a) 交替曝光导致相邻帧亮度不同，标准光度重投影失败→无法估计相机位姿；(b) 2D 先验（跟踪、深度、光流）在亮度变化帧间噪声大；(c) 现有动态方法（SplineGS, MoSca）设计用于恒定亮度视频，直接加 HDR 头效果差；(d) HDR 方法（GaussHDR）需要已知位姿和多视角。

3. **核心矛盾**：亮度交替→位姿估计难→几何不稳定→HDR 外观不一致。一个恶性循环。

4. **切入角度**：先在正交相机坐标空间训练"视频高斯"（回避位姿估计），恢复 HDR 训练帧→用恢复的 HDR 帧做光度重投影估计位姿→变换到世界空间联合优化。

5. **核心idea一句话**：两阶段解耦——先解 HDR（正交空间），再解位姿和 3D（世界空间），通过 Video-to-World变换桥接两个阶段。

## 方法详解

### 整体框架
交替曝光单目视频 → 预处理（DepthCrafter深度 + SpatialTracker轨迹 + RAFT光流）→ **Stage 1**: 正交空间训练视频高斯（4K iter），恢复 HDR 训练帧 → **Video-to-World 变换**：动静分离+位置/旋转/尺度变换 → **Stage 2**: 世界空间联合优化高斯+位姿（11K iter）→ 输出可渲染的 4D HDR 场景。

### 关键设计

1. **正交空间视频高斯（Stage 1）**:
   - 做什么：在正交投影模型下训练全动态高斯，$(x^v, y^v) \in [-1,1]^2$ 为归一化像素坐标，$z^v$ 为深度
   - 核心思路：用正交投影避开需要相机内外参的透视投影→不需要事先知道位姿
   - 设计动机：交替曝光下位姿估计不可靠，先在"无位姿"空间做 HDR 恢复

2. **Video-to-World 高斯变换**:
   - 做什么：将视频空间高斯转换到世界空间，用动态掩码+遮挡检测分离动静
   - 关键创新：**2D 协方差不变性缩放重拟合** — 求解世界空间缩放 $S^w$ 使投影后的 2D 高斯形状与视频空间一致：$\min_{S^w} \sum_t \|\Sigma'^v_t - \Sigma'^w_t\|_2$
   - 设计动机：直接继承视频空间缩放到世界空间会导致高斯形状失真

3. **时间亮度正则化（TLR）**:
   - 做什么：确保 HDR 外观在时间上一致
   - 核心思路：将相邻帧 HDR 图像通过渲染光流 warp 对齐，用归一化差异惩罚不一致：$\mathcal{L}_{tlr} = |V \odot \frac{\tilde{H}_{t-1 \to t} - \tilde{H}_t}{\tilde{H}_{t-1 \to t} + \tilde{H}_t}|_1$
   - 设计动机：交替曝光帧的亮度不同，直接监督无法保证 HDR 域的时间一致性；归一化消除 HDR 辐照度的绝对尺度影响
   - 效果：TLR 主要影响 TAE（时间一致性）而非 PSNR——没有 TLR 时 TAE 从 0.057 恶化到 0.071

4. **HDR 光度重投影损失**:
   - 做什么：在 Stage 2 用 Stage 1 恢复的 HDR 帧做光度重投影，联合优化位姿和世界高斯
   - 设计动机：标准光度重投影在交替曝光下失败，但 HDR 域的亮度一致使其可行

### 损失函数 / 训练策略
$\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{ue}\mathcal{L}_{ue} + \lambda_{dep}\mathcal{L}_{dep} + \lambda_{track}\mathcal{L}_{track} + \lambda_{arap}\mathcal{L}_{arap} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{tlr}\mathcal{L}_{tlr} + \lambda_{pr}\mathcal{L}_{pr}$

动态表示：立方 Hermite 样条（位置）+ 立方多项式（旋转）。总训练约 1.5 小时。

## 实验关键数据

### 主实验

| 方法 | Syn-Exp-3 HDR PSNR↑ | Syn-Exp-3 TAE↓ | FPS↑ |
|------|-------------------|---------------|------|
| GaussHDR† | 31.25 | 0.089 | 51 |
| HDR-HexPlane† | 29.60 | 0.155 | 1 |
| MoSca-HDR‡ | 36.89 | 0.059 | 82 |
| **Mono4DGS-HDR** | **37.64** | **0.057** | **161** |

| 方法 | Real-Exp-2 PSNR↑ | Real-Exp-2 TAE↓ | Real-Exp-3 PSNR↑ | Real-Exp-3 TAE↓ |
|------|-----------------|----------------|-----------------|----------------|
| MoSca-HDR‡ | 30.28 | 0.054 | 27.23 | 0.076 |
| **Mono4DGS-HDR** | **31.82** | **0.046** | **27.65** | **0.067** |

### 消融实验

| 配置 | Syn-Exp-3 HDR PSNR | Syn-Exp-3 TAE |
|------|-------------------|---------------|
| w/o Video Gaussian Init | 36.07 (-1.57) | 0.057 |
| w/o 遮挡处理 | 37.22 (-0.42) | 0.059 |
| w/o 2D协方差不变性 | 37.25 (-0.39) | 0.057 |
| w/o TLR | 37.58 (-0.06) | **0.071** (+0.014) |
| **Full Model** | **37.64** | **0.057** |

### 关键发现
- **Video Gaussian Init 最关键**：去掉后 HDR PSNR 降 1.57 dB——两阶段策略是方法的基石
- **TLR 对时间一致性至关重要**：PSNR 影响小（-0.06）但 TAE 恶化 24.6%
- **FPS 领先**：161 FPS，比 MoSca-HDR（82）快约 2 倍
- SplineGS/GFlow 等恒定曝光方法直接加 HDR 头效果极差（PSNR 17.59/失败）

## 亮点与洞察
- **两阶段解耦的智慧**：将 HDR 恢复和 3D 重建解耦——先在简单空间（正交/无位姿）解决 HDR，再用 HDR 的好处（一致亮度）反过来帮助 3D 重建（位姿估计）。这种互助策略打破了交替曝光的恶性循环
- **2D 协方差不变性**：从视频空间到世界空间的高斯变换中，保持投影后 2D 形状不变——这个约束简洁但对避免形变至关重要
- **归一化差异做时间正则化**：$\frac{H_1 - H_2}{H_1 + H_2}$ 消除 HDR 绝对尺度的影响，仅关注相对变化——适用于任何动态范围

## 局限性 / 可改进方向
- 依赖多个预处理模型（DepthCrafter + SpatialTracker + RAFT），预处理管线复杂且可能引入误差
- 仅支持 2-3 种交替曝光模式，更复杂的曝光策略（如随机曝光）未探索
- 训练 1.5 小时仍然较长，不适合实时应用场景
- 合成数据上 HDR GT 可评估，但真实场景无 HDR GT，只能看 LDR 指标

## 相关工作与启发
- **vs GaussHDR**: GaussHDR 需要已知位姿+多视角，仅做静态 HDR；Mono4DGS-HDR 做动态+无位姿+单目
- **vs MoSca-HDR**: MoSca 基于恒定曝光设计加 HDR 头，效果不如本文专门设计的两阶段策略
- **vs HDR-HexPlane**: 基于 NeRF 做动态 HDR 但渲染仅 1FPS；本文 161FPS

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 开创性地解决了从交替曝光单目视频做 4D HDR 重建的新问题
- 实验充分度: ⭐⭐⭐⭐⭐ 25 个场景（合成+真实）、多种曝光模式、详尽消融
- 写作质量: ⭐⭐⭐⭐ 方法复杂但描述清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 解决了实际HDR视频重建的核心问题，161FPS实时渲染
