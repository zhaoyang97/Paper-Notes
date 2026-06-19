---
title: >-
  [论文解读] ActiveGAMER: Active GAussian Mapping through Efficient Rendering
description: >-
  [CVPR 2025][3D视觉][3D Gaussian Splatting] 提出 ActiveGAMER，首次将 3D Gaussian Splatting 用于主动建图，通过基于渲染的信息增益模块高效选择最优下一视角，结合粗到细探索、后精修和全局-局部关键帧策略，在 Replica 和 MP3D 数据集上大幅超越 NeRF-based 方法的几何精度和渲染保真度。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "主动建图"
  - "Next-Best-View"
  - "实时渲染"
  - "SLAM"
---

# ActiveGAMER: Active GAussian Mapping through Efficient Rendering

**会议**: CVPR 2025  
**arXiv**: [2501.06897](https://arxiv.org/abs/2501.06897)  
**代码**: 无  
**领域**: 三维视觉 / 主动重建  
**关键词**: 3D Gaussian Splatting, 主动建图, Next-Best-View, 实时渲染, SLAM

## 一句话总结

提出 ActiveGAMER，首次将 3D Gaussian Splatting 用于主动建图，通过基于渲染的信息增益模块高效选择最优下一视角，结合粗到细探索、后精修和全局-局部关键帧策略，在 Replica 和 MP3D 数据集上大幅超越 NeRF-based 方法的几何精度和渲染保真度。

## 研究背景与动机

主动重建（Active Reconstruction）是自主机器人的核心能力：机器人需要自主决定观测位置，以实现尽可能完整和精确的三维场景重建。这本质是一个规划+建图的联合优化问题。

近年来辐射场（NeRF）被引入主动重建领域，但面临两大瓶颈：

**计算开销大**：NeRF 的体渲染需要对每条光线密集采样 MLP，导致渲染速度极慢，难以实时评估大量候选视角的信息增益

**光度重建被忽视**：由于渲染慢，现有 NeRF-based 方法（如 NARUTO）主要关注几何重建，对渲染质量（RGB 保真度）优化不足

**运动受限**：许多方法限制为 2D 平面运动或离散跳转，无法在复杂 3D 空间中自由探索

3D Gaussian Splatting（3DGS）提供了一种高效替代：用稀疏高斯椭球显式表示场景，渲染速度比 NeRF 快数量级。本文的核心 idea 是：**利用 3DGS 的实时渲染能力来驱动主动建图**——大量生成候选视角、快速评估信息增益、自主决策下一步观测位置，同时实现几何和光度的高质量重建。

## 方法详解

### 整体框架

ActiveGAMER 的 pipeline 可概括为一个循环：
1. **输入**：HabitatSim 模拟器提供 posed RGB-D 图像（$680 \times 1200$）
2. **高斯建图**：用 SplaTAM 的简化 3DGS 增量更新高斯地图
3. **基于渲染的规划**：在候选视角上渲染 silhouette mask 评估信息增益 → 选择 next-best-view
4. **路径规划**：用 RRT 在自由空间规划无碰撞路径
5. **执行动作**：机器人移动到目标位姿，获取新观测 → 回到步骤 1
6. **后精修**：探索完成后用全局关键帧进一步优化渲染质量

支持无约束 6DoF 运动，不限于 2D 平面。

### 关键设计

1. **简化高斯表示与实时渲染**:

    - 功能：用各向同性高斯（颜色 $c$、位置 $\boldsymbol{\mu}$、半径 $r$、不透明度 $o$）表示场景，降低参数量
    - 核心思路：渲染时将 3D 高斯投影到图像平面，前后排序后 alpha blending：
    $C(\mathbf{p}) = \sum_{i=1}^{n} c_i f_i(\mathbf{p}) \prod_{j=1}^{i-1}(1 - f_j(\mathbf{p}))$
      深度图、silhouette mask 同理渲染。优化损失：
    $L = \sum_{\mathbf{p}} (S(\mathbf{p}) > 0.99)(L_1(D(\mathbf{p})) + 0.5 L_1(C(\mathbf{p})))$
    - 设计动机：NeRF 每帧渲染需数秒，而 3DGS 达到实时——这使得在一步内评估数百个候选视角成为可能

2. **基于渲染的信息增益模块**:

    - 功能：给每个候选视角计算一个信息增益分数，选出最优下一视角
    - 核心思路：在候选位姿处渲染 silhouette mask $S$，统计缺失像素数 $N_{S_i}$（值为0的像素数），同时考虑移动代价：
    $\mathcal{I} = (1 - \sigma(l_i)) \cdot \sigma(\log(N_{S_i}))$
      其中 $l_i = \|T_{i,x} - T_{t,x}\|_2$ 为距离，$\sigma$ 为 softmax 归一化
    - 设计动机：缺失像素多 = 该视角可观测到更多未重建区域。乘以距离衰减确保在信息量相同时选近处目标，减少总行程

3. **粗到细探索策略**:

    - 功能：分两个阶段高效覆盖整个场景
    - 核心思路：
        - **粗探索**：候选在单一高度面上采样（$v_1=1$m 间距，$v_2=5$ 个朝向），快速覆盖大范围
        - **细探索**：多高度层，更密采样（$v_1=0.5$m，$v_2=15$），精修遗漏区域
    - 维护**探索候选池**：根据 occupancy grid 增量采样候选，已充分观测的候选（$N_{S_i} < 0.5\%$ 像素总数）移出池。粗→细切换时重新采样全自由空间
    - 设计动机：过密采样候选增加评估开销，粗→细策略在效率和完整性间取得平衡

4. **全局-局部关键帧选择**:

    - 功能：改进 SplaTAM 仅用局部关键帧优化的策略，缓解局部过拟合
    - 核心思路：SplaTAM 选 $k$ 帧局部重叠最大的关键帧优化地图。本文改为一半局部帧 + 一半全局帧。全局帧选择标准：
        - 完整性型：silhouette mask 中新像素 > 10%
        - 质量型：渲染质量低于阈值
    - 设计动机：纯局部关键帧导致视锥内但在主表面后方的高斯被过度压低不透明度。全局帧提供远程监督，防止局部过拟合

5. **后精修（Post-Refinement）**:

    - 功能：探索结束后，用全局关键帧进一步优化高斯地图的渲染质量
    - 核心思路：增加优化迭代次数（15→60次），使用全分辨率图像进行 densification
    - 设计动机：探索阶段每步仅 15 次迭代 + 低分辨率，渲染质量受限。后精修牺牲少量几何完整性（剪枝冗余高斯）换取显著的光度提升

### 损失函数 / 训练策略

- 高斯地图优化：$L_1$ 深度损失 + $0.5 \times L_1$ 颜色损失，仅在 $S > 0.99$ 的像素上计算
- Densification mask：综合低密度区域（$S<0.5$）和深度误差过大区域（$> 50 \times$ 中位深度误差）
- 无学习组件——全部基于规则和渲染评估

## 实验关键数据

### 主实验：几何重建（MP3D）

| 方法 | Accuracy (cm) ↓ | Completion (cm) ↓ | Comp. Ratio (%) ↑ |
|------|-----------------|-------------------|-------------------|
| FBE | / | 9.78 | 71.18 |
| ANM | 7.80 | 9.11 | 73.15 |
| NARUTO | 6.31 | 3.00 | 90.18 |
| **ActiveGAMER** | **1.66** | **2.30** | **95.32** |

### 渲染质量（Replica 8场景平均）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1-Depth ↓ |
|------|--------|--------|---------|------------|
| SplaTAM (被动) | 29.08 | 0.95 | 0.14 | 1.38 |
| NARUTO | 26.01 | 0.89 | 0.41 | 9.54 |
| **ActiveGAMER** | **32.02** | **0.97** | **0.11** | **1.12** |

### 消融实验（Replica）

| 配置 | Comp. (cm) ↓ | Comp. Ratio ↑ | PSNR ↑ | L1-D ↓ |
|------|-------------|--------------|--------|--------|
| 仅粗探索 | 1.77 | 94.53 | 29.77 | 1.80 |
| 无全局关键帧 | 2.19 | 94.87 | 30.73 | 1.23 |
| 无后精修 | 1.56 | **96.50** | 30.67 | 1.42 |
| **完整系统** | 1.80 | 95.45 | **32.02** | **1.12** |

### 关键发现

- **3DGS 在主动建图中全面优于 NeRF**：几何精度提升 ~4x（MP3D Accuracy 6.31→1.66cm），PSNR 提升 6dB+
- 粗到细策略使粗探索阶段就达到 94%+ 完整度，细探索补充 ~2% 并改善渲染
- 后精修显著提升 PSNR（30.67→32.02），但会略降几何完整度（冗余高斯被剪枝）
- 全局关键帧对防止过拟合必不可少：移除后完成度下降，深度误差增大

## 亮点与洞察

- **首个基于 3DGS 的主动建图系统**：充分发挥 3DGS 实时渲染优势，使基于渲染的信息增益评估变得实际可行
- **系统工程设计优秀**：粗细探索、候选池管理、全局/局部关键帧、后精修——每个组件都有明确的设计动机和消融验证
- **几何+光度双重优化**：不像先前方法只关注一方面，同时追求完整几何和高保真渲染
- **6DoF 自由运动**：不限制为 2D 平面或离散跳转，更接近真实机器人场景

## 局限与展望

- **假设已知定位**：真实场景中需要集成 SLAM 定位模块
- **忽略双面物体**：渲染 silhouette mask 看不到物体背面，导致背面不被探索（figure 9 展示了典型失败案例）
- **表面附近候选采样受限**：为避免渲染器忽略近处高斯，刻意避开靠近表面的候选→某些区域无法覆盖
- **运动约束缺失**：未考虑真实机器人的运动学约束（如轮式机器人无法任意 6DoF 移动）

## 相关工作与启发

- NARUTO 开创了 6DoF 主动 NeRF，但受制于渲染速度；ActiveGAMER 用 3DGS 解决了核心瓶颈
- AG-SLAM 同期工作用 Fisher Information + 3DGS，但侧重 SLAM 而非纯重建
- 对 embodied AI 的启示：实时高质量渲染使"边建图边规划"真正可行，是主动感知系统的重要基石

## 评分
- 新颖性: ⭐⭐⭐⭐ 3DGS+主动建图的组合新颖，但各组件方法论创新有限
- 实验充分度: ⭐⭐⭐⭐ Replica+MP3D 双数据集、几何+渲染双指标、详细消融和运行时分析
- 写作质量: ⭐⭐⭐⭐ 系统描述清晰，算法流程完整，但部分章节偏冗长
- 价值: ⭐⭐⭐⭐ 对主动重建领域有重要推动，验证了 3DGS 作为主动视觉基础的可行性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](../../CVPR2026/3d_vision/magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[CVPR 2025\] ERUPT: Efficient Rendering with Unposed Patch Transformer](erupt_efficient_rendering_with_unposed_patch_transformer.md)
- [\[CVPR 2025\] Learnable Infinite Taylor Gaussian for Dynamic View Rendering](learnable_infinite_taylor_gaussian_for_dynamic_view_rendering.md)
- [\[CVPR 2026\] Paparazzo: Active Mapping of Moving 3D Objects](../../CVPR2026/3d_vision/paparazzo_active_mapping_of_moving_3d_objects.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](../../ICLR2026/3d_vision/3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)

</div>

<!-- RELATED:END -->
