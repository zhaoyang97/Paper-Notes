# VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM

**会议**: CVPR 2025  
**arXiv**: [2603.09673](https://arxiv.org/abs/2603.09673)  
**代码**: https://anhthuan1999.github.io/varsplat/  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, SLAM, 不确定性建模, RGB-D, 位姿估计

## 一句话总结
VarSplat 在 3DGS-SLAM 框架中为每个 Gaussian splat 学习外观方差 $\sigma^2$，通过全方差定律推导出可微分的逐像素不确定性图 $V$，并将其用于 tracking、loop detection 和 registration，在 Replica/TUM/ScanNet/ScanNet++ 四个数据集上取得了更鲁棒的位姿估计和有竞争力的重建质量。

## 研究背景与动机
1. **领域现状**：3DGS-SLAM 系统（SplaTAM, Gaussian-SLAM, LoopSplat）通过光栅化各向异性高斯进行快速可微渲染，已在密集 RGB-D SLAM 中取得了很好的效果。
2. **现有痛点**：现有方法对测量可靠性的建模是隐式的——对所有像素使用均匀的光度损失权重。这导致在低纹理区域、反射表面、深度不连续处等场景下，位姿估计容易漂移。
3. **核心矛盾**：几何侧的不确定性（深度方差、概率滤波器）已有研究，但外观不确定性——直接反映 3DGS 渲染不稳定性的量——从未在在线密集 SLAM 中被当作一等公民对待。
4. **本文要解决什么？** 如何在 3DGS-SLAM 中显式量化外观不确定性，并利用它改善位姿估计（tracking + loop + registration）的鲁棒性？
5. **切入角度**：为每个 splat 增加一个可学习的外观方差参数 $\sigma_i^2$，通过全方差定律（law of total variance）在 alpha compositing 中传播，得到逐像素不确定性图。
6. **核心idea一句话**：学习 per-splat 外观方差并通过 alpha compositing 传播为 per-pixel 不确定性，作为 tracking/loop/registration 的置信度权重。

## 方法详解

### 整体框架
输入 RGB-D 流 → 增量式 submap 构建（每个 submap 包含带 $\sigma^2$ 的 3D Gaussians）→ 可微渲染得到颜色 $\hat{I}$、深度 $\hat{D}$、不确定性图 $V$ → 三个下游模块使用 $V$ 或 $\sigma^2$ 进行加权：tracking（帧间位姿）、registration（submap 对齐）、loop detection（长程回环检测）。

### 关键设计

1. **Per-splat 外观方差 $\sigma_i^2$**:
   - 做什么：为每个 Gaussian splat 新增一个 3 维参数 $\sigma_i^2 \in \mathbb{R}^3$，表示该 splat 颜色均值周围的方差
   - 核心思路：与位置 $\mu_i$、协方差 $\Sigma_i$、SH 颜色 $c_i$ 等一起端到端优化。$\sigma_i^2$ 不同于几何协方差 $\Sigma_i$（定义空间范围），也不同于 SH 系数（定义均值颜色），而是建模"该 splat 的颜色在不同视角下的波动幅度"
   - 设计动机：在深度不连续处、遮挡边界、反射/透明表面上，由于 alpha 权重随视角剧烈变化，splat 贡献不稳定，$\sigma^2$ 自然会学到较大值

2. **Per-pixel 不确定性渲染（全方差定律）**:
   - 做什么：将 per-splat 方差传播为 per-pixel 方差 $V$
   - 核心思路：利用全方差定律 $\text{Var}[X] = \mathbb{E}[\text{Var}[X|Z]] + \text{Var}(\mathbb{E}[X|Z])$，令 $X$ 为像素颜色，$Z$ 为 Gaussian 索引。经 alpha compositing 推导得到：$V = \sum_i w_i(\sigma_i^2 + c_i^2) - (\sum_i w_i c_i)^2$
   - 设计动机：方差渲染与颜色/深度共享同一次光栅化 pass，无需额外解码器或预训练模型，保持实时性

3. **端到端方差学习（Gaussian NLL）**:
   - 做什么：通过高斯负对数似然损失 $\mathcal{L}_{\text{var}}$ 训练方差
   - 核心思路：$\mathcal{L}_{\text{var}} = \frac{1}{2V}(\|\hat{I}-I\|_2^2 + \|\hat{D}-D\|_2^2) + \log(V)$。选用 MSE 而非 L1 与高斯假设一致，方差梯度 $\frac{\partial \mathcal{L}}{\partial \sigma_i^2} = \frac{\partial \mathcal{L}}{\partial V} \cdot w_i$，通过 alpha 权重自然传播
   - 设计动机：不依赖预训练模型（如 DINOv2），从头学习方差使其与当前场景匹配

### 下游使用

- **Tracking**：per-pixel 权重 $\tilde{w}_p = \exp[-(\log V - \tilde{V})/\tau]$（中位数居中对数缩放），在光度损失上加权，Tracking 期间冻结方差参数
- **Loop Detection**：计算 submap 的 opacity ratio $r = \frac{\sum_j \tilde{w}_s \alpha_j}{\sum_j \alpha_j}$，调制 cross similarity，避免在不可靠区域触发假回环
- **Registration**：匹配 submap 后用 per-pixel 权重加权光度损失进行位姿精细化

### 损失函数
$$\mathcal{L}_{\text{map}} = \lambda_{\text{color}} \cdot \mathcal{L}_{\text{color}} + \lambda_{\text{depth}} \cdot \mathcal{L}_{\text{depth}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}}$$

其中 $\mathcal{L}_{\text{color}}$ 混合 L1 和 SSIM，$\mathcal{L}_{\text{depth}}$ 为深度 L1，$\mathcal{L}_{\text{reg}}$ 控制 Gaussian scale。

## 实验关键数据

### 主实验

| 数据集 | 指标 (ATE RMSE ↓ cm) | VarSplat | LoopSplat | CG-SLAM | 提升 |
|--------|----------------------|----------|-----------|---------|------|
| Replica (8 scenes) | ATE RMSE | **0.23** | 0.26 | 0.27 | -11.5% |
| ScanNet++ (5 scenes) | ATE RMSE | **1.69** | 2.05 | — | -17.6% |
| TUM-RGBD (5 scenes) | ATE RMSE | **3.20** | 3.33 | 4.0 | -4.0% |
| ScanNet (6 scenes) | ATE RMSE | **6.5** | 7.7 | 8.1 | -15.6% |

渲染质量：Replica 上 PSNR 37.15（LoopSplat 36.63），ScanNet++ NVS PSNR 21.33（LoopSplat 21.30），重建 F1 90.2% 与 LoopSplat 90.4% 基本持平。

### 消融实验

| 配置 | ATE RMSE ↓ | 说明 |
|------|-----------|------|
| No uncertainty | 8.20 | 基准，无不确定性 |
| +Tracking only | 7.63 | 加 tracking 权重降 7% |
| +Tracking+Loop | 7.49 | 回环检测进一步降 |
| +Loop+Registration | 7.51 | 无 tracking 也有效 |
| **Full (T+L+R)** | **6.53** | 三者结合最优，降 20.4% |

方差训练消融（ScanNet）：tracking 冻结方差（7.55→6.53 降 13.5%）、加入深度残差到方差损失（7.17→6.53 降 8.9%）、使用 MSE 而非 L1（7.38→6.53 降 11.5%）。

### 关键发现
- 不确定性在三个阶段（tracking/loop/registration）的效果互补，全部启用效果最佳
- Tracking 时冻结方差避免与位姿优化冲突是关键设计
- 运行时间与 LoopSplat 相当（Mapping 1.9s/fr vs 1.2s/fr，Tracking 2.0s/fr vs 1.8s/fr）
- 在纹理不良和反射表面场景下提升最显著

## 亮点与洞察
- **全方差定律 + alpha compositing 的优雅结合**：将统计推导嵌入现有渲染管线中，无需额外网络或采样，单次光栅化即可获得不确定性图，既理论优美又实用高效
- **Tracking 冻结、Mapping 训练的分离策略**：方差在 mapping 阶段与其他参数一起学习，但在 tracking 和 registration 阶段冻结，避免两个目标互相干扰
- **不确定性可视化直觉正确**：高不确定性自然集中在深度不连续、遮挡边界、反射/透明表面等直觉上不可靠的区域，说明方差学习有效反映了测量不确定性

## 局限性 / 可改进方向
- 方差渲染增加了约 60% 的 Mapping 时间（1.9s vs 1.2s per frame），在资源受限场景下可能是瓶颈
- 仅建模外观方差，未考虑几何（位置/尺度）不确定性——两者联合建模可能提供更完整的置信度
- 全局精化阶段移除了不确定性权重，可能在该阶段仍有改进空间
- 动态场景处理未涉及——方差可能被运动物体扰动

## 相关工作与启发
- **vs LoopSplat [ECCV'24]**: 同为 submap-based 3DGS-SLAM，但 LoopSplat 无不确定性建模。VarSplat 在其框架上增加了方差学习，在 ScanNet++ 上 ATE 从 2.05 降到 1.69
- **vs CG-SLAM**: CG-SLAM 建模深度驱动的几何不确定性，VarSplat 建模外观不确定性，两者互补
- **vs Uni-SLAM**: Uni-SLAM 用 ray-termination probability 的方差，VarSplat 的方差从光栅化器端到端学习，逐帧更新
- **思路迁移**：per-splat variance 的做法可以推广到 3D Gaussian 的其他应用（NeRF synthesis、动态场景、语义分割），作为一种通用的不确定性量化方式

## 评分
- 新颖性: ⭐⭐⭐⭐ 全方差定律在 3DGS 中的应用是新颖的，但整体框架基于 LoopSplat
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、多个 baseline、详细消融、运行时间分析
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，数学和直觉解释都到位
- 价值: ⭐⭐⭐⭐ 为 3DGS-SLAM 提供了一种简洁有效的不确定性建模方案
