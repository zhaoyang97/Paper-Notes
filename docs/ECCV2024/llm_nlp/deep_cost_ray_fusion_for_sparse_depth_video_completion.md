# Deep Cost Ray Fusion for Sparse Depth Video Completion

**会议**: ECCV 2024  
**arXiv**: [2409.14935](https://arxiv.org/abs/2409.14935)  
**代码**: 无  
**领域**: 3D Vision / Depth Completion  
**关键词**: depth completion, cost volume fusion, ray-wise attention, RGB-D video, temporal fusion

## 一句话总结

本文提出 RayFusion 框架，通过在 cost volume 上沿射线方向施加 self-attention 和 cross-attention 实现时序融合，以仅 1.15M 参数在 KITTI、VOID、ScanNetV2 三个数据集上全面超越或持平 SOTA 稀疏深度补全方法。

## 研究背景与动机

1. **领域现状**: 深度传感器（LiDAR、RealSense 等）已广泛部署于移动设备和自动驾驶场景，但其采集的深度图往往稀疏或存在大面积缺失。
2. **现有痛点**: 主流深度补全方法仅利用单帧 RGB-D 图像，专注于提取多模态特征，忽略了视频序列中丰富的时序信息。少数利用多帧的方法（如 ConvLSTM、时空卷积）需要将前一帧特征图 warp 到当前帧对齐，但这种对齐依赖前一帧的深度预测，一旦预测有误就会引入误差传播。
3. **核心矛盾**: 直接在 2D 特征图上做时序融合受深度预测误差影响；而如果在 3D cost volume 上做全局 attention 融合，内存占用为 $D^2H^2W^2$，完全不可行。
4. **切入角度**: 观察到 cost volume 中沿每条射线的特征是关于深度假设平面的概率分布信息，这些分布本身蕴含可利用的内在属性（如熵），因此可以沿射线方向做 attention，将复杂度从 $D^2H^2W^2$ 降至 $D^2HW$。
5. **核心 idea**: 在 3D cost volume 的射线维度上施加 self-attention（利用概率分布的内在特征）和 cross-attention（跨帧融合），实现高效且精确的时序 cost volume 融合。

## 方法详解

### 整体框架

输入 RGB-D 视频序列 → 每帧通过 Cost Volume Creation 模块 $C_\theta$ 生成 cost volume $\mathbf{V}_t$ → 与前一帧更新后的 cost volume $\mathbf{V}'_{t-1}$ 通过 Ray-based Fusion 模块 $F_\theta$ 融合 → 融合后的 cost volume $\mathbf{V}'_t$ 经 Depth Regression 模块 $R_\theta$ 回归深度 → 最后经 NLSPN 深度细化模块 $H_\theta$ 输出最终深度图。

### 关键设计

1. **Cost Volume Creation ($C_\theta$)**:
   - 做什么：从单帧 RGB-D 图像构建 3D cost volume
   - 核心思路：构建三种特征体——occupancy volume $\mathbf{V}_o$、residual volume $\mathbf{V}_r$（来自稀疏深度）和 RGB feature volume $\mathbf{V}_i$（来自多尺度图像特征），拼接后送入 3D 卷积 U-Net 推断 cost volume。体素建立在 $D$ 个均匀采样的深度假设平面上，$\mathbf{V} \in \mathbb{R}^{D \times C \times H \times W}$
   - 设计动机：基于 CostDCNet 的改进版本，去掉独立几何特征提取器，增加多尺度图像特征，使得 cost volume 能同时编码 RGB 外观和稀疏深度先验

2. **Ray-based Fusion ($F_\theta$)**:
   - 做什么：融合当前帧和前一帧的 cost volumes
   - 核心思路：
     - 首先通过相对位姿逆映射将 $\mathbf{V}'_{t-1}$ 对齐到当前视角坐标系
     - 对每个像素位置 $(h,w)$，提取射线特征 $\mathbf{F}_t = \mathbf{V}_t(:,:,h,w) \in \mathbb{R}^{D \times C}$，视 $D$ 个深度假设为 $D$ 个 token
     - 先经 2 层 3D 卷积聚合局部空间信息
     - 对每条射线分别做 **self-attention**: $\mathbf{SA}_t = \text{Attn}(\mathbf{F}_t, \mathbf{F}_t, \mathbf{F}_t)$，利用概率分布的内在属性（如熵/不确定性）精炼当前帧的深度假设
     - 再做 **cross-attention**: $\mathbf{CA}_t = \text{Attn}(\mathbf{SA}_t, \mathbf{SA}_{t-1}, \mathbf{SA}_{t-1})$，实现跨帧融合
     - 添加正弦位置编码注入深度平面索引的相对位置信息
   - 设计动机：射线方向的 attention 只需 $D^2HW$ 注意力条目，比全 volume attention 的 $D^2H^2W^2$ 高效得多；self-attention 能感知概率分布熵等内在属性，cross-attention 利用前帧积累的时序信息

3. **Depth Regression & Refinement ($R_\theta, H_\theta$)**:
   - 做什么：从融合后的 cost volume 回归密集深度图
   - 核心思路：融合 cost volume 经 3D 卷积 + pixel shuffle 转为概率体 $\mathbf{P}'_t$，softmax 归一化后加权求和得深度：
     $$\mathbf{D}_t(h,w) = \sum_{i=1}^{D} d_i \times \mathbf{p}^i_{h,w}$$
   - 最后通过 NLSPN（非局部空间传播网络）在图像域精细化深度
   - 设计动机：概率体回归的方式允许同时输出深度和置信度，为后续精细化提供额外指导

### 损失函数 / 训练策略

- **$L_1$ 深度损失**: $\mathcal{L}_{L1} = \frac{1}{|\mathbb{P}|}\sum|D_t - D_{gt}|$
- **交叉熵损失**: $\mathcal{L}_{CE} = -\mathbf{p}_{gt}^T \log \mathbf{p}$，使用**软标签**（找 GT 深度最近的两个假设平面，计算归一化权重），而非 one-hot 硬标签
- **总损失**: $\mathcal{L}_{total} = \mathcal{L}_{L1} + \mathcal{L}_{CE}$
- 使用 AdamW 优化器，batch size 4，3 × RTX 3090 训练，深度假设平面数 $D=16$，图像下采样 4 倍处理 cost volume

## 实验关键数据

### 主实验

| 数据集 | 指标 | RayFusion (1.15M) | 之前 SOTA | 提升 |
|--------|------|------|----------|------|
| ScanNetV2 | MAE (m) | **0.0160** | 0.0244 (CostDCNet, 1.8M) | -34.4% |
| ScanNetV2 | RMSE (m) | **0.0554** | 0.0759 (CostDCNet, 1.8M) | -27.0% |
| ScanNetV2 | F-score↑ | **0.9161** | 0.8998 (CostDCNet) | +1.6% |
| VOID (0.5%) | MAE (mm) | **24.51** | 25.84 (CostDCNet, 1.8M) | -5.1% |
| VOID (0.5%) | RMSE (mm) | **65.46** | 76.28 (CostDCNet) | -14.2% |
| KITTI | MAE (mm) | **176.23** | 188.80 (LRRU, 21.0M) | -6.7% |
| KITTI | RMSE (mm) | **720.63** | 729.50 (LRRU) | -1.2% |

### 消融实验 (KITTI 验证集, 20% 训练数据)

| 配置 | MAE↓ | RMSE↓ | 说明 |
|------|------|-------|------|
| A / $L_1$ only | 203.73 | 855.07 | 基线：无融合、只用 $L_1$ 损失 |
| A / $L_{CE}+L_1$ | 203.05 | 834.09 | CE 损失带来 MAE/RMSE 平衡改善 |
| A+B2 (GRU) / $L_{CE}$ | 228.85 | 799.03 | 局部卷积融合 MAE 反而变差 |
| A+B1 (Ray) / $L_{CE}+L_1$ | **198.51** | 777.37 | Ray attention 全面优于 GRU |
| A+B1+C (full) | **188.88** | **768.11** | 加入 NLSPN 进一步提升 |

### 关键发现

- 仅使用 self-attention（单帧模式）即可在 VOID 和 ScanNetV2 上达到 SOTA，说明 cost volume 内在属性的利用本身就很有价值
- 模型仅 1.15M 参数，比 LRRU (21.0M) 小 94.5%，比 ComplFormer (83.5M) 小 98.6%
- 跨数据集泛化能力突出：ScanNetV2 训练 → VOID 测试，MAE 仅 29.08mm，远优于 NLSPN (158.60mm) 和 ComplFormer (65.90mm)
- 对不同稀疏度也有很强鲁棒性：0.5% 训练 → 0.05% 测试，性能衰减远小于竞争方法

## 亮点与洞察

- **射线级 attention 是一个极其聪明的设计平衡**：既获取了全局信息（沿射线方向所有深度假设的交互），又避免了 3D volume 全局 attention 的天价开销
- **Self-attention 能隐式利用概率分布的熵作为不确定性先验**，这个发现可以迁移到其他需要处理概率体/cost volume 的任务
- **交叉熵损失 + 软标签**用于 cost volume 概率体的监督，比传统 $L_1+L_2$ 组合更稳定，这一训练技巧值得借鉴
- **极致参数效率**：1.15M 参数即可超越 21-83M 参数的方法，展示了方法设计优于暴力堆参数

## 局限性 / 可改进方向

- **高内存占用**: 3D 卷积 + cost volume 本身占用大量显存，限制了分辨率和深度假设数
- **持续错误预测难以纠正**: 如果某区域连续多帧都预测失败，时序融合也无法修复
- **深度假设平面均匀采样**: 对远近场景不够灵活，可考虑自适应采样策略
- **未探索更强的 3D backbone**: 如 Sparse 3D Conv 或者 3D Gaussian 表示可能进一步提升效率

## 相关工作与启发

- **vs CostDCNet**: RayFusion 在 CostDCNet 的单帧 cost volume 基础上增加了时序融合，且去掉了独立几何特征提取器，参数更少但性能更好
- **vs LRRU**: LRRU 用 21.0M 参数在 KITTI 上达接近 SOTA，但 RayFusion 以 94.5% 更少的参数超越之
- **vs DeepVideoMVS**: 都利用视频信息做深度估计，但 RayFusion 在 3D cost volume 空间做融合而非 2D 特征图 warp，避免了深度预测误差传播
- **vs SimpleRecon (MVS)**: MVS 方法不利用稀疏深度先验，虽然视觉效果好但定量指标差

## 评分

- 新颖性: ⭐⭐⭐⭐ 射线级 attention 融合 cost volume 的思路简洁有效，但 cost volume + attention 的组合不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集（室内/室外/不同传感器），丰富的消融和泛化实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，公式规范
- 价值: ⭐⭐⭐⭐ 极致参数效率 + 多数据集 SOTA，射线 attention 思路可迁移性强
