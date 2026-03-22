# Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots

**会议**: CVPR2025  
**arXiv**: [2603.13108](https://arxiv.org/abs/2603.13108)  
**代码**: PanoMMOcc (待开源)  
**领域**: autonomous_driving  
**关键词**: occupancy prediction, panoramic perception, multimodal fusion, quadruped robot, 3D scene understanding

## 一句话总结
首个面向四足机器人的全景多模态语义占用预测框架 VoxelHound，提出 PanoMMOcc 数据集（全景 RGB + 热成像 + 偏振 + LiDAR），通过垂直抖动补偿（VJC）和多模态信息提示融合（MIPF）模块达到 23.34% mIoU。

## 研究背景与动机
1. 全景相机提供 360° 无盲区视觉覆盖，对移动机器人感知至关重要，但现有占用预测方法主要面向轮式自动驾驶平台
2. 四足机器人相比轮式平台具有低传感器视点、频繁自遮挡和步态引起的强烈自运动等独特挑战
3. 仅依赖 RGB 模态在光照变化、低纹理区域和远距离感知场景下鲁棒性不足
4. 现有全景数据集主要聚焦 2D 视觉任务（检测/分割），缺乏 3D 占用标注
5. 现有占用基准数据集（SemanticKITTI、Occ3D 等）为自动驾驶设计，不考虑全景成像或四足平台
6. 缺少一个同时包含全景、热成像、偏振和 LiDAR 四种模态的真实四足机器人数据集

## 方法详解

### 整体框架
VoxelHound 是一个全景多模态语义占用预测框架。给定全景 RGB 图像、热图像、偏振图像和 LiDAR 点云，通过各自编码器提取特征，统一投影到 BEV 空间进行跨模态融合，最终由占用头恢复高度维度生成 3D 语义占用预测 $\mathbf{O} \in \mathbb{R}^{X \times Y \times Z}$。

### 关键设计

**1. 多模态融合网络**
- **Camera 分支**：三个图像模态（全景 RGB、热成像、偏振）分别由独立 2D backbone 提取多尺度特征（1/8, 1/16, 1/32），经 FPN 聚合后通过 2D→BEV 视角变换投影到 BEV 空间
- **LiDAR 分支**：点云体素化（每个体素最多 10 个点取均值），通过稀疏 3D 卷积编码器（stride 8）提取层级几何特征，splat 到 BEV 空间
- **融合分支**：多模态 BEV 特征通过融合模块聚合，经 SECOND-FPN BEV 编码器精化，占用头 reshape 通道维为垂直 bin

**2. 垂直抖动补偿（VJC）模块**
- 针对四足步态导致的垂直方向图像抖动
- 沿宽度维度平均池化得到垂直结构信息 $\mathbf{F}_v \in \mathbb{R}^{C \times H}$
- 两层 Conv1D + ReLU 编码，自适应平均池化 + 线性层估计全局垂直偏移 $\Delta h$
- 构建偏移采样网格 $\mathcal{G}(h,w) = \mathcal{G}_0(h,w) + (0, \Delta h)$，双线性网格采样补偿
- 插入在图像编码器和 BEV 视角变换之间，轻量且即插即用

**3. 多模态信息提示融合（MIPF）模块**
- 非对称融合原则：几何主导 + 语义补充
- 各模态 1×1 卷积投影到共享嵌入空间
- 图像模态通过 GAP + MLP 压缩为模态级语义 prompt $\mathbf{p}_m$
- LiDAR BEV 特征作为 query，prompt 栈作为 key/value 做几何引导注意力
- 残差调制：$\mathbf{F}_f = \tilde{\mathbf{F}}_l + \sigma(\gamma(\mathbf{F}_{attn})) \odot \tilde{\mathbf{F}}_l$，确保几何结构不被覆盖

### 损失函数
标准语义占用预测损失（交叉熵 + Lovász-softmax），沿用 EFFOcc 框架设定。

## 实验关键数据

### PanoMMOcc 数据集统计
| 属性 | 数值 |
|------|------|
| 序列数 | 54 (42 标注) |
| 总帧数 | 21,600 |
| 模态 | 全景 RGB + LiDAR + 热成像 + 偏振 |
| 体素分辨率 | 64×64×16, 每个 0.4m³ |
| 类别数 | 12 |
| 平台 | Unitree Go2 四足机器人 |

### 主实验表（mIoU on PanoMMOcc）
| 方法 | 模态 | mIoU |
|------|------|------|
| MonoScene | C | 8.94 |
| EFFOcc-C | C | 4.47 |
| EFFOcc-L | L | 18.77 |
| EFFOcc-T | C+L | 19.18 |
| VoxelHound | C | 5.79 |
| VoxelHound | C+T+P | 6.14 |
| VoxelHound | C+L | 22.87 |
| **VoxelHound** | **C+L+T+P** | **23.34 (+4.16%)** |

### 关键发现
- 四模态融合（C+L+T+P）比双模态（C+L）提升约 0.5 mIoU，个别类别（terrain +2.13, mannade +2.18）提升显著
- LiDAR 是占用预测的关键模态，纯 Camera 方法在全景场景下性能有限
- VJC 有效缓解步态引起的 BEV 特征错位
- MIPF 非对称设计优于简单拼接/加法/对称注意力

## 亮点与洞察
1. **首创性**：首个面向四足机器人的全景多模态占用数据集和方法，填补具身智能占用预测空白
2. **四模态互补设计**：热成像增强低光照鲁棒性，偏振揭示材料线索，LiDAR 提供精确几何，全景 RGB 提供丰富语义
3. **VJC 模块轻量实用**：用 1D 卷积估计全局垂直偏移，无需显式 IMU 即可补偿步态抖动
4. **MIPF 非对称融合**：以 LiDAR 几何为锚，用 prompt 机制避免密集跨模态注意力的高计算代价

## 局限性
1. mIoU 绝对值低（23.34%），pedestrian 和 pillar 类别预测接近 0，长尾问题严重
2. 数据集规模较小（21.6K 帧），场景多样性有限
3. 全景 ERP 畸变对 2D→BEV 变换质量的影响未深入分析
4. 四足低视点限制远距离感知，未讨论感知范围瓶颈

## 相关工作与启发
- 与 SemanticKITTI、SurroundOcc 等车载占用数据集互补，为具身智能社区提供测试平台
- VJC 垂直抖动补偿思路可推广到无人机、水下机器人等非稳定平台
- MIPF 的 prompt 融合范式可用于其他需要非对称模态融合的任务（如 RGB-Thermal 分割）
- 展示了四足机器人感知的独特挑战，为该方向后续研究奠定基础

## 评分
- 新颖性: ⭐⭐⭐⭐ (首个四足全景多模态占用数据集+框架)
- 实验充分度: ⭐⭐⭐ (数据集规模有限，消融可更详细)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，数据集描述详尽)
- 价值: ⭐⭐⭐⭐ (填补具身智能占用预测空白)
