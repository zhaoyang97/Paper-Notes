# SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining

**会议**: AAAI 2026  
**arXiv**: [2511.17993](https://arxiv.org/abs/2511.17993)  
**代码**: [https://github.com/Aster-1024/SD-PSFNet](https://github.com/Aster-1024/SD-PSFNet)  
**领域**: 图像去雨 / 物理感知图像复原  
**关键词**: image deraining, point spread function, physics-aware, multi-stage restoration, dynamic filtering

## 一句话总结

提出基于动态 PSF 机制的级联 CNN 去雨网络 SD-PSFNet，通过多尺度可学习 PSF 字典建模雨滴光学效应，配合自适应门控融合的序列化修复架构，在 Rain100H 达 33.12 dB、RealRain-1k-L 达 42.28 dB 均为 SOTA，对比基线 MPRNet 累计提升 5.04 dB（13.5%）。

## 研究背景与动机

图像去雨是计算机视觉的基础低层任务，对目标检测、自动驾驶、监控系统等下游应用至关重要。现有方法面临三个核心挑战：

1. **物理建模缺失**：多数深度学习方法仅从数据学习映射，忽视雨滴的光学物理特性（方向、密度、折射等），导致无法动态适应不同雨型，去雨不彻底且缺乏可解释性
2. **传统 PSF 的局限**：显式物理建模依赖固定先验假设（如静态 PSF 模板），无法捕获雨滴退化的多尺度分布和光学差异
3. **效率瓶颈**：Transformer/GAN/Diffusion 方法虽然性能强，但参数量庞大（Restormer 26.10M、HINet 88.67M），计算开销阻碍实时部署

核心 insight：将 PSF 从静态预设转为**数据驱动的可学习形式**，用有限维的退化模式字典近似空间变化的退化函数。

## 方法详解

### 整体框架

SD-PSFNet 采用三阶段序列化修复架构（Stage In → τ 个 Stage Mid → ORStage），灵感来自 LSTM 的序列建模思想和 MPRNet 的多阶段设计。与 MPRNet 的空间分割策略不同，SD-PSFNet 每阶段处理完整图像，在 UNet 的多尺度下采样结构中动态预测 PSF 特征。各阶段通过自适应门控融合传递物理先验和特征信息。

### 关键设计

1. **动态 PSF 机制**：将空间变化的退化函数建模为可学习退化模式字典的线性组合：$K(x,y) \approx \sum_{j=1}^{K_c} w_j(x,y) \cdot k_j$，其中 $k_j$ 为可学习 2D 卷积核代表基本退化模式，$w_j$ 由神经网络从图像特征自适应映射得到
2. **Multi-Scale PSF Head**：3×3 / 5×5 / 7×7 三个预测头分别捕获高频细节退化、平衡局部-全局信息、宏观低频模式，通过通道注意力块（CAB）自适应融合后经 1×1 卷积投影为 $K_c=40$ 通道 PSF 表示，每通道做空间归一化确保能量守恒
3. **PSF-Aware Attention**：双路径机制——(a) 通道调制：PSF 编码器提取紧凑特征生成 $\gamma, \beta$ 参数，$x_{mod} = x \odot \gamma + \beta$；(b) 空间注意力：上采样单通道 PSF 到特征分辨率，与调制特征结合生成空间权重，突出不同退化程度区域
4. **门控跨阶段融合**：$F^{(t)} = G_\theta \cdot F_{current} + (1-G_\theta) \cdot F_{prev}$，在三处操作——阶段输入处自适应融合浅层特征、编码器层级特征更新、增强版 CSFF 双门控单元精确控制跨阶段信息流

### 损失函数

$$\mathcal{L}_{total} = \sum_{s=1}^{\tau+1} (\mathcal{L}_{char}(I_s, T) + 0.05 \cdot \mathcal{L}_{edge}(I_s, T) + 0.01 \cdot \mathcal{L}_{freq}(I_s, T))$$

Charbonnier loss 负责像素级监督，边缘感知 loss 保持高频细节，频域 loss 对齐 PSF 退化特性。AdamW 优化器，lr=1e-4，3 epoch 线性 warmup + 余弦退火至 1e-6，FP16 混合精度，梯度裁剪 2.0。Patch 128×128，训练 2000 epochs，单 RTX 4090。

## 实验关键数据

### 主实验：与 SOTA 方法对比

| 方法 | 类型 | 参数量(M) | Rain100L PSNR | Rain100H PSNR | RealRain-1k-L PSNR | RealRain-1k-H PSNR |
|------|------|-----------|---------------|---------------|---------------------|---------------------|
| MPRNet | CNN | 3.64 | 36.40 | 30.41 | 36.29 | 34.74 |
| HINet | CNN | 88.67 | 37.28 | 30.65 | **41.98** | **40.82** |
| M3SNet | CNN | 16.70 | 40.04 | 30.64 | 41.55 | 40.01 |
| Restormer | Transformer | 26.10 | 38.99 | 31.46 | 40.90 | 39.57 |
| DRSFormer | Transformer | 33.66 | 41.32 | 32.07 | 41.52 | 40.21 |
| NeRD-Rain-S | Transformer | 10.53 | **42.00** | **32.86** | 38.64 | 36.69 |
| **SD-PSFNet** | **CNN** | **9.63** | 41.47 | **33.12** | **42.28** | **41.08** |

### 消融实验：组件增量贡献（RealRain-1k-L）

| 模型配置 | PSNR (dB) | SSIM | 增量 |
|----------|-----------|------|------|
| MPRNet (Baseline) | 37.24 | 0.9754 | — |
| + Gate 机制 | 38.65 | 0.9773 | +1.41 |
| + 层级间更新 | 40.41 | 0.9792 | +1.76 |
| + 增强 CSFF | 41.41 | 0.9850 | +1.00 |
| + 1-channel PSF | 41.63 | 0.9855 | +0.22 |
| + 40-channel PSF (Ours) | 42.28 | 0.9872 | +0.65 |
| **累计提升** | | | **+5.04** |

### 关键发现

- **阶段数 τ 的影响**：τ=0 (40.81 dB, 3.64M) → τ=3 (41.54 dB, 9.63M)，性能随阶段数单调提升；CNN 架构高效管理参数增长
- **跨域泛化**：Rain100H 训练 → RealRain-1k-L 测试 26.98 dB，超越 Rain13K 训练的 Restormer (26.59 dB) 和 NeRD-Rain-S (26.67 dB)
- **合成-真实域差距**：合成训练模型在真实测试集上性能严重下降（如 Rain100L 训练 → RealRain-1k-L 仅 27.54 dB），反映数据驱动方法学到数据集特定特征而非普遍去雨原理

## 亮点与洞察

- **CNN + 物理先验可媲美 Transformer**：9.63M 参数的 CNN 在真实数据集上超越 88.67M 的 HINet 和 26.10M 的 Restormer，证明物理建模的价值
- **动态 PSF 字典的优雅设计**：有限维字典 + 数据驱动权重组合，既有物理可解释性（每个基核对应一种退化模式），又有数据适应性
- **序列化修复的信息流控制**：双门控 CSFF 机制在跨阶段特征传递中精确控制浅层细节和深层语义的流动

## 局限性

- 合成到真实域的泛化差距仍然明显，PSF 的线性退化假设可能对非线性退化（雨+雾组合）不足
- 未在视频去雨场景验证时序一致性
- τ=3 时 MACs 达 244.83G，相对 DRSFormer 等轻量 Transformer 未必有效率优势

## 相关工作

| 方法类别 | 代表方法 | 特点 | 局限 |
|----------|---------|------|------|
| 早期 CNN | DerainNet, DDN | 局部特征提取 | 感受野受限 |
| 多尺度 CNN | RESCAN, DID-MDN | 多分辨率特征融合 | 缺乏物理建模 |
| Transformer | Restormer, DRSFormer | 长距离依赖建模 | 参数量大，部署困难 |
| GAN/Diffusion | SSCGAN, DCDGAN | 生成真实感结果 | 模式崩塌，训练不稳定 |
| **SD-PSFNet** | **本文** | **物理 PSF + 序列化修复** | **CNN 中唯一在合成+真实同时达 SOTA** |

## 评分

- 新颖性: ⭐⭐⭐⭐ 动态 PSF 机制将物理建模与数据驱动有机结合
- 实验充分度: ⭐⭐⭐⭐ 丰富消融、跨域评估、特征可视化
- 写作质量: ⭐⭐⭐⭐ 物理模型与网络设计对应关系清晰
- 价值: ⭐⭐⭐⭐ 证明 CNN + 物理先验可媲美 Transformer
