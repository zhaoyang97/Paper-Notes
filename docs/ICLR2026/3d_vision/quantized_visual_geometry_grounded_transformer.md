# Quantized Visual Geometry Grounded Transformer

**会议**: ICLR 2026  
**arXiv**: [2509.21302](https://arxiv.org/abs/2509.21302)  
**代码**: https://github.com/wlfeng0509/QuantVGGT  
**领域**: 3D 视觉 / 模型压缩  
**关键词**: VGGT, post-training quantization, 3D reconstruction, Hadamard rotation, calibration

## 一句话总结
针对十亿级 3D 重建模型 VGGT 的部署需求，提出首个专用 PTQ 框架 QuantVGGT，通过双重平滑细粒度量化（Hadamard 旋转 + 通道平滑）解决特殊 token 导致的重尾分布，以及噪声过滤多样化采样解决校准不稳定问题，4-bit 量化实现 3.7× 内存压缩和 2.5× 加速，保持 98%+ 精度。

## 研究背景与动机

1. **领域现状**：VGGT 是 1.2B 参数的统一 3D 重建模型，单次前向即完成深度估计、点图回归、相机位姿预测和点跟踪。性能卓越但计算/内存开销巨大，限制了实际部署。
2. **现有痛点**：PTQ 在 LLM 和 2D 视觉模型上成熟，但对 VGGT 存在两个独特挑战：(1) 数据无关的特殊 token（camera/register token）导致极端重尾激活分布；(2) 3D 多视图数据的语义复杂性使校准样本选择高度不稳定。
3. **核心矛盾**：特殊 token 是 VGGT 多任务推理的关键设计，但其与常规图像 token 的分布差异导致量化时大量 bit 被浪费在极端值上。
4. **本文要解决什么？** 设计 VGGT 专用的 PTQ 方案，在低 bit 量化下保持重建精度。
5. **切入角度**：从分布分析入手，发现特殊 token 是重尾根源，多视图帧间关系是校准的关键结构。
6. **核心 idea 一句话**：全局 Hadamard 旋转分散特殊 token 的尖峰 + 局部通道平滑降低旋转后残余方差，配合帧感知多样化采样构建稳健校准集。

## 方法详解

### 整体框架
QuantVGGT 包含两个核心组件：(1) DSFQ（Dual-Smoothed Fine-Grained Quantization）——先全局 Hadamard 旋转平滑重尾分布，再局部通道缩放降低幅间方差，配合细粒度量化粒度；(2) NFDS（Noise-Filtered Diverse Sampling）——用深层激活统计过滤异常样本，用帧感知相关性聚类构建多样化校准集。

### 关键设计

1. **Pre-Global Rotation（全局 Hadamard 旋转）**:
   - 做什么：分散特殊 token 导致的激活尖峰
   - 核心思路：对激活 $\mathbf{X}$ 和权重 $\mathbf{W}$ 同时左乘随机 Hadamard 矩阵 $\mathbf{H}$，利用中心极限效应将重尾分布近似为高斯分布。$\mathbf{XW}^\top = (\mathbf{XH})(\mathbf{WH})^\top$
   - 设计动机：Hadamard 变换将少数 channel 的极端值均匀分散到所有 channel

2. **Post-Local Smooth（局部通道平滑）**:
   - 做什么：降低旋转后残余的通道间方差
   - 核心思路：在旋转后的空间中计算缩放因子 $\hat{c}_i = \frac{\max(|\mathbf{X}_i\mathbf{H}|)^\alpha}{\max(|\mathbf{W}_i\mathbf{H}|)^{1-\alpha}}$，$\alpha=0.5$
   - 设计动机：旋转只分散全局尖峰，不消除局部通道差异。先旋转再缩放比先缩放再旋转更稳定（后者会破坏缩放的收益）

3. **Fine-Grained Quantization Granularity**:
   - 做什么：降低量化粒度以减少误差
   - 核心思路：权重按 $d_{out}$ 维度量化，激活按 token 维度量化（利用矩阵乘法的内积求和只在 $d_{in}$ 上进行）
   - 设计动机：μ-coherent 理论表明更细粒度的量化能显著降低量化难度

4. **Noise-Filtered Diverse Sampling（NFDS）**:
   - 做什么：构建稳健的校准数据集
   - 核心思路：两步流程——(a) 从深层激活统计计算每个样本的 noise score（均值和方差的标准分 z-score 的 L2 范数），过滤高分异常样本；(b) 利用 VGGT 的帧间相关性（第一帧 vs 后续帧的归一化相似度向量 $c_t^i$）做 K-means 聚类，均匀采样构建校准集
   - 设计动机：Theorem 3.2 证明校准集应该在数据空间的各子域按尺度比例采样；帧间关系是 VGGT 的归纳偏置核心

## 实验关键数据

### 主实验（Camera Pose Estimation on CO3Dv2）

| 配置 | W/A bit | 精度保持 | 内存压缩 | 加速 |
|------|---------|---------|---------|------|
| Full FP16 | 16/16 | 100% | 1× | 1× |
| W8A8 QuantVGGT | 8/8 | ~99% | 2× | 1.5× |
| W4A4 QuantVGGT | 4/4 | ~98% | 3.7× | 2.5× |
| W4A4 SmoothQuant | 4/4 | ~85% | 3.7× | 2.5× |
| W4A4 QuaRot | 4/4 | ~90% | 3.7× | 2.5× |

### 消融实验

| 组件 | 精度变化 | 说明 |
|------|---------|------|
| 仅 Hadamard 旋转 | +5% vs naive | 分散尖峰 |
| + 通道平滑 | +3% | 降低残余方差 |
| + 细粒度量化 | +2% | 更精细的量化粒度 |
| + NFDS | +2% | 稳健校准 |
| Full QuantVGGT | 98% FP | 所有组件协同 |

### 关键发现
- **特殊 token 是量化的最大障碍**：前 5 个 token（camera+register）的激活幅度比普通 patch token 大 10 倍以上
- **旋转→平滑的顺序很重要**：先平滑再旋转会破坏平滑的收益；先旋转使分布更均匀后再平滑更稳定
- **帧感知聚类优于标签聚类**：t-SNE 可视化显示 3D 场景的语义标签无法有效区分校准子域，但帧间关系可以
- **4-bit 量化在硬件上可行**：实测 RTX 4090 上 2.5× 推理加速

## 亮点与洞察
- **首个十亿级 3D 模型量化工作**：填补了量化在 3D 重建领域的空白
- **双重平滑的"先全局后局部"设计**：简洁优雅地分两步解决重尾问题，且无额外运行时开销（缩放因子可融入 LayerNorm）
- **NFDS 的帧感知校准**：利用了 VGGT "第一帧 vs 后续帧"的独特归纳偏置，体现了"理解模型才能更好压缩"的理念
- **Theorem 3.2 的理论贡献**：给出了校准集构建的形式化指导原则

## 局限性 / 可改进方向
- 仅针对 VGGT 一个模型，未验证对 DUSt3R/MASt3R 等其他 3D 模型的适用性
- 4-bit 下精度仍有 2% 损失，对高精度需求场景可能不够
- NFDS 的噪声阈值和聚类数需要调参
- 未探索 INT2/INT3 等极低比特量化

## 相关工作与启发
- **vs SmoothQuant**: 仅做全局平滑，没有考虑 VGGT 特殊 token 的影响
- **vs QuaRot**: 仅做 Hadamard 旋转，没有后续的局部通道平滑
- 对其他包含特殊 token 的大模型（如 VLM 中的 [CLS]）的量化也有启发

## 评分
- 新颖性: ⭐⭐⭐⭐ 组件不算全新（Hadamard、SmoothQuant），但针对 VGGT 的组合和分析有新意
- 实验充分度: ⭐⭐⭐⭐ 多基准多 bit-width 评估，消融完整
- 写作质量: ⭐⭐⭐⭐ 动机分析深入，可视化清晰
- 价值: ⭐⭐⭐⭐ 首个 3D 大模型量化工作，实用意义强
