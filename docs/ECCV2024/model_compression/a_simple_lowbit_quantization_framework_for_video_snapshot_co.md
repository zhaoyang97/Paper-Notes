# A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging

**会议**: ECCV 2024  
**arXiv**: [2407.21517](https://arxiv.org/abs/2407.21517)  
**代码**: [https://github.com/mcao92/QuantizedSCI](https://github.com/mcao92/QuantizedSCI)  
**领域**: 模型压缩 / 计算成像  
**关键词**: 网络量化, 视频快照压缩成像, Transformer量化, 低比特推理, 高效重建  

## 一句话总结

首个面向视频快照压缩成像（Video SCI）重建任务的低比特量化框架Q-SCI，通过高质量特征提取模块、精确视频重建模块和Transformer分支的query/key分布偏移操作，在4-bit量化下实现7.8倍理论加速且性能仅下降2.3%。

## 背景与动机

视频快照压缩成像（Video SCI）利用低速2D相机通过编码掩模将高速场景压缩为快照测量值，再通过重建算法恢复高速视频帧。当前基于深度学习的SOTA重建方法（如EfficientSCI、STFormer等）虽然性能出色，但参数量和计算量依然很大——EfficientSCI-S有3.78M参数、563.87 GFLOPs，难以部署在手机、自动驾驶等资源受限设备上。网络量化是降低计算开销最直接有效的手段之一，但此前从未有人探索过Video SCI重建任务中的量化问题。直接将现有重建网络量化到低比特会带来巨大的性能崩塌（4-bit直接量化掉4.11 dB），必须针对SCI重建任务的特殊结构设计专门的量化策略。

## 核心问题

1. **低比特量化的性能崩塌问题**：直接将端到端Video SCI重建网络（由特征提取→特征增强→视频重建三个模块组成）量化到4-bit会导致严重的质量下降，核心原因是特征提取模块在低比特下丢失了大量高质量特征信息。
2. **Transformer分支的分布失真**：量化后Transformer中query和key的激活分布发生偏移，导致注意力权重计算失真，且分布形态与标准视觉Transformer不同（非钟形），无法直接套用已有方法（如Q-ViT）。

## 方法详解

整体思路是：先通过经验分析定位性能崩塌的根源，再针对性地设计三个轻量改进模块来弥补量化带来的质量损失。

### 整体框架

Q-SCI以端到端Video SCI重建网络（以EfficientSCI-S为backbone）为基础。输入是2D压缩测量值和编码掩模，输出为重建的多帧高速视频。整个网络分为三个阶段：特征提取模块 → 特征增强模块（ResDNet，内含CFormer的Transformer分支）→ 视频重建模块。Q-SCI在这三个阶段分别引入针对性改进，用极少的额外参数换取显著的量化性能恢复。

### 关键设计

1. **高质量特征提取模块（FEM）**：经验分析发现特征提取模块是性能崩塌的主要来源（单独量化到4-bit掉2.22 dB，远超其他模块的0.5 dB左右）。核心原因是低比特量化导致初始特征质量严重退化，后续模块无法弥补。解决方案很直接：在特征提取模块中加入若干1×1×1卷积作为shortcut连接（含pixel shuffle用于空间尺寸对齐），并将这些shortcut卷积设为8-bit精度，从而在低比特主干中保持一条高质量特征传播通道。这个设计贡献了最大的性能提升（+2.35 dB）。

2. **偏移Transformer分支（Shifted Transformer Branch, RDM）**：量化后Transformer的query和key分布发生明显偏移（8-bit模型中query均值漂移了1.207），导致注意力计算失真。与Q-ViT不同，SCI中量化后的分布不是钟形，不能用Q-ViT的方法。Q-SCI引入可学习的偏移偏置$\beta_q$、$\beta_k$，对query和key做$\tilde{q} = q + \beta_q$、$\tilde{k} = k + \beta_k$，让量化后的分布重新对齐全精度模型。这个操作几乎不增加计算量，提升0.53 dB。

3. **精确视频重建模块（VRM）**：与FEM设计思路一致，在视频重建模块也加入8-bit精度的1×1×1 shortcut卷积，保证高质量特征能一直传播到网络末端输出。提升0.43 dB。

### 损失函数 / 训练策略

- **损失函数**：标准MSE损失，$\mathcal{L}_{MSE} = \frac{1}{T \cdot n_x \cdot n_y} \sum_{t=1}^{T} \|\hat{X}_t - X_t\|_2^2$
- **量化方案**：激活用非对称量化，权重用对称量化，训练时scale和zero-point作为可学习参数优化（QAT方式）
- **训练流程**：用全精度EfficientSCI-S初始化，Adam优化器，先128×128 crop训练100 epoch（lr=1e-4），再256×256训练20 epoch，最后lr降到1e-5再训练20 epoch
- **变体**：8/4/3/2-bit四个量化等级；8-bit只加shifted Transformer branch；4/3/2-bit同时使用全部三个改进模块

## 实验关键数据

| 方法 | PSNR (avg) | SSIM (avg) | Params (M) | OPs (G) |
|------|-----------|-----------|-----------|---------|
| EfficientSCI-S (全精度) | 35.51 | 0.970 | 3.78 | 563.87 |
| Q-ViT (8-bit) | 35.17 | 0.967 | 0.95 | 141.04 |
| **Q-SCI (8-bit)** | **35.57** | **0.969** | **0.95** | **140.95** |
| **Q-SCI (4-bit)** | **34.69** | **0.963** | **0.48** | **72.69** |
| Q-SCI (3-bit) | 33.62 | 0.953 | 0.37 | 37.47 |
| Q-SCI (2-bit) | 31.62 | 0.928 | 0.25 | 19.85 |
| BIRNAT | 33.31 | 0.951 | 4.13 | 390.56 |
| RevSCI | 33.92 | 0.956 | 5.66 | 766.95 |
| Dense3D-Unfolding | 35.26 | 0.968 | 61.91 | 3975.83 |

- Q-SCI (8-bit) 超过全精度EfficientSCI-S 0.06 dB，OPs仅为1/4
- Q-SCI (4-bit) 与全精度仅差0.82 dB（2.3%性能差距），理论加速7.8×
- Q-SCI (4-bit) 比BIRNAT高1.38 dB，OPs仅为其1/5.4

### 消融实验要点

| 配置 | PSNR | SSIM | 增量 |
|------|------|------|------|
| 4-bit Baseline（直接量化） | 31.40 | 0.931 | — |
| + Shifted Transformer (RDM) | 31.93 | 0.929 | +0.53 |
| + 高质量特征提取 (FEM) | 34.28 | 0.959 | +2.35 |
| + 精确视频重建 (VRM) | 34.71 | 0.963 | +0.43 |

- **FEM贡献最大**（+2.35 dB），印证了高质量初始特征对量化模型的关键作用
- 三个模块总计恢复3.31 dB，额外计算开销仅增加3.14%
- 泛化验证：在STFormer-S上，FEM带来+3.23 dB，VRM进一步+0.25 dB

## 亮点

- **首创性**：Video SCI领域第一个网络量化工作，开辟了SCI重建效率优化的新方向
- **分析驱动设计**：先做系统的性能分析（逐模块量化实验 + 特征可视化），精准定位瓶颈再设计方案，方法论值得学习
- **极简有效的shortcut策略**：仅用少量8-bit的1×1×1卷积作为shortcut就能大幅恢复量化损失，这一trick对其他low-level视觉任务的量化也有启示
- **Transformer分布偏移的修正**：可学习shift bias的方法非常轻量，适用于任何含Transformer分支的量化网络
- **良好的泛化性**：Q-SCI框架可迁移到不同的端到端SCI重建方法（EfficientSCI、STFormer），不是绑定单一架构

## 局限性 / 可改进方向

- **未部署验证**：论文仅报告理论加速比（OPs），没有实际芯片/GPU上的延迟测试，实际加速效果存疑
- **量化策略较传统**：采用的是标准的均匀量化+QAT，没有探索混合精度、分层自适应比特分配等更先进的量化技术
- **shortcut用8-bit可能不是最优**：固定将shortcut设为8-bit是手工设计，没有自动搜索最优的混合精度配置
- **仅验证两种backbone**：泛化实验只在EfficientSCI和STFormer上做了，未测试更多架构（如deep unfolding方法）
- **损失函数简单**：只用MSE损失，感知损失或对抗损失可能进一步提升视觉质量
- 🔗 可延伸方向：将级联量化+熵编码的思路（参考 [Cascaded Quant-Entropy Compression](../../../ideas/model_compression/20260317_cascaded_quant_entropy.md)）应用于SCI重建模型，进一步压缩存储

## 与相关工作的对比

- **vs Q-ViT**：Q-ViT是通用ViT量化框架，在SCI任务上直接应用效果不如Q-SCI（8-bit下差0.4 dB）。原因是Q-ViT假设量化后分布为钟形，但SCI网络的Transformer分支不满足这一假设。Q-SCI通过learnable shift替代Q-ViT的固定re-parameterization，更灵活。
- **vs PAMS/CADyQ/BBCU等low-level量化方法**：这些方法针对超分辨率、去噪等任务设计，没有考虑SCI特有的feature extraction→enhancement→reconstruction三阶段结构和3D卷积特性。Q-SCI的shortcut设计是针对SCI pipeline特点定制的。
- **vs EfficientSCI**：EfficientSCI追求轻量化架构设计，Q-SCI从量化角度切入，两者正交互补。Q-SCI (8-bit)性能与EfficientSCI-S持平，OPs仅为1/4。

## 启发与关联

- **高质量特征通道思想的迁移**：在量化低级视觉网络时，保留少量高精度的shortcut通道来传输高质量特征，这一策略可推广到图像超分辨率、去噪、去模糊等任务的量化。
- **与级联量化-熵编码idea的关联**：Q-SCI表明8-bit量化几乎无损（甚至微微超越全精度），如果在8-bit量化基础上再做熵编码压缩存储（如 [Cascaded Quant-Entropy](../../../ideas/model_compression/20260317_cascaded_quant_entropy.md) 所提设想），可能在SCI场景下同时获得高精度和极致压缩。
- **混合精度自动搜索**的可能性：Q-SCI手动选择shortcut为8-bit，如果引入NAS或可微搜索来自动确定各层最优比特宽度，可能发现更优的精度-效率配置。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个SCI量化工作，但量化技术本身并非全新
- 实验充分度: ⭐⭐⭐⭐ 仿真+实拍数据、两种backbone泛化、完整消融，缺实际部署延迟
- 写作质量: ⭐⭐⭐⭐ 分析驱动、逻辑清晰，图表质量高
- 价值: ⭐⭐⭐⭐ 为SCI高效部署开了先河，shortcut量化策略有实用参考价值
