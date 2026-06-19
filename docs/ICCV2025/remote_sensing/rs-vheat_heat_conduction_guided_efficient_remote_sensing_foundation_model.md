---
title: >-
  [论文解读] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model
description: >-
  [ICCV2025][遥感][遥感基础模型] 首次将物理热传导过程引入遥感基础模型，提出 RS-vHeat，用热传导算子（HCO）替代注意力机制来建模遥感图像中的局部区域相关性，在 4 个任务 10 个数据集上取得优异性能的同时，相比注意力基线减少 84% 显存、24% FLOPs、提升 2.7 倍吞吐量。
tags:
  - "ICCV2025"
  - "遥感"
  - "遥感基础模型"
  - "热传导"
  - "自监督学习"
  - "频域掩码"
  - "多模态"
---

# RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model

**会议**: ICCV2025  
**arXiv**: [2411.17984](https://arxiv.org/abs/2411.17984)  
**代码**: [iecashhy/RS-vHeat](https://github.com/iecashhy/RS-vHeat)  
**领域**: 遥感  
**关键词**: 遥感基础模型, 热传导, 自监督学习, 频域掩码, 多模态

## 一句话总结

首次将物理热传导过程引入遥感基础模型，提出 RS-vHeat，用热传导算子（HCO）替代注意力机制来建模遥感图像中的局部区域相关性，在 4 个任务 10 个数据集上取得优异性能的同时，相比注意力基线减少 84% 显存、24% FLOPs、提升 2.7 倍吞吐量。

## 研究背景与动机

遥感基础模型（RSFM）已从传统的任务特定模型范式中脱颖而出，提供了跨任务的可扩展性。但现有方法面临两大挑战：

**计算效率与感受野的矛盾**：遥感图像中包含大尺度目标，需要模型对足够大的区域做出响应。CNN 受限于固定卷积核，缺乏全局感受野；基于注意力的模型（ViT/Swin）虽能全局建模，但注意力机制带来二次方复杂度，计算开销巨大。对于高分辨率遥感图像（如 1024×1024），现有 RSFM 难以同时实现快速推理和高精度。

**物理可解释性不足**：遥感目标通常呈不规则多边形形状，现有模型虽能提取特征，却无法结合物理原理解释特征如何传播。缺乏可解释性使研究者难以有效调整学习策略。

作者从**热传导**物理过程中获得灵感——热量从高温区向低温区自然扩散的过程，可以类比于神经网络中的特征提取过程。复杂的包含遥感目标的区域对应高温区（热量积聚），稀疏区域对应低温区（热量容易扩散）。这一物理类比为遥感图像处理提供了天然的可解释性框架。

## 方法详解

RS-vHeat 由三个核心组件组成：频域分层掩码策略、热传导视觉编码器和多域重建解码器。

### 1. 频域分层掩码策略（Frequency Domain Hierarchical Masking）

传统的空间域掩码直接遮挡图像块，容易丢失遥感图像中的小目标信息。RS-vHeat 采用频域分层掩码策略：

- 对多模态输入（光学图像 $I_o \in \mathbb{R}^{H \times W \times 3}$、SAR 图像 $I_s \in \mathbb{R}^{H \times W \times 1}$），先通过离散余弦变换（DCT）将图像从空间域转换到频域
- 使用随机生成的扇形掩码 $\tilde{M}$，将频域图像分离为低频分量 $\tilde{I}_{low}$ （携带整体结构信息）和高频分量 $\tilde{I}_{high}$（携带细节信息）
- 掩码率设置为 20%-30%，灵活处理图像
- 通过逆离散余弦变换（IDCT）将高低频信号转回空间域，拼接后恢复原始维度

关键公式为：

$$\tilde{I}_{low}(u,v), \tilde{I}_{high}(u,v) = \tilde{M} \odot \tilde{I}(u,v)$$

这种频域掩码的好处是：即使在掩码后，目标的空间结构也不会完全消失，从而保留了小目标信息。

### 2. 热传导视觉编码器（Heat-Conduction-Based Visual Encoder）

编码器包含 4 个阶段，每个阶段包含 $L$ 个 block（配置为 2, 2, 18, 2，与 Swin-B 一致）。每个 block 包含两个关键模块：

#### (a) HCO Block —— 频域热扩散

热传导方程的通解为：

$$u(x,y,t) = \mathcal{F}^{-1}\left(\tilde{f}(\omega_x, \omega_y) \cdot e^{-k(\omega_x^2 + \omega_y^2)t}\right)$$

将其离散化到视觉特征处理中：

$$U_t^m = \text{IDCT}_{2D}\left(\text{DCT}_{2D}(U_0^m) \cdot e^{-k(\omega_x^2 + \omega_y^2)t}\right)$$

其中 $e^{-k(\omega_x^2 + \omega_y^2)t}$ 充当频域自适应滤波器，执行热传导计算。热扩散系数 $k$ 通过可学习的**频率值嵌入（FVE, Frequency Value Embeddings）**预测，$W_{FVEs} \in \mathbb{R}^{M \times N \times C}$ 与图像维度对齐，能根据遥感场景的特定信息自适应处理不同频域的多模态图像。

HCO 的复杂度为 $O(N^{1.5})$，低于注意力机制的 $O(N^2)$，同时具有全局感受野。

#### (b) 空间校正学习器（Correction Learner）—— 空间域信息调整

通过预测**空间校正嵌入（SCE, Spatial Correction Embeddings）** $W_{SCEs} \in \mathbb{R}^{M \times N \times C}$，与现有温度场交互并激活。SCE 对目标边界进行自适应调整，增强或抑制边缘特征：

$$Z'^{p}_m = CL(Z^p_m, W_{SCEs})$$

SCE 帮助捕捉局部细节和更广泛的上下文区域，辅助模拟热扩散速率。

### 3. 多域重建解码器（Multi-Domain Reconstruction Decoders）

预训练阶段使用三种损失函数的组合：

- **频域重建损失** $\mathcal{L}_{Fre}$：在频率域中计算重建图像与原始图像的 L1 差异
- **空间域重建损失** $\mathcal{L}_{Spa}$：融合编码器第三和第四阶段的输出，上采样到原始尺寸后在空间域计算 L1 损失
- **对比损失** $\mathcal{L}_{Con}$：利用余弦相似度约束不同模态的高低频特征在热空间中的语义一致性

总损失：$\mathcal{L}_{total} = \mathcal{L}_{Con} + \mathcal{L}_{Spa} + \mathcal{L}_{Fre}$

### 4. 预训练与微调

- **预训练数据**：300 万多模态遥感图像（含 45 万对配对的光学-SAR 图像），来自全球六大洲，分辨率 0.3m-30m，统一裁剪至 448×448
- **预训练设置**：8 块 A100 (80G) GPU，200 epochs，224 尺寸图像，基础学习率 2e-4，余弦退火
- **下游微调**：直接迁移嵌入层、热传导视觉编码器的结构和权重，对于不同尺寸的下游任务图像，通过插值调整 FVE 和 SCE

## 实验关键数据

### 语义分割（4 个数据集）

| 数据集 | 指标 | RS-vHeat | 对比最优 RSFM |
|--------|------|----------|---------------|
| Potsdam | mF1 | **92.82** | SkySense 93.99 (>702M params) |
| iSAID | mIoU | **68.72** | SkySense 70.91 (>702M params) |
| AIR-PolSAR-Seg | mIoU | **57.46** | DANet 51.93 |
| WHU-OPT-SAR (多模态) | OA | **83.9** | MCANet 82.9 |

RS-vHeat 仅用 148M 参数、921G FLOPs 即接近 SkySense（>702M params、>2708G FLOPs）的精度。

### 目标检测（3 个数据集）

| 数据集 | 指标 | RS-vHeat | 对比 |
|--------|------|----------|------|
| DIOR | mAP50 | **82.30** | SkySense 78.73, Scale-MAE 73.81 |
| FAIR1M-2.0 | mAP | 48.29 | SkySense 54.57 |
| SAR-AIRcraft-1.0 | mAP50 | **87.1** | SA-Net 77.7 |

在 DIOR 上超越 SkySense 3.57%，同时 FLOPs 仅 266G（SkySense >1679G）。

### 图像分类（2 个数据集）

| 数据集 | 训练比例 | RS-vHeat | Scale-MAE (ViT-L) |
|--------|----------|----------|---------------------|
| AID | 20% | **96.81** | 96.44 |
| AID | 50% | 97.58 | 97.58 |
| NWPU-RESISC45 | 10% | 92.01 | 92.63 |
| NWPU-RESISC45 | 20% | **95.66** | 95.04 |

RS-vHeat 仅 150M/340G，Scale-MAE 为 310M/2070G，FLOPs 减少超 6 倍。

### 变化检测

| 数据集 | 指标 | RS-vHeat | SkySense |
|--------|------|----------|----------|
| LEVIR-CD | F1 | **93.48** | 92.58 |

### 计算效率（1024×1024 图像，单 A100）

| 指标 | RS-vHeat vs Swin-B 基线 |
|------|-------------------------|
| 吞吐量 | **2.7 倍** |
| 显存占用 | **减少 84%** |
| FLOPs | **减少 24%** |

### 消融实验

损失函数组合消融（DIOR mAP50）：
- 仅空间域重建：78.2
- +频域重建：79.5（+1.3）
- +对比损失：82.3（+2.8）→ 三者缺一不可

## 亮点与洞察

1. **物理启发的架构设计**：将热传导方程直接映射到视觉特征处理，HCO 在频域中执行热扩散计算，既有全局感受野又有 $O(N^{1.5})$ 的亚二次复杂度，是一种兼顾效率和建模能力的巧妙设计。

2. **频域掩码保护小目标**：相比空间域掩码可能完全遮挡小目标，频域掩码保留了目标的空间结构，对遥感场景中的小目标（如飞机、船只）尤为重要。

3. **多模态统一热空间**：光学和 SAR 图像都被投射到同一热空间中处理，通过对比损失约束两种模态的深层语义一致性，实现了自然的多模态融合。

4. **效率优势随分辨率增大而放大**：在处理大尺度遥感图像时，RS-vHeat 的效率优势更加显著——这正是遥感领域最需要的特性。

5. **可解释性**：热传导过程提供了直觉解释——包含复杂目标的区域积聚热量成为高温区，稀疏区域热量扩散成为低温区，这与遥感目标检测的语义逻辑高度一致。

## 局限与展望

1. **FAIR1M 上表现相对弱**：在 FAIR1M-2.0 细粒度目标检测上（mAP 48.29 vs SkySense 54.57），RS-vHeat 未能超越最强基线，说明在多类别、多尺度、多朝向的复杂场景中仍有提升空间。

2. **分类任务未能全面超越**：在 NWPU-RESISC45（10% 标注）和 AID（50% 标注）上，RS-vHeat 与 Swin-based 基线（如 RingMo）仍有差距，说明场景级分类可能不是热传导范式的最佳适用场景。

3. **预训练数据规模较大但未完全公开**：300 万图像中包含高分二号卫星等非公开数据，可能影响可复现性。

4. **固定尺寸 FVE/SCE 的插值瓶颈**：下游任务中不同图像尺寸需要插值调整 FVE 和 SCE，这可能引入信息损失，尤其在分辨率差异较大时。

5. **仅设计了 Base 版本**：缺乏不同规模版本（Tiny/Small/Large）的完整实验，无法评估模型的 scaling 特性。

## 相关工作与启发

- **vHeat**（Wang et al., 2024）：本文的基础架构来源，首次将热传导方程引入视觉模型，RS-vHeat 在此基础上针对遥感场景设计了频域掩码预训练和 SCE 模块
- **RingMo**（Sun et al., TGRS 2022）：遥感基础模型代表作，使用 Swin-B 配合不完整掩码策略
- **SkySense**（Guo et al., CVPR 2024）：当前 SOTA RSFM（2.6B 参数），RS-vHeat 用远少的参数在多个任务上接近甚至超越其精度
- **Scale-MAE**（Reed et al., ICCV 2023）：在空间域掩码基础上重建高低频图像，RS-vHeat 的频域掩码策略是对此思路的进一步发展
- **VMamba**（Liu et al., NeurIPS 2024）：状态空间模型在视觉领域的探索，与 RS-vHeat 同属注意力替代方案，但热传导范式在物理可解释性上更有优势

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[CVPR 2026\] ORSATR-X: A Foundation Model based on Differential-and-Excitation Networks for Optical Remote Sensing Object Recognition](../../CVPR2026/remote_sensing/orsatr-x_a_foundation_model_based_on_differential-and-excitation_networks_for_op.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[CVPR 2026\] CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation](../../CVPR2026/remote_sensing/crossearth-gate_fisher-guided_adaptive_tuning_engine_for_efficient_adaptation_of.md)

</div>

<!-- RELATED:END -->
