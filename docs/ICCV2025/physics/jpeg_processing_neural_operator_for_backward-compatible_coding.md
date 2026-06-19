---
title: >-
  [论文解读] JPEG Processing Neural Operator for Backward-Compatible Coding
description: >-
  [ICCV 2025][物理/科学计算][JPEG] 提出JPNeO，一个完全后向兼容JPEG格式的下一代编解码器，通过在编码和解码阶段分别引入神经算子(JENO和JDNO)以及可训练量化矩阵，显著提升JPEG重建质量（尤其是色度分量），同时保持低内存和少参数量的优势。 领域现状 尽管基于DNN的非线性变换编码在有损压缩中表…
tags:
  - "ICCV 2025"
  - "物理/科学计算"
  - "JPEG"
  - "神经算子"
  - "后向兼容"
  - "图像压缩"
  - "色度保持"
---

# JPEG Processing Neural Operator for Backward-Compatible Coding

**会议**: ICCV 2025  
**arXiv**: [2507.23521](https://arxiv.org/abs/2507.23521)  
**代码**: [github.com/WooKyoungHan/JPNeO](https://github.com/WooKyoungHan/JPNeO)  
**领域**: 科学计算  
**关键词**: JPEG, 神经算子, 后向兼容, 图像压缩, 色度保持

## 一句话总结

提出JPNeO，一个完全后向兼容JPEG格式的下一代编解码器，通过在编码和解码阶段分别引入神经算子(JENO和JDNO)以及可训练量化矩阵，显著提升JPEG重建质量（尤其是色度分量），同时保持低内存和少参数量的优势。

## 研究背景与动机

### 领域现状

尽管基于DNN的非线性变换编码在有损压缩中表现出色，但JPEG标准已广泛嵌入图像信号处理器(ISP)中，是不可避免的图像处理管线组成部分。DNN压缩方法的标准化仍需相当长时间，且大量已有图像已用传统编解码器压缩。

### 现有痛点

**编码端的信息损失不可逆**：JPEG的两大损失来源——量化和色度下采样——在编码端就丢失了信息，传统解码器受限于编码端约束的互信息

**色度分量恢复差**：现有JPEG伪影去除方法主要关注亮度分量，色度分量（CbCr）因下采样而严重退化

**编码和解码独立优化**：之前的工作要么优化编码要么优化解码，没有共享信息实现协同增效

**兼容性问题**：DNN编解码器无法与现有JPEG基础设施互操作

### 核心矛盾

如何在不改变JPEG源编码协议（文件格式）的前提下，通过神经网络增强编码和解码质量？

### 切入角度

从互信息理论出发，分析JPEG编解码过程中的信息损失，分别在编码端嵌入图像先验增加互信息$I(\mathbf{X}';\varphi)$，在解码端通过训练参数获取额外互信息$I(\tilde{\mathbf{X}};\hat{\theta})$，同时保持完全后向兼容。

## 方法详解

### 整体框架

JPNeO由三个组件构成：
- **JENO (编码神经算子)**：处理色度下采样问题的辅助编码器
- **可训练量化矩阵** $\mathbf{Q}_\psi$：替代标准量化矩阵的预训练查找表
- **JDNO (解码神经算子)**：替代传统JPEG解码器，直接从DCT频谱解码高质量图像

关键设计：JENO、$\mathbf{Q}_\psi$和JDNO可灵活替换传统JPEG编解码器的对应组件。

### 关键设计

#### 1. **可训练量化矩阵 ($\mathbf{Q}_\psi$)**

- **功能**：用一个线性层学习优化的量化矩阵，替代JPEG标准的默认矩阵
- **核心思路**：以标准量化矩阵(quality factor=50)为输入，通过线性层映射得到优化矩阵。用3阶近似处理不可微的取整操作：$\lfloor x \rceil \simeq \lfloor x \rceil + (\lfloor x \rceil - x)^3$
- **损失函数**：$\mathcal{L} = \lambda \cdot \|\mathbf{X} - \tilde{\mathbf{X}}\|_2 + \|1/\mathbf{Q}_\psi\|_1$，通过超参$\lambda$控制失真与比特率的权衡
- **设计动机**：训练后仅存储结果矩阵作为查找表，不增加运行时计算。不同$\lambda$对应17种不同的$\mathbf{Q}_\psi$

#### 2. **JENO (编码神经算子)**

- **功能**：解决色度下采样引起的信息损失，学习原始图像的高频分量
- **核心思路**：
    - 用EDSR-baseline提取RGB图像特征$\mathbf{z} \in \mathbb{R}^{H \times W \times K}$
    - 在下采样坐标处采样并用Galerkin注意力机制处理
    - 加上传统下采样结果形成残差：$\hat{\mathbf{X}} = \mathcal{G}_\phi(\mathcal{S}(f_\xi(\mathbf{X}), \delta)) + \mathbf{X}'$
- **关键性质**：JENO学习的实质是高通滤波器——$U(E_\varphi(\mathbf{X})) \simeq HPF(\mathbf{X})$，补偿下采样丢失的高频信息
- **训练目标**：$\hat{\varphi} = \arg\min_\varphi \|\mathbf{X} - U(\hat{\mathbf{X}}_\varphi)\|_1$

#### 3. **JDNO (解码神经算子)**

- **功能**：直接从DCT频谱解码高质量图像，替代传统JPEG解码器
- **核心思路**：
    - **Group Embedding**：将亮度和色度频谱嵌入为统一表示，支持4:2:0/4:2:2/4:4:4
    - **特征提取**：使用SwinV2注意力模块提取特征
    - **余弦神经算子(CNO)**：利用连续余弦函数公式化解码：$\mathbf{T}_\rho(\mathbf{z}', \delta; \mathbf{Q}) = \mathbf{A} \otimes (\cos(\pi\mathbf{F}_h \otimes \delta_h) \odot \cos(\pi\mathbf{F}_w \otimes \delta_w))$，其中$\mathbf{A} = h_q(\mathbf{Q}) \odot h_a(\mathbf{z}')$融合了量化矩阵先验
    - 最终通过Galerkin注意力完成解码
- **设计动机**：JDNO感知量化矩阵，可以根据压缩程度自适应解码

### 损失函数 / 训练策略

- JENO和JDNO均使用L1损失
- 训练数据：DIV2K + Flickr2K（3450张图像），裁剪为112×112
- JENO随机选择4:2:0和4:2:2色度下采样模式训练
- JDNO同时使用标准和预训练量化矩阵训练，增强鲁棒性
- 4×RTX 3090训练1000 epochs

## 实验关键数据

### 主实验（与JPEG伪影去除方法对比，LIVE-1数据集）

| 方法 | 参数量 | q=0 PSNR/PSNR-B | q=10 PSNR/PSNR-B | q=40 PSNR/PSNR-B |
|------|--------|-----------------|-------------------|-------------------|
| JPEG | — | 20.89/19.73 | 25.69/24.20 | 30.28/28.84 |
| QGAC | 259.4M | 16.33/15.99 | 27.65/27.43 | 32.08/31.64 |
| FBCNN | 70.1M | 21.70/21.19 | 27.77/27.51 | 32.34/31.80 |
| JDEC | 38.9M | 20.76/20.07 | 27.95/27.71 | 32.50/31.98 |
| **JPNeO** | **29.7M** | **23.15/22.64** | **28.15/27.55** | **32.83/31.91** |

**色度分量对比**（LIVE-1，$\mathbf{X}_C$-PSNR）：

| 方法 | q=0 | q=10 | q=40 |
|------|-----|------|------|
| DnCNN | 29.27 | 34.47 | 38.98 |
| FBCNN | 29.86 | 37.35 | 41.23 |
| JDEC | 28.75 | 37.95 | 41.92 |
| **JPNeO** | **32.30** | **38.56** | **43.47** |

### 消融实验

| 配置 | bpp↓ | PSNR↑ | SSIM↑ | 说明 |
|------|------|-------|-------|------|
| JPEG+Q+JPEG | 0.262 | 19.91 | 0.559 | 基线 |
| JPEG+$Q_\psi$+JPEG | 0.260 | 21.21 | 0.581 | 量化矩阵优化有效 |
| JPEG+Q+JDNO | 0.262 | 22.27 | 0.643 | 解码器优化大幅提升 |
| JPEG+$Q_\psi$+JDNO | 0.260 | 23.10 | 0.661 | 两者叠加效果 |
| JPNeO (完整) | 0.260 | **23.36** | **0.680** | JENO进一步提升 |

**计算效率对比**：

| 方法 | 参数(M) | 内存(GB) | 时间(ms) | q=0 PSNR |
|------|---------|---------|---------|----------|
| FBCNN | 70.1 | 0.61 | 71.95 | 21.70 |
| JDEC | 38.9 | 1.76 | 224.79 | 20.76 |
| JPNeO- (轻量) | 8.0 | **0.09** | 222.95 | 22.98 |
| JPNeO | 29.7 | 0.26 | 562.42 | **23.15** |

### 关键发现

- **低比特率优势显著**：q=0时JPNeO超越FBCNN 1.45dB，超越JDEC 2.39dB
- **色度恢复是核心贡献**：JPNeO在$\mathbf{X}_C$-PSNR上领先3-5dB，远超其他方法
- **JENO在高bpp有效，JDNO在低bpp有效**：编码器提升上界，解码器提升下界，量化矩阵决定两者间路径
- **互信息验证**：实验证实JENO在高质量时增加更多互信息，JDNO在低质量时增加更多互信息
- **轻量版JPNeO-仅8M参数**即超越70M参数的FBCNN和39M参数的JDEC
- **t-SNE可视化**：JENO编码的图像在潜空间中更接近GT分布

## 亮点与洞察

1. **完全后向兼容**：JPNeO的编解码器可以灵活替换——JENO编码的文件可用标准JPEG解码器打开，标准JPEG文件也可用JDNO解码
2. **信息论视角**：从互信息角度分析JPEG编解码过程中的信息损失和恢复，理论框架优雅
3. **色度分量的重点关注**：首次系统性解决JPEG色度下采样的信息损失问题
4. **ISP友好**：低内存(0.26GB)和少参数(29.7M)使JPNeO适合嵌入边缘设备的ISP管线
5. **量化矩阵作为查找表**：训练后直接使用整数矩阵，零推理开销

## 局限与展望

1. **推理速度较慢**：562ms远慢于FBCNN(72ms)，主要来自Galerkin注意力计算
2. **仅L1损失**：未使用感知损失或GAN损失，感知质量可能有提升空间
3. **训练需要预设$\lambda$**：17种量化矩阵需分别训练和存储
4. **与端到端学习压缩的差距**：虽然在JPEG框架内最优，但与VVC等现代编码标准相比仍有结构性差距
5. **编码端需要额外计算**：JENO增加了编码时间，对实时拍照场景不友好

## 相关工作与启发

- **JDEC** 首次提出直接从DCT频谱解码而绕过传统JPEG解码器，但不支持4:2:2和4:4:4
- **FBCNN** 提出盲质量因子伪影去除，是当前最实用的JPEG增强方法
- **Strümpler et al.** 提出可训练量化矩阵+预编辑的思路，本文扩展并与神经算子结合
- 神经算子从PDE求解器到图像编解码的迁移是一个有趣的交叉方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 编解码双端神经算子+量化矩阵优化的完整方案，信息论分析有洞察力
- **实验充分度**: ⭐⭐⭐⭐ — 模块消融、RD曲线、色度分析、计算效率、互信息验证全面
- **写作质量**: ⭐⭐⭐⭐ — 信息论推导清晰，但符号较多读起来密集
- **价值**: ⭐⭐⭐⭐ — 对JPEG生态系统有实际价值，即插即用的后向兼容增强方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DiffFNO: Diffusion Fourier Neural Operator](../../CVPR2025/physics/difffno_diffusion_fourier_neural_operator.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](../../NeurIPS2025/physics/score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](../../NeurIPS2025/physics/integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/physics/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](../../NeurIPS2025/physics/from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)

</div>

<!-- RELATED:END -->
