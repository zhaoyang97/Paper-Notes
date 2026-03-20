# Adaptive Morph-Patch Transformer for Aortic Vessel Segmentation

**会议**: AAAI 2026
**arXiv**: [2511.06897](https://arxiv.org/abs/2511.06897)
**代码**: [https://github.com/iCherishxixixi/MPTransformer](https://github.com/iCherishxixixi/MPTransformer)
**领域**: 医学图像分割 / Transformer
**关键词**: 主动脉分割, 形态感知 patch, 语义聚类注意力, 微分同胚变形, 速度场

## 一句话总结

提出 Morph-Patch Transformer (MPT)，通过基于速度场的自适应 patch 划分策略生成形态感知 patch（保持血管拓扑完整性），并引入语义聚类注意力（SCA）动态聚合语义相似 patch 的特征，在 AVT、AortaSeg24 和 TBAD 三个主动脉分割数据集上均达 SOTA。

## 研究背景与动机

主动脉血管分割对心血管疾病的诊断和治疗至关重要，直接影响计算流体建模、手术规划和疾病进展监测的可靠性。Transformer 在此领域已成为主导范式，但面临两个根本挑战：

1. **固定矩形 patch 破坏血管完整性**：传统 Transformer 将图像切分为固定大小的矩形 patch，而血管结构细长弯曲、形态复杂，矩形框难以包裹纤细血管，导致语义信息被截断。即使 DPT（Deformable Patch Transformer）引入可学习变形，仍限于矩形 patch，无法适配血管形态
2. **缺乏跨尺度语义相似性建模**：SwinTransformer 的层次化窗口注意力能提取多尺度特征，但固定窗口仍无法建模不同尺度 patch 间的语义相似性。现有方法（含动态 Snake 卷积等）虽增强了细长结构的特征提取，但仍缺少语义聚类机制

核心 insight：用速度场驱动的微分同胚变形（diffeomorphic deformation）生成形态感知 patch，天然保持拓扑连续性；用 Soft K-means 做语义聚类注意力，动态聚合相似 patch。

## 方法详解

### 整体框架

基于 3D UNet 的编解码结构，核心创新在两处：(1) Morph Partition Block 替代固定 patch 划分；(2) Spatial + Semantic Transformer Block 融合空间关系（窗口注意力）和语义关系（聚类注意力）。提供三个版本：MPT（纯 3D ViT）、MPT-UNETR（混合 3D ViT-CNN）、MPTUNet（轻量 2D）。

### 关键设计

1. **Morph Partition Block（形态 patch 划分）**：
   - 用 CNN 预测速度场 $\upsilon$，通过缩放平方法（scaling and squaring）积分得到微分同胚变形场 $\phi^{(1)}$
   - 核心公式：$y(p_0) = \sum_{p_n \in \mathcal{R}} w(p_n) \cdot x(p_0 + p_n + \phi(p_0 + p_n))$
   - 与传统可变形卷积（直接预测偏移场）不同，速度场积分保证变换**光滑、可逆、保拓扑**。变形场的每个点代表坐标偏移，通过双线性插值从原始输入生成变形特征
   - 递推关系：$\phi^{(1/2^{n-1})} = \phi^{(1/2^n)} \circ \phi^{(1/2^n)}$，迭代 n 步后得到 $\phi^{(1)}$

2. **语义聚类注意力（SCA）**：
   - 用可微 Soft K-means 提取核心语义特征 $F_{core}$，根据 patch 特征到核心特征距离的 softmax 权重聚类
   - 更新公式：$f_{newcore}^s = \sum_{i=1}^m g_s(f^i) \cdot (f^i - f_{core}^s)$，$g_s$ 为可微化的隶属度函数
   - $\lambda, \mu$ 和 $F_{core}$ 均由网络学习，$g_s$ 通过 $e^{\lambda^s f^i + \mu^s}$ 参数化确保可微
   - 最终计算 SCA = softmax(QK^T/√d) · V，Q 来自 patch，K/V 来自更新后的语义中心

3. **融合策略**：Morph Partition Block 的形态 patch + SwinTransformer 的窗口注意力（空间关系）+ SCA（语义关系），三者在 Transformer Block 中融合

### 损失函数

使用 Dice 损失。Adam 优化器，lr=5e-5。聚类数设为 32。训练 1000 epochs，基于 nnU-Net 框架，NVIDIA RTX 3090。

## 实验关键数据

### 主实验：AVT 和 TBAD 数据集

| 模型 | 骨干类型 | AVT Dice | AVT mIoU | AVT clDice | TBAD Dice | TBAD mIoU | TBAD clDice |
|------|---------|----------|----------|-----------|-----------|-----------|------------|
| MedNeXt | CNN | 0.809 | 0.718 | 0.724 | 0.926 | 0.871 | 0.880 |
| SegMamba | Mamba | 0.829 | 0.730 | 0.711 | 0.932 | 0.881 | 0.918 |
| nnFormer | 3D ViT | 0.835 | 0.743 | 0.732 | 0.926 | 0.871 | 0.895 |
| DPT | 2D ViT-CNN | 0.886 | 0.800 | 0.825 | 0.924 | 0.868 | 0.917 |
| MambaVision | Mamba | 0.882 | 0.795 | 0.795 | 0.929 | 0.874 | 0.914 |
| TransFuse | 2D ViT-CNN | 0.880 | 0.794 | 0.796 | 0.927 | 0.872 | 0.895 |
| **MPT** | **3D ViT** | **0.856** | **0.762** | 0.757 | **0.933** | **0.881** | 0.915 |
| **MPTUNet** | **2D ViT-CNN** | **0.896** | **0.815** | **0.839** | **0.930** | **0.877** | **0.920** |

### AortaSeg24 数据集（23 类精细分割）

| 模型 | Dice | mIoU | clDice |
|------|------|------|--------|
| 3DUXNet | 0.784 | 0.666 | 0.964 |
| nnFormer | 0.779 | 0.666 | 0.923 |
| SwinUNETR | 0.781 | 0.664 | 0.937 |
| DSCViT | 0.788 | 0.673 | 0.965 |
| DPT | 0.778 | 0.662 | 0.959 |
| MambaVision | 0.795 | 0.682 | 0.960 |
| **MPT** | **0.804** | **0.690** | 0.926 |
| **MPTUNETR** | **0.809** | **0.695** | **0.955** |
| **MPTUNet** | **0.796** | **0.686** | **0.966** |

### 关键发现

- **MPTUNet 在 AVT 上 Dice 0.896**，超越所有方法（含 DPT 0.886、MambaVision 0.882、TransFuse 0.880），且方差最小（0.046）说明稳定性强
- **AortaSeg24 的 23 类精细分割**极具挑战性，MPTUNETR 达 Dice 0.809 / clDice 0.955，领先第二名 MambaVision (0.795/0.960)
- **clDice 指标**专门衡量血管拓扑完整性：MPTUNet 在 TBAD 上 clDice 0.920 为最高，验证了微分同胚变形对拓扑保持的有效性
- **模型效率**：MPT 系列在 FLOPs 和参数量上与 DPT 等相当，性能显著更优

## 亮点与洞察

- **速度场 → 微分同胚变形 → 保拓扑 patch 划分**：不同于直接预测偏移的 DCN，通过 ODE 积分得到的变形天然光滑可逆，特别适合血管这种需要保持连续性的结构
- **Soft K-means 做可微语义聚类**：将 K-means 的硬分配通过指数核 $e^{-\beta \|f-f_{core}\|^2}$ 软化，$\lambda, \mu, F_{core}$ 全部可学习，比固定窗口的注意力更灵活
- **三个版本覆盖不同需求**：MPT（纯 ViT 追求精度）、MPT-UNETR（混合架构平衡效率）、MPTUNet（2D 轻量化）

## 局限性

- 速度场预测的 CNN 本身引入额外计算开销，未详细分析推理速度
- 聚类数固定为 32，未探索自适应聚类数策略
- 仅在主动脉相关数据集验证，未拓展到视网膜血管、冠状动脉等其他血管分割任务
- 变形场的 scaling and squaring 步数（n steps）对变形精度 vs 计算量的权衡未充分讨论

## 相关工作

| 方法类别 | 代表 | patch 策略 | 语义建模 | 拓扑保持 |
|----------|------|-----------|---------|---------|
| 标准 Transformer | UNETR, SwinUNETR | 固定矩形 | 窗口注意力 | 无 |
| 可变形 Transformer | DPT | 可变形矩形 | 标准注意力 | 弱 |
| Snake 卷积混合 | TTCNet, DAU-Net | 固定+Snake | 卷积特征 | 隐式 |
| **MPT（本文）** | **MPT/UNETR/UNet** | **微分同胚变形** | **SCA 聚类注意力** | **强（保拓扑）** |

## 评分

- 新颖性: ⭐⭐⭐⭐ 速度场驱动的形态 patch + 可微语义聚类注意力
- 实验充分度: ⭐⭐⭐⭐ 三个数据集 + 17 个对比方法 + 三个模型版本
- 写作质量: ⭐⭐⭐⭐ 方法推导清晰，问题动机充分
- 价值: ⭐⭐⭐⭐ 对医学图像中复杂形态结构的分割有普适启发
