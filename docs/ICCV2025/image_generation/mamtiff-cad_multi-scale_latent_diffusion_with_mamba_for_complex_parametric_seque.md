---
title: >-
  [论文解读] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence
description: >-
  [图像生成] 提出MamTiff-CAD框架，结合Mamba+编码器与Transformer解码器的自编码器学习CAD命令序列的潜表示，再用多尺度Transformer扩散模型生成，首次实现60-256命令长度的复杂CAD模型生成。 参数化CAD通过命令序列（草图、拉伸、布尔运算等）构建3D模型，是工业设计的核心…
tags:
  - "图像生成"
---

# MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence

## 元信息
- **会议**: ICCV 2025
- **arXiv**: [2511.17647](https://arxiv.org/abs/2511.17647)
- **代码**: 未公开
- **领域**: 扩散模型 · CAD生成
- **关键词**: 参数化CAD, Mamba, 长序列建模, 多尺度Transformer, 扩散模型

## 一句话总结
提出MamTiff-CAD框架，结合Mamba+编码器与Transformer解码器的自编码器学习CAD命令序列的潜表示，再用多尺度Transformer扩散模型生成，首次实现60-256命令长度的复杂CAD模型生成。

## 研究背景与动机

参数化CAD通过命令序列（草图、拉伸、布尔运算等）构建3D模型，是工业设计的核心。现有深度学习方法（如DeepCAD）受Transformer二次复杂度限制，仅能处理短序列（<60命令），无法生成工业级复杂CAD模型。

**核心挑战**：
1. 长序列建模瓶颈：Transformer的$O(n^2)$复杂度限制序列扩展
2. 局部-全局约束平衡：CAD模型同时包含局部几何细节和全局拓扑约束
3. 数据集缺失：现有数据集（DeepCAD平均15命令）不含复杂长序列CAD

## 方法详解

### 整体框架（两阶段）

**阶段一**：Mamba+编码器 + Transformer解码器的自编码器，将CAD序列编码为潜表示$Z$

**阶段二**：多尺度Transformer扩散模型在潜空间学习分布，生成新CAD

### CAD参数化表示

每个命令$m_i = (C_i, p_i)$，其中$C_i$为6种命令类型之一，$p_i \in \mathbb{R}^{16}$包含坐标、角度、拉伸参数等。连续参数归一化到$2 \times 2 \times 2$立方体并量化为256级离散token，序列固定长度256（不足则用EOS填充）。

### Mamba+编码器（带遗忘门）

核心创新——**双分支+遗忘门**：

$$G_f = 1 - G_{b2}$$

$$x'' = G_f \cdot x'$$

$$h_{\text{out}} = x'' + h_{\text{SSM}}$$

- 分支b1：1D卷积 + SSM块提取序列特征
- 分支b2：SiLU激活生成控制信号
- 遗忘门$G_f$调节历史信息保留，防止长距离依赖中关键信息丢失

4层Mamba+块堆叠，实现从60到256长度的CAD序列高效编码。

### Transformer解码器（非自回归）

4层Transformer块，输入为潜向量$Z$和可学习位置嵌入。非自回归解码并行生成所有256个命令位置的预测：

$$p(\hat{M} | z, \Theta) = \prod_{i=1}^{N_c} p(\hat{C}_i, \hat{p}_i | z, \Theta)$$

### 自编码器训练损失

$$L = \sum_{i=1}^{N_c} \ell(p_i(t_i)) + \beta \sum_{i=1}^{N_c} \sum_{j=1}^{N_p} \ell(q_{i,j}(a_{i,j}))$$

$\beta=2$平衡参数损失和命令类型损失。跳过填充命令和未使用参数。

### 多尺度Transformer扩散生成器（MST-D）

三个并行注意力分支，分别捕获不同尺度的依赖：
- 窗口64：局部几何约束
- 窗口128：中程拓扑依赖
- 窗口256：全局语义一致性

自适应融合：
$$\mathbf{H} = \text{MLP}(\sigma(\mathbf{W}_g [\mathbf{H}_l \| \mathbf{H}_m \| \mathbf{H}_g]) \odot [\mathbf{H}_l \| \mathbf{H}_m \| \mathbf{H}_g])$$

标准DDPM噪声预测损失：
$$L_{\text{diff}} = \mathbb{E}_{t,Z_0,\epsilon}[\|\epsilon - \epsilon_\theta(Z_t, t)\|_2^2]$$

## 实验

### 自编码重建（ABC-256数据集）

| 方法 | 命令精度↑ | 参数精度↑ | MCD↓ | 无效率↓ | STEP率↑ |
|------|---------|---------|------|--------|--------|
| DeepCAD | 92.24% | 75.93% | 41.02 | 33.11% | 70.46% |
| MT-CAD | 89.72% | 66.87% | 121.35 | 39.89% | 63.97% |
| **MamTiff-CAD** | **99.99%** | **99.93%** | **0.75** | **8.50%** | **93.93%** |

命令精度和参数精度均接近100%，MCD从41.02降至0.75（降低98%），无效率从33.11%降至8.50%。

### 无条件生成

| 方法 | MMD↓ | JSD↓ | COV↑ | Unique↑ | Novel↑ | STEP率↑ |
|------|------|------|------|---------|--------|--------|
| DeepCAD | 2.66 | 6.49 | 56.66% | 75.8 | 88.0 | 23.96% |
| SkexGen | 2.31 | 4.53 | 57.76% | 80.5 | 96.9 | 75.26% |
| **MamTiff-CAD** | **1.43** | **3.19** | **64.16%** | **90.8** | **95.6** | **85.38%** |

JSD（分布差异）3.19为最优，STEP转换成功率85.38%远超DeepCAD的23.96%。

### ABC-256数据集贡献

13,705个CAD模型，平均序列长度99（DeepCAD的6.6倍），序列范围60-256。训练集10,964、验证集1,370、测试集1,371。

## 亮点与洞察

1. **长序列突破**：首次实现256命令的工业级CAD生成
2. **Mamba+的遗忘门设计**：有效解决长距离依赖的信息遗忘问题
3. **多尺度扩散**：局部-中程-全局三级注意力同步几何和拓扑约束
4. **数据集贡献**：ABC-256填补了长序列CAD数据集的空白

## 局限性

- 固定256长度限制超长序列（>256）的建模
- 仅支持无条件生成，缺乏文本/图像引导
- 评估仅限几何质量，未评估工程可用性
- 计算开销：二阶段训练共需300+200K epochs

## 相关工作

- **CAD生成**: DeepCAD, SkexGen, HNC-CAD
- **长序列模型**: Mamba, 稀疏注意力
- **3D扩散生成**: 3DShape2VecSet, DiT-3D, DiffCAD

## 评分
- 新颖性：★★★★☆ — Mamba+与多尺度扩散的组合针对性强
- 技术深度：★★★★☆ — 架构设计合理，实验验证充分
- 实用性：★★★★☆ — 面向工业级CAD生成的实际需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](../../CVPR2025/image_generation/multi-party_collaborative_attention_control_for_image_customization.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
