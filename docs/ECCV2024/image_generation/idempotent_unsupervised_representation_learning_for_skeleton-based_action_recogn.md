---
title: >-
  [论文解读] Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition
description: >-
  [ECCV 2024][图像生成][幂等生成模型] 提出幂等生成模型（IGM），从理论上建立生成模型与最大熵编码（谱对比学习）的等价关系，通过在骨架数据的特征空间施加幂等约束，使生成模型的特征更紧凑、更适合识别任务，在 NTU 60 xsub 上将准确率从 84.6% 提升至 86.2%。 骨架数据以 3D 关节坐标表示人体…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "幂等生成模型"
  - "骨架动作识别"
  - "自监督学习"
  - "对比学习"
  - "扩散模型"
---

# Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition

**会议**: ECCV 2024  
**arXiv**: [2410.20349](https://arxiv.org/abs/2410.20349)  
**代码**: [GitHub](https://github.com/LanglandsLin/IGM)  
**领域**: 图像生成  
**关键词**: 幂等生成模型, 骨架动作识别, 自监督学习, 对比学习, 扩散模型

## 一句话总结

提出幂等生成模型（IGM），从理论上建立生成模型与最大熵编码（谱对比学习）的等价关系，通过在骨架数据的特征空间施加幂等约束，使生成模型的特征更紧凑、更适合识别任务，在 NTU 60 xsub 上将准确率从 84.6% 提升至 86.2%。

## 研究背景与动机

骨架数据以 3D 关节坐标表示人体运动，相比 RGB 视频具有紧凑、高效的优势，广泛用于动作识别任务。现有自监督预训练方法分为两大类：

**生成式学习**（如 MAE、MAMP）：通过预测/重建被 mask 的骨架数据学习时空相关性，但保留了过多与识别无关的外观信息，与骨架数据"空间稀疏、时间一致"的天然特性矛盾

**对比学习**（如 AimCLR、CMD）：通过数据增强构造正样本对，在嵌入空间维持一致性，但增强过程会丢失大量细粒度运动细节

这两种范式优势互补但此前研究通常分开探索。关键问题是：**能否统一生成模型和对比学习的优势？**

作者从信息论角度出发，发现了一条理论路径：
- 生成模型等价于最大熵编码（Maximum Entropy Coding）
- 在生成模型上施加幂等约束后，其损失等价于谱对比学习（Spectral Contrastive Learning）
- 这为在生成框架内引入对比学习提供了理论基础

## 方法详解

### 整体框架

IGM 由三个核心组件构成：

1. **编码器** $f(\cdot)$：对骨架数据施加增强后提取条件特征 $\mathbf{z}$
2. **生成器** $g(\cdot)$：基于扩散模型的条件去噪生成器，以 $\mathbf{z}$ 为条件重建骨架
3. **适配器** $h(\cdot)$：将编码器的高级语义特征投影并融合到生成器的特征空间中

训练使用两类损失：扩散噪声预测损失 + 幂等特征约束。推理时只需编码器 $f(\cdot)$ 用于下游识别任务。

### 关键设计

#### 1. 理论基础：生成模型 = 最大熵编码

自条件生成模型的重建损失 $\mathcal{L} = H(\mathbf{x}|\mathbf{z})$ 本质上是在最大化互信息 $I(\mathbf{z}; \mathbf{x})$。由于编码过程是确定性的，$H(\mathbf{z}|\mathbf{x}) \to 0$，所以最大化互信息等价于最大化特征空间的熵 $H(\mathbf{z})$。

通过有损编码长度（rate-distortion）作为连续随机变量熵的代理，并进行 Taylor 展开，可以证明生成模型主要在**减小特征空间中数据间的相似度**：

$$L = -\frac{\mu\lambda^2}{2}\sum_{i,j}(\mathbf{z}_i^T\mathbf{z}_j)^2 - \mathbf{R}$$

#### 2. 幂等生成模型 = 谱对比学习

幂等性指重编码的稳定性：$f(\hat{\mathbf{x}}) = \mathbf{z}$，即对生成数据再编码应得到相同特征。

幂等损失为 $\mathcal{L}_{\text{ide}} = \|f(\hat{\mathbf{x}}) - \mathbf{z}\|^2 = 2 - 2f(\hat{\mathbf{x}})^Tf(\mathbf{x})$

将幂等损失与熵最大化目标结合，可以推导出：

$$\mathcal{L} = \|\mathbf{A} - \mathbf{F}^T\mathbf{F}\|_F^2 + \mathbf{R} + \mathbf{C}$$

其中 $\mathbf{A}$ 是由数据生成过程定义的邻接矩阵。这恰好是**谱对比学习**的损失形式！而且相比谱对比学习，IGM 还额外优化了高阶残差项 $\mathbf{R}$。

#### 3. 与 MAE 的关系

MAE 通过随机 mask 过程 $M(\cdot)$ 隐式最大化同一数据不同 mask 样本间的特征相似性，但变换后的数据可能偏离真实分布。而幂等生成模型通过生成过程 $G(\cdot)$ 实现类似目标，生成的数据更接近真实分布。

#### 4. 下游任务误差界

根据谱对比学习理论，下游线性评估的错误率有界：

$$P_e \le c_1\sum_{i=d+1}^{m}\lambda_i^2 + c_2\alpha$$

其中 $\alpha$ 是聚类纯度相关项。这意味着需要增加生成数据的多样性（减小邻接矩阵的小奇异值），同时保持运动语义（保持聚类纯度）。扩散模型的噪声采样过程天然提供了这种多样性。

#### 5. 流形解耦特征融合模块（Manifold Decoupled Feature Fusion）

识别任务关注高频运动细节，而生成任务主要优化主成分空间（低频信息），两者在不同特征子空间操作。适配器通过以下高频提取实现解耦：

$$\hat{\mathbf{z}} \Leftarrow (1+\eta)\mathbf{z} - \eta\text{SoftMax}(\mathbf{z}^T\mathbf{z})\mathbf{z}$$

这等价于对比学习均匀性损失的梯度更新，过滤掉序列中的均值等低频信息，保留对识别更重要的语义信息。然后通过 Adaptive LayerNorm（AdaLN）将高频条件注入生成器。

### 损失函数 / 训练策略

总损失包含两个部分：

**1. 噪声预测损失：**

$$\mathcal{L}_{\text{gen}} = \|g(\mathbf{x}_t, h(\mathbf{z}), t) - \varepsilon\|^2$$

**2. 幂等约束（双重约束）：**

**(a) 特征幂等约束** — 确保生成数据再编码的特征与原始特征一致：

$$\mathcal{L}_{\text{ide\_feat}} = -f(\mathbf{x})^T f(\mathbf{x}_0, \mathbf{z}_{t'}, t, t')$$

其中 $\mathbf{x}_0$ 是通过一步去噪估计得到的生成数据。由于生成数据可能含噪，额外输入噪声特征和时间步作为辅助信息。

**(b) 分布幂等约束** — 确保生成数据的特征流形结构与原始数据一致：

$$\mathcal{L}_{\text{ide\_dist}} = \mathcal{D}(\mathcal{P}(\mathbf{x}_0), \mathcal{P}(\mathbf{x}))$$

其中 $\mathcal{P}(\mathbf{x}) = f(\mathbf{x})^Tf(\mathbf{X})$ 表示特征间的相似度结构。这不仅连接同一数据的不同生成样本，还连接具有相似特征的不同数据，构建更紧致的聚类。

## 实验关键数据

### 主实验

NTU RGB+D 数据集上与无监督方法的对比：

| 方法 | 架构 | NTU 60 xview | NTU 60 xsub | NTU 120 xset | NTU 120 xsub |
|------|------|:---:|:---:|:---:|:---:|
| 3s-AimCLR（对比） | GCN | 83.4 | 77.8 | 66.7 | 67.9 |
| 3s-CMD（对比） | GRU | 90.9 | 84.1 | 76.1 | 74.7 |
| MAMP（生成） | Transformer | 89.1 | 84.9 | 79.1 | 78.6 |
| PCM3（混合） | GRU | 90.4 | 83.9 | 77.5 | 76.3 |
| **IGM（本文）** | Transformer | **91.2** | **86.2** | **81.4** | **80.0** |

在所有四个评测协议上均取得最优，NTU 60 xsub 上从之前最佳 84.9% 提升到 86.2%（+1.3%），NTU 120 xsub 从 78.6% 提升到 80.0%（+1.4%）。

### 消融实验

KNN 评估（NTU 60 数据集）：

| 方法 | xview | xsub |
|------|:---:|:---:|
| IGM w/o $\mathcal{L}_{\text{ide}}$ | 67.2 | 64.7 |
| IGM w/ $\mathcal{L}_{\text{ide\_feat}}$ | 70.7 | 68.4 |
| IGM w/ $\mathcal{L}_{\text{ide\_dist}}$ | 72.1 | 69.0 |
| **IGM（完整）** | **72.6** | **69.3** |

### 关键发现

1. **幂等约束至关重要**：去掉幂等约束后 KNN xsub 从 69.3 降到 64.7（-4.6%），验证了理论分析的正确性
2. **分布幂等优于特征幂等**：分布约束（69.0）比特征约束（68.4）更有效，因为它捕获了更丰富的结构信息
3. **两种幂等约束互补**：同时使用达到最佳效果（69.3），说明特征级和分布级约束关注不同层面的一致性
4. **IGM 在零样本适应场景中表现出色**：在之前不可识别的场景中也能取得可观结果
5. **统一框架优于单独范式**：在所有数据集上超越了纯对比学习和纯生成式方法

## 亮点与洞察

1. **理论贡献突出**：首次严格证明了生成模型（带幂等约束）与谱对比学习的等价性，为两个领域的统一提供了理论基础
2. **巧妙利用扩散模型的噪声采样**：解决了自条件生成中多样性不足的矛盾——普通生成过程受限于与原始数据的距离约束导致多样性有限，而扩散模型的噪声采样天然提供多样性
3. **流形解耦设计**：通过高通滤波提取对识别重要的高频信息，避免了生成模型特征偏向主成分空间的维度坍缩问题
4. **理论到实践的完整链条**：从信息论分析出发，推导出幂等约束的必要性，再设计具体的特征和分布级约束

## 局限与展望

1. **仅针对骨架模态**：虽然理论框架通用，但实验仅在骨架数据上验证，对 RGB 视频等其他模态的效果未知
2. **扩散采样的计算开销**：训练阶段需要扩散过程生成多样化数据，增加了训练成本
3. **NTU 数据集局限**：主要在 NTU 和 PKUMMD 上评估，缺少更大规模数据集的验证
4. 高阶残差项 $\mathbf{R}$ 的理论贡献在实验中未被明确量化
5. 可探索将此框架推广到其他自监督学习场景（如视频理解、点云分析）

## 相关工作与启发

- **MAE / MAMP**：生成式预训练的代表，IGM 在此基础上引入幂等约束弥补识别能力不足
- **MCR²（最大率降维编码）**：提供了有损编码作为熵代理的理论工具
- **谱对比学习**：IGM 证明了幂等生成模型与之等价，统一了两个看似不同的范式
- **启发**：在其他模态的自监督学习中（如图像、视频），也可以考虑在生成模型上施加幂等约束来提升识别性能

## 评分

- **创新性**: ★★★★★ — 理论贡献出色，建立了生成模型与对比学习的等价桥梁
- **实验充分度**: ★★★★☆ — 消融详尽但数据集范围有限
- **写作质量**: ★★★★☆ — 理论推导清晰但部分符号较密集
- **实用价值**: ★★★★☆ — 在骨架动作识别领域有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bridging the Skeleton-Text Modality Gap: Diffusion-Powered Modality Alignment for Zero-shot Skeleton-based Action Recognition](../../ICCV2025/image_generation/bridging_the_skeleton_text_modality_gap_diffusion_powered_modality_alignment_for.md)
- [\[ECCV 2024\] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_vi.md)
- [\[ECCV 2024\] MacDiff: Unified Skeleton Modeling with Masked Conditional Diffusion](macdiff_unified_skeleton_modeling_with_masked_conditional_diffusion.md)
- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)

</div>

<!-- RELATED:END -->
