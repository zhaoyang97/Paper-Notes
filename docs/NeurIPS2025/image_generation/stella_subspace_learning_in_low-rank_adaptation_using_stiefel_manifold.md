---
title: >-
  [论文解读] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold
description: >-
  [NeurIPS 2025 Spotlight][图像生成][LoRA] 提出StelLA，通过将LoRA的适配矩阵分解为 $USV^\top$ 三因子形式，并将 $U$、$V$ 约束在Stiefel流形上进行黎曼优化，实现训练过程中对低秩子空间的显式学习，在多个下游任务上一致超越现有LoRA变体。
tags:
  - "NeurIPS 2025 Spotlight"
  - "图像生成"
  - "LoRA"
  - "Stiefel流形"
  - "子空间学习"
  - "黎曼优化"
  - "三因子分解"
---

# StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.01938](https://arxiv.org/abs/2510.01938)  
**代码**: [GitHub](https://github.com/SonyResearch/stella)  
**领域**: 参数高效微调 / 低秩适配  
**关键词**: LoRA, Stiefel流形, 子空间学习, 黎曼优化, 三因子分解

## 一句话总结

提出StelLA，通过将LoRA的适配矩阵分解为 $USV^\top$ 三因子形式，并将 $U$、$V$ 约束在Stiefel流形上进行黎曼优化，实现训练过程中对低秩子空间的显式学习，在多个下游任务上一致超越现有LoRA变体。

## 研究背景与动机

LoRA作为大模型参数高效微调的主流方法，通过学习低秩矩阵 $BA^\top$ 来适配预训练权重，但其性能与全参数微调之间仍存在差距。已有工作（如PiSSA、MiLoRA）尝试利用SVD分解来改进LoRA的初始化，但这些方法仅在训练开始时提供指导，对后续优化过程的影响有限。

作者观察到一个关键问题：**不同的启发式子空间选择策略**（选择主成分还是次要成分、基于权重还是梯度）给出了不同的结论，说明手动选择子空间是次优的。这自然引出一个问题：能否在训练过程中**直接学习最优子空间**？

此外，LoRA的两因子分解 $BA^\top$ 将输入/输出子空间与缩放因子耦合在一起，难以进行结构化的几何优化。作者受SVD结构启发，提出将方向（子空间）与幅度（缩放）解耦，这与DoRA中将权重分解为大小和方向的思想一致。

## 方法详解

### 整体框架

StelLA将每个线性层的低秩适配表示为三因子形式：

$$\tilde{W} = W + \frac{\alpha}{r} U S V^\top$$

其中 $U \in \text{St}(r,m)$ 和 $V \in \text{St}(r,n)$ 分别是输出和输入子空间的正交基（约束在Stiefel流形上），$S \in \mathbb{R}^{r \times r}$ 学习两个子空间间的映射。这种设计显式分离了子空间方向和缩放幅度，使得可以在保持正交性的同时进行子空间优化。

### 关键设计

1. **Stiefel流形约束与黎曼优化**: 为保证 $U$ 和 $V$ 在训练过程中始终为列正交矩阵，将其约束在Stiefel流形 $\text{St}(k,n) = \{Y \in \mathbb{R}^{n \times k} \mid Y^\top Y = I_k\}$ 上。优化分为三步：(a) 将欧几里得梯度转换为黎曼梯度 $\text{grad}_Y = \nabla_Y - Y(\nabla_Y)^\top Y$；(b) 将优化器产生的扰动梯度投影回切空间 $\pi_Y(\Delta) = \Delta - Y \text{symm}(Y^\top \Delta)$；(c) 通过极分解回缩映射 $\rho_Y(\Delta) = \text{uf}(Y + \Delta)$ 将更新后的点拉回流形。这种设计允许任何现有欧几里得优化器（如Adam）无缝转化为黎曼优化器。

2. **模块化几何优化设计**: 算法通过optimizer hook实现：黎曼梯度转换作为pre-hook，投影和回缩作为post-hook。与现有黎曼优化器（如Riemannian Adam）不同，StelLA将几何约束与优化器逻辑解耦——不需要修改优化器内部的动量或自适应学习率机制，只需将优化器的更新方向视为对黎曼梯度的扰动，再通过投影操作修正即可。实现上采用批量SVD策略（将不同层中相同形状的 $U$/$V$ 堆叠），实现15-20倍加速。

3. **梯度缩放策略**: 由于 $U$ 和 $V$ 的列是单位向量，其元素量级分别为 $1/\sqrt{m}$ 和 $1/\sqrt{n}$。当 $m \neq n$（如LLM的FFN层中隐藏维度放大4倍），Adam优化器的梯度归一化会导致两者学习速度不平衡。StelLA在投影操作前分别对 $U$ 和 $V$ 的梯度乘以 $\sqrt{d/m}$ 和 $\sqrt{d/n}$（$d$ 为输入token的隐藏维度）来补偿这种差异。

### 损失函数 / 训练策略

训练目标为标准的任务损失 $\mathcal{L}$，但优化路径不同：$U$ 和 $V$ 在Stiefel流形上受约束优化，$S$ 在欧几里得空间中无约束优化。初始化采用随机列正交矩阵初始化 $U$/$V$、单位矩阵初始化 $S$，无需像PiSSA那样修改预训练权重。

## 实验关键数据

### 主实验

**常识推理（LLaMA3-8B, rank=32）**

| 方法 | 参数量(%) | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | 平均 |
|------|-----------|-------|------|-----------|------------|-------|-------|------|
| LoRA | 0.700 | 75.16 | 88.14 | 95.41 | 86.74 | 90.84 | 78.70 | 85.27 |
| DoRA | 0.710 | 75.38 | 88.01 | 95.35 | 86.29 | 90.54 | 79.69 | 85.16 |
| ScaledAdamW | 0.700 | 75.24 | 88.57 | 95.81 | 85.11 | 91.09 | 80.55 | 85.40 |
| **StelLA** | **0.702** | **75.91** | **89.86** | **96.41** | **87.82** | **91.98** | **82.34** | **86.72** |

StelLA在LLaMA2-7B和LLaMA3-8B上均提升约+1.3个百分点。

**文本到图像生成（SD 1.5, rank=4, FID↓）**

| 数据集 | LoRA | DoRA | PiSSA | StelLA |
|--------|------|------|-------|--------|
| BarbieCore | 175.48 | 175.04 | 299.49 | **170.25** |
| Expedition | 156.34 | 155.80 | 291.22 | **146.12** |
| Hornify | 180.48 | 179.58 | 295.15 | **167.53** |

FID最大降低达12个点。

### 消融实验

| 配置 | 平均准确率 | 说明 |
|------|-----------|------|
| StelLA（默认） | 86.72 | 非零初始化 + 梯度缩放 |
| 欧几里得几何（无正交约束） | 84.4 | 说明正交约束的必要性 |
| 商空间几何 | 85.7 | 次于Stiefel乘积流形 |
| 零初始化 | 86.5 | 小S导致U/V梯度小，收敛慢 |
| 伪零初始化 | 84.2 | 污染预训练权重，性能最差 |
| SVD-major初始化 | 86.7 | 与随机初始化接近，说明几何优化可自动学习子空间 |
| 无梯度缩放 | 86.4 | 缩放带来+0.3提升 |
| 极回缩 vs 指数映射 | 86.72 vs 86.76 | 性能接近，极回缩更高效 |

### 关键发现

- Stiefel流形约束相比无约束的欧几里得三因子分解（TriLoRA/MoSLoRA）明显更优(86.7 vs 84.4)
- StelLA对初始化策略鲁棒：随机初始化、SVD主成分、SVD次要成分初始化性能接近
- 额外参数量仅增加 $r^2$（rank的平方），可忽略不计
- 在图像分类（ViT-Base/Large, 8个数据集）上也一致最优

## 亮点与洞察

- **设计优雅**：三因子分解 + Stiefel流形约束形成了一个几何上自然的框架，将SVD的结构直接嵌入训练过程
- **模块化**：通过optimizer hook实现，可与任何现有优化器组合，无需手动实现黎曼Adam等特殊优化器
- **批量SVD加速**：将瓶颈操作（极分解）通过跨层批处理加速15-20倍，解决了实际部署的效率问题
- 梯度缩放策略虽然简单，但针对FFN层的非对称维度提供了有效的学习速率平衡

## 局限与展望

- 三因子分解和回缩操作引入了额外计算开销（虽然通过批量SVD缓解）
- 未与AdaLoRA等自适应rank方法结合（可将 $S$ 约束为对角矩阵实现）
- 未在70B规模模型或Mistral/LLaVA等其他模型族上验证
- 未探索与QLoRA（量化）的联合使用

## 相关工作与启发

- 可与AdaLoRA的rank自适应策略结合，将 $S$ 设为对角矩阵，通过奇异值裁剪实现动态rank
- 正交约束对对抗鲁棒性也有潜在好处
- 几何优化框架为其他流形约束（如Grassmann流形）在微调中的应用打开了思路

## 评分

- 新颖性: ⭐⭐⭐⭐ 将Stiefel流形优化引入LoRA具有原创性，三因子分解虽非首创但几何约束设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖NLU、NLG、视觉分类、图像生成四个领域，消融详尽
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导清晰，算法描述规范，消融设计有说服力
- 价值: ⭐⭐⭐⭐ 提供了即插即用的LoRA改进方案，代码已开源，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[ICML 2025\] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape](../../ICML2025/image_generation/flat-lora_low-rank_adaptation_over_a_flat_loss_landscape.md)
- [\[ICML 2025\] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](../../ICML2025/image_generation/intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[CVPR 2026\] Low-Rank Residual Diffusion Models](../../CVPR2026/image_generation/low-rank_residual_diffusion_models.md)

</div>

<!-- RELATED:END -->
