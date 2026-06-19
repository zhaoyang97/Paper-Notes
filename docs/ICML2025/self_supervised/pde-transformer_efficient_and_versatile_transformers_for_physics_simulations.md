---
title: >-
  [论文解读] PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations
description: >-
  [ICML 2025][自监督学习][偏微分方程] 提出 PDE-Transformer，一种面向物理模拟的改进 Transformer 架构，通过分离通道嵌入、移位窗口注意力和多尺度 U 形结构，在 16 种 PDE 类型上超越现有 SOTA，并展现出强大的下游任务迁移能力。 物理模拟的机器学习代理模型面临几个核心挑战：(…
tags:
  - "ICML 2025"
  - "自监督学习"
  - "偏微分方程"
  - "Transformer"
  - "物理模拟代理模型"
  - "基础模型预训练"
  - "多尺度注意力"
---

# PDE-Transformer: Efficient and Versatile Transformers for Physics Simulations

**会议**: ICML 2025  
**arXiv**: [2505.24717](https://arxiv.org/abs/2505.24717)  
**代码**: [tum-pbs/pde-transformer](https://github.com/tum-pbs/pde-transformer)  
**领域**: 自监督学习  
**关键词**: PDE求解, Transformer架构, 物理模拟代理模型, 基础模型预训练, 多尺度注意力

## 一句话总结

提出 PDE-Transformer，一种面向物理模拟的改进 Transformer 架构，通过分离通道嵌入、移位窗口注意力和多尺度 U 形结构，在 16 种 PDE 类型上超越现有 SOTA，并展现出强大的下游任务迁移能力。

## 研究背景与动机

物理模拟的机器学习代理模型面临几个核心挑战：(1) 物理系统固有的多尺度特性；(2) 数据表示与数值方法紧密耦合（规则网格/网格/粒子）；(3) 不同 PDE 类型的物理通道数量和动力学差异巨大；(4) 模型需在精度或速度上超越传统数值方法才有实用价值。

现有方法的不足：

- **Diffusion Transformer (DiT)**：使用全局自注意力导致计算量随 token 数量二次增长，无法直接处理高分辨率原始数据；采用固定通道数嵌入，不同 PDE 共享 token 时信息密度不一致
- **scOT**：虽引入层次化结构和移位窗口，但并非专门为 PDE 物理模拟设计，性能存在提升空间
- **MPP**：基于轴向 ViT 的多物理预训练，但缺乏对通道间交互的精细控制

PDE-Transformer 的核心动机是：设计一个 **既高效可扩展、又能统一处理多类 PDE** 的通用 Transformer 骨干网络，使其适合作为物理科学领域的基础模型。

## 方法详解

### 整体框架

PDE-Transformer 在 DiT 架构基础上进行了五项关键改造，形成了面向 PDE 模拟的专用架构：

1. **直接操作原始数据**（非潜空间）——通过 Patch 将输入分割为时空 token
2. **多尺度 U 形结构**——PixelShuffle 上下采样 + 跳跃连接
3. **移位窗口注意力**——替换全局注意力，线性扩展到高分辨率
4. **分离通道 (SC) 嵌入**——每个物理通道独立嵌入，通道间通过轴向注意力交互
5. **深度条件化机制**——adaLN-Zero 块注入 PDE 类型、通道类型等条件信息

整体流程：输入 $T_p$ 个时间步的快照 $\mathbf{u}_{\text{in}}$，经 Patch 嵌入→多尺度 Transformer 编解码→输出下一时间步预测 $\mathbf{u}_{\text{out}}$。

### 关键设计

#### 1. Patch 嵌入与膨胀率

给定 patch 大小 $p$，输入 $T \times H \times W$ 被分割为 $H/p \cdot W/p$ 个大小为 $T \times p \times p$ 的 patch，线性映射到 $d$ 维 token 向量。定义 **膨胀率**（expansion rate）：

$$E(p) = \frac{d}{p^2 \cdot T}$$

膨胀率控制 token 信息密度：低膨胀率利于可扩展性（token 少），但小 patch（高膨胀率）可获得更高精度。论文系统探索了该精度-计算量权衡。

#### 2. 多尺度 U 形结构

与原始 DiT 的平坦结构不同，PDE-Transformer 引入层次化设计：

- 在每个 Transformer 阶段末尾使用 **PixelShuffle** 和 **PixelUnshuffle** 进行 token 下采样和上采样
- 相同分辨率的编码器和解码器阶段之间有 **跳跃连接**
- 形成类似 UNet 的多尺度结构，天然契合物理系统的多尺度特性
- 与 Bao (2023) 和 Hoogeboom (2023) 不同，本文使用 adaLN 而非交叉注意力进行条件化，效率更高

#### 3. 移位窗口注意力 (Shifted Window Attention)

为避免全局自注意力的 $O(N^2)$ 计算瓶颈，采用 Swin Transformer 式的移位窗口机制：

- 窗口大小 $w$：每个窗口包含 $w \times w$ 个时空 token
- 相邻层之间窗口偏移 $w/2$，防止窗口边界处的不连续性
- **不使用绝对位置编码**，改用 token 在窗口内的 **对数间距相对位置**，配合前馈网络计算注意力分数（源自 Swin V2）
- 优势：增强平移不变性（对 PDE 学习至关重要），提升不同窗口分辨率间的泛化

#### 4. 混合通道 (MC) vs 分离通道 (SC) 表示

这是本文最核心的设计创新，解决不同 PDE 物理通道数不同的问题：

**混合通道 (MC)**：定义最大通道数 $C_{\max}$，将所有通道拼接为 $T \times C_{\max} \times p \times p$ 的 patch，不足的用零填充。问题：(a) 膨胀率被 $1/C_{\max}$ 压缩，token 表示过度压缩；(b) 将不同物理含义的通道混合。

**分离通道 (SC)（本文提出）**：

- 每个物理通道 **独立嵌入** 为 token 序列
- 空间维度上使用窗口自注意力（不同通道的 token 不交互）
- 引入额外的 **通道维度轴向自注意力 (channel-wise axial MHSA)**：不同通道的同一空间位置 token 通过该机制交互
- 每个通道的膨胀率保持一致，计算量随通道数线性增长
- 通道类型（速度、密度、涡度等）作为条件信息注入

SC 设计使得不同 PDE 类型共享相同的 token 信息密度，显著提升多 PDE 联合学习和迁移能力。

#### 5. 条件化机制 (adaLN-Zero)

继承 DiT 的自适应层归一化机制，扩展条件信息范围：

- **PDE 类型标签**：对应 DiT 中的类别标签
- **物理通道类型标签**（SC 版本）：密度、涡度、速度等
- **扩散时间步**（当作为扩散模型训练时）
- 所有条件嵌入相加后，通过前馈网络回归出 scale 和 shift 向量
- 残差块初始化为恒等函数（Zero 初始化），加速训练收敛
- 所有标签以 10% 概率 dropout，使模型同时支持有条件和无条件推理

#### 6. 边界条件处理

显式支持周期性和非周期性边界条件：

- 移位注意力窗口时，token 沿 x、y 轴滚动排列（模拟周期边界条件）
- 非周期情况下，通过 mask 注意力分数来禁止跨边界 token 交互

#### 7. 算法改进

- 对自注意力的 Q、K 使用 **RMSNorm** 归一化，防止注意力熵失控增长导致的不稳定
- 学习率从 DiT 的 $1.0 \times 10^{-4}$ 调整为 $4.0 \times 10^{-5}$
- 使用 AdamW 优化器，权重衰减系数 $10^{-15}$（bf16 精度训练推荐）
- 基于梯度 EMA 的 **梯度裁剪**，消除训练 loss 的尖峰

### 损失函数 / 训练策略

**监督训练**：对于确定性任务（如确定性求解器的代理模型），使用 MSE 损失单步推理：

$$\mathcal{L}_S = \mathbb{E} \left[ \| \mathcal{M}_\Theta(\mathbf{u}_{\text{in}}, \mathbf{c}) - \mathbf{u}_{\text{out}} \|_2^2 \right]$$

**扩散训练**：对于后验分布较宽的任务，训练为扩散模型，支持生成式推理。将扩散时间步作为额外条件注入 adaLN-Zero。

**预训练策略**：在 PDEBench+ 数据集（16 种 PDE）上进行自回归 next-step prediction 预训练，仅依赖前几个快照（不提供粘度、域范围等模拟参数），模型需从观测数据中隐式推断。

## 实验关键数据

### 主实验

在 PDEBench+（16 种 PDE，包括 Navier-Stokes、扩散方程、Burgers 方程、浅水方程等）上进行预训练评估：

| 模型 | 参数量 | VRMSE (20步) | 架构类型 |
|------|--------|------|----------|
| **PDE-Transformer-SC** | ~110M | **最优** | 分离通道 + U形 + 移位窗口 |
| PDE-Transformer-MC | ~110M | 次优 | 混合通道 + U形 + 移位窗口 |
| DiT (全局注意力) | ~110M | 较差 | 平坦 + 全局注意力 |
| scOT | ~110M | 中等 | Swin + U形 |
| MPP | ~110M | 中等 | 轴向 ViT |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| Patch 大小 p=2 vs p=4 | p=2 显著优于 p=4 | token 数量增 4 倍但精度提升明显 |
| 窗口大小 w=4 vs w=8 | w=8 略优 | 更大窗口覆盖更广空间上下文 |
| SC vs MC 表示 | SC 全面优于 MC | 信息密度一致性是关键 |
| 有/无 U 形结构 | U 形显著优于平坦 | 多尺度归纳偏置契合物理特性 |
| 移位窗口 vs 全局注意力 | 移位窗口精度相当但计算量大幅降低 | 可扩展性关键设计 |
| 有/无相对位置编码 | 相对位置优于绝对位置 | 平移不变性对 PDE 很重要 |
| adaLN-Zero vs 交叉注意力 | adaLN-Zero 更高效 | 加速训练且表现相当 |

### 关键发现

1. **SC 表示是多 PDE 联合训练的核心**：保持各通道 token 信息密度一致，避免混合通道带来的表示压缩问题，在预训练和微调中均优于 MC
2. **U 形多尺度结构对物理模拟至关重要**：与 DiT 平坦结构相比，多尺度结构提供强归纳偏置，显著提升性能
3. **预训练有效提升下游任务表现**：在多个挑战性下游任务（不同 PDE 参数、更高分辨率、更长时间步）上，预训练后微调的模型全面超越从头训练
4. **Patch 大小存在精度-效率最佳平衡点**：p=2 精度最优但计算量大，p=4 是实用选择
5. **20 步自回归预测中误差积累可控**：得益于架构设计，长程预测质量保持稳定

## 亮点与洞察

1. **分离通道 (SC) 设计理念精妙**：将"不同物理量用不同 token 表示 + 轴向注意力交互"的思路，优雅地解决了多 PDE 联合建模的通道异构问题，同时保持了信息密度的一致性
2. **从 DiT 到 PDE-Transformer 的改造路径清晰**：每一步修改（U 形结构、移位窗口、SC 表示、条件化机制）都有明确的物理动机和消融验证
3. **直接操作原始数据而非潜空间**：避免了预训练 VAE 带来的额外复杂性和信息损失，通过架构设计（移位窗口 + 多尺度）解决了高分辨率下的计算瓶颈
4. **边界条件的显式处理**：通过 token 滚动和 mask 注意力分数优雅地支持周期和非周期边界，这对物理模拟至关重要但常被忽略
5. **同时支持监督和扩散两种训练模式**：通过 adaLN-Zero 的灵活条件化机制实现，拓宽了应用场景

## 局限与展望

1. **仅支持 2D 规则网格**：当前架构限于二维空间数据，未扩展到 3D 或非结构化网格
2. **预训练数据仅 16 种 PDE**：作为"基础模型"，PDE 类型覆盖面仍有限，距离真正通用的物理基础模型有距离
3. **缺乏与传统求解器的直接速度对比**：论文主要与其他 ML 方法比较，未深入讨论相对于传统数值方法的加速比
4. **SC 版本计算量随通道数线性增长**：对于通道数很多的复杂系统，可能存在计算瓶颈
5. **扩展方向**：3D 扩展、非规则网格支持、更大规模预训练数据集、与传统求解器的混合方法

## 相关工作与启发

- **DiT (Peebles & Xie, 2023)**：PDE-Transformer 的基础骨干，本文在其上做了五项关键改造
- **Swin Transformer (Liu et al., 2021)**：移位窗口注意力机制的来源
- **scOT (Herde et al., 2024)**：层次化 ViT + 移位窗口用于物理，PDE-Transformer 显著超越
- **MPP (McCabe et al., 2023)**：多物理预训练的先驱工作，使用轴向 ViT
- **FNO (Li et al., 2021)**：神经算子方法的代表，在频域操作
- **启发**：SC 的通道独立嵌入 + 轴向交互思路可推广到其他多模态/多通道 Transformer 架构设计

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 创新性 | 8 | SC 表示和面向 PDE 的 DiT 改造有显著新意 |
| 实用性 | 8 | 开源代码、强泛化能力、监督+扩散双模式 |
| 实验充分性 | 9 | 详尽的消融实验、16 种 PDE、多个下游任务 |
| 写作质量 | 8 | 架构设计动机清晰，图示直观 |
| **综合** | **8** | 物理模拟 Transformer 架构设计的扎实工作 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Test-Time Training Provably Improves Transformers as In-Context Learners](test-time_training_provably_improves_transformers_as_in-context_learners.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[CVPR 2025\] Transformers without Normalization](../../CVPR2025/self_supervised/transformers_without_normalization.md)
- [\[CVPR 2025\] MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining](../../CVPR2025/self_supervised/map_unleashing_hybrid_mamba-transformer_vision_backbones_potential_with_masked_a.md)
- [\[ECCV 2024\] PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer](../../ECCV2024/self_supervised/posformer_recognizing_complex_handwritten_mathematical_expression_with_position_.md)

</div>

<!-- RELATED:END -->
