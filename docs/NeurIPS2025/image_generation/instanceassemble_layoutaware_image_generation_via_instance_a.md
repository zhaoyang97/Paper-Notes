---
title: >-
  [论文解读] InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention
description: >-
  [NeurIPS 2025][图像生成][layout-to-image] 提出 InstanceAssemble，在 DiT-based T2I 模型（SD3 和 Flux）的 Transformer 块中注入"实例组装注意力"机制，通过将每个 bounding box 区域的 image token 独立与对应的 layout hidden state 做 cross-attention 来实现精确的实例级空间控制，同时以 LoRA 轻量适配方式保持与现有风格 LoRA 的兼容性，并提出包含 5K 图像/90K 实例的 DenseLayout 基准和多维度的 Layout Grounding Score（LGS）评估指标。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "layout-to-image"
  - "注意力机制"
  - "DiT"
  - "LoRA"
  - "DenseLayout benchmark"
---

# InstanceAssemble: Layout-Aware Image Generation via Instance Assembling Attention

**会议**: NeurIPS 2025  
**arXiv**: [2509.16691](https://arxiv.org/abs/2509.16691)  
**代码**: [https://github.com/FireRedTeam/InstanceAssemble](https://github.com/FireRedTeam/InstanceAssemble)  
**领域**: 目标检测  
**关键词**: layout-to-image, instance assembling attention, DiT, LoRA, DenseLayout benchmark

## 一句话总结

提出 InstanceAssemble，在 DiT-based T2I 模型（SD3 和 Flux）的 Transformer 块中注入"实例组装注意力"机制，通过将每个 bounding box 区域的 image token 独立与对应的 layout hidden state 做 cross-attention 来实现精确的实例级空间控制，同时以 LoRA 轻量适配方式保持与现有风格 LoRA 的兼容性，并提出包含 5K 图像/90K 实例的 DenseLayout 基准和多维度的 Layout Grounding Score（LGS）评估指标。

## 研究背景与动机

**领域现状**：Layout-to-Image（L2I）是可控图像生成中的核心任务——用户指定一组 bounding box 及其文字/视觉描述，模型需要在精确位置生成对应内容。随着 DiT（Diffusion Transformer）架构取代 UNet 成为主流文本到图像生成骨架（如 Stable Diffusion 3、Flux.1 系列），L2I 研究也需要从 UNet 时代的 GLIGEN/ControlNet 等方法迁移到 DiT 架构上。

**现有痛点**：当前 L2I 方法主要存在三个层面的问题。第一，**多实例特征串扰（feature leakage）**严重——在全局注意力机制中，当布局包含大量密集实例时，不同实例的文本条件信息会在 attention 层中互相泄漏，导致生成的物体错位、混淆甚至丢失。这一问题在实例数量超过 10 个时尤为突出。第二，**内容控制维度单一**——绝大多数现有方法只支持文本描述来控制每个实例的外观，缺乏视觉参考图像控制能力，而纯文本在精确描述细粒度外观特征（颜色渐变、纹理细节、特定姿态等）方面存在天然局限。第三，**架构适配代价高**——直接修改 DiT 架构需要大量额外参数和全量微调，不仅训练成本高昂，而且会破坏原始 T2I 模型与社区已有的风格 LoRA 生态之间的兼容性。

**核心矛盾**：问题的根源在于，现有方法试图在全局注意力空间中同时处理所有实例的布局条件——这从结构上就无法避免不同实例之间的信息交叉干扰。传统的做法包括在全局 attention 中加入 layout mask 或 cross-attention 条件注入（如 GLIGEN 的 gated cross-attention、Layout Diffusion 的 cross-attention mask），但这些方法只是在"软约束"层面调节注意力分布，无法从根本上隔离不同实例的特征交互。

**本文目标** 核心目标是：(1) 设计一种能在 DiT 注意力层中精确控制每个实例位置和内容的机制，且在密集布局下不发生特征串扰；(2) 支持文本和视觉两种模态的内容控制；(3) 实现为轻量级 LoRA 插件，保持与现有模型生态的兼容性；(4) 构建一套严格的密集布局评估体系。

**切入角度**：作者的核心观察是——如果我们能把全局注意力问题分解为多个局部的实例级注意力问题，让每个实例区域的 image token 仅与其对应的 layout 描述做交互，就能从结构上彻底消除跨实例的特征泄漏。这不是后处理修补，而是从注意力机制的"原子操作"层面进行隔离。

**核心 idea**：在 DiT 的 Transformer 块中增加一个并行的"实例组装注意力"分支，将每个 bounding box 内的 image token 提取出来，仅与该实例的 layout hidden state 做 cross-attention，然后通过 scatter-add 将结果聚合回原始 hidden state，实现实例级的空间控制而不干扰全局 attention 流。

## 方法详解

### 整体框架

InstanceAssemble 的整体架构是在已有 DiT-based T2I 模型（SD3 或 Flux）的基础上做"最小化侵入"式修改。整体 pipeline 如下：

**输入**：全局文本 prompt + 一组 layout 条件（每个条件包括 bounding box 坐标和实例描述文本/视觉参考）。

**处理流程**：首先，Layout Encoder（`TextBoundingboxProjection` 模块）将每个实例的 bbox 坐标进行 Fourier 位置编码，并与经过 text embedder 投影后的文本 embedding 拼接，通过 MLP 映射成统一维度的 layout hidden state。然后，在 DiT 的每个（或部分）Transformer 块中，除了执行原始的全局 self-attention 和 cross-attention 之外，额外执行一次"Instance-Assembling Attention"操作：根据每个 bbox 对应的空间区域索引（`img_idxs`），从当前 hidden state 中提取该区域的 image token，与对应的 layout hidden state 做 cross-attention，输出再通过一个零初始化的线性层投影后，用 scatter-add 聚合回原始 hidden state。整个过程通过 LoRA 权重控制，冻结原始 DiT 参数。

**输出**：正常的去噪扩散过程输出图像。

一个关键的推理策略是"**Grounding Ratio**"——布局控制仅在去噪过程的前 $r \times T$ 步（默认 $r=0.3$，即前 30%步）施加，之后将 `layout_scale` 设为 0，让模型自由地细化图像细节。这避免了布局控制对最终图像质量的过度约束。

### 关键设计

1. **TextBoundingboxProjection（Layout Encoder）**:

    - 功能：将每个实例的 bounding box 坐标和文本描述编码为统一的 layout hidden state 向量
    - 核心思路：该模块首先对 bbox 坐标进行**密集采样**——每个 bbox 不只用 $(x_1, y_1, x_2, y_2)$ 四个角点表示，而是在 bbox 内部均匀采样 $6 \times 6 = 36$ 个网格点，生成 72 维的坐标向量（36 个点的 xy 坐标），这使得位置信息更加精细地覆盖了实例的空间范围。然后对这 72 维坐标向量进行 **Fourier 位置编码**：用 8 个不同频率（$100^{k/8}$, $k=0,...,7$）分别对每个坐标值做 sin/cos 变换，生成 $72 \times 8 \times 2 = 1152$ 维的 position embedding。文本描述通过 CLIP text encoder 编码后，再经一个 `PixArtAlphaTextProjection`（两层 MLP + SiLU 激活）投影到与 DiT 内部维度一致的 positive embedding。最后将 text embedding 和 position embedding 拼接（维度为 $d_{inner} + 1152$），通过另一个 MLP 映射到 $d_{inner}$ 维，得到最终的 layout hidden state。所有操作都乘以一个二值 mask $m_i \in \{0, 1\}$，标记该 slot 是否有实例存在
    - 设计动机：使用密集采样的 Fourier 编码而非简单的四角坐标，能更精确地表达不同尺寸/长宽比的 bbox 在隐空间中的空间位置。同时，将文本信息和空间信息在 embedding 层就融合，使得后续 attention 操作时每个实例的 layout hidden state 同时携带"是什么"和"在哪里"两类信息

2. **Instance-Assembling Attention（核心贡献）**:

    - 功能：在 DiT 的 Transformer 块中，对每个有效实例独立执行局部 cross-attention，使每个 bbox 区域的 image token 仅受其对应 layout 描述控制
    - 核心思路：具体来说，在每个 Layout Transformer Block 的前向传播中，当 `attention_type == "layout"` 且 `layout_scale != 0` 时，会执行以下步骤：(a) 对 layout hidden states 和 image hidden states 分别做 AdaLayerNorm（使用与原始 attention 相同的时间步条件 temb），得到 normalized 的表示；(b) 找出所有有效实例的索引 `valid_mask = (layout_masks == 1)`；(c) 对每个有效实例 $(i, j)$（batch $i$，实例 $j$），提取该 bbox 覆盖区域的 image token 索引 `img_idxs = img_idxs_list_list[i][j]`，从 `norm_hidden_states[i, img_idxs]` 中取出局部 image token，与 `norm_layout_hidden_states[i, j]` 做 cross-attention——image token 做 Q，layout token 做 KV；(d) attention 输出通过一个**零初始化的线性层** `layout_forward`（即 `zero_module(nn.Linear(dim, dim))`）投影后，乘以 `layout_scale` 进行缩放；(e) 所有实例的输出通过 **scatter-add** 聚合到一个全局大小的 tensor 上，对于有多个实例 bbox 重叠的 image token 位置，取平均值（除以计数 `img_add_cnt`）；(f) 将聚合后的结果加回到原始 hidden states 上
    - 设计动机：这一设计从根本上解决了多实例特征串扰问题——每个实例的 attention 操作完全独立，不存在跨实例的 key-value 交互。零初始化的线性层确保训练开始时 layout 控制对原始生成过程无影响（即"渐进式"注入），避免初始化不当导致生成崩溃。scatter-add + 平均的聚合策略允许处理任意数量的实例和 bbox 重叠的情况。整个过程包裹在 `enable_lora()` 上下文管理器中，确保仅 LoRA 参数被更新

3. **LoRA 轻量适配与风格兼容**:

    - 功能：以 LoRA 模块的形式插入 DiT 的 attention 层和 norm 层，只训练低秩增量参数，冻结原始 DiT 权重
    - 核心思路：InstanceAssemble 的权重文件由两部分组成——`pytorch_lora_weights.safetensors`（LoRA 增量权重）和 `layout.pth`（Layout Encoder 和 zero-init linear 的参数）。推理时，先从预训练的 DiT 模型初始化一个 `LayoutTransformer`（从原始 transformer 的 config 创建，但将 `attention_type` 设为 `"layout"`），加载原始权重（`strict=False`），然后依次加载 LoRA 权重和 layout 权重。关键的一步是加载后立即**将 LoRA scales 清零** (`_zero_out_lora_scales`)——这使得用户可以叠加自定义的 style LoRA 而不会与 layout LoRA 冲突。LoRA 在需要时通过 `enable_lora()` 上下文管理器临时激活，仅在执行 layout 相关计算时生效
    - 设计动机：全量微调不仅计算成本高，更致命的是会破坏 DiT 的原始能力以及与社区风格 LoRA 的兼容性。通过 LoRA 适配，用户可以同时使用"布局控制 LoRA + 风格 LoRA"实现"在指定位置、以指定风格生成指定内容"的组合控制。代码中可以看到，Flux 版本的 layout attention 仅注入到每 3 个 joint transformer block 的第 1 个（`i % 3 == 0`）和 single transformer block 的第 1 个（`i == 0`），进一步减少了 LoRA 参数量

### 损失函数 / 训练策略

训练采用标准的扩散去噪目标——给定一张有完整 layout 标注（bbox + 实例描述）的图像，前向扩散加噪后，模型预测噪声，计算 MSE loss。训练时仅更新 LoRA 参数、Layout Encoder 参数和 zero-init linear 参数，DiT 的原始参数完全冻结。训练数据需要包含多实例 bbox 标注的图像（如 COCO 格式），每个实例有类别名和可选的详细文字描述。InstanceAssemble 支持 SD3（18 层 joint transformer blocks）和 Flux（19 层 double blocks + 38 层 single blocks）两种骨架，且同一套 layout 模块模式可无缝适配两种架构。

一个有趣的训练选择是 `layout_pre_only` 参数——在 SD3 版本中，除最后一个 Transformer block 外，所有 block 都为 layout hidden states 保留完整的 AdaLayerNormZero 和 FFN 更新路径，而最后一个 block 使用简化的 `layout_pre_only=True` 模式（仅做 norm，不做 FFN）。这与 SD3 原始架构中对 text hidden states 的处理方式（`context_pre_only` 在最后一层为 True）保持一致，体现了设计上的对称性。

## 实验关键数据

### 主实验

论文在两个基准上进行了全面评估：标准的 COCO-style 稀疏布局基准和自建的 DenseLayout 密集布局基准。评估指标包括传统的 AP（检测准确率）、FID（生成质量）以及新提出的 LGS（Layout Grounding Score）。

| 评估维度 | 方法 | 性能 | 备注 |
|----------|------|------|------|
| 稀疏布局 (COCO-style) | GLIGEN (UNet) | 基线 | 基于 UNet，需全量微调 |
| 稀疏布局 (COCO-style) | InstanceDiffusion | 较好 | DiT 适配但参数量大 |
| 稀疏布局 (COCO-style) | **InstanceAssemble (Flux)** | **SOTA** | 轻量 LoRA 适配，仅需少量额外参数 |
| 密集布局 (DenseLayout) | 现有方法 | 急剧下降 | 实例数 >10 时性能崩塌，物体丢失严重 |
| 密集布局 (DenseLayout) | **InstanceAssemble (Flux)** | **SOTA** | 即使 20+ bbox 也保持稳定 |
| 密集布局 (DenseLayout) | **InstanceAssemble (SD3)** | **次优** | SD3 骨架略弱于 Flux 但仍远优于基线 |
| 风格 LoRA 兼容 | 其他方法 | 差/中 | 全量微调破坏兼容性 |
| 风格 LoRA 兼容 | **InstanceAssemble** | **强** | LoRA scale 管理机制保证叠加使用 |

InstanceAssemble 支持两种 DiT 骨架——Flux.1-dev（28步推理）、Flux.1-schnell（4步快速推理）和 SD3-medium（50步推理）。在 DenseLayout 数据集上，该方法展现出显著优势：当实例数量从 5 增长到 20+ 的密集场景下，传统方法因全局 attention 中的特征串扰而性能急剧下滑（物体错位、混淆甚至完全消失），而 InstanceAssemble 由于其实例级隔离的 attention 机制得以保持稳定的空间控制精度。特别值得注意的是在 Flux 骨架上的表现优于 SD3，这与 Flux 更强的基础生成能力一致。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Full model (Instance-Assembling Attention + LoRA) | 最优 | 完整模型 |
| 全局 attention + layout mask 替代 | 下降 | 无法完全隔离特征串扰 |
| 无密集采样（仅 4 角点） | 下降 | 位置编码精度受限 |
| 无 Grounding Ratio（全程施加 layout） | 质量下降 | 过度约束影响细节生成 |
| layout_forward 非零初始化 | 不稳定 | 训练初期容易崩溃 |
| 不同 Grounding Ratio (10%/30%/50%) | 30% 最优 | 平衡布局准确性和生成质量，10% 位置不够准，50% 细节损失 |

### 关键发现

- **Instance-Assembling Attention 是核心贡献**——去掉它改回全局 attention+mask 方式后，密集布局场景的性能显著下降，说明实例级隔离机制是解决特征串扰的关键
- **密集采样的 Fourier 位置编码贡献显著**——36 个网格点（$6 \times 6$）的密集采样比简单的 4 角点表示提供了更精确的空间位置信息，特别是对于长条形或极小的 bbox
- **Grounding Ratio 30% 是一个很好的默认值**——前 30% 的去噪步骤施加布局控制足以确定物体位置，之后的 70% 步骤让模型自由细化细节，在布局准确性和生成质量之间取得了良好平衡
- **LoRA 兼容性**——与多种风格 LoRA（动漫、水墨、3D 等）叠加后，布局控制能力不衰减，这对实际应用场景（如设计工具、电商素材生成）非常重要
- **在 Flux 架构上，layout attention 仅注入到部分 block**——joint blocks 中每 3 个注入 1 个，single blocks 仅注入第 1 个，说明不需要在所有层都施加布局控制就能获得良好效果

## 亮点与洞察

- **实例级注意力隔离的思想非常巧妙**。传统方法试图在全局 attention 中通过 mask 或条件注入来"引导"注意力分布，但这本质上是一种软约束，无法完全防止 key-value 在不同实例之间的信息泄漏。InstanceAssemble 直接将问题分解为 N 个独立的局部 attention 操作，从结构上杜绝了串扰。这种"分治"思想可以推广到其他需要细粒度空间控制的生成任务中

- **零初始化线性层（zero_module）作为渐进注入机制**。这个设计确保了训练开始时 layout 分支对原始生成流程的影响为零，允许模型从预训练好的 DiT 权重出发渐进地学习布局控制能力，避免了"从零训练 layout 分支导致的初期不稳定"问题。这是一个在条件生成领域广泛适用的技巧（ControlNet 也用了类似思路），但在实例级 attention 中的应用是新颖的

- **Grounding Ratio 的设计展示了对扩散过程的深刻理解**。扩散模型的去噪过程中，早期步骤决定全局结构和物体位置，后期步骤负责精细细节。InstanceAssemble 利用这一特性，仅在前 30% 步骤施加布局约束，之后"放手"让模型自由发挥，这是一种优雅的控制策略

- **DenseLayout 基准和 LGS 指标填补了重要的评估空白**。现有的 L2I 基准（如基于 COCO 的评估）布局密度较低（通常 3-5 个实例），无法暴露密集场景下的问题。DenseLayout 包含平均每张图 18 个实例的数据（5K 图像/90K 实例），而 LGS 评估不仅衡量空间定位准确性（IoU），还通过 VLM 自动判断颜色、纹理、形状是否与描述一致，比简单的 FID/AP 更全面

## 局限与展望

- **计算复杂度随实例数线性增长**。由于每个有效实例需要独立执行一次 attention 操作（包含 QKV 投影和全连接），当实例数非常大时（如 50 个 bbox，DenseLayout 基准支持最多 100 个），推理时间会显著增加。代码中的循环 `for k in range(valid_indices.size(0))` 是逐实例串行执行的，没有进行批处理优化，这在 GPU 并行计算能力得不到充分利用的同时还引入了 Python 循环开销。一个可能的改进是将所有实例的 attention 操作通过 padding + batched attention 来批处理执行，或者利用 CUDA 自定义算子实现并行化

- **视觉参考控制尚未公开**。GitHub 仓库的 TODO 显示 "additional-visual control version" 尚未发布，目前只有文本控制版本。视觉参考控制对于需要精确外观一致性的应用场景（如角色一致性生成）至关重要

- **训练代码未开源**。当前仅发布了推理代码和预训练权重，训练代码尚未公开，限制了社区对该方法的验证和改进

- **仅在 SD3 和 Flux 两个 DiT 骨架上验证**。未探索在其他 DiT 变体（如 SD3.5、Flux.2 等后续版本）或 UNet-based 模型上的适用性。此外，将 instance-assembling attention 思想迁移到视频生成模型（如 OpenSora）或 3D 生成模型也是值得探索的方向

- **LGS 指标依赖 GroundingDINO 和 VLM 的能力**。LGS 的 IoU 计算依赖 GroundingDINO 的检测准确性（box_threshold=0.35, text_threshold=0.25），特别是对小物体和罕见类别的检测可能不准确。颜色/纹理/形状判断依赖 MiniCPM-V-2.6 的 VQA 能力，通过向 VLM 提问"图中的 [物体] 是否与描述中的 [颜色/纹理/形状] 一致"来评分。如果 IoU 低于阈值（默认 0.5）则该实例的所有属性得分均为 0。这些外部模型的局限性会成为 LGS 可靠性的瓶颈

- **bbox 重叠区域的处理为简单平均**。当多个实例的 bbox 有重叠时，重叠区域的 image token 会收到多个实例的控制信号，目前的处理是通过 `img_add_cnt` 计数器做简单平均（`attn_output_add / img_add_cnt`）。这在密集重叠场景下（如多个物体堆叠、前后遮挡等）可能导致控制冲突，生成结果在重叠区域出现模糊或不一致

## 相关工作与启发

- **vs GLIGEN**：GLIGEN 基于 UNet 架构，通过在 cross-attention 层插入额外的 gated attention 模块来注入 layout 条件。它需要对整个 UNet 进行全量微调，参数量大且不兼容风格 LoRA。InstanceAssemble 改进了两点：(1) 适配 DiT 架构时代的主流骨干；(2) 使用实例级隔离 attention 代替全局条件注入，消除了特征串扰

- **vs InstanceDiffusion**：InstanceDiffusion 同样关注实例级控制，但其 attention 机制仍在全局空间操作，通过 attention mask 来约束不同实例的交互范围。InstanceAssemble 更进一步——完全将不同实例的 attention 操作物理隔离，每个实例有自己独立的 Q/K/V 计算路径

- **vs IP-Adapter**：IP-Adapter 通过解耦 cross-attention 实现图像提示控制——一路 cross-attention 处理文本，另一路处理图像参考。InstanceAssemble 将类似的"解耦 attention"思想推广到了实例级别——每个实例有自己独立的 cross-attention 通路，且同时支持文本和视觉两种条件

- **vs ControlNet**：ControlNet 通过克隆整个 encoder 来注入空间条件（如边缘图、深度图），参数量巨大（与原始 UNet/DiT 的 encoder 相当）。InstanceAssemble 的 layout encoder 仅由两个 MLP（PixArtAlphaTextProjection）和 zero-init linear 组成，参数量极小，且架构设计上与 ControlNet 完全正交——代码中保留了 `controlnet_block_samples` 和 `controlnet_single_block_samples` 接口，意味着用户可以同时使用 ControlNet（提供全局空间条件如深度图）和 InstanceAssemble（提供实例级布局控制），实现多层次的空间控制组合

- **启发方向**：(1) 实例级隔离 attention 可推广到视频生成中的 object-tracking-aware generation——让每帧的物体追踪信息指导 attention 的空间分组；(2) dense sampling + Fourier encoding 的位置表达方式可用于需要精确空间控制的其他任务（如 inpainting、超分辨率的空间引导）；(3) Grounding Ratio 的"前期约束+后期释放"策略可迁移到其他条件生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 实例级隔离 attention 的思想清晰且有效，从结构上解决了多实例串扰问题；Fourier 密集采样位置编码和 Grounding Ratio 策略也有独到之处
- 实验充分度: ⭐⭐⭐⭐ 提出了新 benchmark（DenseLayout）和新指标（LGS），在两种 DiT 骨架（SD3、Flux）上验证，有消融实验和风格 LoRA 兼容性验证；遗憾的是论文全文 HTML 版本不可用，无法验证定量结果的具体数值
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，从"特征串扰"问题出发推导出"实例级隔离"的解决方案逻辑链流畅；代码结构清晰，开源质量高
- 价值: ⭐⭐⭐⭐⭐ DiT 时代 L2I 的实用方案——轻量（LoRA）、灵活（兼容风格 LoRA）、效果好（密集布局 SOTA），适用于设计工具、电商素材、游戏场景等需要精确布局控制的实际应用场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](instance-level_composed_image_retrieval.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](../../ICML2026/image_generation/occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)
- [\[ICCV 2025\] FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation](../../ICCV2025/image_generation/ficgen_frequency-inspired_contextual_disentanglement_for_layout-driven_degraded_.md)
- [\[CVPR 2025\] Learning Flow Fields in Attention for Controllable Person Image Generation](../../CVPR2025/image_generation/learning_flow_fields_in_attention_for_controllable_person_image_generation.md)

</div>

<!-- RELATED:END -->
