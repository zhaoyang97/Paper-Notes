---
title: >-
  [论文解读] ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer
description: >-
  [NEURIPS2025][图像生成][指令图像编辑] ICEdit 提出一种基于大规模 Diffusion Transformer (DiT) 的上下文编辑范式，通过 in-context prompt + 最小化 LoRA-MoE 微调 + VLM 早期筛选推理时缩放，仅用 0.1% 训练数据即达到 SOTA 编辑性能。
tags:
  - "NEURIPS2025"
  - "图像生成"
  - "指令图像编辑"
  - "Transformer"
  - "上下文学习"
  - "LoRA-MoE"
  - "推理时缩放"
---

# ICEdit: Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer

**会议**: NEURIPS2025  
**arXiv**: [2504.20690](https://arxiv.org/abs/2504.20690)  
**代码**: [项目主页](https://river-zhang.github.io/ICEdit-gh-pages)  
**领域**: 图像生成  
**关键词**: 指令图像编辑, 扩散 Transformer, 上下文学习, LoRA-MoE, 推理时缩放

## 一句话总结

ICEdit 提出一种基于大规模 Diffusion Transformer (DiT) 的上下文编辑范式，通过 in-context prompt + 最小化 LoRA-MoE 微调 + VLM 早期筛选推理时缩放，仅用 0.1% 训练数据即达到 SOTA 编辑性能。

## 背景与动机

指令引导的图像编辑（instruction-based image editing）是近年来图像生成领域的热门方向，其核心是根据自然语言指令对图像进行精确修改。现有方法分为两大类：

1. **微调方法**：如 InstructPix2Pix、EmuEdit、UltraEdit 等，需要大规模编辑数据集（45万~1000万样本）和全量微调，计算代价高昂，且往往需要修改模型架构（如额外的条件编码器、通道调整等）。
2. **免训练方法**：如 SDEdit、StableFlow、RF-Solver 等，通过图像反演或注意力操控实现编辑，计算高效但难以理解复杂指令，编辑成功率低。

这两类方法面临**精度-效率的两难困境**。与此同时，大规模 DiT 模型（如 FLUX）展现出两个关键特性：(1) 卓越的文本-图像对齐能力；(2) 内在的上下文感知能力——通过注意力机制实现参考图与生成内容的双向交互。这启发了作者思考：能否直接利用 DiT 的固有能力来完成指令编辑，而非依赖外部复杂度？

## 核心问题

直接用 DiT 做指令编辑会遇到两个根本性限制：

1. **指令理解能力差**：DiT 能理解描述性 prompt（"一只红色的猫"），但无法理解编辑指令（"把猫变成红色"），因为描述性 prompt 和编辑指令的嵌入空间存在 gap。
2. **布局不稳定**：在重新生成场景时，模型经常改变不需要编辑的区域，导致编辑成功率低。

## 方法详解

ICEdit 由三个核心组件构成：

### 3.1 In-Context 编辑范式

核心思想是将编辑任务转化为 DiT 天然擅长的"diptych 生成"任务。具体做法是构建一个左右并排的图像对：左侧放原图，右侧由模型根据指令生成编辑结果。

**关键创新 — In-Context Edit Prompt**：设计了一种固定格式的 prompt 模板：

> "A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}."

这一 prompt 设计将编辑指令嵌入到 DiT 能理解的描述性格式中，实验表明编辑成功率提升约 **70%**。作者对比了三种 prompt 格式：直接指令（效果最差）、IC prompt（显著提升）、全局描述性 prompt（效果最好但不实用，需要精确的目标图描述）。

作者还提出了两种免训练框架：基于 T2I DiT（需要图像反演，较慢）和基于 Inpainting DiT（更简洁，用固定 mask 直接操作）。最终选择 inpainting 框架作为后续微调的基础。

### 3.2 LoRA-MoE 高效微调

免训练框架虽有潜力但效果不够好，作者用仅 **5万** 样本（来自 MagicBrush 9K + OmniEdit 40K）进行轻量微调。

**普通 LoRA 的局限**：单一 LoRA 结构难以同时处理多种编辑任务（风格迁移、物体移除等），因为不同任务需要不同的潜空间特征操控。

**解决方案 — Mixture of LoRA Experts**：在 DiT block 的多模态注意力输出投影层引入 MoE 结构。其形式为：

$$\text{Output} = \text{BaseLayer}(x) + \frac{\alpha}{r} \sum_{i=1}^{N} G(x)_i \cdot B_i \cdot A_i \cdot x$$

其中 $G(x)_i$ 是路由概率，采用 Top-K 稀疏选择（K=1）。其他模块使用标准 LoRA（rank=32）。这一设计使模型仅用 **0.2B 可训练参数**（对比 FLUX 12B 全量参数），GPT 评分相比纯 LoRA 提升 **13%**。

训练细节：4 张 A800 GPU 训练一天，batch size=2（梯度累积），512×512 分辨率，Prodigy 优化器。

### 3.3 Early Filter 推理时缩放

核心观察：初始噪声对编辑结果影响极大，不同噪声的编辑质量差异显著。

**关键洞察**：在 rectified flow 模型中，编辑是否成功在极少的去噪步数内就能判断。这是因为模型在潜空间中高效遍历，少量步数即可生成足够评估的粗略结果。

**具体策略**：

1. 采样 $M$ 个不同的初始噪声（默认 M=6）
2. 每个噪声执行 $m$ 步快速去噪（默认 m=10，远小于完整的 n=50 步）
3. 使用 VLM（Qwen-VL-72B）通过冒泡排序式的逐对比较，选出最佳噪声
4. 对最佳噪声执行完整的 $n$ 步去噪得到最终结果

计算开销为 $\text{NFE} = n + M \times m = 50 + 6 \times 10 = 110$，相比完整推理所有噪声（$M \times n = 300$）大幅降低。

## 训练与推理细节

### 数据集构成

训练数据共 50K 样本，来源于两个公开数据集，**未经精细筛选**（仍含低质量样本）：

| 任务类型 | 移除 | 添加 | 替换 | 属性修改 | 风格迁移 | 总计 |
|---------|------|------|------|---------|---------|------|
| 样本数 | 13,272 | 11,938 | 5,823 | 11,484 | 10,530 | 53,047 |

选用 MagicBrush（9K）弥补其在编辑类型均衡性上的不足，OmniEdit（40K 随机抽样）补充风格和领域多样性。

### 训练配置

- **骨干模型**：FLUX.1 Fill（12B 参数的 rectified flow inpainting DiT）
- **LoRA 配置**：rank=32，alpha=rank，4 个 MoE 专家，TopK=1
- **路由网络**：单层线性层，无额外负载均衡 loss（实验发现路由自然均衡）
- **优化器**：Prodigy（safeguard warmup + bias correction，weight decay=0.01）
- **有效 batch size**：2（batch=1 × 梯度累积 2 步）
- **训练分辨率**：512×512 → 拼接为 512×1024 diptych
- **硬件**：4× A800 (80G) GPU，训练一天

**显存消耗**（无 gradient checkpointing）：

| 分辨率 | 显存 |
|--------|------|
| 512×512 | 60 GB |
| 768×768 | 77 GB |
| 1024×1024 | OOM |

启用 gradient checkpointing 可降至 37/39/42 GB，但训练速度显著下降。作者选择在 512 分辨率下不用 gradient checkpointing 以平衡效率。

### 推理配置

- **guidance scale**：50
- **去噪步数**：50
- **IC prompt 格式**（训练和推理一致）："A diptych with two side-by-side images of the same scene. On the right, the scene is exactly the same as on the left but {instruction}."
- **推理缩放**：6 个随机噪声 × 10 步快速推理，Qwen2.5-VL-72B 通过 API 做逐对评估
- **推理硬件**：单张 A100 (40G)

## 实验关键数据

### Emu Edit 测试集

| 方法 | 基础模型 | 可训练参数 | 训练数据 | CLIP-I↑ | DINO↑ | GPT↑ |
|------|---------|-----------|---------|---------|-------|------|
| InstructPix2Pix | SD 1.5 | 0.9B | 0.45M | 0.856 | 0.773 | 0.36 |
| EmuEdit | 闭源 | 2.8B | 10M | 0.877 | 0.844 | 0.72 |
| UltraEdit | SD 3 | 2.5B | 3M | 0.880 | 0.847 | 0.54 |
| FluxEdit | Flux.1 dev | 12B | 1.2M | 0.852 | 0.760 | 0.22 |
| ACE++ | Flux.1 Fill | 12B | 54M | 0.791 | 0.687 | 0.24 |
| **ICEdit (ours)** | Flux.1 Fill | **0.2B** | **0.05M** | **0.907** | **0.866** | **0.68** |

### MagicBrush 测试集

| 方法 | L1↓ | CLIP-I↑ | DINO↑ |
|------|-----|---------|-------|
| UltraEdit | 0.066 | 0.904 | 0.852 |
| **ICEdit (ours)** | **0.060** | **0.928** | **0.853** |

### VIE-Score 对标商业模型

ICEdit（含推理缩放）VIE-Score 达到 **78.2**，超过商业模型 SeedEdit 的 75.7。推理缩放带来 SC 分数 **19%** 提升和整体 VIE-Score **16%** 提升。

### 消融实验关键发现

- IC prompt vs 直接指令：GPT 分数提升 **70%**（0.14→0.24）
- 加入 LoRA 微调：GPT 分数再提升 **150%**（0.24→0.60）
- LoRA-MoE vs 普通 LoRA：GPT 分数提升 **13%**（0.60→0.68）
- 数据效率：仅 10K 样本即可显著超越免训练方法，50K 达到性能饱和
- 推理缩放中 CLIP 评估器 vs VLM 评估器：VLM 显著优于 CLIP（0.78 vs 0.65）

### MoE 专家配置消融

| 专家数量 | 专家 Rank | 参数量 | CLIP-I↑ | GPT↑ |
|---------|----------|--------|---------|------|
| 1 | 32 | 120M | 0.892 | 0.59 |
| 4 | 8 | 120M | 0.920 | 0.58 |
| **4** | **32** | **214M** | **0.907** | **0.68** |
| 6 | 32 | 270M | 0.914 | 0.66 |
| 8 | 32 | 335M | 0.907 | 0.61 |

关键发现：同参数量下（120M），4 专家 rank=8 的 CLIP-I 优于 1 专家 rank=32，但 GPT 分数相近；增大到 4×rank32 后 GPT 大幅提升；但继续增加专家数（6/8）反而性能下降，说明路由网络在更多专家下训练困难，需更复杂的路由设计和均衡约束。

### 评估方法论补充

作者发现传统 CLIP text-image direction similarity 指标**与人类偏好严重不一致**——成功编辑可能得低分，失败编辑反而得高分。因此采用 GPT-4o 进行 SC（指令遵循+未编辑区域保持）和 PQ（感知质量）双维度评估，最终通过阈值二值化得到 $\text{Overall} = \sqrt{\text{SC} \times \text{PQ}}$ 的 VIE-Score。

## 亮点

1. **极致数据效率**：仅需 0.05M（5万）训练样本，是 EmuEdit 的 0.5%、InstructPix2Pix 的 11%，却达到或超越 SOTA 性能，颠覆了"编辑需要海量数据"的认知。
2. **零架构修改**：不像之前方法需要额外的位置编码器、条件编码器或通道修改，ICEdit 完全保留原始 DiT 结构，仅通过 prompt 工程和轻量 LoRA 适配。
3. **IC Prompt 设计精妙**：将"编辑指令理解"问题转化为"描述性生成"问题，巧妙绕开了指令嵌入空间与生成嵌入空间的 gap。
4. **推理时缩放策略创新**：利用 rectified flow 模型"早期可判断"的特性，结合 VLM 评判，用极低计算代价（NFE=110 vs 350）获得接近全量搜索的效果。
5. **MoE-LoRA 设计有效**：在注意力输出投影层引入多专家路由，用更少参数实现更好的多任务编辑能力。

## 局限与展望

1. **物体移动失败**：涉及空间位置重定位的指令（如"把椅子移到角落"）效果差，源于训练数据中此类样本不足。
2. **语义歧义**：T5 编码器的语义理解有限，难以处理多义词（如"mouse"既指鼠标又指老鼠），未来可整合 MLLM 模块提升语义保真度。
3. **VLM 推理开销**：推理缩放依赖 72B 参数的 Qwen-VL，小模型（7B）判断不准确。可通过蒸馏 VLM 来降低开销。
4. **训练分辨率限制**：训练在 512×512 分辨率上进行，高分辨率场景效果有待验证。
5. **MoE 路由扩展性**：当前 4 专家 + TopK=1 即可，更多专家反而不提升，路由网络设计和负载均衡 loss 需进一步研究。
6. **数据集未精心筛选**：50K 训练数据中仍含低质量样本，精细数据筛选有望进一步提升性能。

## 与相关工作的对比

| 维度 | ICEdit | InstructPix2Pix | EmuEdit | UltraEdit | FluxEdit | ACE++ |
|------|--------|----------------|---------|-----------|----------|-------|
| 架构修改 | 无（保留原始 DiT） | 通道调整 | 条件编码器 | 通道调整 | 全量微调 | 位置+条件编码器 |
| 训练数据量 | 50K | 450K | 10M | 3M | 1.2M | 54M |
| 可训练参数 | 0.2B | 0.9B | 2.8B | 2.5B | 12B | 12B |
| 基础模型 | FLUX Fill (DiT) | SD 1.5 (UNet) | 闭源 | SD 3 (DiT) | FLUX dev (DiT) | FLUX Fill (DiT) |
| 编辑范式 | In-context diptych | 条件注入 | 条件注入 | 条件注入 | 全量微调 | 上下文填充 |

**与 InstructPix2Pix 系列的本质区别**：InstructPix2Pix 通过修改 UNet 输入通道将原图作为条件注入，需要大量数据训练模型理解编辑指令。ICEdit 则将编辑指令包装为 DiT 已经理解的描述性 prompt，利用 diptych 结构实现图到图的映射，从根本上绕开了指令嵌入空间的 gap 问题。

**与 ACE++ 的对比**：ACE++ 同样基于 FLUX 但使用了 54M 编辑对训练、额外的位置/条件编码器，GPT 分数仅 0.24，远低于 ICEdit 的 0.68。这说明单纯堆数据和参数并不能解决问题，关键在于编辑范式的设计。

**与免训练方法（RF-Solver Edit, StableFlow）的对比**：免训练方法依赖图像反演和注意力操控，需要精心设计的 source/target caption，无法直接使用编辑指令。ICEdit 的 IC prompt 让 DiT 在免训练阶段就能理解指令（GPT 0.24），加上轻微调后达到 0.68，证明了"范式 > 数据量"的设计哲学。

## 启发与关联

1. **"将难题转化为模型已会的任务"的通用思路**：ICEdit 的核心洞察不是教模型学新能力，而是将"指令编辑"这个难题重新表述为"描述性 diptych 生成"这个 DiT 已经擅长的任务。这种思路在 LLM 领域的 prompt engineering 中极为常见（如 Chain-of-Thought），本文将其成功迁移到视觉生成领域。

2. **Rectified Flow 的"早期可判断"特性值得深挖**：作者发现 flow matching 模型在极少步数内就能暴露生成质量，这一特性的潜在应用远超噪声筛选——可用于自动质量控制、主动学习中的样本筛选、甚至训练阶段的课程学习。

3. **MoE-LoRA 作为通用多任务适配方案**：在注意力输出投影层引入路由专家的设计，可推广到其他需要单模型处理多种子任务的场景（如多风格文本生成、多任务视觉理解等）。4 专家 + TopK=1 已足够，说明编辑子任务的差异可被少量专家覆盖。

4. **与 In-Context Learning 的深层联系**：本文的 diptych 范式可视为视觉领域的 few-shot in-context learning——左图作为"示例输入"，右图作为"示例输出"。这与 GPT 系列的 in-context learning 在精神上一脉相承，暗示 DiT 规模足够大后可能涌现出更强的上下文学习能力。

5. **VLM 作为生成模型"裁判"的趋势加速**：用 Qwen-VL-72B 做推理时评估选择，与 RLHF 中用 reward model 做 best-of-N sampling 如出一辙。未来 VLM 评判可以嵌入到更多生成流程中，如自动迭代编辑、生成质量自动过滤等。

## 评分
- 新颖性: ★★★★☆ — In-context diptych 编辑范式新颖且优雅，IC prompt 设计巧妙，但 LoRA-MoE 和推理缩放各有前作基础
- 实验充分度: ★★★★★ — 双 benchmark 评测 + VIE-Score 对标商业模型 + 全面消融（prompt 类型、MoE 配置、数据量、推理缩放参数）+ GPT-4o 人类偏好对齐评估
- 写作质量: ★★★★☆ — 结构清晰，motivation 到方法的推导逻辑自然，图表丰富；但符号使用偶有不一致
- 价值: ★★★★★ — 以极小训练代价达到 SOTA，开源可复现，方法论对社区有广泛启发意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LaVin-DiT: Large Vision Diffusion Transformer](../../CVPR2025/image_generation/lavin-dit_large_vision_diffusion_transformer.md)
- [\[CVPR 2026\] MRT: Masked Region Transformer for Layered Image Generation and Editing at Scale](../../CVPR2026/image_generation/mrt_masked_region_transformer_for_layered_image_generation_and_editing_at_scale.md)
- [\[ICML 2026\] Self-Prompting Diffusion Transformer for Open-Vocabulary Scene Text Editing via In-Context Learning](../../ICML2026/image_generation/self-prompting_diffusion_transformer_for_open-vocabulary_scene_text_editing_via_.md)
- [\[NeurIPS 2025\] CAMILA: Context-Aware Masking for Image Editing with Language Alignment](camila_contextaware_masking_for_image_editing_with_language.md)
- [\[NeurIPS 2025\] UltraHR-100K: Enhancing UHR Image Synthesis with A Large-Scale High-Quality Dataset](ultrahr-100k_enhancing_uhr_image_synthesis_with_a_large-scale_high-quality_datas.md)

</div>

<!-- RELATED:END -->
