---
title: >-
  [论文解读] AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs
description: >-
  [AAAI 2026 Oral][多模态VLM][visual abductive reasoning] 受人类认知中"语言溯因+图像想象"双模式启发，提出 AbductiveMLLM，通过 Reasoner（因果对比学习筛选假设）和 Imaginer（扩散模型图像化推理）两个协同组件增强 MLLM 的视觉溯因推理能力，在 VAR 和 YouCookII 基准上取得 SOTA。
tags:
  - "AAAI 2026 Oral"
  - "多模态VLM"
  - "visual abductive reasoning"
  - "MLLM"
  - "扩散模型"
  - "对比学习"
  - "pictorial thinking"
---

# AbductiveMLLM: Boosting Visual Abductive Reasoning Within MLLMs

**会议**: AAAI 2026 Oral  
**arXiv**: [2601.02771](https://arxiv.org/abs/2601.02771)  
**代码**: [https://github.com/ChangPtR/AbdMLLM](https://github.com/ChangPtR/AbdMLLM)  
**领域**: 多模态VLM  
**关键词**: visual abductive reasoning, MLLM, diffusion model, contrastive learning, pictorial thinking

## 一句话总结

受人类认知中"语言溯因+图像想象"双模式启发，提出 AbductiveMLLM，通过 Reasoner（因果对比学习筛选假设）和 Imaginer（扩散模型图像化推理）两个协同组件增强 MLLM 的视觉溯因推理能力，在 VAR 和 YouCookII 基准上取得 SOTA。

## 研究背景与动机

视觉溯因推理（VAR）要求 AI 从不完整的视觉观测中推断最可能的解释，是人类认知的核心能力之一。当前面临的核心问题：

1. **MLLM 溯因能力不足**：虽然 GPT-4o 等 MLLM 在 VQA 等任务上表现优秀，但在因果推理方面与人类存在显著差距——GPT-4o-mini 在 VAR 上 CIDEr 仅 7.30，远低于人类的 147.79
2. **现有方法局限**：传统小模型（REASONER、UPD-Trans）专注于语言推理，忽略了人类认知中的"图像化思维"——人类不仅能用语言推理，还能在脑中**想象**可能的场景
3. **核心切入点**：模拟人类认知中语言推理（verbal abduction）与图像想象（pictorial abduction）的协同作用

## 方法详解

### 整体框架

AbductiveMLLM 包含两个端到端联合训练的组件：
1. **Reasoner**（语言域）：用盲 LLM 生成候选假设 → 因果对比学习筛选 → 作为先验指导 MLLM 推理
2. **Imaginer**（图像域）：基于 Stable Diffusion 的扩散模型，利用 Reasoner 的输出嵌入和视觉观测生成"想象"场景，反哺语言推理

任务定义：给定视频序列 $\mathcal{V}=\{O_1,\dots,O_{t-1},H,O_t,\dots,O_{T-1}\}$，其中 $H$ 是未观测事件，目标是推断 $H$ 的最可能语言解释 $E_h$。

### 关键设计

**设计一：因果感知假设生成与筛选（CHG）**

分两步实现：

**Step 1 - 候选假设生成**：用预训练 MLLM 为每个观测事件生成视频描述 $\mathcal{C}=\{C_t\}_{t=1}^{T-1}$，然后以高温度（1.4）多次提示 GPT-4o-mini 生成 $L$ 个多样化候选假设 $\mathcal{Y}=\{Y_i\}_{i=1}^{L}$。

**Step 2 - 因果对比学习筛选**：将视频序列分为初始段 $\mathcal{I}$、过程段 $\mathcal{P}$、终结段 $\mathcal{F}$，通过视觉编码器 $\Phi_V$ 和文本编码器 $\Phi_T$ 映射到联合因果空间。训练使用 NT-Xent 损失：

$$\mathcal{L}_{\text{Contrast}}=-\log\frac{\exp(\langle \boldsymbol{X}_{\mathcal{I}}+\boldsymbol{X}_{\mathcal{P}}^{+}, \boldsymbol{X}_{\mathcal{F}}\rangle/\tau)}{\sum_{i=1}^{M}\exp(\langle \boldsymbol{X}_{\mathcal{I}}+\boldsymbol{X}_{\mathcal{P}}^{i-,+}, \boldsymbol{X}_{\mathcal{F}}\rangle/\tau)}$$

推理时对每个候选假设计算因果相关度分数 $\text{Score}(Y_i)=\langle \boldsymbol{X}_{\mathcal{I}}+\boldsymbol{X}_{Y_i}, \boldsymbol{X}_{\mathcal{F}}\rangle$，选取 top-$k$（$k=3$）假设。与标准对比学习的本质区别在于：这里的"正样本"基于因果关系而非表面相似性——即使假设与视频内容相似但因果不成立，也会被排除。

**设计二：Imaginer 扩散模型的图像化推理**

在 Stable Diffusion 的 U-Net 中引入三种轻量级适配器：

1. **V-Adapter（视觉交叉注意力）**：注入观测视频的视觉先验。采用局部-全局混合表示：
    - 局部表示：用 CLIP 计算每帧与解释 $E_h$ 的相似度 $\gamma^i$，保留高分帧拼接为 $\boldsymbol{c}_{local}$
    - 全局表示：加权平均 $\boldsymbol{c}_{global}=\sum_{i=1}^{N}\gamma^i \boldsymbol{c}_v^i$
    - 交叉注意力：$\text{V-Adapter}(\boldsymbol{Q},\boldsymbol{K}_v,\boldsymbol{V}_v)=\text{Softmax}(\frac{\boldsymbol{Q}\boldsymbol{K}_v^{\top}}{\sqrt{d_k}})\boldsymbol{V}_v$

2. **T-Adapter（时间卷积）**：建模帧间时间依赖，使用深度可分离 3D 卷积：$\text{T-Adapter}(\boldsymbol{x})=\boldsymbol{x}+\text{Conv3D}_{up}(\text{Conv3D}_{down}(\boldsymbol{x}))$

3. **F-Adapter（FFN 适配器）**：增强空间表示，与 FFN 并行：$\text{F-Adapter}(\boldsymbol{x})=\boldsymbol{x}+\text{FC}_{up}(\text{GELU}(\text{FC}_{down}(\boldsymbol{x})))$

**设计三：两阶段端到端训练**

- **Stage I**：分别训练——MLLM 用 LoRA 微调（$\mathcal{L}_{CE}$），Imaginer 冻结 SD 权重只训练适配器（$\mathcal{L}_{Diffusion}$），加 Min-SNR 加权策略
- **Stage II**：联合端到端微调——$\mathcal{L}=\mathcal{L}_{CE}+\alpha\mathcal{L}_{Diffusion}$，其中 $\alpha=5$

### 损失函数 / 训练策略

总损失：$\mathcal{L}=\mathcal{L}_{CE}+\alpha\mathcal{L}_{Diffusion}$，$\alpha=5$ 时效果最佳。Stage I 训练 2 个 epoch，对比学习模块训练 10 个 epoch（每个正样本 100 个难负例），Stage II 联合微调 1 个 epoch。使用 4 张 A800 80GB GPU。

## 实验关键数据

### 主实验

VAR 测试集结果：

| 方法 | BLEU@4 | METEOR | ROUGE | CIDEr | BERT-S |
|------|--------|--------|-------|-------|--------|
| Human | 11.35 | 19.36 | 36.92 | 147.79 | 40.59 |
| REASONER | 3.44 | 9.05 | 22.89 | 30.75 | 30.64 |
| UPD-Trans | 5.40 | 11.16 | 25.62 | 41.66 | 30.80 |
| GPT-4o-mini | 0.63 | 7.38 | 13.64 | 7.30 | 12.27 |
| Qwen2VL-7B | 2.41 | 11.29 | 21.61 | 29.25 | 30.01 |
| Qwen2VL-7B (FT) | 5.67 | 12.77 | 27.11 | 50.82 | 36.03 |
| **AbductiveMLLM** | **6.54** | **13.41** | **27.95** | **57.04** | **36.80** |

YouCookII 测试集结果：

| 方法 | BLEU@4 | METEOR | ROUGE | CIDEr | BERT-S |
|------|--------|--------|-------|-------|--------|
| REASONER | 3.54 | 9.47 | 24.62 | 32.99 | 23.19 |
| Qwen2VL-7B (FT) | 5.66 | 12.62 | 28.64 | 68.44 | 29.09 |
| **AbductiveMLLM** | **6.16** | **13.46** | **30.06** | **77.70** | **30.77** |

### 消融实验

核心组件消融（VAR 测试集）：

| CHG | Imaginer | BLEU@4 | METEOR | ROUGE | CIDEr | BERT-S |
|-----|----------|--------|--------|-------|-------|--------|
| ✗ | ✗ | 5.67 | 12.77 | 27.11 | 50.82 | 36.03 |
| ✓ | ✗ | 6.33 | 12.96 | 27.21 | 53.60 | 36.31 |
| ✗ | ✓ | 6.35 | 13.07 | 27.52 | 55.00 | 36.40 |
| ✓ | ✓ | **6.54** | **13.41** | **27.95** | **57.04** | **36.80** |

Imaginer 适配器消融：

| 变体 | CIDEr | BERT-S |
|------|-------|--------|
| 完整模型 | 57.04 | 36.80 |
| 去掉 V-Adapter | 54.51 | 36.68 |
| 去掉 T-Adapter | 54.99 | 36.68 |
| 去掉 F-Adapter | 54.52 | 36.63 |

Top-$k$ 假设数量：$k=3$ 最优（CIDEr 57.04），$k=10$ 时下降至 53.66。

### 关键发现

- CHG 和 Imaginer 各自独立贡献约 +2.78 和 +4.18 CIDEr，联合使用达到 +6.22
- Imaginer（图像化推理）在语义指标（METEOR/ROUGE）上贡献更大，说明视觉想象能丰富语言表达
- 即使最强 MLLM（Qwen2VL-7B FT）也远低于人类表现（57.04 vs 147.79 CIDEr）
- $\alpha$ 系数在 1-9 范围内变化不敏感，模型鲁棒性好

## 亮点与洞察

- 首次将"图像化思维"引入视觉溯因推理，模拟人类双模式认知
- 因果对比学习（而非表面相似性匹配）是筛选假设的关键，捕捉了从前提到结论的因果链
- 扩散模型不是为了生成高质量图像，而是作为推理引导——潜空间的去噪损失迫使模型收敛到视觉上合理的结果
- 轻量级适配器设计（V/T/F-Adapter）使得在 Stable Diffusion 上做视频推理成为可能

## 局限与展望

- 与人类表现差距仍然巨大（CIDEr 57.04 vs 147.79），说明溯因推理依然是 AI 的重大挑战
- Imaginer 基于 SD-v1-4（256×256 分辨率），升级到更强的生成模型可能进一步提升
- 假设生成依赖 GPT-4o-mini，受限于其知识和推理能力
- 仅在两个数据集上验证，泛化性有待进一步检验

## 相关工作与启发

- **vs REASONER**: REASONER 是传统小模型+因果解码器，AbductiveMLLM 用 MLLM+扩散模型实现语言和图像双模式推理，CIDEr 从 30.75 提升至 57.04
- **vs UPD-Trans**: UPD-Trans 引入概率蒸馏但仍限于语言推理，AbductiveMLLM 通过 Imaginer 补充图像化思维，全面超越（+15.38 CIDEr）
- **vs KN-VLM**: KN-VLM 引入外部知识库增强，但仍是传统架构；AbductiveMLLM 利用 MLLM 的内在知识和生成模型的想象能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将图像化思维引入 VAR，Reasoner+Imaginer 双模式设计有创新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、完整消融（组件/假设数/系数/适配器），分析详尽
- 写作质量: ⭐⭐⭐⭐ 从人类认知出发的动机清晰，方法阐述详细
- 价值: ⭐⭐⭐⭐ 扩散模型作为推理引导而非生成器的思路值得推广，但与人类差距仍大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](astar_boosting_multimodal_reasoning_with_automated_structure.md)
- [\[CVPR 2026\] Boosting Reasoning in Large Multimodal Models via Activation Replay](../../CVPR2026/multimodal_vlm/boosting_reasoning_in_large_multimodal_models_via_activation_replay.md)
- [\[CVPR 2026\] Boosting Visual Reprogramming for CLIP with Dual Granularity Alignment](../../CVPR2026/multimodal_vlm/boosting_visual_reprogramming_for_clip_with_dual_granularity_alignment.md)
- [\[CVPR 2026\] POINTS-Long: Adaptive Dual-Mode Visual Reasoning in MLLMs](../../CVPR2026/multimodal_vlm/points-long_adaptive_dual-mode_visual_reasoning_in_mllms.md)
- [\[CVPR 2026\] Boosting Document Parsing Efficiency and Performance with Coarse-to-Fine Visual Processing](../../CVPR2026/multimodal_vlm/boosting_document_parsing_efficiency_and_performance_with_coarse-to-fine_visual_.md)

</div>

<!-- RELATED:END -->
