---
title: >-
  [论文解读] Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression
description: >-
  [NeurIPS 2025][多模态VLM][delta compression] 提出 UltraDelta——首个无数据 delta 权重压缩流水线，通过方差引导的混合稀疏分配、分布感知压缩和迹范数引导缩放三个组件，在 LLM/NLP/视觉/多模态模型上实现最高 224× 的超高压缩比且性能不降甚至超越微调模型。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "delta compression"
  - "模型剪枝"
  - "量化"
  - "无数据压缩"
  - "多任务部署"
---

# Breaking the Compression Ceiling: Data-Free Pipeline for Ultra-Efficient Delta Compression

**会议**: NeurIPS 2025  
**arXiv**: [2505.13563](https://arxiv.org/abs/2505.13563)  
**代码**: [xiaohuiwang000/UltraDelta](https://github.com/xiaohuiwang000/UltraDelta)  
**领域**: 多模态VLM  
**关键词**: delta compression, 模型剪枝, 量化, 无数据压缩, 多任务部署

## 一句话总结

提出 UltraDelta——首个无数据 delta 权重压缩流水线，通过方差引导的混合稀疏分配、分布感知压缩和迹范数引导缩放三个组件，在 LLM/NLP/视觉/多模态模型上实现最高 224× 的超高压缩比且性能不降甚至超越微调模型。

## 研究背景与动机

- **多模型部署的存储瓶颈**：微调范式下每个下游任务对应一个完整模型副本，多任务部署带来巨大存储开销。Delta compression 只存一个预训练模型 + 压缩后的差值权重，是缓解该问题的有效途径。
- **现有方法的压缩天花板**：剪枝方法（DARE、Magnitude Pruning）在高稀疏率下性能急剧下降；量化方法（BitDelta、Delta-CoMe）受限于 1-bit 精度无法进一步压缩。
- **忽视层间差异**：现有方法对所有层施加统一稀疏率，忽略不同层对模型性能的贡献差异，导致关键信息丢失。
- **破坏层内分布**：激进的量化或剪枝会扭曲层内权重分布形状，而分布形状对性能至关重要。
- **高压缩下稳定性不足**：极端稀疏下标准缩放因子 $1/(1-s)$ 不足以维持模型稳定性，且现有方法依赖数据来调节，限制了实用性。
- **缺乏统一无数据方案**：此前没有方法能同时实现无数据、超高压缩比和强性能，三者之间存在未被突破的 trade-off。

## 方法详解

### 整体框架

UltraDelta 是一个 **混合（剪枝+量化）** 的无数据 delta 压缩流水线，从三个维度（层间、层内、全局）分别设计组件来最小化冗余、最大化信息保留和增强稳定性。流程为：① 计算 delta 权重 $\Delta\theta = \theta_{ft} - \theta_{pre}$ → ② 按方差分配层级稀疏率 → ③ 均匀量化后按值分组剪枝 → ④ 迹范数引导的全局缩放 → ⑤ 推理时重构。

### 关键设计一：Variance-Based Mixed Sparsity Allocation (MSA, 层间)

- **功能**：基于各层 delta 权重的方差，将所有层分为低/中/高方差三组，对高方差层分配更低的稀疏率以保留更多信息。
- **核心思路**：理论推导表明层方差与信息熵正相关——方差越大，信息量越高，在有损压缩下需要更多比特才能维持给定失真度 $R(D) = \frac{1}{2}\log(\sigma^2/D)$。因此高方差层应获得更低的稀疏率。
- **设计动机**：现有统一稀疏策略对所有层"一视同仁"，导致信息密集层被过度剪枝。MSA 以方差为无数据代理指标实现自适应分配，无需校准数据。

### 关键设计二：Distribution-Aware Compression (DAC, 层内)

- **功能**：先做低比特（4-bit）均匀量化，将参数映射到离散值；然后按量化值分组，在每组内独立执行随机剪枝。
- **核心思路**：按值分组后组内随机剪枝能保持各量化值的相对比例不变，从而保留层内分布形状。相比 Magnitude Pruning 优先删小值导致分布严重偏移，DAC 的分布保持特性更优。
- **设计动机**：先前研究表明保留权重分布形状对压缩后性能至关重要。DAC 将量化与剪枝巧妙结合：量化降低比特宽度，分组剪枝进一步减少参数数量，二者协同实现超高压缩比。

### 关键设计三：Trace-Norm-Guided Rescaling (TNGR, 全局)

- **功能**：在标准缩放因子 $1/(1-s)$ 上引入额外因子 $\gamma$，最终缩放为 $\gamma/(1-s)$，其中 $\gamma \in [0.5, 1.0]$ 与 delta 权重的迹范数成反比。
- **核心思路**：理论分析表明激活误差方差为 $\text{Var}(\varepsilon) = \frac{\gamma^2 s}{1-s} \odot a^2$，在高稀疏下方差爆炸。通过引入 $\gamma < 1$ 可有效抑制误差。实验发现迹范数大的 delta 权重需要更小的 $\gamma$，且对 $\gamma$ 更敏感。
- **设计动机**：DARE 的标准缩放在极端稀疏（≥95%）下会导致性能不稳定。TNGR 利用迹范数作为无数据的自适应估计指标，无需通过数据搜索最优缩放因子。

## 损失函数/训练策略

UltraDelta 是 **纯后处理方法**，不涉及训练或微调，完全无数据（data-free）。压缩流程仅依赖 delta 权重本身的统计量（方差、迹范数），推理时通过 $\theta^{final} = \theta_{pre} + \frac{\gamma}{1-s} \cdot \hat{\Delta\theta}^*$ 重构完整模型。压缩后的稀疏 delta 权重使用 Golomb 编码存储零游程长度以实现高效编码。

## 实验关键数据

### 表1：大语言模型 LLaMA-2 系列（3个任务平均）

| 方法 | 压缩比(7B) | 压缩比(13B) | Avg(7B) | Avg(13B) |
|:---|:---:|:---:|:---:|:---:|
| Fine-tuned | 1× | 1× | 45.37 | 50.94 |
| BitDelta | 16× | 16× | 41.89 | 48.42 |
| Delta-CoMe | 16× | 16× | 42.52 | 48.71 |
| DARE (同压缩比) | 31.7× | 50.1× | 33.97 | 40.23 |
| MP (同压缩比) | 31.7× | 50.1× | 27.02 | 15.36 |
| **UltraDelta** | **32.9×** | **50.9×** | **45.57** | **52.05** |

UltraDelta 在 32.9×/50.9× 压缩比下**超越微调模型**本身（45.57 vs 45.37, 52.05 vs 50.94），暗示压缩可能引入正则化效益。

### 表2：T5-base 与 ViT-L/14 的极端压缩

| 模型 | 方法 | 压缩比 | 平均性能 |
|:---|:---|:---:|:---:|
| T5-base (8 NLP任务) | Fine-tuned | 1× | 86.37 |
| | BitDelta | 16× | 84.68 |
| | DARE (同压缩比) | 220.5× | 84.30 |
| | **UltraDelta** | **224.6×** | **86.74** |
| ViT-L/14 (8 视觉任务) | Fine-tuned | 1× | 94.4 |
| | BitDelta | 16× | 94.1 |
| | DARE (同压缩比) | 127.6× | 89.1 |
| | **UltraDelta** | **132.5×** | **94.4** |

T5-base 上 224.6× 压缩仍超越微调模型；ViT-L/14 上 132.5× 压缩实现**完全无损**性能。

### 表3：消融实验（ViT-B/32, 8任务）

| 配置 | 压缩比 | 平均准确率 |
|:---|:---:|:---:|
| DARE (97% sparsity) | 23.7× | 89.7 |
| + DAC | 50.9× (↑27.2×) | 89.7 |
| + DAC + MSA | 50.9× | 90.3 (↑0.6) |
| + DAC + MSA + TNGR | 50.9× | 90.7 (↑0.4) |

DAC 在保持性能的同时将压缩比翻倍；MSA 和 TNGR 分别贡献稳定的性能提升。

## 亮点与洞察

- **压缩后反超原模型**：在 LLaMA-2 和 T5-base 上实现了"压缩 > 微调"的现象，作者推测可能是压缩带来的正则化效应，这一观察非常有启发性。
- **理论驱动的设计**：MSA 基于信息论（方差-熵关联 + 率失真理论），TNGR 基于激活误差方差分析，不是纯经验调参。
- **极致的无数据约束**：完全不需要校准数据或验证数据，仅靠 delta 权重自身统计量（方差、迹范数），比 BitDelta、DAREx 等方法更加实用。
- **跨模态一致性**：同一流水线在 LLM、NLP、视觉、多模态四类模型上均有效，泛化能力突出。

## 局限性

- **分类为三组的设计较粗糙**：MSA 将层均分为低/中/高三组，分组粒度和组数的选择缺乏自适应机制。
- **$\gamma$ 的设定依赖经验**：TNGR 中 $\gamma$ 与迹范数的反比关系是启发式的，缺乏严格的最优性保证。
- **仅压缩线性层**：为公平比较，所有方法仅压缩 Transformer 中的线性层，对 embedding / normalization 层未做处理。
- **未涉及推理加速**：工作专注存储压缩，未讨论稀疏+量化后的实际推理加速收益。

## 相关工作与启发

- **DARE**：随机剪枝 + $1/(1-s)$ 缩放的开创性工作，UltraDelta 在其基础上全面改进（分布感知剪枝、自适应稀疏、改进缩放）。
- **BitDelta / Delta-CoMe**：1-bit 量化路线的代表，压缩比上限为 16×，被 UltraDelta 大幅超越。
- **DeltaZip**：混合方法先驱但信息保留不足。UltraDelta 的 DAC 通过按值分组剪枝避免了分布畸变。
- **启发**：delta 权重的"可压缩性"远超预期，关键在于理解并利用 delta 权重的统计结构（方差分布、迹范数特性）而非简单粗暴剪枝。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个无数据超高压缩 delta 流水线，三个组件各有理论支撑，MSA+DAC+TNGR 组合新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 LLM(7B/13B+新架构)、NLP、视觉、多模态四大类，含消融和分布分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，理论推导与实验分析并重，图表丰富
- **价值**: ⭐⭐⭐⭐ — 对多模型部署有直接实用价值，无数据约束使其即插即用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](../../AAAI2026/multimodal_vlm/llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[ICML 2026\] DCER: Robust Multimodal Fusion via Dual-Stage Compression and Energy-Based Reconstruction](../../ICML2026/multimodal_vlm/dcer_dual-stage_compression_and_energy-based_reconstruction.md)
- [\[CVPR 2026\] VLIC: Vision-Language Models As Perceptual Judges for Human-Aligned Image Compression](../../CVPR2026/multimodal_vlm/vlic_vision-language_models_as_perceptual_judges_for_human-aligned_image_compres.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)

</div>

<!-- RELATED:END -->
