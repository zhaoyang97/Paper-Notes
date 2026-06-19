---
title: >-
  [论文解读] Cross-Lingual Optimization for Language Transfer in Large Language Models
description: >-
  [ACL 2025][多语言/翻译][cross-lingual transfer] 提出 Cross-Lingual Optimization (CLO)，通过修改 DPO 损失函数实现跨语言偏好优化——给目标语言输入时偏好目标语言回复、给英语输入时偏好英语回复——在 5 个模型 × 6 种语言上一致超越 SFT，低资源语言中仅 3,200 样本的 CLO 即超越 6,400 样本的 SFT。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "cross-lingual transfer"
  - "DPO"
  - "multilingual"
  - "low-resource language"
  - "language adaptation"
  - "注意力机制"
---

# Cross-Lingual Optimization for Language Transfer in Large Language Models

**会议**: ACL 2025  
**arXiv**: [2505.14297](https://arxiv.org/abs/2505.14297)  
**代码**: 无  
**领域**: 多语言翻译  
**关键词**: cross-lingual transfer, DPO, multilingual, low-resource language, language adaptation, attention-only tuning

## 一句话总结

提出 Cross-Lingual Optimization (CLO)，通过修改 DPO 损失函数实现跨语言偏好优化——给目标语言输入时偏好目标语言回复、给英语输入时偏好英语回复——在 5 个模型 × 6 种语言上一致超越 SFT，低资源语言中仅 3,200 样本的 CLO 即超越 6,400 样本的 SFT。

## 研究背景与动机

**领域现状**：英语为中心的 LLM（Llama、Mistral、Qwen 等）在其他语言上表现显著较差，标准做法是用目标语言指令数据做 SFT 进行语言迁移。

**现有痛点**：SFT 在数据稀缺场景下存在严重的"英语偏向"问题——模型可能理解了目标语言的输入，但仍然默认用英语回复（如 Llama3 Chat 可以理解斯瓦希里语但用英语作答）。对中低资源语言，SFT 的效果尤其不稳定。

**核心矛盾**：如何在保留模型英语能力的同时有效迁移到目标语言？SFT 会导致两难：英语数据过多则目标语言迁移不足，目标语言数据过多则英语能力退化，且低资源语言往往没有足够的高质量指令数据。

**本文目标**：用可获取的英语 SFT 数据 + 翻译模型，在数据受限环境下实现高效跨语言迁移，同时保持英语性能。

**切入角度**：将问题从"学习目标语言知识"转化为"学习输入输出语言的对应关系"——抑制"目标语言输入→英语回复"，增强"目标语言输入→目标语言回复"，反之亦然。

**核心 idea**：用跨语言偏好对（同一问题的目标语言回复为 chosen、英语回复为 rejected）结合修改版 DPO 训练模型"用对的语言回复"。

## 方法详解

### 整体框架

英语 SFT 数据 $(x_{en}, y_{en})$ → 翻译模型（M2M100-1.2B）生成目标语言数据 $(x_\ell, y_\ell)$ → 构建跨语言偏好对 → CLO 损失 = $\lambda \cdot \mathcal{L}_{SFT} + (1-\lambda) \cdot \mathcal{L}_{CL}$ → 仅微调 attention 层参数。

### 关键设计

#### 1. 跨语言数据构建（Cross-Lingual Dataset Preparation）

- **功能**：将英语 SFT 数据翻译为目标语言，构建两组偏好对
- **为什么**：让模型显式学习"输入语言-输出语言"的对应关系，而非仅学内容
- **怎么做**：
    - 英语输入 $x_{en}$：chosen = $y_{en}$（英语回复），rejected = $y_\ell$（目标语言回复）→ 保持英语能力
    - 目标语言输入 $x_\ell$：chosen = $y_\ell$（目标语言回复），rejected = $y_{en}$（英语回复）→ 迁移目标语言能力
    - 使用 M2M100-1.2B 翻译模型，从 6,400 条英语数据生成共 12,800 条跨语言对

#### 2. CLO 损失函数（Cross-Lingual Optimization Loss）

- **功能**：结合 NLL 和修改版 DPO 的联合优化目标
- **为什么**：基座模型本身不会回答查询，NLL 教会模型生成回复；修改版 DPO 教会模型选择正确语言
- **怎么做**：
    - $\mathcal{L}_{CLO} = \lambda \cdot \mathcal{L}_{SFT} + (1-\lambda) \cdot \mathcal{L}_{CL}$
    - $\mathcal{L}_{SFT}$：仅在**目标语言输出**上计算 NLL（关键设计：排除英语 NLL 以缓解英语偏差）
    - $\mathcal{L}_{CL}$：跨语言 DPO 损失，由两部分组成——英语输入时偏好英语回复抑制目标语言回复 + 目标语言输入时偏好目标语言回复抑制英语回复
    - 消融实验验证：加入英语 NLL 会导致模型偏向英语输出（斯瓦希里语 Win Rate 从 83.0 降至 74.5）

#### 3. 仅 Attention 层微调

- **功能**：CLO 训练中仅更新 attention 层参数，冻结其余层
- **为什么**：基于 Zeping & Sophia (2024) 的发现，语言相关能力主要存储在 attention 层，重要神经元集中在深层
- **怎么做**：仅对 Q/K/V/O 投影矩阵做梯度更新
- **效果**：与全参数微调效果相当（中文/韩语 Win 率约 50:50），但训练 GPU 显存比 DPO 低约 30%，仅比 SFT 高约 55%
- **例外**：斯瓦希里语等极低资源语言中 attention-only 训练性能显著低于全参数训练（Win 29.7% vs 70.3%），因为模型内嵌的语言知识不足

### 损失函数 / 训练策略

- 英语种子数据：OpenAssistant 前 6,400 条高排名单轮数据
- 翻译模型：M2M100-1.2B
- 训练目标：$\mathcal{L}_{CLO}$ 联合损失
- 参考模型：基座模型自身（与 DPO 一致）
- 可训练参数：仅 attention 层

## 实验关键数据

### 主实验 1：AlpacaEval 指令跟随能力（Win Rate %，vs SFT 基线）

| 模型 | 中文(高) | 德语(高) | 韩语(中) | 印尼语(中) | 斯瓦希里语(低) | 约鲁巴语(低) |
|------|---------|---------|---------|-----------|-------------|-------------|
| Llama3-8B CLO Δ | +11.3 | +2.9 | +15.3 | +1.5 | **+17.6** | +11.6 |
| Llama2-7B CLO Δ | +2.0 | +9.2 | +1.3 | +5.0 | +0.9 | **+23.6** |
| Llama2-13B CLO Δ | +5.9 | +3.2 | +2.3 | +9.3 | **+17.0** | **+23.8** |
| Mistral-7B CLO Δ | +0.6 | +0.9 | +2.0 | +1.1 | **+15.9** | +0.2 |
| Qwen2.5-3B CLO Δ | +5.7 | +1.0 | +10.1 | +1.1 | **+23.0** | **+23.0** |

**关键发现**：CLO 在所有 30 个 (模型, 语言) 组合中均超越 SFT+DPO，低资源语言提升最显著（Δ 最高达 +23.8）。英语性能同步保持甚至提升。

### 主实验 2：BELEBELE 阅读理解准确率

| 模型 | 方法 | 斯瓦希里语 | 约鲁巴语 | 韩语 |
|------|------|-----------|---------|------|
| Llama3-8B | SFT | 42.0 | 29.6 | 46.7 |
| Llama3-8B | CLO | 42.6 | 29.8 | **57.7** |
| Qwen2.5-3B | SFT | 51.7 | 45.9 | 52.3 |
| Qwen2.5-3B | CLO | **74.7** | **68.9** | **62.4** |

### 主实验 3：MMMLU 推理能力

| 模型 | 方法 | 中文 | 韩语 | 斯瓦希里语 | 英语 |
|------|------|------|------|-----------|------|
| Llama3-8B | SFT | 39.36 | 25.31 | 27.59 | 53.00 |
| Llama3-8B | CLO | **41.99** | **32.73** | **33.38** | **57.55** |
| Qwen2.5-3B | SFT | 46.34 | 35.90 | 26.22 | 55.70 |
| Qwen2.5-3B | CLO | **52.10** | **41.94** | **29.80** | **60.52** |

### 消融实验

| 消融配置 | 斯瓦希里语 Win Rate | 英语 Win Rate |
|---------|-------------------|--------------|
| CLO (NLL 仅目标语言, ours) | **83.0** | 65.4 |
| CLO (NLL 目标+英语) | 74.5 | 67.8 |

| Attention-only vs Full (Llama2) | 目标语言 Win% | 英语 Win% |
|------|---------|---------|
| 中文 | 50.7 vs 49.1 | 54.4 vs 45.6 |
| 韩语 | 52.7 vs 46.4 | 52.7 vs 47.3 |
| 斯瓦希里语 | 29.7 vs **69.6** | 50.1 vs 49.9 |

### 数据效率实验

| 训练数据量 | SFT 斯瓦希里语 | CLO 斯瓦希里语 |
|-----------|--------------|--------------|
| 1,600 对 | 远低于 6,400 SFT | ≈ 6,400 SFT |
| 3,200 对 | 仍低于 6,400 SFT | **> 6,400 SFT** |
| 6,400 对 | 基线 | 远超基线 |

**关键结论**：CLO 在斯瓦希里语中仅用 1,600 对data即可匹配 SFT 使用 6,400 对的效果，数据效率提升 4 倍。

### 关键发现总结

- **CLO 在所有语言和模型上一致优于 SFT**，低资源语言优势最大（Δ 最高 +23.8%）
- **数据效率**：CLO 3,200 > SFT 6,400（低资源）；CLO 400 ≈ SFT 6,400（Llama3 斯瓦希里语）
- **SFT 对低资源语言数据量极敏感**（3,200→6,400 有巨大跳变），CLO 则平滑提升
- **CLO 同时保持甚至提升英语能力**，而 SFT 在加入目标语言数据后英语能力常退化
- **NLL 仅算目标语言**是关键设计，加入英语 NLL 会重新引入英语偏差

## 亮点与洞察

- **跨语言偏好对设计**非常优雅——将"语言选择"问题转化为经典的偏好优化问题，利用翻译数据构建"语言正确性"偏好信号，无需任何人工标注
- **仅微调 attention 层**是一个实用的发现，验证了语言能力主要存储在 attention 层的假说，大幅降低训练成本
- **低资源语言的数据效率提升**对实际多语言部署有直接价值——很多语言连 1 万条高质量指令数据都难以获取
- **"SFT 的英语偏向"的系统性论证**：通过 SFT-eng、SFT-tgt、SFT 三个变体对比，清晰展示了 SFT 在中低资源语言中的失效模式
- **Qwen2.5-3B 虽是最小模型但多语言迁移效果最好**，说明预训练的多语言数据覆盖比模型大小更重要

## 局限与展望

- **仅支持单语言迁移**：每次只能迁移到一种目标语言，不支持同时多语言迁移
- **依赖翻译模型质量**：M2M100 翻译质量对低资源语言可能不佳，但论文论证 SFT 和 CLO 共享相同翻译数据，比较是公平的
- **语言特定评估不足**：评估用的是翻译的 AlpacaEval/MMMLU，无法捕捉语言特异性文化语境
- **仅在 DPO 上验证**：未探索 CLO 框架与其他偏好优化算法（KTO、SimPO 等）的兼容性
- **Attention-only 对极低资源语言失效**：斯瓦希里语中 attention-only 训练大幅落后全参数训练，需要自适应策略
- **改进方向**：多语言同时迁移、与 KTO/SimPO 结合、自适应层选择策略、更强的翻译模型

## 相关工作与启发

- **vs 标准 SFT（Lee et al. 2023; Shaham et al. 2024a）**：SFT 单纯通过最大似然模仿目标语言输出，无法显式学习语言选择策略；CLO 通过偏好优化教模型"选对语言"，在低资源场景优势显著
- **vs 继续预训练 + SFT（Cui et al. 2023; Zhao et al. 2024a）**：继续预训练需要大量目标语言语料（通常百万级），CLO 仅需 6,400 条翻译数据，成本差距巨大
- **vs InstructionCP（Chen & Lee 2024）**：InstructionCP 需要大规模目标语言指令数据和复杂架构分析，CLO 仅需英语数据+翻译模型，且方法更简洁

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 DPO 创新性应用于跨语言迁移，"语言选择偏好"的视角新颖；但核心技术（修改 DPO + NLL）并不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 5 模型 × 6 语言 × 3 评估基准，消融充分（NLL 变体、attention-only、数据量曲线），实验规模在该领域属上乘
- 写作质量: ⭐⭐⭐⭐ 两个假设清晰，方法推导严谨，图示直观；但 Limitations 部分写得过于冗长
- 价值: ⭐⭐⭐⭐⭐ 对低资源语言 LLM 部署有直接实用价值，方法简洁易复现，仅需公开英语数据+翻译模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models](crosslingual_pitfalls.md)
- [\[ACL 2025\] Bridging the Language Gaps in Large Language Models with Inference-Time Cross-Lingual Intervention](bridging_the_language_gaps_in_large_language_models_with_inference-time_cross-li.md)
- [\[ACL 2025\] Disentangling Language and Culture for Evaluating Multilingual Large Language Models](disentangle_language_culture.md)

</div>

<!-- RELATED:END -->
