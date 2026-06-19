---
title: >-
  [论文解读] Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation
description: >-
  [AAAI 2026][模型压缩][多域机器翻译] 提出 CANEFT，通过互信息（MI）识别 LLM 中跨域一致对齐的神经元（consensus-aligned neurons），仅微调这些神经元即可实现多域机器翻译的高效适应，在 3 个 LLM、10 个翻译域上超越 LoRA 等 PEFT 基线，且无需额外参数。
tags:
  - "AAAI 2026"
  - "模型压缩"
  - "多域机器翻译"
  - "神经元选择"
  - "互信息"
  - "参数高效微调"
  - "LLM"
---

# Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation

**会议**: AAAI 2026  
**arXiv**: [2602.05694](https://arxiv.org/abs/2602.05694)  
**代码**: [GitHub](https://github.com/fortunatekiss/CANEFT)  
**领域**: 多语言翻译  
**关键词**: 多域机器翻译, 神经元选择, 互信息, 参数高效微调, LLM

## 一句话总结
提出 CANEFT，通过互信息（MI）识别 LLM 中跨域一致对齐的神经元（consensus-aligned neurons），仅微调这些神经元即可实现多域机器翻译的高效适应，在 3 个 LLM、10 个翻译域上超越 LoRA 等 PEFT 基线，且无需额外参数。

## 研究背景与动机

### 领域现状

**领域现状**：领域现状**：多域机器翻译（MDMT）要求单一模型覆盖法律、医学、字幕等多个领域。LLM 的通用翻译能力强但域适应仍是挑战。

**现有痛点**：

### 现有痛点

**现有痛点**：LoRA 在多域微调时存在参数干扰——适应某域后其他域性能下降

### 核心矛盾

**核心矛盾**：Adapter 方法为每个域引入独立模块——域数增长时参数和内存成本线性增加

### 解决思路

**解决思路**：ICL 依赖高质量域内示例，MDMT 效果不佳

**核心矛盾**：能否找到不引入额外参数、不产生域间干扰的 PEFT 方法？

**切入角度**：受神经科学启发——共识性对话增强群体成员间的神经对齐，类比到 LLM 中应存在跨域一致编码翻译知识的神经元。找到并仅微调这些神经元。

**核心 idea**：用互信息衡量神经元重要性与域标签的关联，选择所有域上 MI 都高的"共识对齐神经元"，仅微调这些神经元。

## 方法详解

### 整体框架
三步流程：(1) 通过激活-梯度分析识别 MDMT 任务相关神经元 (2) 用互信息选择跨域共识对齐神经元 (3) 仅微调这些神经元的参数

### 关键设计

1. **任务相关神经元识别**:

    - 功能：找出 FFN 层中与翻译任务相关的神经元
    - 核心思路：神经元重要性 $I_{l,j}^{(d)} = \mathbb{E}[|A_{l,j}^{(d)} \cdot G_{l,j}^{(d)}|]$（激活值×梯度的绝对值）
    - 理论支撑：通过一阶 Taylor 展开证明该指标近似于去掉该神经元后的损失变化

2. **互信息选择共识神经元**:

    - 功能：从任务相关神经元中选出跨域一致重要的子集
    - 核心思路：将重要性分数离散化后计算神经元与域标签的互信息 $MI_{l,j}$
    - 选择标准：$\mathcal{N}_{MDCA} = \{(l,j) | \min MI_{l,j} \geq \gamma\}$——在所有域上 MI 都达到阈值
    - 设计动机：不选仅在某些域上重要的神经元（会导致域偏），而选跨域一致重要的

3. **神经元高效微调**:

    - 构建二值 mask $M$，仅允许共识神经元对应参数的梯度通过
    - $\nabla W_m \leftarrow \nabla W_m \odot M$
    - 覆盖 FFN 的 up/down/gate 三个投影矩阵
    - 不引入任何额外参数——比 LoRA 更轻量

### 损失函数 / 训练策略
标准翻译 CE 损失 + 梯度 mask。在 LLaMA2-7B-Chat、Qwen2.5-7B、Gemma2-9B 上验证。

## 实验关键数据

### 主实验（De→En，5个域）

| 方法 | 可训参数 | IT BLEU | Law BLEU | Med BLEU | 平均 |
|------|---------|---------|----------|----------|------|
| Base (零样本) | 0 | 30.0 | 44.1 | 35.5 | 32.8 |
| Full FT | 7B | 47.9 | 47.1 | 35.3 | 38.9 |
| LoRA | ~20M | 35.8 | 49.9 | 33.5 | 36.2 |
| **CANEFT** | 极少 | **最优** | **最优** | **最优** | **+1.3 BLEU** |

CANEFT 在已见和未见域上都超越 Full FT 和 LoRA。

### 关键发现
- LoRA 的参数干扰确实存在——Medical 域训练后 IT 域明显下降
- 仅微调共识神经元不产生域间干扰，且泛化到未见域
- MI-based 选择比纯梯度/激活选择更好——因为后者容易选到域特异性神经元
- 在 3 个 LLM 上效果一致，方法对模型架构不敏感

## 亮点与洞察
- **"共识对齐"的概念**受神经科学启发，类比优雅——群体共识对话→跨域共识神经元
- **不引入额外参数的 PEFT**——比 LoRA/Adapter 更轻量，理论上也更不容易干扰
- **MI 选择标准的设计**——"所有域上都重要"而非"任何域上重要"，有效避免域偏

## 局限与展望
- 神经元识别需要所有域的数据做前向/反向传播——初始化成本不低
- 阈值 $\gamma$ 需要调参
- 仅在机器翻译上验证，其他多任务场景有待探索
- 仅微调 FFN 神经元，注意力层未涉及

## 相关工作与启发
- **vs LoRA**: LoRA 有参数干扰问题，CANEFT 通过神经元选择避免；且 CANEFT 无额外参数
- **vs Adapter**: Adapter 每域独立模块，域数增长开销大；CANEFT 统一一组共识神经元
- **vs 语言特异神经元**: LAPE 等找语言特异神经元，CANEFT 找跨域共识神经元——目标相反

## 评分
- 新颖性: ⭐⭐⭐⭐ MI-based 共识神经元选择是新颖的 PEFT 范式
- 实验充分度: ⭐⭐⭐⭐⭐ 3个LLM+10域+De→En和Zh→En+未见域泛化
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论推导完整
- 价值: ⭐⭐⭐⭐ 实用的无额外参数 PEFT 方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[ACL 2025\] FedEx-LoRA: Exact Aggregation for Federated and Efficient Fine-Tuning of Large Language Models](../../ACL2025/model_compression/fedex_lora_federated_exact_aggregation.md)
- [\[CVPR 2026\] S2FT: Parameter-Efficient Fine-Tuning in Sparse Spectrum Domain](../../CVPR2026/model_compression/s2ft_parameter-efficient_fine-tuning_in_sparse_spectrum_domain.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](../../ICLR2026/model_compression/abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)

</div>

<!-- RELATED:END -->
