---
title: >-
  [论文解读] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals
description: >-
  [NeurIPS 2025][信号/通信][掩码符号建模] 本文提出 Masked Symbol Modeling（MSM），将 BERT 的掩码预测范式应用于通信物理层——将脉冲成形产生的符号间贡献重新定义为"上下文信息"，训练 Transformer 在干净过采样基带信号上学习波形结构，推理时利用学到的上下文来恢复被冲激噪声破坏的符号。
tags:
  - "NeurIPS 2025"
  - "信号/通信"
  - "掩码符号建模"
  - "通信物理层"
  - "Transformer"
  - "脉冲成形"
  - "冲激噪声"
---

# Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals

**会议**: NeurIPS 2025  
**arXiv**: [2512.01428](https://arxiv.org/abs/2512.01428)  
**代码**: [https://github.com/OguzBedir/Masked_Symbol_Modeling](https://github.com/OguzBedir/Masked_Symbol_Modeling)  
**领域**: 信号与通信  
**关键词**: 掩码符号建模, 通信物理层, Transformer, 脉冲成形, 冲激噪声

## 一句话总结

本文提出 Masked Symbol Modeling（MSM），将 BERT 的掩码预测范式应用于通信物理层——将脉冲成形产生的符号间贡献重新定义为"上下文信息"，训练 Transformer 在干净过采样基带信号上学习波形结构，推理时利用学到的上下文来恢复被冲激噪声破坏的符号。

## 研究背景与动机

**领域现状**：Transformer 架构在通信系统中的应用日益增多，主要用于解决信道估计、均衡等传统信号处理问题。然而，现有工作大多将 Transformer 当作黑盒工具，未深入探索物理波形中"上下文"的本质含义。

**现有痛点**：在脉冲成形的过采样系统中，相邻符号的脉冲会在时域上重叠，产生符号间贡献（inter-symbol contribution）。传统方法将这种重叠视为需要消除的干扰（如均衡器），而非可利用的信息源。同时，冲激噪声（如Middleton Class-A噪声）因其突发性和高幅值特性，对传统检测器造成严重问题——高斯噪声假设下设计的检测器在冲激噪声环境中性能急剧恶化。

**核心矛盾**：脉冲成形产生的符号间重叠其实蕴含了丰富的确定性结构信息——每个采样点都包含来自多个相邻符号的贡献。但现有方法没有系统地利用这种结构来实现更强的信号恢复。

**本文目标** 如何将 NLP 中的"上下文理解"能力迁移到通信物理层？具体地，能否训练模型理解脉冲成形波形的"语法"，进而利用上下文来恢复被噪声破坏的符号？

**切入角度**：作者将"脉冲成形产生的符号间贡献"类比于"自然语言中单词的上下文"——就像一个词的含义可以从周围词推断，一个被遮挡的符号的身份也可以从周围未遮挡的采样点推断。

**核心 idea**：将 BERT 的掩码预测范式移植到通信物理层，把脉冲成形的符号间重叠视为上下文信息而非干扰，自监督地学习波形的"潜在语法"。

## 方法详解

### 整体框架

输入是过采样的复基带信号（每符号8个采样点），其中随机15%的符号被掩码（对应时域采样区间置零）。Transformer 模型根据周围未掩码的采样点预测被掩码位置的符号标识符。训练阶段仅使用干净（无噪声）信号；推理阶段，先识别被冲激噪声影响的符号位置，将其掩码后输入模型，利用学到的上下文进行符号恢复。

### 关键设计

1. **掩码符号建模（Masked Symbol Modeling）**:

    - 功能：自监督地学习过采样基带波形的结构表征
    - 核心思路：定义一个离散词汇表，为所有考虑的调制方式（BPSK到QAM256）的每个星座点分配唯一标识符（共272个标识符）。训练时随机掩码15%的符号（将对应的采样区间置零），模型预测被掩码符号的标识符。损失仅在掩码位置计算交叉熵，使用逆频率加权处理类别不平衡
    - 设计动机：掩码比例选为15%与BERT一致，且通过干净信号训练让模型专注学习波形结构而非噪声模式。符号间的脉冲重叠天然提供了"完形填空"所需的上下文

2. **Reformer Transformer 架构**:

    - 功能：高效处理长序列（1024个采样点）的波形
    - 核心思路：2通道输入（I/Q分量）通过1D可学习线性投影映射到512维嵌入，加上正弦位置编码后经过6层Reformer块处理。Reformer使用局部敏感哈希注意力（桶大小64，4个哈希），共享权重和可逆层以节省显存。对每个掩码符号，在其采样跨度（8个采样点）上做均值池化后送入线性分类头（$\mathbb{R}^{512} \to \mathbb{R}^{272}$）
    - 设计动机：使用Reformer而非标准Transformer是为了处理1024长度序列时的计算效率。局部敏感哈希注意力将注意力复杂度从 $O(n^2)$ 降低到 $O(n\log n)$

3. **半合成冲激噪声推理策略**:

    - 功能：在推理阶段利用学到的上下文恢复被冲激噪声破坏的符号
    - 核心思路：推理时并非将整个含噪波形直接送入模型，而是先识别被冲激噪声影响的符号位置，仅对这些位置进行掩码，未受影响的部分保持不变。然后模型利用周围完好的采样点来推断被掩码位置的符号标识符。冲激噪声的 impulsive index $A$ 根据目标符号命中率15%来校准
    - 设计动机：这种选择性掩码策略将问题从"全局噪声鲁棒性"转化为"局部缺失恢复"，充分利用了模型在干净信号上学到的上下文理解能力

### 损失函数 / 训练策略

仅用交叉熵损失，只在被掩码符号位置计算，加逆频率权重处理类别不平衡。训练完全自监督，基于在线数据生成的 IterableDataset，无需外部数据集。Adam 优化器 ($lr=10^{-3}$)，单卡 A100 训练24小时（37551步），batch size 64。

## 实验关键数据

### 主实验

| 调制方式 | SER（无噪声） | SER（$\Gamma=10^{-6}$, 强冲激） | SER（$\Gamma=10^{-3}$, 中等冲激, 高SNR） |
|---------|-------------|-------------------------------|---------------------------------------|
| BPSK | ~0.001 | ~0.001 | ~0.001 |
| QPSK | ~0.02 | ~0.02 | ~0.02 |
| QAM16 | ~0.05 | ~0.05 | ~0.05 |
| QAM64 | ~0.15 | ~0.15 | ~0.15 |
| QAM256 | ~0.35 | ~0.35 | ~0.35 |

### 消融实验

| 配置 | 观察 | 说明 |
|------|------|------|
| $\Gamma=10^{-6}$（高斯分量可忽略） | SER 不随 SNR 变化 | 掩码消除冲激后高斯噪声极小 |
| $\Gamma=10^{-3}$（高斯分量不可忽略） | 低 SNR 时 SER 显著上升 | 高斯噪声影响未掩码位置的上下文质量 |
| 简单 vs 复杂调制 | BPSK 最好，QAM256 最差 | 星座点越多，掩码预测越难 |
| 不同滤波器跨度和滚降因子 | 性能稳定 | 模型泛化到各种脉冲成形参数 |

### 关键发现

- 在强冲激噪声（$\Gamma=10^{-6}$）下，模型的SER几乎不受SNR影响，因为掩码有效消除了冲激分量，且高斯背景噪声几乎为零。这证明了"掩码+上下文恢复"策略对冲激噪声的天然免疫力
- 在中等冲激噪声（$\Gamma=10^{-3}$）下，低SNR区间性能明显下降，因为高斯噪声会同时影响掩码和未掩码区域的信号质量
- 模型在多种调制方式（BPSK到QAM256）和脉冲成形参数（4种滤波器跨度×6种滚降因子）上都保持稳定性能，展现了跨配置泛化能力
- 符号命中率的统计分析表明，使用 $A^\star = -\ln(0.85)/L$ 可以精确控制平均15%的符号受冲激噪声影响

## 亮点与洞察

- **将NLP中context的概念迁移到通信物理层**是本文最核心的洞察。把脉冲成形产生的符号间重叠——传统上被视为需要消除的ISI——重新定义为可利用的上下文信息，这种视角转换非常巧妙。它暗示了一种新的接收机设计范式：不是检测（detect）信号，而是理解（interpret）信号
- **训练与推理的分离设计**：训练阶段使用干净信号避免学到噪声模式，推理阶段通过选择性掩码将噪声恢复问题转化为完形填空问题。这种设计使得模型对噪声类型具有天然的适应性，只要能检测到受噪声影响的位置
- 从BERT到通信信号的类比打开了一扇门：波形有"语法"，SAE有"词汇表"，掩码预测学的是波形的"语言模型"

## 局限与展望

- 当前依赖事先识别冲激噪声影响的符号位置（"半合成"设置），在真实系统中噪声位置检测本身就是一个难题。未来需要让模型直接处理含噪波形（不需要显式掩码）
- 输入表示较为简单（原始I/Q通道+线性投影），论文自身提出可以改为量化+嵌入的方式，更忠实于BERT的token化设计
- 缺乏与传统通信方法（如最优非线性解调器、深度学习基线）的对比
- 仅在单天线、单用户、无多径的简化场景下验证，真实通信环境中的适用性未知
- 没有系统的消融实验分析架构选择（深度、头数、嵌入维度）对性能的影响

## 相关工作与启发

- **vs 传统均衡器（如MMSE、ZF）**: 传统方法将符号间贡献视为ISI并试图消除。本文反其道而行之，将ISI视为信息源。两种范式可能互补——均衡器消除多径ISI，MSM利用脉冲成形ISI
- **vs 基于DL的信号检测（DeepSIG等）**: 大多数DL通信方法做端到端监督学习。本文采用自监督预训练方法，不需要含噪信号/标签对，训练数据生成更简单
- **vs BERT原始论文**: MSM是BERT在连续信号域的直接移植，但关键区别在于"token"不是离散符号而是连续波形段，"掩码"是时域置零而非特殊token替换

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将NLP掩码建模范式迁移到通信物理层的概念创新令人眼前一亮
- 实验充分度: ⭐⭐⭐ 验证了基本可行性但实验偏初步，缺乏与传统方法对比和系统消融
- 写作质量: ⭐⭐⭐⭐ 类比清晰，物理动机阐述充分，但部分描述可以更精炼
- 价值: ⭐⭐⭐⭐ 为通信物理层的表征学习开辟了新方向，但距实际部署还需大量工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] WirelessMathBench: A Mathematical Modeling Benchmark for LLMs in Wireless Communications](../../ACL2025/signal_comm/wirelessmathbench_a_mathematical_modeling_benchmark_for_llms_in_wireless_communi.md)
- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](../../ICLR2026/signal_comm/mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [\[NeurIPS 2025\] Feature-aware Modulation for Learning from Temporal Tabular Data](feature-aware_modulation_for_learning_from_temporal_tabular_data.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)
- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)

</div>

<!-- RELATED:END -->
