---
title: >-
  [论文解读] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
description: >-
  [NeurIPS 2025][音频/语音][test-time adaptation] 提出首个面向语音基础模型的无反向传播测试时自适应框架 E-BATS，通过轻量级 prompt 自适应、多尺度损失函数和测试时 EMA 机制，在保持高精度的同时实现 2.0×–6.4× 的 GPU 显存节省。 语音基础模型（Speech F…
tags:
  - "NeurIPS 2025"
  - "音频/语音"
  - "test-time adaptation"
  - "speech foundation model"
  - "backpropagation-free"
  - "提示学习"
  - "CMA-ES"
---

# E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.07078](https://arxiv.org/abs/2506.07078)  
**代码**: [JiahengDong/E-BATS](https://github.com/JiahengDong/E-BATS)  
**领域**: 音频语音  
**关键词**: test-time adaptation, speech foundation model, backpropagation-free, prompt tuning, CMA-ES

## 一句话总结

提出首个面向语音基础模型的无反向传播测试时自适应框架 E-BATS，通过轻量级 prompt 自适应、多尺度损失函数和测试时 EMA 机制，在保持高精度的同时实现 2.0×–6.4× 的 GPU 显存节省。

## 背景与动机

语音基础模型（Speech Foundation Models, SFM）如 Wav2Vec2、HuBERT 在干净数据上表现优异，但在真实部署场景中面临严重的声学域偏移问题——背景噪声、说话人口音、麦克风特性等因素导致性能显著下降。

测试时自适应（Test-Time Adaptation, TTA）是一种在推理阶段利用无标签测试数据适配新域的方案，无需访问源数据或标签。现有 TTA 方法分为两大类：

- **基于反向传播（BP-based）**：如 TENT、SUTA、DSUTA 等，通过熵最小化或伪标签的梯度更新参数，精度较好但显存开销大，即便只更新 BN 层也需存储中间梯度
- **无反向传播（BP-free）**：如 LAME、T3A、FOA 等，仅通过前向传播更新模型，显存效率高但精度不足，且主要为视觉任务设计

关键矛盾：**现有 BP-free 方法均为视觉任务定制**，语音任务在模型架构（CNN+Transformer 混合）、任务形式（序列到序列）、噪声特性（时域动态变化）和批处理要求（单条语音处理）方面差异巨大，直接迁移效果差。

## 核心问题

1. **架构差异**：SFM 使用 LayerNorm 而非 BatchNorm，且包含 CNN 特征编码器 + Transformer 编码器的混合架构，现有 BN 统计量调整的 BP-free 方法不适用
2. **任务差异**：语音识别是序列到序列映射，噪声在时间维度动态变化，需要多尺度自适应而非单一图像级别适配
3. **批次限制**：语音 TTA 需逐条语音处理（batch size=1），无法依赖大 batch 的统计估计
4. **显存瓶颈**：BP-based 方法在长语音上显存随时长急剧增长，限制资源受限场景部署

## 方法详解

E-BATS 由三个核心模块组成：

### 1. Lightweight Prompt Adaptation (LPA)

**核心观察**：在不同声学条件下，源域与目标域隐空间嵌入的均值偏移量最高达协方差偏移量的 **7.8 倍**。这表明域偏移主要表现为隐空间中的几何平移。

**设计思路**：不同于传统 prompt tuning 在 Transformer 输入端拼接 prompt，E-BATS 在 CNN 编码器输出的隐特征 $\mathbf{Z}_t$ 上直接叠加一个可学习的 prompt 向量 $\mathbf{s}_t$：

$$\hat{\mathbf{Z}}_t = \mathbf{Z}_t + \mathbf{s}_t \cdot \mathbf{1}_N^\top$$

选择在 CNN 层而非 Transformer 层注入 prompt 的原因是：CNN 捕获局部频谱特征（音高、共振峰），对声学域偏移更敏感；Transformer 侧重全局上下文依赖，不擅长建模细粒度声学变化。消融实验证实 CNN 层注入效果远优于 Transformer 层（WER 24.0 vs 34.2）。

### 2. 多尺度损失函数

总损失为三项加权组合：$L_{adapt} = \alpha L_{ent} + \beta L_{utt} + c \cdot L_{token}$

**（a）排除 blank token 的熵最小化 $L_{ent}$**：  
CTC 解码中大量帧预测为 blank 类别，造成类别不平衡。仅对非 blank 预测帧计算 Shannon 熵。单独使用熵最小化会导致退化解（所有帧预测 blank）。

**（b）语音级别隐嵌入对齐 $L_{utt}$**：  
在每个 Transformer 层计算源域与目标域语音级嵌入质心间的欧氏距离平方和。语音级嵌入通过对帧嵌入取平均得到。有效防止熵最小化的退化解，存储开销仅 $L \times d$。

**（c）自适应置信度 token 级对齐 $L_{token}$**：  
按伪标签将帧分组到各 token 类别，对齐源域和目标域各 token 类别嵌入的均值和标准差。引入自适应置信度系数 $c$：当域偏移大或熵高时降低 $c$，避免不可靠伪标签的误导；偏移小时提高 $c$ 加强对齐。

### 3. Prompt 优化：CMA-ES

使用无梯度优化算法 CMA-ES（协方差矩阵自适应演化策略）优化 prompt 向量：每次迭代采样 $J=50$ 个候选 prompt，按 $L_{adapt}$ 排序后更新搜索分布参数（均值 $\mathbf{m}$、协方差 $\mathbf{C}$、步长 $\sigma$），迭代直至收敛，选最优 prompt。

### 4. Test-time EMA (T-EMA)

跨语音流的稳定自适应机制：处理完每条语音后，用 EMA 更新 CMA-ES 的搜索分布参数：

$$\mathbf{m}_{ema} = \gamma \mathbf{m}_{ema} + (1-\gamma)\mathbf{m}_t^{(K)}$$

协方差和步长同理更新。平衡历史知识保留与新语音适配，避免遗忘也避免过拟合。

## 实验关键数据

**数据集**：4 个噪声语音数据集，16 种声学条件  
- LibriSpeech + 高斯噪声（σ=0.0~0.02）  
- CHiME-3 单域 / 混合域  
- CommonVoice（口音多样性）  
- TEDLIUM-v2（演讲风格多样性）

**主要结果（Wav2Vec2-Base）**：

| 指标 | E-BATS 表现 |
|------|------------|
| BP-free 基线提升 | WER 降低 4.1%–13.5%（绝对值） |
| BP-based 最优对比 | 3/5 数据集达最低 WER，最大相对提升 30.7% |
| 显存节省 vs BP-based | 2.0×–6.4×（相比 DSUTA 节省 3.3×） |
| 高噪声场景（σ=0.02） | WER 25.3，比最强 BP-free 基线 FOA（45.3）降低 20.0 |

**HuBERT-Large 结果**：WER 比 BP-free 基线降低 1.8%–17.1%，显存节省 2.4×–6.8×

**显存随语音时长变化**（TED 数据集，HuBERT-Large）：BP-based 方法在 30 秒语音时达 6–12GB，E-BATS 仅 ~1.9GB，呈近线性增长。

**消融实验核心结论**：
- CNN 层 prompt 注入 >> Transformer 层（WER 24.0 vs 34.2）
- 三项损失缺一不可：仅 $L_{ent}$ 导致退化（WER 49.6）；加入 $L_{utt}$ 大幅修正（25.5）；再加 $L_{token}$ 继续提升（25.4）
- T-EMA > 无重置连续适配 > 每次重置（WER 24.3 vs 25.4 vs 26.5）

## 亮点

1. **首个面向 SFM 的 BP-free TTA 方法**，填补领域空白，且效果不输甚至超越 BP-based 方法
2. **隐空间偏移分析有说服力**——均值偏移远大于协方差偏移的实验观察，为"平移即可对齐"的设计提供了坚实依据
3. **多尺度损失设计巧妙**——从语音级到 token 级的层次化对齐，加上自适应置信度控制，解决了伪标签不可靠的难题
4. **显存优势随模型增大更显著**——HuBERT-Large 上节省 6.8×，实际部署价值大

## 局限与展望

1. **推理延迟**：CMA-ES 的迭代优化引入额外延迟，当前实现未充分利用 GPU 并行化，不适用实时场景
2. **仅验证语音识别**：未扩展到说话人识别、情感检测等其他语音任务
3. **源域统计量依赖**：需预收集源域各层嵌入的统计量，部分场景可能不可用
4. **CMA-ES 种群大小**：$J=50$ 的采样数在高维空间搜索效率存疑，可考虑引入更高效的无梯度优化器

## 与相关工作的对比

| 方法 | 类别 | 关键特点 | 相比 E-BATS 劣势 |
|------|------|---------|-----------------|
| SUTA/CEA/SGEM | BP-based 语音 TTA | 每条语音独立重置，熵最小化+语音特定损失 | 无法跨语音积累知识；显存开销大 |
| DSUTA | BP-based 语音 TTA | 连续适配，快慢双模型 | 频繁参数更新导致灾难性遗忘；CommonVoice 上 WER 比 E-BATS 高 5.5 |
| FOA | BP-free 通用 TTA | CMA-ES + prompt tuning | prompt 注入 Transformer 层不适合声学域偏移 |
| T3A/LAME | BP-free 通用 TTA | 仅调整分类器 | 适配能力不足，甚至劣于源模型 |

## 启发与关联

- **"均值偏移主导"的观察具有通用性**：类似分析可推广至其他模态（如视频、多模态模型）的域偏移理解
- **CNN 层 vs Transformer 层的 prompt 注入位置选择**对混合架构模型的自适应设计有重要参考意义
- 自适应置信度控制 token 级损失的思路可迁移到其他利用伪标签的半监督/自适应方法中
- 无梯度优化器在 TTA 中的应用值得关注，CMA-ES 以外还可尝试 Natural Evolution Strategies 或 OpenAI-ES

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个 BP-free 语音 TTA，问题定义和方法设计均有创新
- 实验充分度: ⭐⭐⭐⭐ — 4 数据集 16 条件 + 2 种骨干 + 13 基线 + 详细消融
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰，图表丰富，动机论证有数据支撑
- 价值: ⭐⭐⭐⭐ — 对资源受限场景的语音系统部署有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/audio_speech/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[NeurIPS 2025\] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)

</div>

<!-- RELATED:END -->
