---
title: >-
  [论文解读] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training
description: >-
  [ACL 2025][LLM效率][Positional Encoding] 提出 LaMPE（Length-aware Multi-grained Positional Encoding），通过 **参数化 scaled sigmoid 函数** 自适应确定最优映射长度，并设计 **三区域多粒度注意力机制**（head 精细局部 + middle 线性归一化压缩 + tail 恢复长程依赖），实现无训练即插即用的 LLM 上下文窗口外推，在五大长上下文基准上全面超越现有方法。
tags:
  - "ACL 2025"
  - "LLM效率"
  - "Positional Encoding"
  - "RoPE"
  - "Context Window Extension"
  - "training-free"
  - "Length Extrapolation"
  - "注意力机制"
---

# LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training

**会议**: ACL 2025  
**arXiv**: [2508.02308](https://arxiv.org/abs/2508.02308)  
**代码**: [https://github.com/scar-on/LaMPE](https://github.com/scar-on/LaMPE)  
**作者**: Sikui Zhang, Guangze Gao, Ziyun Gan, Chunfeng Yuan, Zefeng Lin, Houwen Peng, Bing Li, Weiming Hu  
**领域**: LLM / NLP — 长上下文建模、位置编码  
**关键词**: Positional Encoding, RoPE, Context Window Extension, training-free, Length Extrapolation, Multi-grained Attention

## 一句话总结

提出 LaMPE（Length-aware Multi-grained Positional Encoding），通过 **参数化 scaled sigmoid 函数** 自适应确定最优映射长度，并设计 **三区域多粒度注意力机制**（head 精细局部 + middle 线性归一化压缩 + tail 恢复长程依赖），实现无训练即插即用的 LLM 上下文窗口外推，在五大长上下文基准上全面超越现有方法。

## 研究背景与动机

- **领域现状**: RoPE（Rotary Position Embedding）已成为主流 LLM 的标准位置编码方式（Llama、Qwen、Mistral 等均采用），但模型的有效上下文受限于预训练阶段的窗口长度（如 Llama2 的 4K、Llama3 的 8K）。
- **现有痛点**: 当输入超过预训练窗口时，RoPE 遇到 OOD（Out-of-Distribution）的相对位置，导致注意力崩溃。现有外推方法（SelfExtend、DCA）采用 **固定映射策略**——不管输入多长，分组大小 $G$ 和映射范围都是手动预设的常数。
- **核心矛盾**: ① 固定映射忽视了训练阶段相对位置的 **左偏频率分布**（短距离位置被充分训练，长距离位置严重欠训练），所有位置被同等对待；② 固定分组大小无法适应不同长度的输入，同一个 $G$ 对短序列过度压缩、对长序列又不够压缩。
- **本文目标**: 如何根据输入长度动态确定最优映射长度，并设计位置分辨率随区域变化的多粒度注意力机制？
- **切入角度**: 通过系统性实验发现困惑度随映射长度变化呈 V 形或单调递减模式，且最优映射长度与输入长度之间存在 S 形关系，可用 sigmoid 函数精确建模。
- **核心 idea**: 用 scaled sigmoid 函数自适应确定映射长度 + 三区域多粒度位置编码同时捕获局部精细信息和长程依赖。

## 方法详解

### 整体框架

LaMPE 是一种作用于 RoPE 注意力计算阶段的 **位置索引修改方法**（position indices modified），由两个核心组件构成：

1. **Length-aware Dynamic Mapping Strategy**：根据输入长度 $l$ 通过 scaled sigmoid 函数计算最优映射长度 $m$
2. **Multi-grained Attention Mechanism**：将序列分为 head / middle / tail 三个区域，各区域采用不同粒度的位置编码

推理时仅替换位置索引，**不修改模型参数、不需要训练数据、不需要额外微调**，可直接与 FlashAttention2 集成。

### 关键设计一：Length-aware Dynamic Mapping Strategy（长度感知动态映射）

论文首先在 PG-19 数据集上系统性地探索了 **映射长度（mapping length）与困惑度之间的关系**，发现两个关键模式：

- **短输入的 V 形模式**：困惑度先降后升，存在一个最优映射长度
- **长输入的单调递减模式**：困惑度随映射长度增大持续下降，最优值为预训练窗口上限

基于此，最优映射长度 $m$ 与输入长度 $l$ 的关系呈 **S 形增长趋势**，可用 scaled sigmoid 函数精确建模：

$$m = \frac{L}{1 + e^{-(al + b)}}$$

其中 $L$ 为最大映射长度（设为预训练窗口的 3/4），$a$ 和 $b$ 为通过少量采样点曲线拟合得到的参数。这种设计的核心优势在于：

- 对短输入：映射长度较小，避免浪费位置空间
- 对长输入：映射长度自动增大，充分利用预训练位置
- **全程自适应**，消除了手动调参的负担

### 关键设计二：Multi-grained Attention Mechanism（多粒度注意力机制）

获得最优映射长度 $m$ 后，LaMPE 将序列分为三个区域，各自使用不同的位置编码策略：

**① Head Region（头部区域，$i-j \leq s_1$）**：保持原始的 1:1 精确位置，$PE[i][j] = i - j$。这确保当前 token 与最近邻 token 保持精细位置区分，对连续文本生成至关重要。

**② Middle Region（中间区域，$s_1 < i-j < l - s_2$）**：采用线性归一化压缩，将位置映射到 $[s_1, m - s_2]$ 范围：

$$PE[i][j] = \left\lfloor \frac{m - s_1 - s_2}{l - s_1 - s_2} (i - j - s_1) + s_1 \right\rfloor$$

压缩比 $m/l$ 随输入长度自动调整，远距离 token 的位置粒度自然变粗。

**③ Tail Region（尾部区域，$i-j \geq l - s_2$）**：恢复精细位置，$PE[i][j] = m - l + (i - j)$。这基于关键指令或问题常出现在序列首尾的观察，保留当前 token 与序列开头 token 的精确位置关系。

论文证明了三个区域的边界处满足 **单调性连续性**，不会出现位置跳变。最优超参数为 $s_1$ 取预训练窗口的 1/8 到 1/16，$s_2$ 取 8 到 1024 的小值。

### 关键设计三：与 FlashAttention2 的无缝集成

LaMPE 的三区域分别用不同的 Q/K 位置索引实现：
- Head 区域：标准滑动窗口注意力（window_size = $s_1$）
- Middle 区域：修改 Q、K 的位置索引后计算滑动窗口注意力
- Tail 区域：仅修改 Q 的位置索引，K 保持原始位置，用下三角 mask 的全注意力

三部分通过 log-sum-exp 技巧合并，无需修改 FlashAttention2 的核心实现。

## 实验关键数据

### Table 1: LongBench (16 tasks) + L-Eval (5 tasks)

| 模型 | 方法 | LongBench Avg. | L-Eval Avg. |
|------|------|----------------|-------------|
| Llama2-7B-Chat | 原始 RoPE | 31.52 | 39.53 |
| | + SelfExtend (25K) | 34.30 | 44.27 |
| | + DCA (25K) | 32.48 | 45.59 |
| | + YaRN (25K) | 31.35 | 41.01 |
| | + NTK (25K) | 25.03 | 35.91 |
| | **+ LaMPE** | **35.07** | **48.13** |
| Llama3-8B-Ins | 原始 RoPE | 42.38 | 67.07 |
| | + SelfExtend (32K) | 42.22 | 69.39 |
| | + DCA (32K) | 44.70 | 69.93 |
| | + YaRN (32K) | 45.90 | 70.79 |
| | + NTK (32K) | 44.24 | 68.75 |
| | **+ LaMPE** | **46.99** | **71.78** |

LaMPE 在两个模型上分别超越最佳基线 0.45/1.09（LongBench）和 2.54/0.99（L-Eval）。

### Table 2: ∞Bench（超长上下文，所有输入 >64K tokens）

| 模型 | 方法 | En.MC | En.QA | En.sum | Code | Re.KV | Re.Num | Re.Pass | Avg. |
|------|------|-------|-------|--------|------|-------|--------|---------|------|
| Llama3 (32K) | SelfExtend | 50.66 | 14.06 | 15.13 | 24.87 | 3.60 | 27.12 | 27.12 | 23.22 |
| | DCA | 52.84 | 13.90 | 18.79 | 25.38 | 4.40 | 27.12 | 27.12 | 24.22 |
| | **LaMPE** | **55.02** | **16.36** | **20.49** | 25.63 | **17.40** | 27.12 | 27.12 | **27.02** |
| Llama3 (64K) | SelfExtend | 53.71 | 15.10 | 15.22 | 21.57 | 2.80 | 54.24 | 54.24 | 30.98 |
| | DCA | 50.66 | 14.35 | 18.98 | 24.11 | 2.00 | 52.88 | 54.24 | 31.03 |
| | **LaMPE** | **55.90** | **15.49** | **23.10** | 24.11 | **10.80** | 54.24 | 54.24 | **33.98** |
| Llama3.1 (128K) | 原始 RoPE | 67.25 | 14.57 | 25.42 | 22.08 | 54.80 | 99.49 | 100.00 | 54.80 |
| | STRING | 71.18 | 14.39 | 27.81 | 30.46 | 81.40 | 99.83 | 100.00 | 60.72 |
| | **LaMPE** | 70.30 | **19.51** | **28.54** | 29.19 | **92.60** | 99.83 | 100.00 | **62.85** |

在 KV 检索任务上，LaMPE 在 Llama3.1 上超出原始 RoPE **37.8 个点**（92.60 vs 54.80）。

### PG-19 困惑度 (PPL) 与 RULER 基准

| 模型 | 方法 | PPL Avg. (4K-64K) | RULER 8K | RULER 16K | RULER 64K | RULER 128K |
|------|------|-------------------|----------|-----------|-----------|------------|
| Llama2 | DCA | 7.23 | - | - | - | - |
| | **LaMPE** | **7.00** | - | - | - | - |
| Llama3 | 原始 RoPE | - | 88.76 | - | - | - |
| | SelfExtend | 7.60 | 87.59 | 75.44 | 61.95 | 35.97 |
| | DCA | 7.43 | 89.35 | 72.28 | 47.01 | 15.96 |
| | YaRN | 7.41 | - | 62.93 | 5.02 | 12.17 |
| | **LaMPE** | **7.23** | **90.57** | **87.32** | **69.46** | **59.48** |

LaMPE 在 RULER 128K 上达到 59.48，是第二名 SelfExtend (35.97) 的 **1.65 倍**。YaRN 在 64K 时崩溃至 5.02。

## 亮点与洞察

1. **经验驱动的理论发现**：通过系统性实验发现 PPL 的 V 形 / 单调递减双模式，并将最优映射长度与输入长度的 S 形关系用 sigmoid 精确建模，将启发式调参转化为少量采样点的曲线拟合问题。

2. **三区域设计的精妙分工**：head 保局部连贯性、middle 做自适应压缩、tail 恢复长程依赖——每个区域的设计都有明确的认知动机（相邻 token 需精细区分、中间 token 允许粗粒度、序列首尾的指令/问题需保留精确位置）。

3. **超越原生长上下文模型**：在 ∞Bench 的 KV 检索任务上，LaMPE 应用于 Llama3.1-8B-Instruct-128K 后达到 92.60，大幅超过原生 RoPE 的 54.80，说明位置映射优化可以比单纯扩大训练窗口更有效。

4. **预训练窗口内的性能增益**：LaMPE 不仅在外推时表现好，在原始窗口内也能提升性能（RULER 8K: 90.57 vs 88.76），这得益于对左偏位置频率分布的利用。

5. **无需手动调参**：彻底消除了 SelfExtend 中需要手调 $G$ 和 $w$ 的负担，sigmoid 函数参数通过简单曲线拟合自动确定。

## 局限与展望

1. **仅在 Llama 系列验证**：实验覆盖 Llama2-7B-Chat、Llama3-8B-Instruct、Llama3.1-8B-Instruct 三个模型，但未涉及 Qwen、Mistral、Phi 等其他 RoPE-based 架构，泛化性有待验证。

2. **sigmoid 参数的模型依赖性**：虽然 sigmoid 参数可通过少量采样拟合，但不同模型需要重新拟合，这引入了一次性的探索成本。

3. **KV 检索等精确检索任务仍有瓶颈**：虽然 LaMPE 在 KV 检索上大幅提升，但 middle 区域的线性压缩仍然损失了精确位置信息，在超长序列的多针检索（NIAH_M3）任务上性能仍然较低（128K 时仅 1.20）。

4. **与 base-modified 方法的组合**：论文指出 NTK-RoPE、YaRN 等方法与 LaMPE 正交可组合，但未实际给出组合实验结果。

5. **尾部区域大小 $s_2$ 的设定**：虽然实验表明 $s_2 = 8$ 即可恢复大部分性能，但最优值的确定仍需经验判断，缺乏理论指导。

## 相关工作与启发

| 方法 | 类型 | 是否需训练 | 映射策略 | 长度自适应 | 核心局限 |
|------|------|-----------|---------|----------|---------|
| NTK-RoPE | base-modified | 否（可选微调） | 修改频率基底 | 否 | 有外推上界，64K+ 性能崩溃 |
| YaRN | base-modified | 否（可选微调） | 频率缩放 | 否 | 外推上界低于索引修改方法 |
| SelfExtend | indices-modified | 否 | 固定分组 $G$ | 否（手动 $G$） | 均匀分组粗暴，窗口内性能可能下降 |
| DCA | indices-modified | 否 | chunk-based 映射 | 否（手动设定） | 稳定但无法利用位置频率分布 |
| STRING | indices-modified | 否 | 利用频率分布 | 部分 | 主要增强窗口内性能 |
| **LaMPE** | indices-modified | **否** | **sigmoid 动态映射 + 三区域** | **是** | 仅 Llama 系列验证 |

**启发方向**：
- LaMPE 的三区域思想可迁移到 **KV Cache 压缩**：近距离 KV 保持原精度，远距离 KV 按区域压缩或合并
- sigmoid 建模映射长度的思路可应用于 **多模态长序列**（如视频理解中帧间位置编码的自适应压缩）
- 预训练窗口内性能提升的发现说明，即使不做外推，**优化位置编码利用效率** 本身也是一个值得探索的方向

## 评分

- 新颖性: ⭐⭐⭐⭐ sigmoid 动态映射 + 三区域多粒度机制的组合有新意，V 形/单调模式的经验发现有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 五大基准（LongBench/L-Eval/∞Bench/RULER/PG-19）全覆盖，消融实验和超参分析完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，经验观察→数学建模→实验验证的逻辑线流畅，图表直观
- 价值: ⭐⭐⭐⭐ 无训练长上下文扩展是刚需方向，即插即用+自适应+兼容FlashAttention2的特性实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[ACL 2025\] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)
- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](../../ICML2025/llm_efficiency/nextlong_toward_effective_long-context_training_without_long_documents.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs](ladm_long_context_data.md)

</div>

<!-- RELATED:END -->
