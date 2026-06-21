---
title: >-
  [论文解读] Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off
description: >-
  [ICLR 2026][可解释性][RoPE] 本文揭示 RoPE 中存在由 base 频率 θ 和训练长度 $L_{train}$ 共同决定、且在预训练早期就形成并被位置内插继承的"频率带"，证明带以下的低频维度近乎等价于 NoPE，并据此推翻"调大 θ 一定利于长上下文"的主流直觉——增大 θ 只是把能量重新分配从而提升内插却损害外推。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "RoPE"
  - "频率带 (Frequency Band)"
  - "base frequency θ"
  - "长上下文"
  - "内插-外推权衡"
  - "NoPE"
---

# Frequency Bands in RoPE: Base Frequency and Context Length Shape the Interpolation–Extrapolation Trade-off

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=PR1PPxvG9Q](https://openreview.net/forum?id=PR1PPxvG9Q)  
**代码**: 待确认  
**领域**: 可解释性 / 位置编码分析  
**关键词**: RoPE, 频率带 (Frequency Band), base frequency θ, 长上下文, 内插-外推权衡, NoPE  

## 一句话总结
本文揭示 RoPE 中存在由 base 频率 θ 和训练长度 $L_{train}$ 共同决定、且在预训练早期就形成并被位置内插继承的"频率带"，证明带以下的低频维度近乎等价于 NoPE，并据此推翻"调大 θ 一定利于长上下文"的主流直觉——增大 θ 只是把能量重新分配从而提升内插却损害外推。

## 研究背景与动机
- **领域现状**: RoPE 是当前 LLM 事实标准的位置编码。为支持更长上下文，业界普遍把 base 频率 θ 从默认的 10,000 一路放大到 500,000 乃至 1,000,000，背后的直觉是"更大的 θ 能缓解注意力分数随相对距离的衰减，从而实现外推"。
- **现有痛点**: 这一 θ-scaling 范式存在内在矛盾。一方面 Xiong et al. 用激活衰减理论为放大 θ 辩护；另一方面 Barbero et al. 发现把 RoPE 低频维度替换成 NoPE 几乎不影响性能——既然这些维度可有可无，那放大 θ 究竟在贡献什么？此外，仅放大 θ 常常无法稳健外推，仍需配合带微调的位置内插 (YaRN / LongRoPE)。
- **核心矛盾**: Barbero et al. 观察到的"频率带"(query/key 经 RoPE 后呈现高 L2-norm 的维度) 形成机制尚不清楚，且其分析局限于短文本 (20-token 窗口)，未覆盖长上下文与位置内插场景。
- **本文目标**: 系统回答一个研究问题——**增大 θ 到底是注入了有用的位置信息，还是只是把大量 RoPE 维度推向类 NoPE 的形态而几乎不贡献信息？**
- **核心 idea**: **【频率带视角】** 把分析聚焦在频率带上，发现 θ 与上下文长度的关系比此前认为的紧密得多；并提出**频率匹配干预 (FMRoPE)**——令 $\theta = L_{train}$——把频率带推向最低频，从而揭示一个清晰的内插-外推权衡。

## 方法详解

### 整体框架
论文不是提出新模型，而是一套"诊断 + 理论 + 干预"的分析流程：先在多个真实 LLM 上确认频率带普遍存在且被位置内插继承 (第 3 节)；再通过受控预训练拆解频率带由 θ 和 $L_{train}$ 决定、且在早期形成 (第 4 节)；接着用一个方差最大化的闭式推导精确预测带的位置 (第 5 节)；最后用 FMRoPE 干预把带移到最低频，实证暴露内插-外推权衡 (第 6 节)。

```mermaid
graph LR
    A[多 LLM 诊断<br/>第3节] --> B[频率带普遍存在<br/>被位置内插继承]
    C[受控预训练<br/>第4节] --> D[带位置由 θ×L_train 决定<br/>早期形成]
    E[方差最大化推导<br/>第5节] --> F[闭式预测 j*≈d/2·log_θ(L/x*)]
    B --> G[FMRoPE: θ=L_train<br/>第6节]
    D --> G
    F --> G
    G --> H[内插↑外推↓ 的清晰权衡]
```

### 关键设计

**1. 频率带索引 $i_{band}$：把"高范数维度"量化成可比较的标量**。要谈论带，先得把它定位。论文利用 Cauchy-Schwarz 不等式 $|\langle q_m, k_n\rangle| \le \|q_m\|_2\|k_n\|_2$ 指出，只需分析 query 或 key 的 2-norm 就能刻画影响注意力分数的频率成分。于是对每个 token 位置 $n$ 取范数最大的维度 $idx_n = \arg\max_i \|k^n_i\|_2$，再在整条序列上取众数得到主导维度 $\hat{idx}$，对所有 head 与 layer 求平均即得带索引 $i_{band}$。关键观察是：标准化后的 $i_{band}/d$ 随 θ 增大而减小——θ 越大，带越往高频 (低维度索引) 移动；而 $i_{band}$ 在位置内插前后几乎不变,证明带被继承而非被修正。

**2. p-RoPE 探针：验证带以下的低频维度近乎等价于 NoPE**。p-RoPE 只对 top-$r$ 高频维度施加旋转，$r=0$ 退化为 NoPE、$r=1$ 为完整 RoPE，由此在二者间插值。论文不做任何重训练、直接对预训练模型测困惑度：对 Gemma、Llama 等模型，把带以下的低频维度换成 NoPE (降低 $r$) 几乎不损性能,坐实"这些低频维度没被有效利用"。唯一例外是 Phi-3——其 block-sparse 注意力会真正用到低频维度,降 $r$ 即性能崩溃,反过来说明带的行为依赖注意力结构。

**3. 方差最大化推导：从 $(θ, L_{train}, d)$ 闭式预测带的位置**。这是论文的理论核心。把"哪个 $\theta_i$ 在固定系数范数预算下允许最大的位置相关变化"约化为最大化 cos 坐标在窗口上的方差。设 $m\sim\text{Unif}[0,L_{train}]$、$x:=\omega L_{train}$，由直接积分得方差 $V(x)=\frac{1}{2}+\frac{\sin(2x)}{4x}-\left(\frac{\sin x}{x}\right)^2$。对其求导并数值求解 $V'(x)=0$，得到最小正根 $x^\star \approx 3.657210$ rad (全局最大,$V(x^\star)\approx 0.540 > 1/2$)。最优角频率 $\omega^\star = x^\star/L_{train}$，再选 RoPE 网格中频率最接近 $\omega^\star$ 的维度,得到闭式预测器:
$$j^\star \approx \frac{d}{2}\log_\theta\!\left(\frac{L_{train}}{x^\star}\right), \quad x^\star \approx 3.657210$$
实测中 $i_{band} \approx c \times j^\star$ ($c\approx 1.0\text{–}1.1$) 线性吻合,即带位置由 $(θ, L_{train}, d)$ 提前决定。当 $\theta = L_{train} = 8192$、$d=128$ 时 $j^\star=59$,$c\times j^\star\approx 64.9$ 恰好逼近 RoPE 对数 $d/2=64$,即带被推到最低频维度。

**4. FMRoPE 频率匹配干预：用一个极简设置暴露内插-外推权衡**。基于上面推导,FMRoPE 直接令 $\theta = L_{train}$ (预训练用 θ=512、内插微调用 θ=1512),把频率带从训练之初就推到最低频,让模型从一开始就利用更宽、更有效的频率范围。其作用是把一个抽象结论变成可操作的旋钮:θ 选大还是选小,直接对应内插 vs 外推的取舍。

## 实验关键数据

### 主实验：多 LLM 上的带索引与 p-RoPE 困惑度 (Wikitext-103, L=4096)

| Model | $L_{train}$ | θ | $i_{band}$ | $i_{band}/\frac{d}{2}$ | r=1.0 | r=0.75 | r=0.50 |
|---|---|---|---|---|---|---|---|
| Gemma | 8k | 10000 | 116.68 | 0.91 | 2.52 | 81.66 | >100 |
| Qwen3 | 40k | 1M | 51.04 | 0.79 | 6.22 | 6.22 | 7.46 |
| Llama-2 | 4k | 10000 | 53.53 | 0.84 | 2.54 | >100 | >100 |
| Llama-3 | 8k | 500k | 43.43 | 0.68 | 2.29 | 2.29 | 84.50 |
| Phi-3 | 8k | 1M | 36.67 | 0.57 | 2.84 | 46.36 | >100 |

带在所有模型中普遍出现;θ 越大 $i_{band}/\frac{d}{2}$ 越小 (带往高频移);位置内插模型 (+YaRN/+Llama3/+LongRoPE) 继承带、$i_{band}$ 基本不变。

### 消融：受控预训练拆解 $(θ, L_{train})$ 对带的作用 ($L_{train}=512$)

| θ | $i_{band}$ | $i_{band}/\frac{d}{2}$ | r=1.0 | r=0.25 |
|---|---|---|---|---|
| 512 (=$L_{train}$) | 60.5 | 0.94 | 19.58 | 98.26 |
| 10000 | 30.12 | 0.47 | 19.39 | 63.59 |
| 500000 | 17.00 | 0.26 | 19.35 | 34.46 |
| 1000000 | 15.37 | 0.24 | 19.36 | 30.59 |

固定 θ、增大 $L_{train}$ → 带索引升高;固定 $L_{train}$、增大 θ → 带往低维度 (高频) 移,且 500k 与 1M 差异已很小。带在第 6 个 epoch (快速收敛期) 出现并保持到训练结束。

### 关键发现：FMRoPE 暴露的内插-外推权衡 (困惑度,$L_{train}=512$)

| 设置 | L=512 | L=1512 | L=2512 | L=25512 |
|---|---|---|---|---|
| FMRoPE (θ=512) | 19.58 | 21.19 | 24.20 | >100 |
| FMRoPE θ_inf=3512 | 21.28 | 20.27 | 20.37 | >100 |
| RoPE θ=10000 | 19.39 | 43.63 | 84.45 | >100 |
| RoPE θ=1M | 19.35 | 37.94 | 74.26 | >100 |
| +YaRN FMRoPE | 19.62 | 17.78 | 17.56 | 23.19 |
| +YaRN θ=1M | 19.07 | 17.76 | 17.81 | >100 |

短上下文里常规大 θ 略优 (内插更好),但一旦外推到更长序列,FMRoPE 困惑度显著更低 (例如 L=2512 时 24.20/20.37 vs 84.45/74.26);该权衡在 YaRN 内插后依然存在。

## 亮点与洞察
- **把"频率带"从现象升级为可证伪的定律**:给出 $j^\star \approx \frac{d}{2}\log_\theta(L_{train}/x^\star)$ 闭式预测器,只用 $(θ, L_{train}, d)$ 就能提前算出带位置,并在多模型上线性吻合。
- **正面回击 θ-scaling 迷思**:增大 θ 不是注入新位置信息,而是把能量重新分配——带以下越来越多维度变成类 NoPE,换来内插提升却牺牲外推。
- **实用指导清晰**:外推关键时选 $\theta \approx L_{train}$,训练范围内的内插为主时用更大 θ;位置内插要配合"带感知"的 θ 选择而非盲目套用。
- **诊断工具落地**:$i_{band}$ + p-RoPE 是一套无需重训练、即插即用的频率带诊断探针。

## 局限与展望
- **FMRoPE 需在推理时已知目标序列长度** (如把 $\theta_{inf}$ 调到 1512/3512) 才能拿到最佳外推,实际部署不现实;作者把"动态/自适应 θ 调整"列为未来工作。
- 主实验规模偏小 (受控预训练为 16 层、d=128 的模型;真实 LLM 分析用 4096 窗口),虽在附录补了 1B 模型与下游任务,但与生产级超长上下文仍有差距。
- Phi-3 的 block-sparse 注意力打破了 p-RoPE 趋势,说明结论对注意力结构有依赖,稀疏注意力下的频率带行为需要单独建模。
- 理论推导基于单坐标 cos 方差代理,与完整协方差视角的连接放在附录,严格性上仍是"代理"而非端到端证明。

## 相关工作与启发
- **直接前序**: Barbero et al. (2025) 首次发现 RoPE 频率带并把低频换 NoPE,但只看短文本、未解释带的形成机制——本文补上"带在哪/为什么/何时形成"三问,并扩展到长上下文与位置内插。
- **θ 设计谱系**: Xiong et al. (θ=500k 抑制衰减)、Peng et al. (YaRN 规则放大)、Ding et al. (LongRoPE 搜索),它们都倾向增大 θ;而 Liu et al. (小 θ=500 改善外推)、Takase & Okazaki (LRPE 令 θ=序列长度) 与本文的 $\theta=L_{train}$ 一脉相承。
- **启发**: 长上下文工程里"无脑放大 θ"应被"带感知的 θ 选择"取代;频率带 + p-RoPE 可作为评估任意位置编码方案是否"有效利用频率"的通用诊断框架。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把零散现象统一成由 $(θ, L_{train})$ 决定、可闭式预测的频率带定律,并据此推翻 θ-scaling 直觉,视角足够新。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 5 个真实 LLM + 3 种位置内插 + 受控预训练消融 + 理论验证,链条完整;扣分在规模偏小、超长上下文与下游任务仅在附录。
- **写作质量**: ⭐⭐⭐⭐ 研究问题贯穿全文、每节有 takeaway、理论推导分步清晰,可读性强。
- **价值**: ⭐⭐⭐⭐ 既给出可证伪诊断又给出实用 θ 选择指导,对长上下文 LLM 的位置编码设计有直接参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Probing Rotary Position Embeddings through Frequency Entropy](probing_rotary_position_embeddings_through_frequency_entropy.md)
- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)
- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICLR 2026\] From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning](from_tokens_to_thoughts_how_llms_and_humans_trade_compression_for_meaning.md)
- [\[ICLR 2026\] From Data Statistics to Feature Geometry: How Correlations Shape Superposition](from_data_statistics_to_feature_geometry_how_correlations_shape_superposition.md)

</div>

<!-- RELATED:END -->
