---
title: >-
  [论文解读] FOCUS: Efficient Keyframe Selection for Long Video Understanding
description: >-
  [ICLR 2026][视频理解][关键帧选择] FOCUS 把"在严格 token 预算下挑出与问题最相关的视频帧"重新表述为多臂老虎机里的组合纯探索（CPE）问题——把短时片段当成臂、用经验均值加 Bernstein 置信半径自适应分配打分预算，从而在只看不到 2% 帧的情况下显著提升长视频问答精度。
tags:
  - "ICLR 2026"
  - "视频理解"
  - "关键帧选择"
  - "长视频理解"
  - "多臂老虎机"
  - "纯探索"
  - "训练无关"
---

# FOCUS: Efficient Keyframe Selection for Long Video Understanding

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1OQKqLFcbB](https://openreview.net/forum?id=1OQKqLFcbB)  
**代码**: 待确认  
**领域**: 视频理解 / 多模态大模型  
**关键词**: 关键帧选择, 长视频理解, 多臂老虎机, 纯探索, 训练无关  

## 一句话总结
FOCUS 把"在严格 token 预算下挑出与问题最相关的视频帧"重新表述为多臂老虎机里的组合纯探索（CPE）问题——把短时片段当成臂、用经验均值加 Bernstein 置信半径自适应分配打分预算，从而在只看不到 2% 帧的情况下显著提升长视频问答精度。

## 研究背景与动机
**领域现状**：多模态大模型（MLLM）把图像/视频帧编码成视觉 token 与文本一起送进 LLM。但从单图扩展到一小时长视频时，帧数爆炸（30fps 一小时超过 $10^5$ 帧）会让视觉 token 远超算力上限。主流做法要么是均匀降采样（如从一小时里抽 64 帧），要么是用 CLIP/BLIP 等视觉-语言编码器给每帧打"与 query 的相关性分"再取 Top-K。

**现有痛点**：均匀采样常常错过关键瞬间；而基于相关性打分的关键帧选择方法看似训练无关、即插即用，却有一个隐藏代价——**它们必须先把视频降采样到 1fps 做预过滤（pre-filtering）再打分**。原因很现实：用 BLIP 给一小时视频的全部帧逐帧打分要 $10^{11}\sim10^{12}$ FLOPs（论文实测 AKS 不做预过滤需 255 GPU 小时）。这个预过滤恰恰违背了"从全部帧里找最有信息量帧"的初衷，最有价值的瞬间可能在降采样时就被丢掉了。

**核心矛盾**：穷举打分太贵 vs. 为省钱而预过滤又会漏掉关键帧——本质是"在有限的打分预算下，把算力花在哪些帧上"的资源分配问题。

**本文目标**：设计一个训练无关、模型无关、即插即用的关键帧选择模块，在严格 token 预算下挑出 query 相关帧，且**不需要预过滤**，能直接处理超长视频。

**核心 idea**：作者观察到自然视频有**强时间局部性**——相邻帧在外观/运动上高度相关，这种平滑性会传导到"帧-query 相关性分数"上。论文用自相关函数（ACF）实测 LongVideoBench 和 Video-MME，发现相关性中位数在前 5 秒内都保持在 0.5 以上。**这意味着穷举打分是浪费的：可以把相邻帧聚成片段，用老虎机式的自适应探索快速排除无关区域、把打分集中在有潜力的片段上。**

## 方法详解

### 整体框架
FOCUS（Frame-Optimistic Confidence Upper-bound Selection）把视频沿时间轴切成 $M$ 个等长不重叠的短片段，每个片段当作老虎机的一个"臂"；拉动一个臂 = 在该片段里随机采一帧、用 BLIP 算出它和 query 的相关性分作为 reward。目标是在有限拉动预算下识别出"最有价值的 $m$ 个片段"（组合纯探索 CPE），再在选中的片段里挑出 Top 相关帧凑齐 $k$ 帧关键帧集，最后只把这 $k$ 帧送进 MLLM。整条流程分两步走：先粗探索拿到每个臂的可靠统计量，再对最有潜力的臂做精细打分，最后按无偏均值定选。

```mermaid
flowchart LR
    A[长视频] --> B[切成 M 个等长片段<br/>每段=一个臂]
    B --> C[Stage I 粗探索<br/>每臂拉 q 次<br/>算经验均值+Bernstein半径]
    C --> D[乐观均值 UCB 选 top-αm 臂]
    D --> E[Stage II 精细打分<br/>对选中臂各补 z 次拉动]
    E --> F[按无偏经验均值选 top-m 臂]
    F --> G[臂内插值+按分采样<br/>每臂取 ka 帧]
    G --> H[k 帧关键帧 → MLLM 回答]
```

### 关键设计

**1. 把关键帧选择建模为组合纯探索老虎机：用"预算分配"替代"穷举打分"。** 论文先把任务级效用 $R_\Phi(K\mid V,q)$ 拆成帧级效用之和 $K^\star=\arg\max_{|K|=k}\mathbb{E}[\sum_{t\in K}y_t]$，再假设帧级效用 $y_t$ 可通过视觉-语言编码器间接观测：$r_t=\psi(x_t,q;\theta)=y_t+\epsilon_\psi$，其中噪声 $\epsilon_\psi$ 零均值、方差 $\sigma_\psi^2$，因此相关性分 $r_t$ 是 $y_t$ 的无偏估计。把每个片段 $A_a$ 当臂、其帧级效用服从均值 $\mu_a$ 的分布 $\nu_a$，则"挑最优片段子集"就是经典 CPE 目标 $S^\star=\arg\max_{S\in\mathcal{S}}\sum_{a\in S}\mu_a$（$\mathcal{S}$ 取所有大小为 $m$ 的子集）。这一步是全文的支点——它把"该给哪些帧打分"从启发式预过滤变成了有理论保证的探索-利用决策。

**2. Bernstein 置信半径驱动的"乐观面对不确定性"探索。** 每个臂维护经验均值 $\hat\mu_a(n)$ 和方差自适应的经验 Bernstein 置信半径
$$\beta_a(n)=\sqrt{\frac{2\hat\sigma_a^2\ln n}{\max(1,N_a(n))}}+\frac{3\ln n}{\max(1,N_a(n))},$$
它保证 $P[|\hat\mu_a(n)-\mu_a|\le\beta_a(n)]\ge 1-6/n$。算力被自适应地分给那些"均值高（promising）"或"半径大（uncertain）"的臂：对当前 top-$m$ 内的臂看其下置信界，对 top-$m$ 外的臂看其上置信界，只要存在外部臂的 UCB 还可能超过内部臂的 LCB，就去拉动"最不确定"的那个臂 $a=\arg\max_{a\in(\tilde A_n\setminus A_n)\cup(A_n\setminus\tilde A_n)}\beta_a(n)$，反复迭代直到 top-$m$ 集合不再变化。这个迭代版（Algorithm 1）有收敛到最优 top-$m$ 集的高概率保证。相比方差无关的标准 UCB，用经验 Bernstein 半径能让方差小的片段更快收敛、把宝贵的打分次数留给真正模糊的区域。

**3. 两阶段粗-精规约：把串行迭代变成可并行的批处理。** 迭代版虽有保证却"逐臂拉动、batch size=1"，GPU 利用率极低。论文把它特化成两阶段批处理（Algorithm 2）：**Stage I 粗探索**——并行地给每个臂拉 $q$ 次，得到全体经验均值与置信半径；**Stage II 精细利用**——用乐观分数 $\tilde\mu_a(n)=\hat\mu_a(n)+\beta_a(n)$ 选出 top-$\alpha m$ 个臂 $A_{\text{coarse}}$，对它们各再批量补 $z$ 次拉动。超参 $\alpha$ 控制粗/精探索预算比例。这一步用"乐观均值指导探索、但最终用无偏经验均值定选"（$A_{\text{fine}}=\text{TopM}(\hat\mu,m)$，呼应 δ-PAC 识别的做法），既保留了"乐观面对不确定性"的核心，又彻底去掉了逐步调度开销，可一次性 batch 前向 BLIP。

**4. 臂内插值采样凑齐关键帧。** 选定臂集 $A_{\text{fine}}$ 后，把 $k$ 帧预算均分到各臂（$k_a\approx k/|A_{\text{fine}}|$）。每个臂内只采过少量帧，论文用最近邻插值把已观测 reward 扩散到臂内所有帧 $\hat r_{a,t}$，据此构造臂内采样分布 $p_a$，再无放回地抽 $k_a$ 帧。这样最终关键帧 $K=\bigcup_{a\in A_{\text{fine}}}K_a$ 既落在高相关片段、又在片段内部按相关度分布而非简单取最高，兼顾相关性与覆盖。

## 实验关键数据

### 主实验表格
四个 MLLM 上"FOCUS 选帧 vs 均匀采样"的问答精度（%）：

| 模型 | 帧数 | LongVideoBench | Video-MME |
|------|------|----------------|-----------|
| GPT-4o | 32 | 51.6 → 54.8 (↑3.2) | 61.8 → 62.5 (↑0.7) |
| Qwen2-VL-7B | 32 | 55.6 → 62.3 (↑6.7) | 57.4 → 59.7 (↑2.3) |
| LLaVA-OV-7B | 32 | 54.8 → 60.7 (↑5.9) | 56.5 → 58.3 (↑1.8) |
| LLaVA-Video-7B | 64 | 58.9 → 63.5 (↑4.6) | 64.4 → 65.4 (↑1.0) |

亮点：Qwen2-VL-7B 加 FOCUS 后在 LongVideoBench 上超过 Gemini-1.5-Flash，但只用了后者 1/8 的输入帧。

### 消融实验表格
与 SOTA 训练无关关键帧选择方法在 LLaVA-Video-7B（k=64）上按视频长度对比：

| 方法 | LVB-Short | LVB-Medium | LVB-Long | LVB-Overall | VMME-Overall |
|------|-----------|------------|----------|-------------|--------------|
| Uniform | 67.5 | 57.4 | 51.8 | 58.9 | 64.4 |
| Top-K | 72.3 | 58.0 | 60.5 | 62.3 | 62.9 |
| AKS | 72.3 | 59.2 | 56.1 | 62.1 | 64.6 |
| **FOCUS** | 72.3 | 59.0 | **63.7** | **63.5** | **65.4** |

效率对比（LongVideoBench，单张 H100）：

| 方法 | Frames Seen (%) | GPU 小时 |
|------|-----------------|----------|
| AKS 无预过滤 | 100 | 255 |
| AKS 有预过滤 | 3.7 | 9.3 |
| **FOCUS** | **1.6** | **5.5** |

α 超参权衡：α=0.1 时看 1.1% 帧 / 3.5 GPU 小时 / 62.9%；α=0.25 时 1.6% / 5.5h / 63.5%；α=0.5 时 2.5% / 9.2h / 63.6%（边际收益递减）。

### 关键发现
- **长视频增益最大**：在 LongVideoBench 超过 20 分钟的视频上，FOCUS 比均匀采样高 11.9%、比 Top-K 高 7.6%，说明帧数越多、信息越稀疏，自适应选帧越关键。
- **短视频差异小**：短视频上各选帧方法表现接近且都优于均匀采样，作者归因于底层 MLLM 推理能力在短视频上趋于饱和、输入选择影响变弱。
- **效率与精度双赢**：FOCUS 只看 1.6% 的帧、5.5 GPU 小时，却拿到最高总体精度，免去了 AKS 必需的 1fps 预过滤。

## 亮点与洞察
- **问题重表述很漂亮**：把"关键帧选择"从启发式打分升格为有理论保证的组合纯探索老虎机，让"该给哪些帧花算力"成为可优化、可分析的决策，而非拍脑袋的预过滤。
- **理论与工程的平衡**：迭代版有收敛保证但不可并行，作者用粗-精两阶段把它规约成可批处理的版本，保留了"乐观探索 + 无偏定选"的精髓，这是把 bandit 理论真正落到 GPU 上的关键一招。
- **时间局部性的实证支撑**：用 ACF 量化"相邻帧相关性分高度相关"，为"片段当臂、片段内插值"提供了直接证据，而不是空谈假设。
- **真正即插即用**：训练无关、模型无关，对开源和闭源 MLLM 都直接提升，复用同一个 BLIP 打分器保证公平对比。

## 局限与展望
- **i.i.d. 假设忽略时间依赖**：FOCUS 假设帧级相关性分在片段内 i.i.d.，没有显式建模片段间的时间依赖。作者自己指出这正是它在 Video-MME（问题更全局、关键帧分布更分散）上增益小于 LongVideoBench（问题更具体、关键帧更集中）的原因。引入 Lipschitz/metric bandit 或 contextual bandit 来刻画时间结构是明确的未来方向。
- **依赖打分器质量**：相关性分由 BLIP 提供，整体效果受限于该视觉-语言编码器对 query 的对齐能力，编码器噪声 $\epsilon_\psi$ 直接进入估计。
- **关键帧均分到臂**：每个选中臂等额分配关键帧数，对"信息高度集中在单个片段"的极端情形可能不是最优分配。

## 相关工作与启发
FOCUS 处在长视频 MLLM 的"视觉 token 压缩"主线上，与均匀降采样（VideoLLaVA 等）、相关性打分 Top-K、以及兼顾相关性与覆盖的 AKS、多分辨率自适应的 Q-Frame 同属训练无关关键帧选择。它的独特之处在于把这一切纳入老虎机纯探索框架，从而第一次给出"不需要预过滤"的可扩展方案。对后续工作的启发是：**视频/多模态里凡是"在预算下挑子集"的问题（帧选择、token 剪枝、检索增强里的 chunk 选择）都可能用纯探索/contextual bandit 重新建模**，把启发式取舍换成有理论保证、可自适应分配算力的决策；而把串行 bandit 规约成可并行批处理，是让这类方法真正实用的通用技巧。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把关键帧选择重表述为组合纯探索老虎机、并用粗-精两阶段规约实现可并行，视角新颖且理论扎实。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 4 个 MLLM、2 主 +2 泛化基准，含与 SOTA 的精度/效率对比、按长度分桶、α 权衡与多项消融。
- **写作质量**: ⭐⭐⭐⭐ 动机用 ACF 实证、方法从 oracle 目标层层规约到可执行算法，逻辑清晰；伪代码与图示完整。
- **价值**: ⭐⭐⭐⭐ 训练无关、即插即用、显著降本（仅看 1.6% 帧），对落地长视频 MLLM 有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Efficient Frame Selection for Long Video Understanding via Reinforcement Learning](../../CVPR2026/video_understanding/efficient_frame_selection_for_long_video_understanding_via_reinforcement_learnin.md)
- [\[ICLR 2026\] FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding](floc_facility_location-based_efficient_visual_token_compression_for_long_video_u.md)
- [\[ICLR 2026\] ScaleLong: A Multi-Timescale Benchmark for Long Video Understanding](scalelong_a_multi-timescale_benchmark_for_long_video_understanding.md)
- [\[CVPR 2026\] Wavelet-based Frame Selection by Detecting Semantic Boundary for Long Video Understanding](../../CVPR2026/video_understanding/wavelet-based_frame_selection_by_detecting_semantic_boundary_for_long_video_unde.md)
- [\[CVPR 2026\] Video Panels for Long Video Understanding](../../CVPR2026/video_understanding/video_panels_for_long_video_understanding.md)

</div>

<!-- RELATED:END -->
