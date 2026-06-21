---
title: >-
  [论文解读] BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching
description: >-
  [ICLR 2026][视频生成][Transformer] BWCache 发现视频 DiT 中各个 block 的特征在相邻时间步上呈 U 形相似度曲线（中间时间步高度冗余），于是在 block 粒度上缓存并复用特征，并用一个轻量相似度指标动态决定何时复用，做到训练无关、即插即用、最高 2.6× 加速且画质几乎不掉。
tags:
  - "ICLR 2026"
  - "视频生成"
  - "Transformer"
  - "训练无关加速"
  - "特征缓存"
  - "DiT block"
  - "推理加速"
---

# BWCache: Accelerating Video Diffusion Transformers through Block-Wise Caching

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5bJZtzTFYy](https://openreview.net/forum?id=5bJZtzTFYy)  
**代码**: [https://github.com/hsc113/BWCache](https://github.com/hsc113/BWCache)  
**领域**: 视频生成 / 扩散模型加速  
**关键词**: Diffusion Transformer, 视频生成, 训练无关加速, 特征缓存, DiT block, 推理加速  

## 一句话总结
BWCache 发现视频 DiT 中各个 block 的特征在相邻时间步上呈 U 形相似度曲线（中间时间步高度冗余），于是在 block 粒度上缓存并复用特征，并用一个轻量相似度指标动态决定何时复用，做到训练无关、即插即用、最高 2.6× 加速且画质几乎不掉。

## 研究背景与动机
**领域现状**：Diffusion Transformer（DiT）已成为视频生成的 SOTA 架构（Sora、HunyuanVideo、Wan 2.1 等），但其逐步去噪本质是串行的，每一步都要顺序跑完 N 个 DiT block，推理延迟极高（生成一段 129 帧视频要上千秒），严重限制落地。

**现有痛点**：加速方案分两类，各有硬伤。一类靠**架构改造**（蒸馏、剪枝、量化），虽降复杂度但要大量训练数据和算力做微调，且往往牺牲画质，对大预训练模型不现实；另一类是**训练无关的特征缓存**，但粒度选择是个两难——粗粒度（按 timestep 整步缓存，如 TeaCache）会丢失关键信息，细粒度（按 attention 缓存，如 PAB）又省不下多少计算。Skip-DiT 虽做了 block 级稳定性分析，却要改架构 + 从头重训（Latte 上需 8×H100 训一周）。

**核心矛盾**：缓存有两个未解问题——(1) **缓存什么**：到底哪一层级的特征既值得复用又能带来真加速；(2) **何时复用**：相邻时间步的特征相似度其实随生成内容剧烈波动，无脑复用会糊掉细节。

**本文目标**：在不改架构、不重训的前提下，找到正确的缓存粒度并自适应判断复用时机，实现高加速 + 高保真。

**核心 idea**：**[block 是延迟主因 + U 形冗余]** 作者实测发现 DiT block 占了整个去噪 80% 以上的耗时，且 block 特征在相邻时间步的相对 L1 距离随扩散进程呈 **U 形**——首尾时间步变化大、中间时间步高度相似冗余。于是把缓存粒度定在 **DiT block** 这一层，配合一个相似度阈值指标动态触发复用，在最合适的粒度上吃掉冗余。

## 方法详解

### 整体框架
BWCache 在标准 DiT 视频生成流程（文本 prompt + 初始噪声 → 多步去噪 → VAE 解码成帧）上插入一个缓存模块：每一步算完所有 block 后缓存其输出特征，并计算相邻步特征的平均相对 L1 距离作为相似度指标；当指标低于阈值 $\delta$ 时，后续若干步直接复用缓存的 block 输出而跳过计算，否则重算并更新缓存。为防止长期复用导致 latent 漂移，还引入周期性强制重算。整套机制训练无关、内存友好（只存一份当前 block 特征），可即插即用到任意预训练 DiT。

```mermaid
flowchart LR
    A[文本Prompt + 初始噪声] --> B[第t步: 顺序跑N个DiT block]
    B --> C[缓存block特征<br/>算相邻步相对L1距离]
    C --> D{平均L1/N < δ?}
    D -- 是相似冗余 --> E[后续R步复用缓存<br/>跳过block计算]
    D -- 否变化大 --> F[重算block并更新缓存]
    E --> G[周期性强制重算<br/>防latent漂移]
    F --> H[下一时间步]
    G --> H
    H --> I[VAE解码成视频帧]
```

### 关键设计

**1. Block 级冗余分析：U 形相似度是缓存的依据**。作者先用相对 L1 距离量化每个 block 在相邻时间步的特征变化：$L1_{rel}(h_{t,i}) = \frac{\|h_{t,i} - h_{t+1,i}\|_1}{\|h_{t+1,i}\|_1}$，再把一步内所有 block 的差异聚合成 $ARL1(t) = \sum_{n=1}^{N} L1_{rel}(h_{t,i})$。在 Open-Sora、Latte、Wan 2.1、HunyuanVideo 等五个模型上都观察到一致的 **U 形曲线**：早期低频成分恢复、变化剧烈（U 形左臂），中期趋稳、计算高度冗余（U 形底部），末期恢复高频细节、变化再次变大（U 形右臂）。这个从频域视角解释的规律，正是"中间时间步最该缓存"的物理依据，也把缓存粒度从模糊的"timestep"或"attention"锁定到了"block"。

**2. 自适应相似度指标：何时复用由数据说了算**。不同 prompt（静态 vs 动态场景）的特征相似度差异巨大，固定复用策略必然出错。BWCache 用所有 block 平均相对 L1 距离作为触发条件：当 $\sum_{n=1}^{N} L1_{rel}(h_{t,i})/N < \delta$ 时才复用缓存，其中 $\delta$ 是相似度阈值。这个设计让复用**自适应场景动态**——特征变化大时（动态场景、首尾步）自动少缓存多重算，变化小时（静态场景、中间步）多复用。$\delta$ 越大加速越猛但画质风险越高，默认 $\delta=0.15$。此外，作者发现末期时间步是 latent 从结构噪声过渡到高保真视频的关键阶段，一旦在第 $k$ 步触发缓存，复用只允许发生在剩余步数的前半段，**最后 $k/2$ 步永远强制重算**，守住最敏感的精修阶段。

**3. 周期性缓存重算：防 latent 漂移**。持续复用同一份 block 特征会累积误差、逐渐抹掉细粒度细节（latent drift）。BWCache 借鉴 PAB 的渐进式计算策略，在缓存区间内按复用间隔 $R$ 周期性重算：第 $t$ 步算完后缓存特征并复用 $R$ 步（即 $[t-1, t-R]$），到第 $t-R-1$ 步再更新一次：$OH = \{\dots, \underbrace{h''_t}_{\text{cached}}, \underbrace{h''_t, \dots, h''_t}_{\text{reuse } R \text{ steps}}, \underbrace{h''_{t-R-1}}_{\text{cached}}, \dots\}$。默认 $R$ 取总步数的 10%，在"加速"和"防漂移保真"之间取得平衡。

## 实验关键数据

### 主实验表格
在五个 DiT 视频模型上对比（A800 单卡，画质 LPIPS/SSIM/PSNR 相对原模型计算）：

| 模型 | 方法 | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ | 加速↑ | 延迟(s)↓ |
|------|------|---------|--------|-------|-------|-------|---------|
| Open-Sora | Original | 80.33% | - | - | - | 1× | 44.56 |
| | TeaCache | 79.16% | 0.1496 | 0.8104 | 22.39 | 1.50× | 29.64 |
| | FasterCache | 79.21% | 0.1165 | 0.8435 | 23.99 | 1.35× | 32.03 |
| | **BWCache** | **80.03%** | **0.0879** | **0.8854** | **27.05** | **1.61×** | **27.68** |
| Latte | TeaCache | 77.40% | 0.1969 | 0.7606 | 22.19 | 1.86× | 14.46 |
| | Skip-DiT(需重训) | 75.76% | 0.1403 | 0.8087 | 25.50 | 1.65× | 16.28 |
| | **BWCache** | **78.28%** | 0.1399 | **0.8181** | **26.46** | **1.90×** | **14.16** |
| Wan 2.1 | TeaCache | 81.73% | 0.2407 | 0.6593 | 18.62 | 1.41× | 644 |
| | **BWCache** | **81.99%** | **0.0782** | **0.8539** | **25.86** | **2.00×** | **457** |
| HunyuanVideo | TeaCache | 82.13% | 0.1630 | 0.8052 | 24.37 | 2.27× | 493 |
| | **BWCache** | **82.48%** | **0.0794** | **0.8903** | **29.91** | **2.60×** | **433** |

BWCache 在画质指标上几乎全面领先，HunyuanVideo 上 VBench 甚至超过原模型（82.48% vs 82.29%）、加速达 2.6×。唯一例外是 Open-Sora-Plan 上 TeaCache 加速更猛（4.41× vs 2.24×），因为该模型早期特征震荡剧烈，BWCache 主动跳过早期复用以保画质，牺牲了部分加速潜力。

### 消融实验表格
阈值 $\delta$（复用率）与复用间隔 $R$ 的权衡（Open-Sora）：

| 阈值 $\delta$ | 复用率 | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|------|--------|---------|--------|-------|-------|
| 0.25(Fast) | 59.00% | 79.03% | 0.1935 | 0.7829 | 21.39 |
| 0.20(Mid) | 53.05% | 79.42% | 0.1486 | 0.8267 | 23.45 |
| 0.15(Slow) | 41.38% | 80.03% | 0.0879 | 0.8854 | 27.05 |

| 复用间隔 $R$ | 延迟(s)↓ | VBench↑ | LPIPS↓ | SSIM↑ | PSNR↑ |
|------|---------|---------|--------|-------|-------|
| 5% | 34.70 | 80.28% | 0.0451 | 0.9250 | 30.72 |
| 10% | 27.68 | 80.03% | 0.0879 | 0.8854 | 27.05 |
| 20% | 25.97 | 79.48% | 0.1415 | 0.8465 | 24.82 |
| 25% | 25.76 | 79.31% | 0.1567 | 0.8360 | 24.32 |

### 关键发现
- **缓存 vs 减步数**：同等延迟下，原模型减到 19 步画质崩塌（LPIPS 0.3139、PSNR 16.39），而 BWCache 30 步保持 LPIPS 0.0879、PSNR 27.08，证明缓存比暴力减步数远更优。
- **块复用率随时间步波动**，中间阶段复用率最高，与 U 形冗余分析完全吻合。
- **多 GPU 扩展性强**：配合 Dynamic Sequence Parallelism，Open-Sora 204 帧 8 卡下 BWCache 达 17.2× 加速，远超 TeaCache 的 13.2×。

## 亮点与洞察
- **把"缓存粒度"这个老问题给出了实证答案**：不是 timestep（太粗丢信息）也不是 attention（太细省不动），而是 DiT block——并用占 80% 延迟 + U 形冗余两个观测把这个选择钉死。
- **U 形相似度的频域解释**很漂亮：早期恢复低频、中期稳定、末期恢复高频，物理上解释了为什么中间能复用、首尾不能。
- **完全训练无关 + 即插即用 + 低内存**（只存一份当前 block 特征，不像 ProfilingDiT 要存多步特征），对大预训练模型友好，工程落地门槛极低。
- **"末期强制重算"的细节**抓得准：识别出 latent 从噪声到高保真的关键过渡期不能复用，避免了大多数缓存方法在末期糊细节的通病。

## 局限与展望
- **早期特征剧烈震荡的模型吃亏**：Open-Sora-Plan 上因早期 block 特征不稳，BWCache 被迫跳过早期复用，加速明显落后于 TeaCache，说明该方法对"早期稳定"的模型更友好，对噪声调度/采样器异常的模型适配性受限。
- **阈值 $\delta$ 和间隔 $R$ 需人工设定**：虽然指标自适应触发，但 $\delta=0.15$、$R=10\%$ 这两个超参仍是经验默认值，跨模型/任务可能要重调，缺乏自动搜索。
- **U 形假设的普适性**：虽在五个模型上验证，但对未来更复杂的多模态/超长视频 DiT 是否仍成立、相似度指标是否需要逐 block 而非全局平均，仍待检验。
- **复用率上限**：即使 Fast 配置复用率也只到 59%，加速天花板受限于 block 间的真实冗余度。

## 相关工作与启发
- **训练无关缓存谱系**：DeepCache（缓存高层特征）、T-GATE（缓存 attention 输出）、PAB（attention 冗余，本文借了其周期重算策略）、AdaCache（按内容选择性缓存）、TeaCache（用 timestep embedding 决定缓存）——BWCache 的差异化在于把粒度定在 block 并给出 U 形冗余依据。
- **block 级稳定性**：Skip-DiT 用 Long-Skip-Connection 稳定 block 特征但需重训；ProfilingDiT 也发现中段冗余更高但要存多步特征。BWCache 不改架构、不存历史特征，更轻。
- **启发**：这条"先做细致的逐层/逐模块冗余分析、找到正确缓存粒度、再用自适应指标动态触发"的方法论，可迁移到其他串行推理的生成模型加速（如自回归视频、世界模型）；U 形频域规律也提示后续可结合频域显式建模来设计更精细的复用策略。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — block 粒度缓存本身不算颠覆，但"占 80% 延迟 + U 形冗余"的实证分析把缓存粒度选择讲透了，加上末期强制重算等细节，是缓存加速路线里扎实的一步。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 五个主流 DiT 模型、四类 baseline、多 GPU 扩展、不同长度/分辨率、阈值/间隔消融、缓存 vs 减步数对比，覆盖非常全面，每模型生成 1000+ 视频。
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、分析与方法衔接自然、图表充分；个别公式下标（block 索引 $i$ 与求和变量 $n$）略有混用。
- **价值**: ⭐⭐⭐⭐ — 训练无关、即插即用、最高 2.6× 加速且画质几乎无损，对视频生成落地有直接实用价值，代码开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Flow Caching for Autoregressive Video Generation](flow_caching_for_autoregressive_video_generation.md)
- [\[CVPR 2026\] Accelerating Diffusion-based Video Editing via Heterogeneous Caching: Beyond Full Computing at Sampled Denoising Timestep](../../CVPR2026/video_generation/accelerating_diffusion-based_video_editing_via_heterogeneous_caching_beyond_full.md)
- [\[CVPR 2026\] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching](../../CVPR2026/video_generation/disca_accelerating_video_diffusion_transformers_wi.md)
- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](../../ICML2026/video_generation/worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)
- [\[ICLR 2026\] UltraViCo: Breaking Extrapolation Limits in Video Diffusion Transformers](ultravico_breaking_extrapolation_limits_in_video_diffusion_transformers.md)

</div>

<!-- RELATED:END -->
