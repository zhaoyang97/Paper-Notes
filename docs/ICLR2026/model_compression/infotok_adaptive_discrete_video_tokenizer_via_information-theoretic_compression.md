---
title: >-
  [论文解读] InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression
description: >-
  [ICLR 2026 (Oral)][模型压缩][video tokenizer] InfoTok 把 Shannon 信源编码定理引入视频离散 tokenization，用 ELBO 估计每段视频的信息量来自适应分配 token 数，证明了固定/数据无关的 tokenizer 在表示长度上是有偏次优的，并在同等重建质量下把 token 量省下约 20%~50%、压缩率比启发式自适应方法（ElasticTok）高 2.3×、推理额外开销少 11×。
tags:
  - "ICLR 2026 (Oral)"
  - "模型压缩"
  - "video tokenizer"
  - "adaptive compression"
  - "information theory"
  - "ELBO"
  - "discrete tokenization"
  - "source coding theorem"
---

# InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression

**会议**: ICLR 2026 (Oral)  
**OpenReview**: [https://openreview.net/forum?id=JEYWpFGzvn](https://openreview.net/forum?id=JEYWpFGzvn)  
**代码**: [https://research.nvidia.com/labs/dir/infotok/](https://research.nvidia.com/labs/dir/infotok/)  
**领域**: 视频离散 tokenizer / 自适应压缩  
**关键词**: video tokenizer, adaptive compression, information theory, ELBO, discrete tokenization, source coding theorem  

## 一句话总结
InfoTok 把 Shannon 信源编码定理引入视频离散 tokenization，用 ELBO 估计每段视频的信息量来自适应分配 token 数，证明了固定/数据无关的 tokenizer 在表示长度上是有偏次优的，并在同等重建质量下把 token 量省下约 20%~50%、压缩率比启发式自适应方法（ElasticTok）高 2.3×、推理额外开销少 11×。

## 研究背景与动机
**领域现状**：把视频编码成离散 token 是统一多模态大模型、对接 LLM 的关键一步。一个标准的离散 tokenizer 由 encoder + quantizer（VQ/FSQ/LFQ）+ decoder 组成，训练目标就是在 token 序列约束下最小化重建误差——本质上是一个"压缩器"。

**现有痛点**：绝大多数 tokenizer 用**固定压缩率**，对任意视频都按 $c\cdot THW$ 的固定比例切 token。但视频的空间内容（场景/物体）和时间动态（运动速度/幅度）信息密度差异巨大：一段静止的狗视频和一段激烈打斗的猫视频被分配同样多的 token，导致简单视频冗余、复杂视频信息不足。已有的"柔性 tokenization"（ElasticTok）虽然允许变长，但用的是**数据无关的均匀采样训练 + 推理时二分搜索试错**，既慢又没把信息量真正用上。

**核心矛盾**：自适应 tokenization 的难点不只是"能变长"，而是要**有原则地按信息量分配**——但 token 数 $N_x$ 是离散的、不可直接优化，把期望长度塞进 loss 里做端到端优化是不可行的。

**本文目标**：回答"理论上理想的离散视频 tokenizer 是什么、该如何原则性地训练它"。

**核心 idea**：**用信息论替代试错**。Shannon 信源编码定理告诉我们最优期望 token 长度正比于负对数似然 $-\log p(x)$；于是用神经网络可算的 **ELBO 作为 $-\log p(x)$ 的代理**来决定每段视频该用多少 token，从而绕开"直接优化离散长度"的死结。

## 方法详解

### 整体框架
InfoTok 不重训 tokenizer，而是在任意现成的固定长度 tokenizer 上加两个组件，把它"升级"成自适应版本：一个 **router** 用 ELBO 估计视频信息复杂度、决定该用多少 token $N_x$；一个 **transformer 自适应压缩器 $M_\psi$** 把固定长度 embedding 压成长度 $N_x$ 的序列，量化后再由对称的解压器还原回原长度送进原 decoder 重建。整套流程只比原 tokenizer 多一次 decoder 前向（用来算 ELBO），训练仍只靠重建 loss 端到端驱动。

```mermaid
flowchart LR
    X[视频 x] --> E[Encoder Eφ]
    E --> H[固定长度 embedding h]
    H --> R["Router rβ: 由 ELBO 估 Nx"]
    H --> M["自适应压缩器 Mψ<br/>(8层 Transformer)"]
    R -->|Nx| M
    M --> Q[Quantizer FSQ]
    Q --> Z[长度 Nx 的离散 token]
    Z --> DQ[De-quantize]
    DQ --> MI["解压器 Mψ⁻¹<br/>还原回原长度"]
    MI --> D[Decoder Dθ]
    D --> XR[重建 x̂]
```

### 关键设计

**1. 用信源编码定理把"次优性"证成定理：固定/数据无关 router 必然偏长。** 论文先把 Shannon 信源编码定理重述到 tokenizer 语境（Theorem 2.1）：任何能完美重建数据的 tokenizer，其期望长度都有下界 $\mathbb{E}_{x}[N_x] \ge H_C(\mathcal{D}) \triangleq \mathbb{E}_x[-\log_C p(x)]$，且存在自适应方案能逼近这个熵下界。在此基础上证明（Theorem 2.2）：当 router 是 $\{1,\dots,N\}$ 上的均匀分布时（即 ElasticTok 那类数据无关训练），对任意常数 $\kappa>1$ 都存在数据分布使最优解满足 $\mathbb{E}[N_x] \ge \kappa H_C(\mathcal{D})$——也就是期望长度可以比最优**任意倍地长**。直觉很清楚：均匀 router 要求模型在所有长度下都能重建，却没有任何"缩短期望长度"的激励，不同似然的数据被一视同仁。一个四点分布 $\{2^{-1},2^{-2},2^{-3},2^{-3}\}$（$C=2$）的例子很说明问题：最优该用 $1,2,3,3$ 个 token，但最小化均匀 loss 会让所有视频都用 $2$ 个 token，分布越不均衡这种浪费越严重。这一节把"该自适应"从经验观察升级成了可证的必然性。

**2. ELBO router：用可算的下界代理 $-\log p(x)$ 来定长度。** 理论指出最优长度 $N_x \propto -\log p(x)$，但视频的对数似然不可直接算。InfoTok 用 ELBO 作代理：$\text{ELBO}(x)=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}[q_\phi(z|x)\|p(z)]$，它是 $\log p(x)$ 的可证下界且训练时被直接拉紧。router 定义为确定性分配 $r_\beta(N_x|x)=\delta\!\left(\beta\cdot\frac{\text{ELBO}(x)}{\mathbb{E}[\text{ELBO}(x)]}\right)$，其中 $\beta$ 是"每个 token 该承载多少信息"的平均压缩因子，可按算力预算预先设定，$\mathbb{E}[\text{ELBO}(x)]$ 用训练样本的指数滑动平均归一化。论文进一步证明（Theorem 3.1）：只要 $\beta\ge-\mathbb{E}[\text{ELBO}(x)]$ 且 loss 被良好最小化，则推理期望长度 $\mathbb{E}[N_x]\le H_C(\mathcal{D})+\beta-\mathbb{E}[-\log p(x)]$，即在近似误差范围内达到最优。实践中只需 encode→decode 一次算重建误差（即负 ELBO 的主项，KL 近似正比于它故可省），因此**只多一次 decoder 前向**就能定长，彻底摆脱 ElasticTok 的逐块二分搜索。

**3. INFOTOK-Flex：一个模型横跨多压缩率。** 为避免每个目标压缩率都训一个 tokenizer，把多个 $\beta$ 取值集成进单一压缩器：训练时从 $\mathcal{B}$ 里随机取 $\beta$ 并作为条件输入，router 变成 $r^{\text{flex}}_\beta(N_x|x)=\frac{1}{|\mathcal{B}|}\sum_{\beta\in\mathcal{B}} r_\beta(N_x|x)$（实验中 $\mathcal{B}=\{0.25,0.5,0.75,1.0\}\,N_{\max}$）。这样单个模型在推理时只要指定目标 BPP16 就能换算出 $\beta$、自动给每段视频定长，覆盖一段连续的压缩率谱，且性能与"为每个 $\beta$ 单独训"的 InfoTok 基本持平。

**4. 似然加权的 token 选择：丢信息量最低的 token + 5% mask 开销。** 拿到 $N_x$ 后怎么把长度 $N$ 的 embedding 压成 $N_x$？直接保留前 $N_x$ 个 token 对"时空敏感"的 SOTA tokenizer 并不合理（实验验证）。InfoTok 的压缩器按**逐 token 对数似然（同样用 ELBO 近似）**保留信息量最高的 $N_x$ 个、丢弃信息量最低的，生成二进制 mask $m\in\{0,1\}^N$。妙处在于这个似然项在 router 里已经算过、不增加额外前向；端到端重建 loss 会自然训练压缩器把"要被丢的 token 里的信息"搬运到保留位置。为让解码端知道哪些位置被保留，mask 作为 token 序列的一部分存储，带来约 5% 的长度开销——相对收益是值得的。架构上压缩器/解压器各用 8 层 Transformer（block-causal attention 保持 Cosmos 的因果性），骨干复用 Cosmos Discrete Video Tokenizer 的 3D-CNN encoder/decoder，量化用 FSQ。

## 实验关键数据

### 主实验表格
在 TokenBench 与 DAVIS（均 256×256）上对比固定长度与自适应 tokenizer，压缩用 BPP16（bits-per-16-pixels，越低越省）衡量：

| 方法 | BPP16↓ | TokenBench PSNR↑ | LPIPS↓ | FVD↓ | DAVIS PSNR↑ | FVD↓ |
|------|--------|------------------|--------|------|-------------|------|
| Cosmos-DV4x8x8 (固定) | 1.00 | 30.01 | 0.138 | 49 | 25.92 | 404 |
| ElasticTok (自适应) | 0.81 | 28.26 | 0.244 | 141 | 24.69 | 754 |
| **INFOTOK-Flex** | 0.81 | 29.86 | 0.148 | 54 | 25.69 | 441 |
| **INFOTOK** | 0.81 | 30.08 | 0.145 | 49 | 25.79 | 408 |
| ElasticTok | 0.56 | 27.34 | 0.276 | 194 | 23.76 | 930 |
| **INFOTOK-Flex** | 0.56 | 29.30 | 0.179 | 71 | 24.84 | 581 |
| **INFOTOK** | 0.56 | 29.27 | 0.176 | 70 | 24.52 | 540 |

InfoTok 在 BPP16=0.81 时几乎追平固定率的 Cosmos-DV（PSNR 30.08 vs 30.01）却**省下约 20% token**；同压缩率下相比 ElasticTok，FVD 降 40~60%、LPIPS 降 25~40%、PSNR 高 1.0~2.0。甚至 BPP16=0.56 的 InfoTok 已能超过 BPP16=0.81 的 ElasticTok。

### 消融实验表格
**(a) Router vs 暴力最优搜索**：把 ELBO router 和"对每段视频穷举各 BPP16 再全局求最优分配"的 Optimal 上界对比——两者几乎贴合，验证 ELBO 估长度的有效性：

| BPP16 | 方法 | TokenBench PSNR↑ | FVD↓ | DAVIS PSNR↑ | FVD↓ |
|-------|------|------------------|------|-------------|------|
| 0.81 | INFOTOK-Flex | 29.86 | 54 | 25.69 | 441 |
| 0.81 | Optimal(上界) | 29.92 | 54 | 25.79 | 431 |
| 0.56 | INFOTOK-Flex | 29.30 | 71 | 24.84 | 581 |
| 0.56 | Optimal(上界) | 29.39 | 74 | 24.93 | 601 |

**(b) 推理效率**：ElasticTok 要对每个 4096-token block 二分搜索满足 loss 阈值，额外 $\log_2(4096)-1=11$ 次网络前向；InfoTok 只需 1 次 decoder 前向算 ELBO，**额外 NFEs 少 11×**。

### 关键发现
- 自适应分配在同质量下系统性省 token（20%~50%），且 InfoTok 与穷举最优几乎无差距，说明 ELBO 是 $-\log p(x)$ 的好代理。
- InfoTok-Flex 单模型横跨多压缩率，性能与逐率单训的 InfoTok 持平，部署友好。
- 相比启发式自适应（ElasticTok），同质量下压缩率高 **2.3×**、推理开销低 **11×**，理论与工程双赢。

## 亮点与洞察
- **把"该自适应"从经验直觉升级为可证定理**：Theorem 2.2 给出固定/均匀 router 期望长度可任意倍偏长的反例，Theorem 3.1 给出 ELBO router 的近最优保证，理论闭环干净。
- **绕开离散长度不可优化的巧思**：不去硬优化 $N_x$，而是用信息论指出 $N_x\propto-\log p(x)$，再用现成 tokenizer 已有的 ELBO 直接读出长度，几乎零额外成本。
- **即插即用**：框架建在任意固定长度 tokenizer 之上，复用其 encoder/decoder，能直接吃未来更强 tokenizer 的红利。
- **一个细节复用两处**：逐 token ELBO 既用于 router 定长、又用于压缩器选 token，避免重复前向。

## 局限与展望
- **不覆盖生成下游**：论文明确只做重建/压缩，未训练视频生成模型验证 token 在生成任务上的可用性（作者归因于算力），自适应变长 token 对自回归生成的适配是开放问题。
- **mask 存储开销**：保留-丢弃的二进制 mask 占约 5% 长度，高压缩率下这部分占比相对升高。
- **ELBO≈log-likelihood 的前提**：理论保证依赖 ELBO 足够接近真实对数似然，对欠训练或分布漂移的 tokenizer，近似误差可能放大。
- **评测范围**：仅在 256×256 方形视频、TokenBench/DAVIS 上评测（受 ElasticTok 输入限制），更高分辨率/更长时序的泛化主要放在附录。

## 相关工作与启发
- **固定长度 tokenizer**：VQ-VAE、MAGVIT2、OmniTokenizer、Cosmos Discrete Tokenizer——InfoTok 把它们当作可复用底座。
- **柔性/自适应 tokenization**：ElasticTok（右到左随机 mask + 二分搜索）是直接对标对象，InfoTok 证明其训练有偏、推理低效并给出原则化替代。
- **信息论与压缩**：Shannon 信源编码定理是全文理论支点；把"内容相关压缩优于内容无关压缩"这一经典结论落到神经离散 tokenizer 上是核心贡献。
- **启发**：用 ELBO 当"信息复杂度计量表"来驱动资源分配的思路，可迁移到自适应 patch 化、动态 token 剪枝、KV-cache 预算分配等更广的"按信息量花算力"场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把信源编码定理严格落到视频离散 tokenization、用 ELBO 绕开离散长度优化，是少见的"理论指导设计"的干净工作（Oral 实至名归）。
- **实验充分度**: ⭐⭐⭐⭐ 两数据集多压缩率、与最优搜索的贴合、NFEs 效率都到位；扣分在未验证生成下游、分辨率/时长范围有限。
- **写作质量**: ⭐⭐⭐⭐⭐ 定理-直觉-反例-算法递进清晰，图 1 框架与公式对应紧凑，可读性高。
- **价值**: ⭐⭐⭐⭐ 即插即用、省 20%~50% token 且效率高，对长视频建模与统一多模态 token 化有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[ICLR 2026\] InfoScan: Information-Efficient Visual Scanning via Resource-Adaptive Walks](infoscan_information-efficient_visual_scanning_via_resource-adaptive_walks.md)
- [\[ICLR 2026\] COMI: Coarse-to-fine Context Compression via Marginal Information Gain](comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] Adaptive Nonlinear Compression for Large Foundation Models](adaptive_nonlinear_compression_for_large_foundation_models.md)

</div>

<!-- RELATED:END -->
