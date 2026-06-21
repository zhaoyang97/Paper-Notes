---
title: >-
  [论文解读] Learnable Sparsity for Vision Generative Models
description: >-
  [ICLR 2026][模型压缩][结构化剪枝] EcoDiff 用一个跨越**整个去噪轨迹**的端到端可微掩码目标来给扩散/流匹配模型做结构化剪枝，配合"时间步梯度检查点"把显存从 O(T) 压到 O(1)，仅用 100 条样本、10 个 A100 小时就能在 SDXL/FLUX 上剪掉 20% 参数且几乎不掉质量。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "结构化剪枝"
  - "可微掩码"
  - "扩散模型"
  - "流匹配"
  - "端到端剪枝"
  - "梯度检查点"
---

# Learnable Sparsity for Vision Generative Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=9pNWZLVZ4r](https://openreview.net/forum?id=9pNWZLVZ4r)  
**代码**: 待确认  
**领域**: 模型压缩 / 扩散模型剪枝  
**关键词**: 结构化剪枝, 可微掩码, 扩散模型, 流匹配, 端到端剪枝, 梯度检查点  

## 一句话总结
EcoDiff 用一个跨越**整个去噪轨迹**的端到端可微掩码目标来给扩散/流匹配模型做结构化剪枝，配合"时间步梯度检查点"把显存从 O(T) 压到 O(1)，仅用 100 条样本、10 个 A100 小时就能在 SDXL/FLUX 上剪掉 20% 参数且几乎不掉质量。

## 研究背景与动机
- **领域现状**：视觉生成模型（扩散、流匹配）能力突飞猛进，但代价是参数量爆炸——最新的 FLUX 已达 12B，约为两年前 SD2 的 13 倍。模型越大，推理越慢、部署门槛越高、碳排放越大。剪枝是比知识蒸馏更省算力的压缩路线。
- **现有痛点**：现有扩散剪枝方法几乎都依赖**昂贵的重训练**来恢复质量。Fang et al. 估计扩散模型压缩需要原训练成本的 10%–20%，压缩 SD2 可耗掉 4 万 GPU 小时；即便 BK-SDM 把成本降到约 300 A100 小时也要 0.22M 校准数据。对 SDXL/FLUX 这类更大模型，重训练负担更重。
- **核心矛盾**：重训练贵的根因是剪枝准则太粗——现有方法多用简单启发式或 one-shot 剪枝，选错神经元只能靠后续重训练补救。可微掩码本是经典剪枝/LLM 剪枝里更精细的选择准则，但**直接搬到视觉生成模型上行不通**：扩散/流模型是马尔可夫的，中间某一步的微小改动会沿去噪链产生"涟漪效应"，最终把图像彻底扭曲。
- **本文目标**：在**不依赖大规模重训练**的前提下，为各种架构（U-Net 扩散 + DiT 流匹配）通用、可扩展地学到一个让模型质量损失最小的稀疏掩码。
- **核心 idea**：**【端到端可微掩码】** 不在每一步单独算 loss，而是只对**最终去噪 latent** $z_0$ 做匹配，让掩码学习直接对齐整条生成轨迹的输出；**【时间步梯度检查点】** 用一项工程技巧把端到端反传的天价显存压下来，使大模型也能在单卡上跑通。

## 方法详解

### 整体框架
EcoDiff 给 Transformer block 内的注意力头和 FFN 神经元挂上一组 0/1 结构化掩码 $M$，通过最小化"原模型最终 latent $z_0$"与"被掩码模型最终 latent $\hat z_0$"的差距来学习这组掩码（外加 $L_0$ 稀疏正则）。由于 $L_0$ 不可微，用 hard-concrete 连续松弛把掩码变成可梯度优化的连续变量 $\lambda$，学完再卡阈值得到硬掩码物理删神经元。端到端反传需要存下所有时间步的中间量，显存爆炸，于是用时间步梯度检查点只存每步去噪后的 latent、反传时重算，把显存复杂度从 $O(T)$ 降到 $O(1)$。剪完可选地用 LoRA 或全参微调做轻量恢复。

```mermaid
flowchart LR
    A["文本 prompt + 初始噪声 z_T"] --> B["原模型 ε_θ<br/>完整去噪 → z_0"]
    A --> C["掩码模型 ε_θ^mask(M)<br/>完整去噪 → ẑ_0"]
    B & C --> D["端到端损失<br/>‖z_0 − ẑ_0‖² + β‖M‖₀"]
    D -->|时间步梯度检查点<br/>O(T)→O(1) 显存| E["更新连续掩码 λ"]
    E -->|阈值 τ| F["硬掩码 M → 物理删神经元"]
    F --> G["可选: LoRA / 全参微调恢复"]
```

### 关键设计

**1. 端到端剪枝目标：只对终点 latent 负责，绕开逐步 loss 的近视陷阱** 一个直觉做法是逐步重建损失 $L=\sum_i\sum_t\|f(x_{i,t-1},M)-x_{i,t}\|_2^2$，但它有两个硬伤：把所有时间步同等看待，会低估关键生成步；并且隐含假设每步输入都正确，使剪枝决策只顾"短期精度"而忽略神经元的长期影响，这种近视会显著拖垮整体质量。EcoDiff 改为把完整去噪过程写成嵌套函数 $z_0=F(z_T,y,T)$，只让掩码模型的终点输出贴近原模型终点：
$$\arg\min_{M}\;\mathbb{E}_{z_T,y\sim C}\big[\|F_{\epsilon_\theta}(z_T,y)-F_{\epsilon_\theta^{\text{mask}}}(z_T,y,M)\|^2\big]+\beta\|M\|_0$$
这样掩码天然要为整条轨迹的最终结果负责，自动权衡了哪一步、哪个神经元真正重要，无需像 DiffPruning 那样人工设计逐步 re-weighting 因子。

**2. Transformer block 的结构化神经元掩码：时间无关、即插即用** 剪枝粒度放在 Transformer block 的两类结构上。MHA 按注意力头挂掩码 $M_i\in\{0,1\}$：$\text{MHA}^{\text{mask}}=(M_1\cdot\text{attn}_1\|\dots\|M_h\cdot\text{attn}_h)W^o$；FFN 在激活层之后对神经元挂掩码：$\text{FFN}^{\text{mask}}(x)=(\sigma(xW_1+b_1)\odot M_{\text{ffn}})W_2+b_2$。这种设计不改变模块的输入输出维度，部署时几乎不用改结构、也不需要专用硬件支持（区别于非结构化稀疏）。关键观察是：扩散/流模型每个时间步都复用同一个去噪网络，所以一旦某神经元被结构性删除就对所有 forward 都消失——**学到的掩码天然时间无关**，无需为每个时间步单独学掩码。

**3. 离散掩码的连续松弛：hard-concrete 让 $L_0$ 可梯度优化** $\|M\|_0$ 不可微，EcoDiff 套用 Louizos et al. 的 hard-concrete 采样：$s=\sigma\big((\log(u+\delta)-\log(1-u+\delta)+\lambda)/\alpha\big)$，$\hat M=\min(1,\max(0,s(\zeta-\gamma)+\gamma))$，由 stretch 参数 $\gamma,\zeta$ 与温度 $\alpha$ 控制，从而把可学习量变成连续的 $\lambda\in\mathbb{R}^{|M|}$。重建项写成 $L_E(\lambda)$，$L_0$ 复杂度项有闭式期望 $L_0(\lambda)=\sum_j\text{Sigmoid}(\lambda_j-\alpha\log\frac{-\gamma}{\zeta})$，合成总损失 $L(\lambda)=L_E(\lambda)+\beta L_0(\lambda)$。训练完用阈值 $M(\lambda)=\mathbb{I}(\lambda>\tau)$ 离散化，$\tau$ 决定目标稀疏率——这一步**物理删除**神经元，保证真实的结构缩减和落地时的实际加速，而不仅是掩码模拟。

**4. 时间步梯度检查点：把端到端反传的显存从 O(T) 压到 O(1)** 端到端学掩码会形成一条横跨所有去噪步的超长梯度链，SDXL 上朴素实现需要约 1400GB 显存（相当于 15 张 H100）。传统梯度检查点只在单次 forward 内重算，对需要多次 forward 的扩散过程不够用。EcoDiff 设计的时间步梯度检查点在 forward 时只存每步去噪后的 latent $\hat z_t$，backward 时再逐步重算中间态并累加梯度 $\frac{dL}{d\lambda}\mathrel{+}=\frac{dL}{d\hat z_t}\frac{d\hat z_t}{d\lambda}$。显存复杂度从 $O(T)$ 降到 $O(1)$（与步数无关），代价只是多一次 forward（运行时仍 $O(T)$、约 2× 时间）。实测把 SDXL 的显存从 1400GB 压到 30GB 以下，使最大的 DiT 模型 FLUX 也能在单张 80GB GPU 上端到端剪枝。

## 实验关键数据

### 主实验表格
5,000 张 MS COCO / Flickr30K 评测，统一 10 A100 小时算力预算（FLUX-Lite 除外，它用 1120 H200 小时）：

| 模型 | 方法 | 稀疏率 | 参数 | 加速 | COCO FID↓ | COCO CLIP↑ | Flickr FID↓ |
|------|------|--------|------|------|-----------|------------|-------------|
| SDXL (U-Net) | Original | 0% | 2.6B | 1× | 27.43 | 0.33 | 33.95 |
| | BK-SDM | 20% | 2.1B | 1.25× | 42.87 | 0.30 | 56.17 |
| | DiffPruning | 20% | 2.1B | 1.25× | 83.81 | 0.25 | 96.53 |
| | Per-Step Loss | 20% | 2.1B | 1.25× | 97.36 | 0.22 | 110.53 |
| | **EcoDiff** | 20% | 2.1B | 1.25× | **32.19** | **0.33** | **40.91** |
| FLUX-Dev (DiT) | Dev | 0% | 11.9B | 1× | 28.47 | 0.34 | 37.82 |
| | DiffPruning | 20% | 9.6B | 1.25× | 40.84 | 0.33 | 48.02 |
| | FLUX-Lite | 33% | 8B | 1.49× | 29.36 | 0.34 | 38.17 |
| | **EcoDiff** | 20% | 9.6B | 1.25× | **30.81** | 0.32 | **42.58** |
| FLUX-Schnell | Schnell | 0% | 11.9B | 7× | 30.99 | 0.33 | 39.70 |
| | DiffPruning | 20% | 9.6B | 8.75× | 42.36 | 0.30 | 54.49 |
| | **EcoDiff** | 20% | 9.6B | 8.75× | **31.76** | 0.30 | **43.25** |

EcoDiff 在 SDXL 上把 FID 控制在 32.19（接近原模型 27.43），而 DiffPruning/Per-Step Loss 分别飙到 83.81/97.36；在 FLUX-Dev 上以 10 A100 小时达到与耗费 1120 H200 小时的 FLUX-Lite 相当的质量。

### 消融实验表格
SDXL 不同稀疏率 + 后剪枝恢复策略（Full 全参 / LoRA / 不恢复仅 50 步学掩码）：

| 稀疏率 | 恢复 | COCO FID↓ | COCO CLIP↑ | 迭代 |
|--------|------|-----------|------------|------|
| 0% | – | 27.43 | 0.33 | – |
| 25% | No | 34.61 | 0.32 | 50 |
| 25% | Full | 31.64 | 0.34 | 10k |
| 40% | No | 43.19 | 0.30 | 50 |
| 40% | Full | 33.25 | 0.33 | 10k |
| 50% | No | 81.76 | 0.26 | 50 |
| 50% | LoRA | 53.89 | 0.28 | 10k |
| 50% | Full | 34.87 | 0.33 | 10k |

仅 50 步学掩码（不恢复）即可在 25%–40% 稀疏下保持质量；高稀疏（50%）时 LoRA 表达力不足，全参微调 10k 步即可把 FID 拉回 34.87。

### 关键发现
- **逐步 loss 是最差 baseline**（Per-Step Loss FID 97.36），印证端到端目标的必要性——逐步 loss 的近视会沿去噪链累积误差。
- EcoDiff 的 SSIM 普遍偏低（<0.65），作者解释为它**优先语义保真（高 FID/CLIP）而非像素级复刻**，纹理/细节的轻微位移被像素指标重罚但不损主观质量。
- 能剪步蒸馏模型：FLUX-Schnell 剪 20% 仅掉 0.77 COCO FID，叠加得到相对 FLUX-Dev 的 8.75× 加速。
- 10%–20% 剪枝下个别 prompt（"a cat and a dog playing chess"）语义反而增强，与 FID 下降吻合。

## 亮点与洞察
- **把"剪枝准则"问题转化为"端到端轨迹匹配"**：用只对终点负责的目标，绕开了扩散模型逐步剪枝的涟漪误差，是本文最核心的洞见。
- **掩码时间无关性**的观察很关键——因为去噪网络在所有时间步共享权重，所以一套结构掩码天然覆盖整条轨迹，无需逐步掩码，大幅简化了问题。
- **时间步梯度检查点**把"端到端剪枝显存爆炸"这个工程拦路虎彻底解决（1400GB→30GB），是让方法在 FLUX 这种 12B 模型上可行的关键使能技术。
- 通用性强：同一框架覆盖 U-Net 扩散 + DiT 流匹配 + 步蒸馏模型，且与特征复用、步蒸馏等加速方法正交可叠加。

## 局限与展望
- 稀疏率受限：20% 几乎无损，但更激进（如 50%）就必须全参微调恢复，论文自陈"未来工作可追求更高剪枝率"。
- SSIM 偏低虽被解释为语义优先，但对需要像素级一致性的下游任务（如可控编辑、连续帧一致性）可能是隐患。
- 时间步梯度检查点带来约 2× 运行时开销，是显存换时间的权衡。
- 端到端目标只匹配最终 latent，对中间过程无显式约束，是否影响多步交互式生成（如逐步编辑）未充分讨论。

## 相关工作与启发
- **可微掩码/L0 剪枝谱系**：继承 Louizos et al. 的 hard-concrete 松弛与 LLM-Pruner（Ma et al.）的结构化神经元掩码思路，首次系统性适配到视觉生成模型。
- **扩散剪枝对照**：相对 DiffPruning（梯度代理重要性）、BK-SDM（删冗余 block + 特征蒸馏）、FLUX-Lite（社区后训练剪枝），EcoDiff 的差异化在于"端到端目标 + 极低算力（10 A100h / 100 样本）"。
- **启发**：对任何"马尔可夫式多步生成"过程（视频生成、自回归扩散、世界模型），"只对终点负责的端到端压缩目标 + 跨步梯度检查点"是一个可迁移的范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 端到端可微掩码 + 时间步梯度检查点的组合首次把低成本可微剪枝带到 SDXL/FLUX 级别的视觉生成模型，问题转化的洞见清晰。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 U-Net/DiT/步蒸馏三类模型、多 baseline、稀疏率与恢复策略消融、显存/时间复杂度验证均有，CLIP/FID/SSIM 三指标齐全。
- **写作质量**: ⭐⭐⭐⭐ 动机—挑战（逐步 loss 缺陷）—方法—工程使能技术层层递进，公式与算法伪代码清楚。
- **价值**: ⭐⭐⭐⭐ 极低算力压缩大型生成模型，直击部署成本与碳排放痛点，实用性与可复现性都强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] LEAP: Learnable End-to-End Adaptive Pruning of Large Language Models](../../ICML2026/model_compression/leap_learnable_end-to-end_adaptive_pruning_of_large_language_models.md)
- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] Generative Diffusion Prior Distillation for Long-Context Knowledge Transfer](generative_diffusion_prior_distillation_for_long-context_knowledge_transfer.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](../../ECCV2024/model_compression/isomorphic_pruning_for_vision_models.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](../../NeurIPS2025/model_compression/permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)

</div>

<!-- RELATED:END -->
