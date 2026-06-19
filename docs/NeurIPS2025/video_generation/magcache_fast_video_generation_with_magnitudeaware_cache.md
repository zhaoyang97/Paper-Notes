---
title: >-
  [论文解读] MagCache: Fast Video Generation with Magnitude-Aware Cache
description: >-
  [NeurIPS 2025][视频生成][视频扩散模型加速] 发现视频扩散模型中相邻时间步残差输出的幅度比（magnitude ratio）遵循一条跨模型、跨 prompt 普遍成立的单调递减规律（"统一幅度定律"），由此提出 MagCache：基于幅度比对跳步误差进行精确累积建模，自适应跳过冗余时间步并复用缓存，仅需 1 个样本校准，即可在 Open-Sora、CogVideoX、Wan 2.1、HunyuanVideo 等模型上实现 2.10–2.68× 加速，且在 LPIPS/SSIM/PSNR 三个指标上全面优于 TeaCache 等已有方法。
tags:
  - "NeurIPS 2025"
  - "视频生成"
  - "视频扩散模型加速"
  - "缓存复用"
  - "残差幅度定律"
  - "时间步跳过"
  - "推理加速"
---

# MagCache: Fast Video Generation with Magnitude-Aware Cache

**会议**: NeurIPS 2025  
**arXiv**: [2506.09045](https://arxiv.org/abs/2506.09045)  
**代码**: [https://github.com/Zehong-Ma/MagCache](https://github.com/Zehong-Ma/MagCache)  
**领域**: 图像/视频生成 / 模型压缩  
**关键词**: 视频扩散模型加速, 缓存复用, 残差幅度定律, 时间步跳过, 推理加速

## 一句话总结

发现视频扩散模型中相邻时间步残差输出的幅度比（magnitude ratio）遵循一条跨模型、跨 prompt 普遍成立的单调递减规律（"统一幅度定律"），由此提出 MagCache：基于幅度比对跳步误差进行精确累积建模，自适应跳过冗余时间步并复用缓存，仅需 1 个样本校准，即可在 Open-Sora、CogVideoX、Wan 2.1、HunyuanVideo 等模型上实现 2.10–2.68× 加速，且在 LPIPS/SSIM/PSNR 三个指标上全面优于 TeaCache 等已有方法。

## 研究背景与动机

**领域现状**：扩散模型在视频生成领域已取得令人瞩目的进展，从早期 U-Net 架构（Stable Diffusion、ModelScope）演进到基于 Transformer 的 DiT 架构（Open-Sora、CogVideoX、Wan 2.1、HunyuanVideo），生成质量和时序一致性持续提升。然而，核心瓶颈始终是推理速度——生成过程本质上是序列化的多步去噪，模型越大、分辨率越高、视频越长，耗时就越不可接受。以 Wan 2.1 为例，在单张 A800 GPU 上生成 5 秒 480P 视频需要数分钟。

**现有痛点**：加速视频扩散模型的已有方案各有局限。蒸馏方法（如 VideLCM）需要大量重新训练和额外数据，门槛高且不通用；量化方法（如 PTQ4DM）需要精心校准，且在极低比特下质量损失严重。相比之下，**基于缓存的加速方法**（如 DeepCache、Δ-DiT、PAB）不需要重训练，通过复用相邻时间步的中间特征来减少计算，是一条更轻量的路线。但已有缓存方法存在三个关键问题：（1）DeepCache、PAB 等方法采用**均匀策略**（如每隔 N 步复用一次），完全忽略了不同时间步之间的动态差异——有些时间步之间变化极小可以大胆跳过，有些变化剧烈必须完整计算；（2）TeaCache 虽然尝试了自适应策略，但它基于 time embedding 差异或调制输入差异来构建跳步函数，需要**70 个精心挑选的 prompt 做多项式拟合**，不仅校准成本高，而且容易过拟合到校准集，换一组 prompt 可能失效；（3）FasterCache、DuCa、TaylorSeer 等方法虽然也做了自适应，但需要缓存大量中间状态，**额外显存开销巨大**（TaylorSeer 在 Wan 2.1 上需要 40GB 额外显存），实际部署困难。

**核心矛盾**：问题的根源在于——我们缺少一个既**准确**又**稳定**（跨模型、跨 prompt 不变）的指标来衡量相邻时间步残差输出之间的差异。如果能找到这样一个指标，就可以精确地判断何时可以安全跳过时间步、何时必须重新计算，从而在效率和质量之间取得最优平衡。

**本文目标** 这篇论文将上述问题分解为三个具体子问题：（1）找到一个跨模型、跨 prompt 普遍成立的残差变化指标；（2）基于该指标建立精确的跳步误差模型，支持连续多步跳过时的误差累积估计；（3）设计一种自适应缓存策略，在误差可控的前提下最大化跳步数量。

**切入角度**：作者从一个非常简单但深刻的经验观察出发——分析扩散模型在不同时间步的残差输出（模型预测速度减去输入），发现相邻时间步残差的**幅度比**（magnitude ratio，即 $\gamma_t = \text{mean}(\|\mathbf{r}_t\|_2 / \|\mathbf{r}_{t-1}\|_2)$）呈现出高度规律的单调递减趋势。更关键的是，这个趋势在不同模型（Wan 2.1、Open-Sora）和不同 prompt 下几乎完全一致。这意味着幅度比是一个天然的、无需额外校准的冗余度指标。

**核心 idea**：用残差幅度比这个跨模型跨 prompt 稳定的指标替代 TeaCache 需要大量校准的多项式拟合，实现仅需 1 个样本校准的自适应缓存加速。

## 方法详解

### 整体框架

MagCache 的整体 pipeline 清晰直观：在视频扩散模型的推理过程中，对每个时间步 $t$，计算当前残差 $\mathbf{r}_t = \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{x}_t$（模型预测速度减去输入），然后基于预校准的幅度比曲线 $\{\gamma_i\}$ 估算从上次缓存刷新点 $\hat{t}$ 到当前步 $t$ 的累积跳步误差 $\mathcal{E}_t$。如果累积误差低于阈值 $\delta$ 且跳步数未超过上限 $K$，则直接复用缓存的旧残差 $\mathbf{r}_{\hat{t}}$ 跳过当前步；否则重新计算残差、更新缓存、重置误差计数器。整个过程不需要任何训练，是一个即插即用的推理加速框架。

输入是噪声视频 latent $\mathbf{x}_T$ 和文本 prompt，输出是去噪后的视频 latent $\mathbf{x}_0$。中间的去噪过程本来需要完整走 $T$ 步（如 50 步），MagCache 通过跳过大量冗余步将实际计算步数减少到 $T/2$ 甚至更少，从而实现 2× 以上加速。

### 关键设计

1. **统一幅度定律（Unified Magnitude Law）的发现与验证**:

    - 功能：揭示了一条在不同视频扩散模型和不同 prompt 下普遍成立的经验规律——相邻时间步残差的幅度比 $\gamma_t$ 呈单调递减趋势
    - 核心思路：定义每步幅度比为 $\gamma_t = \text{mean}(\|\mathbf{r}_t\|_2 / \|\mathbf{r}_{t-1}\|_2)$，其中 $\mathbf{r}_t = \mathbf{v}_\theta(\mathbf{x}_t, t) - \mathbf{x}_t$ 是第 $t$ 步的残差。作者通过实验观察到三个关键现象：**第一**，在前 80% 的时间步中，$\gamma_t$ 从接近 1 缓慢平稳递减，同时 token 级余弦距离极小（接近 0），说明相邻步残差之间的差异主要来自**幅度缩放**而非**方向变化**。这意味着 $\|\mathbf{r}_t - \mathbf{r}_{t-1}\| \approx |\|\mathbf{r}_t\| - \|\mathbf{r}_{t-1}\||$，即幅度比就能精确刻画残差差异。**第二**，在后 20% 的时间步中，$\gamma_t$ 急剧下降，余弦距离也显著增大，说明最终生成阶段变化剧烈、不适合跳过。**第三**，也是最关键的——不同 prompt 产生的 $\gamma_t$ 曲线几乎完全重合，标准差极低，跨模型（Wan 2.1 vs Open-Sora）的趋势也完全一致。这就是"统一"的含义：规律是模型无关、prompt 无关的
    - 设计动机：这一发现直接解决了 TeaCache 的核心痛点——TeaCache 需要 70 个精心挑选的 prompt 做多项式拟合来预测残差差异，而 MagCache 基于这条统一规律，只需要 1 个随机 prompt 前向传播一次就能得到适用于所有场景的幅度比曲线。这不仅将校准成本从"70 个 prompt × 完整推理" 降低到 "1 个 prompt × 1 次推理"，更重要的是消除了过拟合风险——因为规律本身就是 prompt 无关的

2. **基于幅度比的精确误差建模（Error Modeling）**:

    - 功能：精确估算跳过连续多个时间步所引入的累积误差，为自适应决策提供可靠依据
    - 核心思路：设上次刷新缓存的时间步为 $\hat{t}$，跳过 $\hat{t}+1, \ldots, t$ 后，跳步误差为 $\varepsilon_{\text{skip}}(\hat{t}, t) = 1 - \text{mean}(\|\mathbf{r}_t\|_2 / \|\mathbf{r}_{\hat{t}}\|_2) \approx 1 - \prod_{i=\hat{t}+1}^{t} \gamma_i$。这个公式非常优雅：由于幅度比的乘法链式关系，连续跳多步的误差可以直接用各步幅度比的乘积来表达，不需要像 TeaCache 那样每步都做独立预测。然后维护一个运行累积误差 $\mathcal{E}_t = \mathcal{E}_{t-1} + \varepsilon_{\text{skip}}(\hat{t}, t)$，初始化 $\mathcal{E}_{\hat{t}} = 0$。由于前中期 $\gamma_i$ 的标准差极低（Figure 1(b)），这个累积估计是紧致且可靠的
    - 设计动机：TeaCache 在连续跳多步时表现很差，因为它的多项式拟合本质上是逐步预测，连续跳步时预测误差会快速累积失控。MagCache 的误差模型天然支持多步跳过——幅度比的乘积关系保证了即使跳 3-4 步，误差估计仍然准确。这是 MagCache 能在相同加速比下显著胜出 TeaCache 的数学基础

3. **自适应缓存策略（Adaptive Caching Strategy）**:

    - 功能：基于误差模型的估计，自适应决定每个时间步是跳过（复用缓存）还是重新计算
    - 核心思路：在每个时间步 $t$，MagCache 检查两个条件：（1）累积误差 $\mathcal{E}_t \leq \delta$（用户设定的总误差阈值）；（2）$t - \hat{t} \leq K$（从上次刷新起的跳步数不超过最大跳步长度 $K$）。只有两个条件同时满足时才跳过当前步并复用缓存 $\mathbf{r}_{\hat{t}}$；任一条件违反则重置：$\hat{t} \leftarrow t$，$\mathcal{E}_t \leftarrow 0$，重新计算 $\mathbf{r}_t$ 并更新缓存。这是一个双重保险机制。此外，遵循已有工作的做法，前 20% 的去噪步保持不变（不跳过），因为这些初始步对整体生成质量至关重要，且实验观察到这些步的幅度比变化相对较大
    - 设计动机：引入最大跳步长度 $K$ 的约束看似保守，实则是一个精巧的工程设计——虽然幅度比误差模型在统计上非常准确，但毕竟是近似，长序列连续跳步时微小建模误差仍可能累积。$K$ 提供了一个"定期校正"机制，确保模型不会偏离真实残差轨迹太远。实践中，$K=2$ 对应 slow 模式（质量优先），$K=4$ 对应 fast 模式（速度优先），用户通过这两个直观的超参数 $(K, \delta)$ 即可覆盖绝大多数速度-质量需求

### 损失函数 / 训练策略

MagCache 是一个完全 **training-free** 的推理加速框架，不涉及任何训练过程。唯一的"校准"步骤是用 1 个随机 prompt 前向传播一次扩散模型，记录每步的幅度比 $\{\gamma_t\}_{t=1}^{T}$，整个过程耗时等同于生成一个视频。校准得到的幅度比曲线可以复用于同一模型的所有后续推理，无需重复校准。

实现上，MagCache 只需存储一份缓存残差（约 0.5GB 额外显存），相比 TaylorSeer 需要 40GB 额外显存、FasterCache 和 DuCa 也需要数十 GB 额外显存，MagCache 的显存开销极低，真正做到了"即插即用"。

## 实验关键数据

### 主实验

实验覆盖了五个主流生成模型：Open-Sora 1.2（视频）、Wan 2.1 1.3B（视频）、HunyuanVideo（视频）、CogVideoX 2B（视频）和 Flux（图像）。对比方法包括 PAB、T-GATE、Δ-DiT、FasterCache、DuCa、TeaCache、TaylorSeer 等。效率指标使用 FLOPs 和延迟，质量指标使用 LPIPS（越低越好）、SSIM（越高越好）和 PSNR（越高越好）。

| 模型 / 方法 | FLOPs(P) | 加速比 | 延迟(s) | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| **Open-Sora 1.2 (51帧, 480P)** | | | | | | |
| 原始模型 (T=30) | 3.15 | 1× | 44.56 | - | - | - |
| TeaCache-slow | 2.40 | 1.40× | 31.69 | 0.1303 | 0.8405 | 23.67 |
| TeaCache-fast | 1.64 | 2.05× | 21.67 | 0.2527 | 0.7435 | 18.98 |
| **MagCache-slow** | 2.40 | 1.41× | 31.48 | **0.0827** | **0.8859** | **26.93** |
| **MagCache-fast** | 1.64 | **2.10×** | 21.21 | 0.1522 | 0.8266 | 23.37 |
| **Wan 2.1 1.3B (81帧, 480P)** | | | | | | |
| 原始模型 (T=50) | 8.21 | 1× | 187.21 | - | - | - |
| TeaCache-slow | 5.25 | 1.59× | 117.20 | 0.1258 | 0.8033 | 23.35 |
| TeaCache-fast | 3.94 | 2.14× | 87.55 | 0.2412 | 0.6571 | 18.14 |
| **MagCache-slow** | 3.94 | **2.14×** | 87.27 | **0.1206** | **0.8133** | **23.42** |
| **MagCache-fast** | 3.11 | **2.68×** | 69.75 | 0.1748 | 0.7490 | 21.54 |
| **HunyuanVideo (129帧, 540P)** | | | | | | |
| 原始模型 (T=50) | 45.93 | 1× | 1163 | - | - | - |
| TeaCache-slow | 27.56 | 1.63× | 712 | 0.1832 | 0.7876 | 23.87 |
| TeaCache-fast | 20.21 | 2.26× | 514 | 0.1971 | 0.7744 | 23.38 |
| **MagCache-slow** | 20.21 | **2.25×** | 516 | **0.0377** | **0.9459** | **34.51** |
| **MagCache-fast** | 18.37 | **2.63×** | 441 | **0.0626** | **0.9206** | **31.77** |
| **CogVideoX 2B (49帧, 480P)** | | | | | | |
| 原始模型 (T=50) | 2.36 | 1× | 74.10 | - | - | - |
| TeaCache | 1.03 | 2.30× | 32.20 | 0.1221 | 0.8815 | 27.08 |
| **MagCache** | 0.99 | **2.37×** | 31.15 | **0.0787** | **0.9210** | **30.44** |

几个特别值得注意的结果：在 HunyuanVideo 上，MagCache-slow 的 LPIPS 仅 0.0377（TeaCache-slow 为 0.1832，差距 4.9 倍），PSNR 高达 34.51（TeaCache-slow 为 23.87，差距超过 10dB），说明在大模型上 MagCache 的误差控制优势更加显著。在 Wan 2.1 上，MagCache-fast 实现了 2.68× 加速，同时质量指标全面超越 TeaCache-fast（LPIPS 0.1748 vs 0.2412），而且 MagCache-slow 匹配 TeaCache-fast 的加速比（2.14×）同时质量甚至好于 TeaCache-slow。MagCache-fast 在 Open-Sora 上的 PSNR 为 23.37，远高于 TeaCache-fast 的 18.98——当 PSNR 低于 20 时视觉失真非常明显，这说明 TeaCache-fast 在高加速比下几乎不可用，而 MagCache 仍能保持可接受的质量。

### 消融实验

| 模式 | K | δ | 加速比 | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|---|---|
| **MagCache-slow (Wan 2.1)** | | | | | | |
| slow | 2 | 0.06 | 2.0× | 0.0940 | 0.8383 | 24.57 |
| slow | 2 | 0.12 | 2.1× | 0.1053 | 0.8275 | 24.32 |
| slow | 2 | 0.03 | 1.9× | 0.0888 | 0.8427 | 24.68 |
| **MagCache-fast (Wan 2.1)** | | | | | | |
| fast | 4 | 0.06 | 2.4× | 0.1375 | 0.7749 | 22.34 |
| fast | 4 | 0.12 | 2.7× | 0.1625 | 0.7571 | 22.25 |
| fast | 4 | 0.03 | 2.0× | 0.1263 | 0.7828 | 22.51 |

校准 prompt 鲁棒性实验（Wan 2.1，slow 模式）：

| 校准策略 | 加速比 | LPIPS↓ | SSIM↑ | PSNR↑ |
|---|---|---|---|---|
| 随机单 prompt（本文默认） | 2.14× | 0.1206 | 0.8133 | 23.42 |
| 全部 944 个 prompt 平均 | 2.14× | 0.1162 | 0.8163 | 23.52 |
| 最远离群 prompt | 2.21× | 0.1209 | 0.8103 | 23.36 |

### 关键发现

- **最大跳步长度 $K$ 是控制速度-质量 trade-off 的主要旋钮**：$K$ 从 2 增加到 4 时，加速比从 2.0× 跳到 2.4×（Wan 2.1，$\delta=0.06$），但 LPIPS 也从 0.0940 上升到 0.1375。$K$ 决定了"加速档位"（slow vs fast），在同一档位内用 $\delta$ 做精细微调
- **误差阈值 $\delta$ 的影响单调可控**：同一 $K$ 下，降低 $\delta$ 总是提升质量、降低速度，关系平滑可预测。例如 $K=2$ 时，$\delta$ 从 0.12 到 0.03，LPIPS 从 0.1053 改善到 0.0888，加速比从 2.1× 降到 1.9×。通常只需 1-2 次调整就能找到满意的配置
- **校准鲁棒性极强**：用 1 个随机 prompt、944 个 prompt 的平均、甚至最远离群 prompt 校准，三种策略的性能几乎完全一致（LPIPS 差异 < 0.005，PSNR 差异 < 0.2dB）。这彻底证实了统一幅度定律的 prompt 无关性——TeaCache 的 70-prompt 校准实际上是不必要的
- **显存效率碾压级优势**：MagCache 仅需约 0.5GB 额外显存，而 TaylorSeer 在 Wan 2.1 上需要 40GB 额外显存，FasterCache 和 DuCa 也被标注为 "not memory-efficient"。这使得 MagCache 是唯一能在消费级 GPU 上实际部署的高加速比方案
- **在大模型上优势更明显**：在参数量和计算量更大的 HunyuanVideo 上，MagCache 的质量领先优势更加显著（LPIPS 0.0377 vs 0.1832，PSNR 34.51 vs 23.87），说明统一幅度定律在更深、更复杂的模型上同样成立甚至更加精确，这对未来更大规模模型的部署有重要参考价值

## 亮点与洞察

- **从"拟合"到"发现规律"的范式转变**：TeaCache 的思路是"用数据拟合一个预测函数来判断何时跳步"，但这本质上是在用统计方法逼近一个未知的底层规律。MagCache 直接找到了这个底层规律（统一幅度定律），用第一性原理代替数据驱动的拟合。这就像牛顿发现万有引力定律 vs 用查表法预测天体轨道的区别——前者不仅更准确、更鲁棒，而且简洁得多
- **乘法链式误差建模是关键数学创新**：跳步误差 $\varepsilon_{\text{skip}} \approx 1 - \prod \gamma_i$ 这个公式看似简单，但它优雅地解决了连续多步跳过的误差估计问题。TeaCache 的逐步预测在连续跳步时误差爆炸，而 MagCache 的乘法结构天然保证了多步跳过的精度，使得 $K=4$ 的 fast 模式成为可能
- **"方向不变、幅度递减"的物理直觉可迁移**：这篇论文揭示了扩散模型去噪过程中残差的内在结构——前 80% 的步骤中，残差方向几乎锁定，只有幅度在缩小。这个发现不仅可以用于缓存加速，还暗示了扩散模型学到的"去噪路径"具有某种低维结构，未来可以用于模型压缩、步骤蒸馏等其他加速方向
- **$(K, \delta)$ 双参数设计提供了直观的控制接口**：$K$ 选"档位"（slow/fast），$\delta$ 做"微调"，这种层次化的参数设计让用户只需 1-2 次实验就能找到满意配置，特别适合集成到 ComfyUI 等交互式工具中

## 局限与展望

- **仅验证了视频/图像生成任务**：作者在结论中承认，目前只在视频生成和图像生成模型上验证了统一幅度定律和 MagCache 的有效性，尚未扩展到其他扩散模型任务（如 3D 生成、音频生成、图像编辑等）。幅度定律是否在非生成任务（如扩散模型用于判别任务）中也成立，有待探索
- **前 20% 时间步无法加速**：遵循先前工作的做法，MagCache 保持前 20% 的去噪步不变。虽然有经验理由（这些步对整体质量至关重要且幅度比变化较大），但这意味着理论加速上限被硬性限制。如果能找到在初始阶段也安全跳步的方法，加速比还能进一步提升
- **统一幅度定律是经验规律而非理论证明**：目前幅度比的单调递减特性和 prompt 不变性都是通过实验观察得到的，缺乏理论解释。为什么扩散模型的残差幅度会呈现这种规律？是 flow matching 训练目标的固有属性，还是模型架构带来的？如果未来出现违反这一规律的模型（例如采用完全不同训练范式的扩散模型），MagCache 可能会失效
- **固定校准曲线不适应动态场景**：MagCache 使用一条固定的幅度比曲线来做所有推理的决策。虽然实验证明了对不同 prompt 的鲁棒性，但如果推理过程中引入了额外控制（如 ControlNet、IP-Adapter），残差动态可能改变，固定曲线可能不再适用。一个改进方向是轻量级的在线自适应——在推理过程中实时计算幅度比而非依赖离线缓存
- **与步数减少方法的联合使用未充分探索**：MagCache 与蒸馏、低精度计算的兼容性已在附录中初步验证，但与 consistency model、progressive distillation 等步数减少方法的联合使用尚未探索。如果扩散模型本身只需 4-8 步（如 LCM），MagCache 的跳步收益可能大幅缩水

## 相关工作与启发

- **vs TeaCache**：TeaCache 是 MagCache 最直接的竞争对手和改进目标。TeaCache 用 70 个 curated prompt 拟合多项式来预测跳步函数，MagCache 发现底层的统一幅度定律后只需 1 个样本校准。核心区别在于"拟合 vs 发现规律"——TeaCache 是在拟合一个近似函数，MagCache 是在利用一个内在规律。这使得 MagCache 在校准成本（1 vs 70 prompts）、鲁棒性（跨 prompt 不变 vs 可能过拟合）、多步跳过精度（乘法链式 vs 逐步预测）三个维度全面胜出。在 HunyuanVideo 上，MagCache-slow 的 LPIPS 仅为 TeaCache-slow 的 1/5（0.0377 vs 0.1832），差距悬殊
- **vs DeepCache / PAB**：这些是均匀缓存策略的代表。它们用固定间隔（如每 N 步）复用缓存，不考虑不同时间步的动态差异。在 Open-Sora 上，PAB-slow 的 LPIPS 为 0.1471，MagCache-slow 为 0.0827，说明自适应策略相比均匀策略的优势是根本性的。这些方法设计简洁但天花板明显——当加速比提升时质量下降严重
- **vs FasterCache / DuCa / TaylorSeer**：这些方法虽然也做了自适应缓存，但需要缓存大量中间特征，导致额外显存开销巨大（TaylorSeer 需要 40GB）。MagCache 只缓存一份残差（~0.5GB），在显存效率上有数量级的优势。这在消费级 GPU 部署中至关重要
- **与扩散模型加速的大方向联系**：MagCache 属于"inference-time 免训练加速"范畴，与蒸馏（需重训练、需数据）、量化（需校准、有精度损失）互补而非替代。论文附录已初步验证了与低精度计算的兼容性，未来可以叠加使用进一步提升效率

## 评分

- 新颖性: ⭐⭐⭐⭐ 统一幅度定律是一个简洁优美的发现，但"用残差变化指标决定跳步"这个思路是 TeaCache 已经开创的，MagCache 更多是找到了更好的指标
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖了 5 个模型（4 个视频 + 1 个图像）、7+ 个对比方法，消融全面（$K$、$\delta$、校准策略），并且在每个模型上都展示了一致的优势
- 写作质量: ⭐⭐⭐⭐⭐ 从经验发现 → 理论建模 → 方法设计 → 实验验证的逻辑链非常清晰流畅，Figure 1 的可视化直观有力地支撑了核心 claim
- 价值: ⭐⭐⭐⭐⭐ 实用价值极高——即插即用、无需训练、显存开销极低、2×+ 加速、质量损失可控，已在 GitHub 开源并支持 Wan 2.2、FramePack 等最新模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SURF: Signature-Retained Fast Video Generation](../../CVPR2026/video_generation/surf_signature-retained_fast_video_generation.md)
- [\[CVPR 2026\] Transition Matching Distillation for Fast Video Generation](../../CVPR2026/video_generation/transition_matching_distillation_for_fast_video_generation.md)
- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2025\] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](../../CVPR2025/video_generation/from_slow_bidirectional_to_fast_autoregressive_video_diffusion_models.md)
- [\[CVPR 2026\] FastLightGen: Fast and Light Video Generation with Fewer Steps and Parameters](../../CVPR2026/video_generation/fastlightgen_fast_and_light_video_generation_with_fewer_steps_and_parameters.md)

</div>

<!-- RELATED:END -->
