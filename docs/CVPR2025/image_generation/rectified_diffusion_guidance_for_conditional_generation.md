---
title: >-
  [论文解读] Rectified Diffusion Guidance for Conditional Generation
description: >-
  [CVPR 2025][图像生成][Classifier-Free Guidance] ReCFG 从理论上揭示了标准 Classifier-Free Guidance (CFG) 中两个系数求和为 1 的约束导致生成分布的期望偏移问题，通过放松系数约束并给出 $\gamma_0$ 的闭式解，提供了一种无需重训练、几乎不增加推理开销的后处理方案来校正 CFG 的引导效果。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "Classifier-Free Guidance"
  - "期望偏移"
  - "校正引导"
  - "查找表"
  - "后处理方案"
---

# Rectified Diffusion Guidance for Conditional Generation

**会议**: CVPR 2025  
**arXiv**: [2410.18737](https://arxiv.org/abs/2410.18737)  
**代码**: [https://github.com/thuxmf/recfg](https://github.com/thuxmf/recfg)  
**领域**: 扩散模型 / 条件生成  
**关键词**: Classifier-Free Guidance, 期望偏移, 校正引导, 查找表, 后处理方案

## 一句话总结

ReCFG 从理论上揭示了标准 Classifier-Free Guidance (CFG) 中两个系数求和为 1 的约束导致生成分布的期望偏移问题，通过放松系数约束并给出 $\gamma_0$ 的闭式解，提供了一种无需重训练、几乎不增加推理开销的后处理方案来校正 CFG 的引导效果。

## 研究背景与动机

1. **领域现状**：Classifier-Free Guidance（CFG）是扩散模型条件采样的核心技术，通过在条件和无条件分数函数之间进行插值 $s_{t,\gamma}(x,c) = \gamma \nabla \log q_t(x|c) + (1-\gamma) \nabla \log q_t(x)$ 来增强条件保真度。CFG 被广泛用于几乎所有主流扩散模型（DALL-E、Stable Diffusion、ImageGen 等）。
2. **现有痛点**：尽管 CFG 在实践中极其成功，但从理论角度看存在一个基本缺陷：使用 CFG 的去噪过程无法表示为标准扩散前向过程的逆过程。具体来说，gamma-powered 分布 $q_{t,\gamma}(x|c) = q_t(x|c)^\gamma q_t(x)^{1-\gamma}$ 的分数函数期望非零，违反了扩散模型的基本理论假设。
3. **核心矛盾**：CFG 的两个系数之和为 1（$\gamma + (1-\gamma)=1$）这一约束虽然看起来自然，但实际上导致了生成分布相对于真实条件分布 $q_0(x_0|c)$ 的期望偏移，且引导强度 $\gamma$ 越大偏移越严重。
4. **本文目标**：量化 CFG 的期望偏移现象，并设计一种校正方案使引导过程严格符合扩散理论。
5. **切入角度**：通过 DDIM 理论框架和 toy distribution（$q_0(x_0|c) \sim \mathcal{N}(c,1)$, $q(c) \sim \mathcal{N}(0,1)$），推导出期望偏移的精确闭式表达，从而发现偏移来源是两个分数函数期望的比值。
6. **核心 idea**：放松 $\gamma_1 + \gamma_0 = 1$ 的约束，让 $\gamma_0$ 自由取值。通过零期望条件 $\mathbb{E}[\gamma_1 \epsilon_\theta(x_t,c,t) + \gamma_0 \epsilon_\theta(x_t,t)] = 0$ 推导出 $\gamma_0$ 的闭式解，可从预计算的查找表中直接获取。

## 方法详解

### 整体框架

ReCFG 是 CFG 的一种后处理校正。标准 CFG 使用 $\hat{\epsilon} = \gamma \epsilon_\theta(x_t,c,t) + (1-\gamma) \epsilon_\theta(x_t,t)$，ReCFG 将其替换为 $\hat{\epsilon} = \gamma_1 \epsilon_\theta(x_t,c,t) + \gamma_0 \epsilon_\theta(x_t,t)$，其中 $\gamma_0$ 不再是 $1-\gamma_1$，而是根据条件 $c$ 和时间步 $t$ 从查找表中获取的动态值。查找表通过遍历训练数据预计算得到，推理时直接查表，速度几乎无影响。

### 关键设计

1. **期望偏移的理论证明（Theorem 1 & 2）**:
    - 功能：从理论上精确描述 CFG 导致的生成分布偏移
    - 核心思路：Theorem 1 证明了 CFG 虽然与标准扩散训练目标兼容（up to constant），但其去噪过程不是前向扩散的逆过程，原因是无条件分数函数在条件分布下的期望 $\mathbb{E}_{q_t(x_t|c)}[\nabla \log q_t(x_t)]$ 非零。Theorem 2 在 toy distribution 下精确计算了偏移量：当 $T \to \infty$ 时，CFG 生成分布的期望为 $c \cdot \phi(\gamma)$，其中 $\phi(1)=1, \phi(3)=2, \phi(\gamma) \geq 2$ for $\gamma>3$。这意味着 $\gamma=3$ 时期望已经是真值的 2 倍。
    - 设计动机：将 CFG 的理论缺陷从"已知但被忽视"提升为"被精确量化并可修正"的问题。

2. **校正引导系数 ReCFG（Theorem 3 & 闭式解）**:
    - 功能：放松系数约束，消除期望偏移
    - 核心思路：将引导公式推广为 $s_{t,\gamma_1,\gamma_0}(x,c) = \gamma_1 \otimes \nabla \log q_t(x|c) + \gamma_0 \otimes \nabla \log q_t(x)$，其中 $\gamma_1, \gamma_0 \in \mathbb{R}^D$ 是逐像素的、与条件和时间步相关的函数，$\otimes$ 表示逐元素乘法。设计约束：(1) $\gamma_{1,i} > 1$ 确保条件保真度增强；(2) 期望偏移为零：$\mathbb{E}[\gamma_1 \epsilon(x_t,c,t) + \gamma_0 \epsilon(x_t,t)] = 0$；(3) 方差不大于真实分布（更集中的更好）。由条件 (2) 推导出闭式解：$\gamma_0 = (1-\gamma_1) \cdot \mathbb{E}[\epsilon_\theta(x_t,c,t)] / \mathbb{E}[\epsilon_\theta(x_t,t)]$。Theorem 4 在 toy distribution 下验证了方差确实变小（当 $\gamma_{0,i} \leq 0$ 且 $\gamma_{1,i}+\gamma_{0,i} \geq 1$ 时）。
    - 设计动机：不改变 $\gamma_1$（保持原有的引导强度控制），而是通过调整 $\gamma_0$ 来补偿偏移。这使得 ReCFG 可以无缝替换 CFG，用户仍然只需要调一个 $\gamma_1$ 超参。

3. **预计算查找表**:
    - 功能：实现高效的推理时校正
    - 核心思路：给定条件 $c$，遍历训练数据中 $q_0(x_0|c)$ 的样本，对每个时间步 $t$ 收集 $(\epsilon_\theta(x_t,c,t), \epsilon_\theta(x_t,t))$ 并计算期望比值 $\mathbb{E}[\epsilon_\theta(x_t,c,t)] / \mathbb{E}[\epsilon_\theta(x_t,t)]$，存入查找表。推理时根据 $\gamma_1$ 直接计算 $\gamma_0 = (1-\gamma_1) \times \text{ratio}$。查找表是逐像素的（pixel-wise），每个像素在不同时间步可以有不同的校正系数。
    - 设计动机：闭式解使得 $\gamma_0$ 可以被精确预计算，无需在推理时进行任何额外的前向传播。逐像素的 $\gamma_0$ 实现了比全局 CFG 更灵活精确的引导。

### 损失函数 / 训练策略

ReCFG 不需要任何训练或微调。核心操作是预计算查找表：
1. 对每个类别/条件 $c$，采样若干 $x_0 \sim q_0(x_0|c)$
2. 对每个时间步 $t$，前向加噪得 $x_t$，计算 $\epsilon_\theta(x_t,c,t)$ 和 $\epsilon_\theta(x_t,t)$
3. 计算并存储期望比值
4. 推理时查表乘以 $(1-\gamma_1)$ 即得 $\gamma_0$

## 实验关键数据

### 主实验

ImageNet 512×512 (EDM2):

| 模型 | 方法 | NFE | FD_DINOv2↓ | FID↓ | Precision↑ | Recall↑ |
|------|------|-----|-----------|------|-----------|---------|
| EDM2-S | CFG | 63 | 52.32 | 2.29 | 0.83 | 0.59 |
| EDM2-S | **ReCFG** | 63 | **50.56** | **2.23** | 0.83 | 0.59 |
| EDM2-M | CFG | 63 | 41.98 | 2.12 | 0.81 | 0.60 |
| EDM2-M | **ReCFG** | 63 | **41.55** | **2.06** | 0.81 | 0.61 |
| EDM2-L | CFG | 63 | 38.20 | 1.96 | 0.81 | 0.62 |
| EDM2-L | **ReCFG** | 63 | **36.75** | **1.89** | 0.81 | 0.62 |

CC12M 512×512 (SD3):

| 方法 | γ₁ | NFE | CLIP-S↑ | FD_DINOv2↓ | MPS↑ |
|------|-----|-----|---------|-----------|------|
| CFG | 7.5 | 10 | 0.262 | 1105.51 | 9.828 |
| **ReCFG** | 7.5 | 10 | **0.263** | **1010.14** | **10.250** |
| RescaleCFG + ReCFG | 7.5 | 10 | **0.268** | **979.87** | **11.336** |
| CFG | 5.0 | 10 | 0.268 | 1053.44 | 10.883 |
| **ReCFG** | 5.0 | 10 | **0.269** | **999.48** | **11.031** |

ImageNet 256×256 (DiT-XL/2 & LDM):

| 模型 | 方法 | γ₁ | NFE | FID↓ |
|------|------|-----|-----|------|
| DiT-XL/2 | CFG | 1.50 | 250 | 2.27 |
| DiT-XL/2 | **ReCFG** | 1.50 | 250 | **2.13** |
| LDM | CFG | 5.0 | 20 | 18.87 |
| LDM | **ReCFG** | 5.0 | 20 | **16.95** |
| LDM | CFG | 2.0 | 20 | 5.32 |
| LDM | **ReCFG** | 2.0 | 20 | **4.40** |
| LDM | CFG | 5.0 | 10 | 16.78 |
| LDM | **ReCFG** | 5.0 | 10 | **14.46** |

### 消融实验

| γ₁ 设置 | CFG FID | ReCFG FID | 改善 | 说明 |
|---------|---------|-----------|------|------|
| 5.0 (LDM, 20步) | 18.87 | 16.95 | -1.92 | 大γ下改善最显著 |
| 3.0 (LDM, 20步) | 11.46 | 9.78 | -1.68 | 中等γ也有明显改善 |
| 2.0 (LDM, 20步) | 5.32 | 4.40 | -0.92 | 小γ改善幅度较小但仍一致 |
| 1.5 (LDM, 20步) | 5.36 | 4.78 | -0.58 | 接近1的γ改善最小 |

### 关键发现

- ReCFG 在所有测试的扩散模型（EDM2、SD3、LDM、DiT）上都一致地改善了 FID/FD_DINOv2
- 引导强度 $\gamma_1$ 越大，ReCFG 的改善幅度越大——与理论预测一致（大γ产生更严重的期望偏移）
- ReCFG 与 RescaleCFG 兼容且叠加使用效果更好（SD3 上 MPS 从 9.828 提升到 11.336）
- 查找表的可视化显示：期望比值在不同像素和时间步上差异极大，没有统一的规律，证明了逐像素校正的必要性
- NFE 越少（采样步数少），ReCFG 的改善越明显（10步时比20步改善更大）
- Precision 和 Recall 都获得提升或保持，说明 ReCFG 不是简单的 precision-recall 权衡

## 亮点与洞察

- **严格的理论基础**：不仅指出了 CFG 的缺陷，还通过 5 个定理从多个角度精确描述了问题的本质和解决方案。Theorem 2 给出的 "$\gamma=3$ 时期望偏移 2×" 的结论非常直观震撼。
- **极强的实用性**：完全后处理、零训练成本、零额外推理开销（查表操作可以忽略不计）。可以即时应用到所有使用 CFG 的现有扩散模型上。这种"免费午餐"式的改进非常有吸引力。
- **与 RescaleCFG 的兼容性**：ReCFG 解决期望偏移，RescaleCFG 解决方差问题，两者正交互补，叠加使用效果更佳。
- **逐像素动态系数的启示**：查找表可视化表明，不同空间位置在不同去噪步骤需要不同的引导强度。这挑战了 CFG 使用全局统一系数的做法，暗示未来的引导方法应该更加空间自适应。

## 局限与展望

- 查找表需要遍历训练数据的子集来预计算，对于开放词汇的文本到图像模型，无法覆盖所有可能的条件
- 闭式解基于一阶近似（假设 $\Delta_t=0$ 然后逐步归纳），多步误差可能累积
- Theorem 4 关于方差的保证仅在 toy distribution 下成立，一般情况下方差的行为尚不完全可控
- 可能改进：在线计算期望比值（避免查找表的覆盖限制）、扩展到其他引导形式（如 classifier guidance）、理论分析更一般的分布族

## 相关工作与启发

- **vs Standard CFG**: CFG 使用 $\gamma + (1-\gamma) = 1$ 的约束，会导致期望偏移。ReCFG 放松这一约束并给出闭式校正解，从理论和实验上全面改进。
- **vs RescaleCFG**: RescaleCFG (Lin et al.) 通过缩放输出来缓解 CFG 的过饱和问题，主要处理方差层面。ReCFG 处理期望层面的偏移，两者正交互补。
- **vs APG/CFG++**: 其他改善 CFG 的工作也注意到了引导的理论缺陷，但 ReCFG 是第一个给出精确的闭式修正方案的。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 理论贡献扎实，从已知但被忽视的问题中挖掘出闭式解决方案，数学推导优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 4 个主流扩散模型、多个分辨率和 NFE 设置、类别/文本条件、与其他方法的兼容性
- 写作质量: ⭐⭐⭐⭐⭐ 定理-证明结构严谨，toy example 直观易懂，查找表可视化有启发性
- 价值: ⭐⭐⭐⭐⭐ 即插即用改善所有 CFG 模型，零成本提升 FID，极高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Accelerating Diffusion via Hybrid Data-Pipeline Parallelism Based on Conditional Guidance Scheduling](../../CVPR2026/image_generation/accelerating_diffusion_via_hybrid_data-pipeline_parallelism_based_on_conditional.md)
- [\[CVPR 2025\] OmniFlow: Any-to-Any Generation with Multi-Modal Rectified Flows](omniflow_any-to-any_generation_with_multi-modal_rectified_flows.md)
- [\[CVPR 2025\] Towards Transformer-Based Aligned Generation with Self-Coherence Guidance](towards_transformer-based_aligned_generation_with_self-coherence_guidance.md)
- [\[CVPR 2025\] Conditional Balance: Improving Multi-Conditioning Trade-Offs in Image Generation](conditional_balance_improving_multi-conditioning_trade-offs_in_image_generation.md)
- [\[CVPR 2025\] JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation](janusflow_harmonizing_autoregression_and_rectified_flow_for_unified_multimodal_u.md)

</div>

<!-- RELATED:END -->
