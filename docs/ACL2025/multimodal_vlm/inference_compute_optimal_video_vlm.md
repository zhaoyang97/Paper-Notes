---
title: >-
  [论文解读] Inference Compute-Optimal Video Vision Language Models
description: >-
  [ACL 2025 (Long Paper)][多模态VLM][推理计算优化] 首次系统性研究视频VLM推理计算预算的最优分配问题：在固定推理FLOPs下，通过大规模训练扫描（~100k A100小时）和add-interact参数化建模（$R^2$=0.98），确定语言模型大小 $x_N$、帧数 $x_T$ 和每帧视觉token数 $x_V$ 三个维度的最优权衡策略。
tags:
  - "ACL 2025 (Long Paper)"
  - "多模态VLM"
  - "推理计算优化"
  - "视频VLM"
  - "缩放定律"
  - "帧数"
  - "视觉token数"
  - "模型大小"
  - "参数化建模"
---

# Inference Compute-Optimal Video Vision Language Models

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2505.18855](https://arxiv.org/abs/2505.18855)  
**代码**: [github/vvlm_inference_scaling](https://github.com/tt6746690/vvlm_inference_scaling)  
**领域**: 多模态VLM  
**关键词**: 推理计算优化, 视频VLM, 缩放定律, 帧数, 视觉token数, 模型大小, 参数化建模  
**作者**: Peiqi Wang (MIT), ShengYun Peng (Georgia Tech), Xuewen Zhang, Hanchao Yu, Yibo Yang, Fujun Liu, Qifan Wang (Meta), Lifu Huang (UC Davis)

## 一句话总结

首次系统性研究视频VLM推理计算预算的最优分配问题：在固定推理FLOPs下，通过大规模训练扫描（~100k A100小时）和add-interact参数化建模（$R^2$=0.98），确定语言模型大小 $x_N$、帧数 $x_T$ 和每帧视觉token数 $x_V$ 三个维度的最优权衡策略。

## 研究背景与动机

**领域现状**：视频VLM（如LLaVA-Video、Qwen2-VL）已广泛应用于推荐系统、内容审核等工业场景，每天需处理百万级视频。对于这类应用，预训练成本已被开源社区吸收，微调成本相对推理可忽略不计（论文估算月推理成本可达微调的340倍），因此推理FLOPs成为主导的运营开销。

**现有痛点**：(a) 视频VLM部署前需确定三个关键设计参数——语言模型大小 $x_N$、帧数 $x_T$、每帧token数 $x_V$——但此前缺乏系统性的分配策略；(b) 已有工作（如Du et al., 2024）只研究 $x_T$ vs $x_V$ 的权衡，忽略了模型大小 $x_N$ 这一关键维度；(c) 此前工作普遍忽视视觉编码器的推理计算成本，仅计算LM成本，导致高估增加帧数的收益；(d) 没有工作探索微调数据量 $n$ 如何与缩放因子 $x$ 交互影响最优前沿。

**核心矛盾**：实际部署中需要在有限推理预算内做出最优的三维分配决策，但三个因子间存在复杂的交互效应，单独优化任何一个因子都是次优的。

**本文目标** 给定固定推理计算预算 $c$ 和微调数据量 $n$，如何选择最优的 $(x_N, x_T, x_V)$ 使任务性能最大化？

**切入角度**：借鉴Chinchilla在训练计算最优方面的研究范式——通过大规模扫描+参数化建模+约束优化——将其迁移到推理计算最优问题上，且关键区别在于微调数据量 $n$ 不影响推理成本但可能影响最优配置。

## 方法详解

### 整体框架

推理计算优化问题形式化为：

$$x^*(c; n) = \arg\min_{x \in \mathcal{X},\; c(x) \leq c} f(x, n)$$

其中 $f(x, n)$ 为下游任务误差，$c(x)$ 为推理计算成本。整体流程分三步：(1) 大规模训练扫描收集经验数据点 $(x, n, f)$；(2) 拟合参数化性能模型；(3) 求解约束离散优化问题得到最优前沿 $x^*(c; n)$。

### 关键设计

1. **包含视觉编码器的推理成本模型**:

    - 推理FLOPs计算为：$c(x) = 2x_T(x_M \cdot x_W + x_N \cdot x_V)$
    - 其中 $x_M$ 为视觉模型参数量，$x_W$ 为视觉特征数。对于SoViT-400M，这变为 $c(x) = 2x_T(0.43\text{e}9 \cdot 768 + x_N \cdot x_V)$
    - **关键区别**：此前工作仅计算LM成本 $c_\text{LM} = 2x_N x_T x_V$，忽略了 $c_\text{ViT} = 2x_M x_T x_W$。但对于7B模型+$x_V \approx 50$的配置，视觉模型约占总推理FLOPs的50%
    - 设计动机：忽略视觉编码器成本会系统性地高估增加帧数的收益（因为每增加一帧，视觉编码器也需要额外的计算）

2. **add-interact参数化性能模型**:

    - 将任务误差建模为幂律加法形式加交互项：

    $$f(x, n) = \sum_k \alpha_k x_k^{-a_k} + \sum_k \beta_k x_k^{b_k} n^{-d} + \xi n^{-d} + \varepsilon$$

    - 各项含义：$\alpha_k x_k^{-a_k}$ 为随 $x_k$ 增大可减少的误差（数据无限时的幂律衰减）；$\beta_k x_k^{b_k} n^{-d}$ 为缩放因子与数据量的交互项——当 $b_k > 0$ 时，更大的 $x_k$ 意味着更丰富的视觉信息，需要更多数据来充分利用但每个样本的边际收益更大；$\xi n^{-d}$ 为仅依赖数据量的误差项；$\varepsilon$ 为不可约误差
    - 设计动机：简单的加法幂律（add）或乘法幂律（mult）无法捕捉 $x$ 与 $n$ 之间的交互效应。add-interact通过交互项 $\beta_k x_k^{b_k} n^{-d}$ 建模了"更多视觉细节使数据更丰富但也更复杂"这一关键现象

3. **两类训练扫描策略**:

    - **Star sweep**：以高计算配置 $(7.5\text{B}, 32, 196)$ 为中心，每次固定两个因子变化一个，在三个数据量 $\{0.25\text{M}, 0.5\text{M}, 1\text{M}\}$ 上微调。避免昂贵的全网格搜索，更准确估计各因子的缩放指数
    - **IsoFLOP sweep**：在固定推理FLOPs（2、5、15、30 TFLOPs）下调整 $(x_N, x_T, x_V)$，在 $n=2\text{M}$ 上微调。用于识别给定预算下的最优配置，同时作为验证集评估参数化模型的外推能力
    - 扫描范围：$x_N \in \{1\text{B}, 2.8\text{B}, 7.5\text{B}\}$，$x_T \in \{4, 8, 12, 16, 32\}$，$x_V \in \{4, 16, 25, 36, 49, 100, 196\}$

### 模型拟合与验证

- **拟合方法**：在log空间中最小化MSE，使用L-BFGS优化，500次随机初始化取最优解
- **Bootstrap聚合**：100次bootstrap重采样训练基模型，取中位数聚合，解决~100个数据点拟合参数化模型时的高方差问题
- **模型选择**：比较mult、add、add-interact-s、add-interact四种参数化形式，add-interact在分布内CV和外推上均最优

### 弹性分析

引入经济学中的"弹性"概念量化数据量对最优前沿的影响：

$$e_k(c, n) = \frac{\partial x_k^*(c; n)}{\partial n} \cdot \frac{n}{x_k^*(c; n)}$$

$e_T = 0.1$ 表示数据量增加1%时，最优帧数增加0.1%。使用前向差分近似导数，在300个推理预算和100个数据量上取平均。

## 实验关键数据

### 参数化模型对比

| 模型 | CV MSE | CV $E\%$ | CV $R^2$ | 外推 MSE | 外推 $E\%$ | 外推 $R^2$ |
|------|--------|----------|----------|----------|-----------|-----------|
| mult | 1.21 | 1.62 | 0.88 | 6.73 | 3.55 | 0.45 |
| add | 0.56 | 1.11 | 0.94 | 2.04 | 2.15 | 0.83 |
| add-interact-s | 0.24 | 0.80 | 0.97 | 0.94 | 1.32 | 0.92 |
| **add-interact** | **0.20** | **0.77** | **0.98** | **0.95** | **1.33** | **0.92** |

### 关键发现

1. **三维递减回报**：单独增加 $x_N$、$x_T$ 或 $x_V$ 均呈边际递减效应，且递减速率因任务而异
2. **联合缩放至关重要**：从15到30 TFLOPs，$x_N=1\text{B}$的模型增益微乎其微（受模型容量瓶颈），$x_N=7.5\text{B}$则获得显著提升。$x_T$ 和 $x_V$ 存在类似的瓶颈效应
3. **缩放速率不同**：$x_T$（帧数）的边际收益通常大于$x_V$（token数），提升视觉模型效率比减少LM处理每帧的成本更有价值
4. **任务特异性强**：长视频理解（LongVideoBench）偏好增加 $x_T$，细粒度感知（PerceptionTest）偏好增加 $x_V$——**不存在通用分配策略**
5. **数据量改变最优前沿**：随数据量增加，弹性分析显示 $e_N < 0$（减小模型）、$e_T > 0$（增加帧数）、$e_V > 0$（增加token数），趋势跨任务一致但幅度不同
6. **compute-optimal配置优势**：在同等推理FLOPs下，最优配置相比naive配置可提升5-15%的平均任务性能

### 弹性（数据量敏感性）

| 因子 | 平均弹性 | 含义 |
|------|---------|------|
| $x_N$ | -0.22 | 数据↑ → 最优模型大小↓ |
| $x_T$ | +0.17 | 数据↑ → 最优帧数↑ |
| $x_V$ | +0.79 | 数据↑ → 最优token数↑ |

## 亮点与洞察

- **"推理版Chinchilla"定位精准**：将训练计算最优的经典研究范式——训练扫描+参数化建模+约束优化——迁移到推理场景，方法论严谨且具有高度可复现性
- **视觉编码器成本纠偏**：纠正了此前工作（Li et al., 2024; Du et al., 2024）忽视视觉编码器成本的系统性偏差，对小模型+少token配置影响尤其显著
- **交互项的引入关键**：add-interact相比add的核心改进在于建模了缩放因子与数据量的交互——"更多帧/token使数据更丰富但也更复杂"，这是add和mult模型无法捕捉的，且直接影响了最优前沿对数据量的依赖
- **弹性分析实用性强**：从经济学借用的弹性概念提供了直观且可比较的量化指标——$e_V = 0.79$ vs $e_T = 0.17$ 表明最优token数对数据量远比帧数敏感
- **实践指导价值**：对工业部署给出了明确的指导——(1)不要只增大模型，要联合调整三个因子；(2)低预算用小模型+少帧+少token，高预算三者同步增加；(3)随着持续收集微调数据，应逐步增加视觉信息量、适当缩小模型

## 局限与展望

- **仅在LLaVA-like架构验证**：使用Llama-3.2系列(1B/2.8B/7.5B)，8B到70B之间无可用预训练模型，限制了对LM大小效应的完整刻画。是否适用于动态分辨率架构（如Qwen2-VL）未知
- **参数化形式手动选择**：add-interact的函数形式是从4个候选中选出的，可能不是最优的描述；且在部分任务（LongVideoBench、Next-QA）上外推误差较大（$E\% \geq 5\%$）
- **未考虑的缩放因子**：vision encoder大小、下采样方法、训练策略等均被固定，可能是重要的自由度
- **推理成本估计简化**：仅考虑理论FLOPs的prefill阶段，未涉及硬件利用率、decode阶段的memory-bound特性、量化/投机解码等实际部署技术
- **计算资源门槛极高**：~100k A100小时的实验投入，绝大多数研究者无法复现

## 相关工作与对比

- **vs Chinchilla (Hoffmann et al., 2022)**：Chinchilla优化训练计算在模型大小与预训练数据量间的分配；本文优化推理计算在 $x_N$、$x_T$、$x_V$ 间的分配。关键区别：推理场景下 $n$ 不影响计算成本但影响最优配置
- **vs Du et al. (2024)**：仅研究 $x_T$ vs $x_V$ 的权衡，(1)未引入 $x_N$，(2)未计入视觉编码器成本，(3)假设参数化模型但未直接从训练扫描观察趋势
- **vs Li et al. (2024b)**：得出"VLM需要更少视觉token、更多参数"的结论，但忽略视觉编码器成本且不考虑数据量交互
- **vs EfficientNet (Tan & Le, 2019)**：EfficientNet在CNN中提出compound scaling联合缩放宽度/深度/分辨率，本文将类似思想扩展到视频VLM的 $x_N$/$x_T$/$x_V$
- **vs Sardana et al. (2024)**：同样考虑部署成本，但针对LM预训练而非视频VLM微调场景

## 启发与关联

- 这一"推理计算最优"研究范式可推广到任何多模态模型：如何在视觉分辨率、编码器大小和LLM大小间分配推理预算
- "数据量影响最优前沿"意味着在持续收集数据的场景下，部署配置应动态调整——与固定一次的传统做法不同
- $x_T$ 的缩放收益大于 $x_V$ 的发现提示：Token压缩/合并技术虽减少了 $x_V$，但释放出的预算若分配给更多帧可能效果更好
- 对于AuroraCap等方法的公平对比提出了警示：固定 $x_T x_V$（总token数）但在 $x_T$ vs $x_V$ 的分配上不一致会导致不公平比较

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 在视频VLM领域首次系统性研究推理计算最优分配，方法论完整、定位独特
- 实验充分度: ⭐⭐⭐⭐⭐ ~100k A100小时的大规模训练扫描，8个视频任务，4种参数化模型对比，详尽的消融和弹性分析
- 写作质量: ⭐⭐⭐⭐⭐ 数学建模清晰严谨，与Chinchilla的对比帮助理解，Appendix极为详尽（6个附录涵盖FAQ/实现/扫描/拟合/弹性/结果）
- 价值: ⭐⭐⭐⭐⭐ 对视频VLM工业部署有直接且实用的指导意义，方法论可迁移到其他多模态缩放问题

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SITCOM: Scaling Inference-Time COMpute for VLAs](../../NeurIPS2025/multimodal_vlm/sitcom_scaling_inference-time_compute_for_vlas.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)
- [\[ACL 2025\] Evaluating Multimodal Large Language Models on Video Captioning via Monte Carlo Tree Search](mcts_video_captioning_eval.md)
- [\[ICCV 2025\] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](../../ICCV2025/multimodal_vlm/free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)

</div>

<!-- RELATED:END -->
