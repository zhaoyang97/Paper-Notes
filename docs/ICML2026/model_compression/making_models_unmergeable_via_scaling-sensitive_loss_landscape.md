---
title: >-
  [论文解读] Making Models Unmergeable via Scaling-Sensitive Loss Landscape
description: >-
  [ICML2026][模型压缩][不可合并性] TRAP² 把「不可合并性」在微调阶段直接写进发布的权重更新里——通过对「更新缩放因子 $s$」做对抗优化，让模型在授权的 $s=1$ 处保持高可用、却在 $s\neq1$（合并管线普遍引入的离标缩放）处迅速崩坏，从而既不依赖 Transformer 的架构对称性、也不需要完整权重访问，对 LoRA 适配器和全量 checkpoint、Transformer 和非 Transformer 骨干一视同仁地防止未授权的模型合并。
tags:
  - "ICML2026"
  - "模型压缩"
  - "不可合并性"
  - "模型合并防护"
  - "LoRA"
  - "损失景观"
  - "缩放敏感"
---

# Making Models Unmergeable via Scaling-Sensitive Loss Landscape

**会议**: ICML2026  
**arXiv**: [2601.21898](https://arxiv.org/abs/2601.21898)  
**代码**: 待确认  
**领域**: 模型合并 / 模型保护 / 模型压缩  
**关键词**: 不可合并性, 模型合并防护, LoRA, 损失景观, 缩放敏感

## 一句话总结
TRAP² 把「不可合并性」在微调阶段直接写进发布的权重更新里——通过对「更新缩放因子 $s$」做对抗优化，让模型在授权的 $s=1$ 处保持高可用、却在 $s\neq1$（合并管线普遍引入的离标缩放）处迅速崩坏，从而既不依赖 Transformer 的架构对称性、也不需要完整权重访问，对 LoRA 适配器和全量 checkpoint、Transformer 和非 Transformer 骨干一视同仁地防止未授权的模型合并。

## 研究背景与动机

**领域现状**：Hugging Face、GitHub 等模型仓库让微调更新（从全量 checkpoint 到 LoRA 轻量适配器）广泛流通。由于大量更新基于同一个底座，发布后可以直接组合参数——这就是「模型合并」（Task Arithmetic、TIES-Merging、DARE、以及 LoRA 专用的 KnOTS、Core Space 等）。合并让能力复用变得极其方便。

**现有痛点**：但这种模块化制造了一个**治理缺口**：一旦发布，下游用户可以把权重重新组合成未授权的混合体，绕过安全对齐、许可条款或任务约束。理想的「不可合并性」要求发布的模型在单独使用时保持完整效用，一旦被并入未授权混合就可靠地失效。现有防御几乎都是**事后（post-hoc）**的——对完整权重做保持功能的重参数化来扰乱合并。代表是 PaRaMS 和 Merge-Lock，它们利用 Transformer 注意力投影的成对对称性（见下文公式），保留单独行为却降低权重空间合并的兼容性。

**核心矛盾**：事后防御有两个根本缺陷。第一，它们绑死 Transformer 的架构对称性，转不到 ResNet、ConvNeXt 这类非注意力骨干。第二，它们假设能拿到**完整权重张量**来施加成对变换——可现实的 hub 生态里，大量发布是 LoRA 这种只放适配器、底座权重不可见的形式。只对适配器 $\Delta W$ 做成对变换无法抵消与冻结底座 $W_0$ 相关的项，要么损害单独效用、要么诱导不出预期的不可合并性。

**本文目标**：能不能把不可合并性**直接注入发布的更新本身**，做到跨架构、跨发布格式（仅适配器 / 全量 checkpoint），且不依赖底座访问和架构特定对称性？

**切入角度**：作者观察到一个被忽视的统一视角——几乎所有合并算子本质都在对每个成员更新**重新缩放**（如 $N$ 个更新平均，等于每个乘 $1/N$）。那么只要让更新对「缩放」敏感，就能让合并失败。

**核心 idea**：用「更新缩放因子 $s$」当合并过程的简单代理，训练一个在 $s=1$ 处可用、在 $s\neq1$ 处崩坏的「缩放敏感损失景观」。

## 方法详解

### 整体框架
TRAP²（Training-time Protection via Task-Robust Adversarial Perturbation）是一个在微调阶段就塑造发布更新的训练目标。它的核心直觉是：把更新 $\Delta W$ 正则化成「在标称缩放 $s=1$ 处仍然准确，但在离标缩放 $s\neq1$ 处退化」——而 $s\neq1$ 正是适配器合并（如线性组合）普遍诱导的状态。

形式化地，设 $\ell(W;\xi)$ 为样本损失，缩放后损失定义为 $L_{\text{scaled}}(\Delta W;s):=L(W_0+s\cdot\Delta W)$，$s=1$ 是标称（单独部署）缩放。TRAP² 对一个离标缩放分布 $\mathcal{S}$（支撑集为 $[s_{\min},1-\delta]\cup[1+\delta,s_{\max}]$，$\delta$ 是 $s=1$ 周围的排除边界）采样，最小化的总目标是：

$$J(\Delta W) = L_{\text{nominal}}(\Delta W) - \lambda \cdot L_{\text{off}}(\Delta W),$$

其中 $L_{\text{nominal}}(\Delta W)=L_{\text{scaled}}(\Delta W;1)$ 保住单独部署性能，$L_{\text{off}}(\Delta W)=\mathbb{E}_{s\sim\mathcal{S}}[w(s)\cdot L_{\text{scaled}}(\Delta W;s)]$ 诱导对离标缩放的敏感度，$\lambda>0$ 权衡两者。注意第二项前是**减号**——这是在「最大化离标损失」，本质是一个对抗目标。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["微调更新 ΔW (LoRA 或全量)"] --> B["缩放即合并代理<br/>采样离标 s∈[smin,1-δ]∪[1+δ,smax]"]
    B --> C["缩放敏感对抗目标<br/>min Lnominal − λ·Loff"]
    C --> D["权重函数 w(s)=1/s<br/>校准下缩放梯度"]
    D --> E["受保护更新 ΔW⋆<br/>s=1 可用 / s≠1 崩坏"]
    E -->|被未授权合并(平均≈×1/N)| F["合并模型系统性退化"]
```

### 关键设计

**1. 缩放即合并代理：用一维 $s$ 把「不可合并」变成可优化目标**

不可合并性难做，是因为合并发生在下游、创建者控制不了、协议还五花八门。TRAP² 的破局点是发现「合并 ≈ 重新缩放」这个统一视角：把 $N$ 个适配器线性聚合 $\Delta W_{\text{merged}}=\sum_i s_i\cdot\Delta W_i$，每个成员都被某个系数缩放；均匀平均更是把每个更新乘 $1/N$，直接把合并后的适配器推离它的标称工作点 $s=1$。于是作者不去建模复杂多变的合并算子，而是用一维缩放因子 $s$ 当代理：只要训练出的更新对 $s$ 偏离 1 敏感，它在几乎任何合并下都会脆。这把一个「下游不可控」的难题，压缩成一个「在训练时对 $s$ 做对抗」的可优化目标，也正是它能跨架构、跨格式的根源——缩放是所有合并算子的公约数，与是不是 Transformer、是不是只有适配器无关。

**2. 缩放敏感损失景观 + 排除边界 $\delta$：授权处保高、未授权处挖坑**

目标 $J$ 刻意在不同缩放间制造**不对称**：在 $s=1$ 处压低 $L_{\text{nominal}}$ 保住性能，在 $s\neq1$ 处经 $L_{\text{off}}$ 抬高损失。这就把损失景观塑造成「$s=1$ 是一根尖峰式的可用山头、两侧迅速塌陷」的形状（论文 Figure 2：$s=1$ 处准确率高、$s\neq1$ 处崩塌）。关键的工程细节是排除边界 $\delta$：离标缩放的采样支撑集**挖掉** $s=1$ 附近的 $[1-\delta,1+\delta]$，否则对抗项会紧贴标称点、把单独效用一起拖垮。这样既给「授权使用」留出安全裕度，又让稍微偏离就掉进坑里——而合并诱导的缩放（如 $1/N$）通常远离 1，正好落在挖坑区。

**3. 缩放权重 $w(s)=1/s$：补偿下缩放的梯度消失**

只对 $s$ 采样还不够：在下缩放区（$s<1$），$L_{\text{scaled}}(\Delta W;s)$ 关于 $\Delta W$ 的梯度会随 $s$ 变小而变小，导致训练信号在最关键的「平均合并 $s=1/N<1$」区域微弱。作者引入非负权重函数 $w(s)$ 归一化跨缩放的训练信号，默认取 $w(s)=1/s$ 来补偿这一效应、稳定训练。直觉上 $1/s$ 恰好抵消缩放带来的梯度衰减，让下缩放区（合并最常落入的区域）获得足够强的对抗梯度。算法上（Algorithm 1）每步只对 $s$ 做单样本 Monte Carlo 采样，给出 $J$ 的无偏梯度估计，用 SGD 一阶优化即可；论文附录给出该过程在标准假设下的平稳性保证。

**4. 跨格式与跨架构的统一实例化**

同一目标天然覆盖两种发布格式。适配器版（LoRA）直接训练 $\Delta W=BA$；全量微调版定义随步更新 $\Delta W_t:=W_t-W_0$，每步在 $W_t$ 处算标称损失、在 $W_0+s\cdot\Delta W_t$ 处算离标损失再更新 $W_t$，得到一个架构无关的统一形式，对适配器和全量 checkpoint 都适用。理论侧作者还刻画了两类退化：**下缩放退化**（均匀平均把每个适配器乘 $1/N$，TRAP² 显式放大此处损失，合并必然系统性退化）和**跨适配器退化**（一个 TRAP² 适配器与一个独立训练的无保护适配器合并，会把后者也推离其标称缩放，连无保护适配器都跟着掉点，且退化随两者在参数空间的距离增大）。

### 损失函数 / 训练策略
核心即上文目标 $J(\Delta W)=L_{\text{nominal}}-\lambda\cdot L_{\text{off}}$。超参含缩放范围 $[s_{\min},s_{\max}]$、排除宽度 $\delta$、权衡权重 $\lambda$、权重函数 $w(s)=1/s$。每次迭代采一个 mini-batch 和一个 $s\sim\text{Unif}([s_{\min},1-\delta]\cup[1+\delta,s_{\max}])$，按 $\nabla_{\Delta W}J$ 用 SGD 更新。

## 实验关键数据

### 主实验
在 8 个视觉分类基准（Cars、DTD、EuroSAT、GTSRB、MNIST、RESISC、Aircraft、SVHN）、3 个 CLIP 骨干（ViT-B/32、ViT-L/14、ConvNeXt）上评测。**第一关是单独可用性**：保护后的适配器单独部署时准确率必须不掉。下表为各方法的 8 任务平均准确率（%，越高越好）：

| 骨干 | Zero-Shot | Fine-Tuned | Merge-Lock⋆ | PaRaMS⋆ | PaRaMS† | TRAP²(本文) |
|------|------|------|------|------|------|------|
| ViT-B/32 | 42.48 | 88.12 | 6.25 | 87.64 | 84.39 | **88.13** |
| ViT-L/14 | 60.63 | 92.40 | 4.43 | 85.08 | 83.89 | **93.71** |
| ConvNeXt | 54.56 | 90.61 | — | — | — | **90.66** |

TRAP² 的单独准确率与无保护 Fine-Tuned 持平甚至略高（ViT-L/14 甚至 93.71 > 92.40），而 Merge-Lock 单独部署就直接崩到个位数（说明其适配器变体根本保不住单独效用）；ConvNeXt 上 PaRaMS/Merge-Lock 因依赖 Transformer 对称性直接不适用，TRAP² 却照常工作——这正是「架构无关」的直接证据。

### 不可合并性（合并后退化）
**第二关是合并时崩坏**：把目标任务的受保护适配器与 7 个无保护适配器做 8 路合并，用 TA/TIES/TIES+DARE/TSV/CART 五种算子、Full/KnOTS/Core 三种合并空间，报告合并后平均准确率（%，**越低越好**，对保护方有利）。作者甚至对合并系数 $s\in\{0.1,...,10.0\}$ 做验证集搜索，让结果对「合并者」最优化（即给攻击方最大便利）。

| 配置（ViT-B/32, TA-Full） | 合并后准确率 | 说明 |
|------|------|------|
| Unprotected | 48.27 | 无保护，合并仍可用 |
| TRAP²(本文) | 显著低于无保护并标红 | 合并后大幅退化，逼近 Zero-shot(42.48) |

在多种合并算子和空间下，TRAP² 受保护适配器合并后的准确率被压到 95% 无保护基线以下（论文中标红），印证了缩放敏感设计能可靠破坏未授权合并。Figure 3 进一步显示：沿「无保护 Cars 适配器 ↔ TRAP² GTSRB 适配器」的插值路径，两个任务的准确率都随合并加深而崩——即跨适配器退化是真实发生的。

### 关键发现
- **单独可用与合并崩坏可以兼得**：TRAP² 是唯一同时做到「单独准确率不掉、合并后大幅退化」的方法；事后防御要么保不住单独效用（Merge-Lock），要么换骨干/换格式就失效（PaRaMS 在 ConvNeXt 不适用）。
- **架构无关是最硬的优势**：在 ConvNeXt 这种非 Transformer 骨干上，所有事后基线直接出局，只有 TRAP² 能用——因为它防的是「缩放」而非「注意力对称性」。
- **$w(s)=1/s$ 与排除边界 $\delta$ 是落地关键**：前者补偿下缩放区梯度消失，后者给授权使用留裕度，两者缺一会让单独效用和保护强度顾此失彼。

## 亮点与洞察
- **「合并 ≈ 重新缩放」的统一视角极其漂亮**：把一个下游不可控、协议异构的难题，压缩成对一维缩放因子 $s$ 的对抗，瞬间获得跨架构、跨格式的普适性——这是全文最核心的「啊哈」。
- **训练时防护 vs 事后防护的范式转变**：把保护写进微调过程而非事后改权重，天然兼容只发布适配器、底座不可见的真实 hub 场景，绕开了事后法对完整权重的硬依赖。
- **可迁移的思路**：「找到下游攻击/复用的公约数操作（这里是缩放），在训练时对它做对抗」这套范式，可推广到其他「发布后失控」的保护问题，如防蒸馏、防未授权微调。

## 局限与展望
- **保护与可用的权衡靠超参手调**：$\lambda$、$\delta$、$[s_{\min},s_{\max}]$ 的选择直接决定「单独效用保多少、合并崩多狠」，论文未充分讨论其敏感性与自动选取。
- **缩放代理的覆盖边界**：若某种合并算子并不显著改变成员更新的有效缩放（如某些对齐/投影后再精细配比的方案），缩放敏感性可能不足以触发崩坏——对「不重新缩放」的合并是否仍鲁棒值得验证。
- **自己发现的局限**：实验集中在 CLIP 视觉分类 8 任务，未在大语言模型 LoRA 合并这一最热门、最受治理关切的场景上验证；「让模型在未授权时崩坏」本身是双刃剑，也可能被滥用来锁死本应开放复用的更新。

## 相关工作与启发
- **vs PaRaMS / Merge-Lock**：它们是事后法，靠 Transformer 注意力的成对可逆重参数化（$W_Q\mapsto W_Q R_1, W_K\mapsto W_K R_1^{-\top}, W_V\mapsto W_V R_2, W_O\mapsto R_2^{-1}W_O$）保持功能等价、扰乱合并；但绑死 Transformer 且需完整权重。TRAP² 改在训练时防护、只塑造发布更新，跨架构跨格式，单独效用还更稳。
- **vs Task Arithmetic / TIES / DARE 等合并法**：这些是 TRAP² 要抵御的「攻击面」——它们都通过加权求和聚合更新（常配剪枝/符号冲突消解），正因为都隐含重新缩放，才被 TRAP² 的缩放敏感设计一网打尽。
- **vs KnOTS / Core Space**：LoRA 专用合并法，在低秩子空间做结构化投影/对齐；TRAP² 在实验里把它们当作合并空间（KnOTS/Core）一并测试，验证保护在这些更精细的合并方案下依然成立。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 「合并即缩放」的统一代理 + 训练时对抗，把不可合并性从架构特定一举推到架构/格式无关。
- 实验充分度: ⭐⭐⭐⭐ 8 任务 × 3 骨干 × 5 合并算子 × 3 空间覆盖扎实、且给合并者最优搜索，但缺 LLM 场景。
- 写作质量: ⭐⭐⭐⭐⭐ 问题设定、缩放代理直觉、理论退化分析层层递进，公式与图配合清楚。
- 价值: ⭐⭐⭐⭐⭐ 直击开源模型治理缺口，给「发布即失控」提供了首个跨架构跨格式的训练时防护范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[NeurIPS 2025\] Geometry of Decision Making in Language Models](../../NeurIPS2025/model_compression/geometry_of_decision_making_in_language_models.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](../../ICLR2026/model_compression/lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)

</div>

<!-- RELATED:END -->
