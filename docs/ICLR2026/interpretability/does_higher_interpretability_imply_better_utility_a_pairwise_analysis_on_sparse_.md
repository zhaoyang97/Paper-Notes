---
title: >-
  [论文解读] Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders
description: >-
  [ICLR 2026][可解释性][Sparse Autoencoder] 作者训练了 90 个 SAE 做系统对比，发现"特征更可解释"与"引导效果更好"之间只有弱正相关（τ_b≈0.30），并提出 **ΔToken Confidence** 特征筛选准则把引导分数提升 52.52%；而在筛出的高效特征上，可解释性与引导效用的相关性彻底消失甚至变负。
tags:
  - "ICLR 2026"
  - "可解释性"
  - "Sparse Autoencoder"
  - "LLM Steering"
  - "Interpretability-Utility Gap"
  - "Feature Selection"
  - "Kendall's τ"
---

# Does Higher Interpretability Imply Better Utility? A Pairwise Analysis on Sparse Autoencoders

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q4ooLNOFeR](https://openreview.net/forum?id=Q4ooLNOFeR)  
**代码**: [https://github.com/Xu0615/SAE4Steer](https://github.com/Xu0615/SAE4Steer)  
**领域**: 可解释性 / SAE 特征引导 (Interpretability & Steering)  
**关键词**: Sparse Autoencoder, LLM Steering, Interpretability-Utility Gap, Feature Selection, Kendall's τ  

## 一句话总结
作者训练了 90 个 SAE 做系统对比，发现"特征更可解释"与"引导效果更好"之间只有弱正相关（τ_b≈0.30），并提出 **ΔToken Confidence** 特征筛选准则把引导分数提升 52.52%；而在筛出的高效特征上，可解释性与引导效用的相关性彻底消失甚至变负。

## 研究背景与动机
**领域现状**：稀疏自编码器（SAE）把 LLM 隐藏态分解成稀疏、人类可读的"特征"，是当前可解释性研究的明星工具。一个被默认接受的假设是：既然 SAE 特征可解释，那它们天然就适合用来"引导"（steering）模型行为——往残差流里注入某个特征方向（如把"蛋糕"概念塞进去），就能精确控制输出。大量 SAE-based steering 工作正建立在这个假设上。

**现有痛点**：这个假设从未被严格检验。SAE 的训练目标是重建+稀疏，从来不是下游引导效用；所以"可解释"和"好用"很可能是两回事。但社区普遍用可解释性分数当作引导效用的代理指标，缺乏量化证据支撑。

**核心矛盾**：**更高的可解释性，是否真的意味着更强的引导效用？** 如果两者相关性弱，那么以可解释性为导向训练 SAE 就无助于提升控制能力；如果连"有用的特征"上两者都不相关，那说明可解释性与效用是根本割裂的两个维度。

**本文目标**：用大规模成对（pairwise）实验量化可解释性与引导效用的秩一致性，定位真正能引导的特征，并刻画"可解释性-效用鸿沟"。

**核心 idea**：**(1) 配对秩一致性分析**——训练 90 个 SAE，用 SAEBENCH 测可解释性、AXBENCH 测引导效用，以 Kendall τ_b 衡量两者排序是否一致，并做轴向条件分析剔除混杂因子；**(2) ΔToken Confidence 特征筛选**——不是所有可解释特征都能引导，用"放大单个特征后下一 token 分布的置信度变化量"挑出真正高效用的特征。

## 方法详解

### 整体框架
论文不是提出一个新模型，而是一套四步诊断流程：先给每个 SAE 算可解释性分数和引导分数（S1），做配对分析发现"弱正相关"的鸿沟（S2）；怀疑结论被"很多特征根本不能引导"污染，于是用 ΔToken Confidence 筛出高效用特征（S3）；再在筛选后的特征上重做配对分析（S4），得到"鸿沟反而扩大"的反直觉结论。

```mermaid
flowchart LR
    A[训练90个SAE<br/>3模型×5架构×6稀疏度] --> B[S1 算两个分数<br/>μ可解释性/g引导效用]
    B --> C[S2 配对分析<br/>Kendall τ_b≈0.30<br/>发现弱正相关鸿沟]
    C --> D[S3 ΔToken Confidence<br/>筛选高效用特征]
    D --> E[S4 筛选后重做配对<br/>τ_b≈0 相关性消失]
```

### 关键设计

**1. 配对秩一致性分析：用 Kendall τ_b 量化"可解释性能否预测引导效用"**。作者放弃绝对数值比较，转而问一个更稳健的问题：在一池子 SAE 里，如果 A 比 B 更可解释，A 是否也比 B 更好引导？对每个 SAE θ 记录一对值 $(\mu(\theta), g(\theta))$，其中 μ 是可解释性分数、g 是引导分数。对任意两个 SAE 定义一致性指标 $v_{ij}=\mathrm{sign}(\mu(\theta_i)-\mu(\theta_j))\cdot\mathrm{sign}(g(\theta_i)-g(\theta_j))$，再用 Kendall 的并列校正秩系数 $\tau_b$ 汇总所有无序对的一致性，取值落在 $[-1,1]$。τ_b 越接近 1 说明两个排序越一致。可解释性用 SAEBENCH 的 AutoInterp Score（LLM-as-judge 预测某特征在哪些序列激活的平均精度），引导效用用 AXBENCH 的 Steering Score（对 Concept/Instruction/Fluency 三项各 0-2 打分取调和平均 $\mathrm{HM}(C,I,F)$）。

**2. 轴向条件分析：剔除超参带来的混杂趋势**。全局秩相关可能被"同时影响可解释性和效用"的超参带偏。作者把 SAE 设计空间拆成三条正交轴——架构（A）、稀疏度（B）、基座模型（C），每次只变一条轴、固定其余轴形成匹配分组，组内算 τ_b 再跨组平均得到轴级统计量 $\psi_i=\frac{1}{|\mathcal{G}_i|}\sum_{G\in\mathcal{G}_i}\tau(\{(\mu,g):\theta\in G\})$，最后聚合成轴控制系数 $\Psi=\frac{1}{n}\sum_i\psi_i$。这样能避免"架构整体偏移"之类的跨轴趋势掩盖局部真实关系。结果是全局 τ_b≈0.30，轴控制后 Ψ≈0.25，两者都正但都不强，且严重依赖架构（Gated 甚至为负）、稀疏度（越稀疏越一致，特征越多反而反转）和模型（Qwen 最强、Gemma-2-2B 最弱）。

**3. ΔToken Confidence：用置信度变化量挑出真正能引导的特征**。作者借鉴 LLM 推理里的熵机制思想，认为"放大一个特征后能大幅改变下一 token 分布"的特征才是高效用候选。先定义 top-k token 置信度 $C_k(p)=-\frac{1}{k}\sum_{j\in I_k(p)}\log p_j$（$I_k$ 是概率最大的 k 个 token 下标，C_k 越小越自信），它比熵更直接地刻画头部分布的尖锐度。然后只把某个 SAE 特征 f 的系数放大 α 倍、其余不动，比较干预前后置信度之差 $\Delta C_k(f;\ell,\alpha)=C_k(p^{\mathrm{int}}_{f,\ell,\alpha})-C_k(p^{\mathrm{base}})$。$\Delta C_k<0$ 表示放大该特征让分布更尖锐（模型更确定）。只需一次基线 + 一次干预前向就能算出。最后按 $|\Delta C_k|$ 排序、分档、子集评测、每个 SAE 保留最佳子集——默认 k=1 效果最好。

**4. 对照基线：output-score 选择器**。为公平对比，作者复现 Arad et al. 的输出分数法：用 logit-lens 选出代表 token 集 M，比较干预前后对 M 的聚合支持度 $P(M)=\left(1-\frac{\min_{i\in M}\mathrm{rank}(i)}{|V|}\right)\max_{i\in M}p(i)$，单特征引导分数 $S_{\mathrm{out}}=P_{\mathrm{int}}(M)-P_{\mathrm{base}}(M)$。它衡量"放大特征是否抬高代表 token 的排名和概率"，是当前最强的输出导向选择器，作为 ΔToken Confidence 的主要竞争对手。

## 实验关键数据

### 主实验：特征筛选后的引导分数（Table 2）
在三个 LLM 上用 CONCEPT100 评测，对比无选择、output-score 选择、ΔToken Confidence 选择：

| 方法 | Gemma-2-2B | Qwen-2.5-3B | Gemma-2-9B |
|------|:---:|:---:|:---:|
| SAE-based（无选择） | 0.133 | 0.171 | 0.142 |
| +Output（Arad et al.） | 0.233 | 0.292 | 0.255 |
| **+ΔC_k（本文）** | **0.328** | **0.399** | **0.289** |

ΔToken Confidence 在三模型上全面超越无选择基线和 output-score 选择器，相对最强竞品平均提升 **52.52%**。

### 配对分析：筛选前 vs 筛选后（Table 1 vs Table 3）

| 阶段 | Overall τ_b | 轴控制 Ψ | 结论 |
|------|:---:|:---:|------|
| 筛选前 g_base | 0.2979 | 0.2499 | 弱正相关，存在可解释性-效用鸿沟 |
| 筛选后 g_high | 0.0823 | 0.0681 | 相关性消失，统计上与 0 无异 |

轴向细节（筛选前）：架构 Ψ_A≈0.26（Gated 为 −0.20 拖后腿，JumpReLU 最高 0.42）；稀疏度 Ψ_B≈0.17（L0≈50 最一致 0.54，L0≈520 反转为 −0.22）；模型 Ψ_C≈0.33（Qwen 0.46 最强）。

### 关键发现
- **可解释性是引导效用的弱代理**：τ_b≈0.30，正相关但远不足以当代理指标。
- **ΔToken Confidence 稳定挑出高效用特征**：五种架构中 BatchTopK 提升最稳定、最显著。
- **鸿沟在高效用特征上不缩反扩**：一旦聚焦最有用的特征，可解释性高低完全无法预测引导好坏（τ_b≈0），甚至可能负相关。

## 亮点与洞察
- **戳破了一个社区默认假设**：可解释性 ≠ 可控性。这对"用 SAE 做安全引导"的整条技术路线是重要的警示——别再拿可解释性分数当引导效用的代理。
- **规模化的实证严谨性**：90 个 SAE、3 模型 × 5 架构 × 6 稀疏度，配合置换检验 p 值、bootstrap 置信区间、轴向条件分析剔除混杂，结论可信度远高于小样本观察。
- **ΔToken Confidence 简洁有效**：只需一次额外前向，无需训练或标注，直接从分布尖锐度变化挑特征，既是工程上的实用工具，也佐证了"效用源于对输出分布的实际影响而非语义可读性"。
- **反直觉的核心发现**：筛选后鸿沟反而扩大，说明可解释性与效用是两个正交维度，最有用的特征往往恰恰不是最可读的。

## 局限与展望
- **效用定义被限定**：utility 仅指 AXBENCH 协议下的 steering 效果，未覆盖其他下游任务（如分类探针、知识编辑），结论的外推性需谨慎。
- **固定中间层、固定字典宽度**：SAE 只训在单一中间层、16k 宽度，层选择和宽度对鸿沟的影响未充分探索。
- **只回答"是什么"，未解决"怎么办"**：论文诊断出鸿沟却没给出"既可解释又高效用"的训练范式，作者也明确把"利用效用导向的 SAE 训练目标"留作未来工作。
- **ΔToken Confidence 的理论依据偏经验**：借用熵机制直觉，为何置信度变化能预示引导效用缺乏更深的理论刻画。

## 相关工作与启发
- **表示级引导**（activation steering）：往残差流注入方向控制行为，轻量但因多义性（polysemanticity）而粗糙；本文以稀疏可解释的 SAE 特征为方向并加效用筛选来缓解。
- **SAE-based steering**：用 decoder atom 作引导方向；本文系统质疑了"可解释即好用"的前提。
- **熵/置信度机制**（Fu et al., Wang et al.）：原用于评估 LLM 推理质量，本文创造性迁移到 SAE 特征选择。
- **启发**：评估可解释性方法时，应把"下游效用"作为独立维度单独衡量，而非默认可解释性会自动带来效用；未来 SAE 训练或许需要显式平衡重建、可解释性与效用三个目标。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首次大规模量化"可解释性-效用鸿沟"，并给出反直觉的"筛选后鸿沟扩大"结论，问题本身就极具价值。
- **实验充分度**: ⭐⭐⭐⭐⭐ 90 个 SAE 跨 3 模型 5 架构 6 稀疏度，配合置换检验、bootstrap CI、轴向条件分析，统计严谨。
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰、图表到位，公式与流程交代完整；部分轴向结论较密集需细读。
- **价值**: ⭐⭐⭐⭐⭐ 直接挑战 SAE-steering 的核心假设，对可解释性社区和安全引导实践都有方向性影响，并附带一个即插即用的高效特征选择器。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability](temporal_sparse_autoencoders_leveraging_the_sequential_nature_of_language_for_in.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[ICLR 2026\] AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features](abstopk_rethinking_sparse_autoencoders_for_bidirectional_features.md)
- [\[ICLR 2026\] On the Limits of Sparse Autoencoders: A Theoretical Framework and Reweighted Remedy](on_the_limits_of_sparse_autoencoders_a_theoretical_framework_and_reweighted_reme.md)
- [\[ICLR 2026\] Sparse Autoencoders Trained on the Same Data Learn Different Features](sparse_autoencoders_trained_on_the_same_data_learn_different_features.md)

</div>

<!-- RELATED:END -->
