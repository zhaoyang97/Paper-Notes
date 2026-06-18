---
title: >-
  [论文解读] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
description: >-
  [CVPR2026][多模态VLM][MoE加速] 提出 MoDES，首个面向 MoE 多模态大模型的训练免调专家跳过框架，通过全局调制的局部门控（GMLG）和双模态阈值（DMT）机制自适应跳过冗余专家，在跳过 88% 专家时仍保留 97%+ 原始性能，并实现 2.16× prefill 加速。 MoE MLLM 推理瓶颈：…
tags:
  - "CVPR2026"
  - "多模态VLM"
  - "MoE加速"
  - "专家跳过"
  - "多模态大模型"
  - "训练免调"
  - "推理效率"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

**会议**: CVPR2026  
**arXiv**: [2511.15690](https://arxiv.org/abs/2511.15690)  
**代码**: [ModelTC/MoDES](https://github.com/ModelTC/MoDES)  
**领域**: 多模态VLM  
**关键词**: MoE加速, 专家跳过, 多模态大模型, 训练免调, 推理效率

## 一句话总结

提出 MoDES，首个面向 MoE 多模态大模型的训练免调专家跳过框架，通过全局调制的局部门控（GMLG）和双模态阈值（DMT）机制自适应跳过冗余专家，在跳过 88% 专家时仍保留 97%+ 原始性能，并实现 2.16× prefill 加速。

## 研究背景与动机

**MoE MLLM 推理瓶颈**：MoE 多模态大模型（如 Qwen3-VL-MoE-30B）通过稀疏激活降低计算量，但每个 token 仍需与多个被激活专家交互，推理开销依然可观。

**现有专家跳过方法失效**：NAEE、MC-MoE、DiEP 等方法原为单模态 LLM 设计，直接应用于 MLLM 在跳过 83% 专家时精度下降超 10%。

**层间贡献不均（Insight i）**：浅层专家对最终输出的贡献远大于深层——浅层引入的误差会被后续层放大，但现有方法仅依据层内路由概率做跳过决策，忽视了全局层级重要性。

**模态间行为差异（Insight ii）**：文本 token 与视觉 token 在 FFN 中的更新幅度显著不同——视觉 token 与 FFN 权重更正交（角度接近 90°），因此受 FFN 影响更小，冗余度更高。

**缺乏多模态感知的跳过策略**：先前工作对所有模态采用统一阈值，未考虑文本/视觉 token 的不同特性，导致跳过策略不合理。

**阈值搜索代价高昂**：暴力搜索双模态阈值需要 $\mathcal{O}(ND^2)$ 时间复杂度，对 20-30B 参数模型需数天才能完成。

## 方法详解

### 整体框架

MoDES 想在不重新训练的前提下，给 MoE 多模态大模型做专家跳过加速。它的逻辑是先衡量每个专家"跳了亏不亏"，再按 token 模态决定到底跳谁：全局调制的局部门控（GMLG）算出带全局层级权重的专家重要性分数，双模态阈值（DMT）对文本和视觉 token 用各自的阈值做跳过决策，而两个阈值由一个前沿搜索算法离线高效求出。整套流程训练免调（training-free），校准与搜索都在离线完成，推理时不引入额外开销。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    subgraph OFF["离线准备（训练免调，推理零额外开销）"]
        direction TB
        C["校准集 C<br/>(GQA 1024 样本)"]
        C --> AL["GMLG·全局调制因子 α(l)<br/>逐层量 KL 散度算层级重要性"]
        C --> FS["前沿搜索<br/>借单调性双指针定最优 τt, τv"]
    end
    OFF --> X["输入 token x(l)<br/>MoE top-k 路由 → 局部概率 πi"]
    X --> GM["GMLG·重要性分数<br/>si = α(l) · πi（全局×局部）"]
    GM --> DMT["DMT·双模态阈值跳过<br/>文本用 τt / 视觉用 τv，s 低于阈值即跳"]
    DMT --> OUT["保留专家计算 → 输出"]
```

### 关键设计

**1. GMLG：把全局层级重要性塞进局部路由概率，纠正"浅层乱跳"**

现有方法只看层内路由概率决定跳谁，可浅层专家引入的误差会被后续层层层放大，统一对待浅深层会跳错。GMLG 把局部路由概率 $\pi_i^{(l)}$（softmax 归一化）乘上一个全局调制因子 $\alpha^{(l)}$，得到真正的重要性分数：

$$s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$$

其中 $\alpha^{(l)}$ 衡量"整层跳掉对最终输出的伤害有多大"，做法是在校准集 $\mathcal{C}$ 上量原始模型与跳掉第 $l$ 层专家后模型的输出分布之间的 KL 散度均值：

$$\alpha^{(l)} = \frac{1}{N}\sum_{j=1}^{N}\mathcal{D}_{\text{KL}}(\text{prob}_j \| \text{prob}_j^{(l)})$$

校准只用 GQA 的 1024 个样本离线跑一次。这样浅层因为 $\alpha^{(l)}$ 大而被"保护"住，深层冗余专家则更容易被识别出来跳掉，跳过决策从纯局部升级为全局感知。

**2. DMT：给文本和视觉 token 各自一把跳过尺子**

论文发现视觉 token 与 FFN 权重更正交（夹角接近 90°）、受 FFN 影响更小、冗余度更高，但旧方法对所有模态用同一个阈值，结果要么误伤文本要么放过视觉。DMT 干脆为文本和视觉分别设阈值 $\tau_t$、$\tau_v$，重要性分数低于对应模态阈值的专家才跳：

$$\{\text{Expert}_i^{(l)} \mid s_i^{(l)} < \tau_t \cdot \mathbb{I}_t + \tau_v \cdot \mathbb{I}_v\}$$

视觉 token 因冗余更高通常拿到更大的 $\tau_v$、被跳得更狠，文本 token 则更谨慎，跳过策略由此变得模态感知。

**3. 前沿搜索：把双阈值搜索从数天压到数小时**

暴力搜最优 $(\tau_t, \tau_v)$ 要 $\mathcal{O}(ND^2)$，20-30B 模型得跑好几天。论文把它建成"在目标跳过率 $\rho$ 约束下最小化 KL 散度"的优化问题，利用约束函数 $f$ 与目标函数 $g$ 关于阈值的单调性，用双指针在前沿集合上扫，复杂度降到 $\mathcal{O}(ND)$，相比暴力搜索约 45× 加速，搜索时间从数天压到数小时以内。

## 实验

### 主实验：Kimi-VL-A3B-Instruct 上 13 个基准的对比

| 方法 | 跳过率 | ChartQA | MME | MMBench | LVB | VMMMU | Avg.(%) |
|------|--------|---------|-----|---------|-----|-------|---------|
| 默认 k=6 | 0% | 89.48 | 2207 | 83.16 | 63.13 | 49.33 | 100.00 |
| DiEP | 83% | 78.31 | 2071 | 76.28 | 52.41 | 43.81 | 87.58 |
| MC-MoE | 83% | 80.25 | 2063 | 73.42 | 54.39 | 44.02 | 88.32 |
| **MoDES** | **83%** | **84.20** | **2162** | **81.44** | **62.60** | **47.11** | **96.25** |

### 跨模型泛化：Qwen3-VL-MoE-30B 上 88% 跳过率

| 方法 | ChartQA | MME | MMBench | VMMMU | Avg.(%) |
|------|---------|-----|---------|-------|---------|
| MC-MoE | 71.43 | 2168 | 75.42 | 37.41 | 86.66 |
| DiEP | 70.51 | 2074 | 73.21 | 34.79 | 85.30 |
| **MoDES** | **78.84** | **2403** | **85.57** | **46.56** | **97.33** |

MoDES 在 88% 激进跳过率下比最强基线 MC-MoE 高出 10.67 个百分点。

### 消融实验

| 配置 | ChartQA | MME | MMBench | LVB | VMMMU |
|------|---------|-----|---------|-----|-------|
| 单阈值基线 | 76.74 | 1956 | 65.48 | 54.67 | 40.33 |
| +GMLG | 79.28 | 2107 | 75.19 | 60.02 | 43.87 |
| +DMT | 82.94 | 2081 | 79.42 | 61.16 | 45.08 |
| +GMLG+DMT（完整） | **84.20** | **2162** | **81.44** | **62.60** | **47.11** |

（83% 跳过率，Kimi-VL-A3B-Instruct）GMLG 和 DMT 均有显著且独立的贡献，且跳过率越高增益越大。

### 关键发现

- **推理加速**：MoDES 在 Qwen3-VL-MoE-30B 上实现 prefill 2.16× 加速，decoding 1.26× 加速。
- **与量化兼容**：MoDES + 2.5-bit 量化在 Qwen3 上仍保留 94.43% 原始性能，MC-MoE 仅 89.58%。
- **跳过模式可视化**：深层跳过率远高于浅层；视觉 token 的专家跳过率远高于文本 token，验证了两个核心 insight。
- **校准数据鲁棒**：换用 COCO 或 VMMMU 作为校准集，性能几乎不变。
- **搜索效率**：前沿搜索 vs 暴力搜索加速 ~45×，20-30B 模型总耗时（校准+搜索）20 分钟到 4 小时以内。

## 亮点

- 首个系统分析 MoE MLLM 中层间贡献不均与模态间行为差异的工作，两个 insight 有充分实验支撑
- GMLG 巧妙地将离线全局校准与在线局部路由结合，推理时无额外开销
- DMT 用模态感知的双阈值替代统一阈值，从动机到设计逻辑清晰
- 前沿搜索算法利用单调性将 $\mathcal{O}(ND^2)$ 降到 $\mathcal{O}(ND)$，实用性强
- 实验覆盖 3 个模型系列 × 13 个基准，跳过 88% 专家时精度损失 <3%

## 局限性

- 仅处理文本/视觉两种模态，未扩展到音频等更多模态场景
- $\alpha^{(l)}$ 为层级别粒度，未区分同层内不同专家的全局重要性差异
- 仅在 image/video understanding 任务上评估，未涉及生成类任务（如 image captioning 质量评估有限）
- Decoding 阶段加速有限（~1.2×），因为 decoding 本身是 memory-bound 且仅处理文本 token
- 前沿搜索依赖单调性假设，虽然实践中合理但缺乏严格理论保证

## 相关工作

- **NAEE** [Lu et al.]：基于路由概率比值跳过次要专家，仅考虑层内信息
- **MC-MoE** [Huang et al., 2024]：在 NAEE 基础上加入 attention-aware 专家保护 + 混合精度量化
- **DiEP** [Bai et al., 2025]：可微专家剪枝，联合路由概率与专家相似度做跳过
- 以上方法均为单模态 LLM 设计，直接迁移到 MLLM 效果差；MoDES 首次针对多模态场景提出全局+模态感知的跳过策略

## 评分

- 新颖性: ⭐⭐⭐⭐ — 两个 insight 有说服力，GMLG+DMT 组合设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ — 3 个模型系列 × 13 基准 × 多跳过率，消融完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机→方法→实验逻辑通顺
- 价值: ⭐⭐⭐⭐ — 对 MoE MLLM 部署有直接实用价值，方法简洁高效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](../../ICML2025/multimodal_vlm/dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)
- [\[ICML 2026\] Toward Structural Multimodal Representations: Specialization, Selection, and Sparsification via Mixture-of-Experts](../../ICML2026/multimodal_vlm/toward_structural_multimodal_representations_specialization_selection_and_sparsi.md)
- [\[ICML 2026\] SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](../../ICML2026/multimodal_vlm/same_stabilized_mixture-of-experts_for_multimodal_continual_instruction_tuning.md)
- [\[ICLR 2026\] Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts](../../ICLR2026/multimodal_vlm/capacity-aware_inference_mitigating_the_straggler_effect_in_mixture_of_experts.md)

</div>

<!-- RELATED:END -->
