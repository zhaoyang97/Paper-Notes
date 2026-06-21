---
title: >-
  [论文解读] More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models
description: >-
  [ICLR 2026][VLM Reasoning][VLM] 论文揭示了多模态推理的"双刃剑"本质——更长的推理虽提升逻辑能力却会因"视觉遗忘"削弱感知 grounding，并提出 VAPO（视觉锚定策略优化），用插入视觉锚点 + 感知奖励的方式把推理拉回视觉证据上，得到新 SOTA 的 VAPO-Thinker-7B。
tags:
  - "ICLR 2026"
  - "VLM Reasoning"
  - "VLM"
  - "多模态推理"
  - "GRPO"
  - "视觉遗忘"
  - "感知奖励"
  - "测试时扩展"
---

# More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=XpL5eqjCjF](https://openreview.net/forum?id=XpL5eqjCjF)  
**代码**: 待确认  
**领域**: 多模态推理 / VLM 强化学习  
**关键词**: VLM, 多模态推理, GRPO, 视觉遗忘, 感知奖励, 测试时扩展  

## 一句话总结
论文揭示了多模态推理的"双刃剑"本质——更长的推理虽提升逻辑能力却会因"视觉遗忘"削弱感知 grounding，并提出 VAPO（视觉锚定策略优化），用插入视觉锚点 + 感知奖励的方式把推理拉回视觉证据上，得到新 SOTA 的 VAPO-Thinker-7B。

## 研究背景与动机
- **领域现状**：在 LLM 上，靠 RL（尤其是 GRPO）训练出的长链推理已成为数学、代码等任务的标配，"想得越久越准"几乎被当成普适规律，社区也顺势把推理迁移到 VLM 上，在几何、导航乃至分类检测等视觉任务上取得了不错的效果。
- **现有痛点**：推理在多模态场景的"副作用"几乎没人系统研究过。已有零散信号显示异常——有工作发现显式推理相比直接回答只带来边际收益，还有工作观察到多模态 RL 训练中推理长度会随准确率上升而"塌缩"变短，暗示推理在 VLM 里并非免费午餐。
- **核心矛盾**：本文通过"提前决策（early decision）"细粒度分析发现，推理早期收益显著，但收益会饱和甚至反转——在 MMStar、HallusionBench 这类视觉密集型基准上，准确率会从峰值跌落 2% 以上，几乎抵消推理带来的好处。错误分析显示，失败案例中**超过 50% 是感知错误**（图表读错、幻觉），而非逻辑错误（33%），且其中很大一部分能靠"提前停止推理"被纠正回来。
- **本文目标**：弄清"想得越狠，看得越差"的根因，并提出一个能从训练层面根治、而非测试时打补丁的方案。
- **核心 idea**：**【因果诊断】** 把根因归结为**视觉遗忘（visual forgetting）**——随推理推进，模型对视觉 token 的注意力持续衰减，决策越来越被自己的历史思考主导而非图像；**【训练层修复】** 提出 VAPO，在推理路径上插入"视觉锚点"，用模型对视觉断言的判断作为**感知奖励**，显式激励推理全程保持视觉 grounding。

## 方法详解

### 整体框架
VAPO 是 GRPO 的多模态替代品：保留 GRPO 的 group-relative 优化骨架，额外引入一条"感知奖励"通路。流程上，先用 GPT-5 为每张图生成一组正误均衡的视觉断言；训练 rollout 时把这些断言作为"锚点"随机插入推理轨迹的不同位置，让模型在每个锚点判断断言真伪，从而探测它当时的感知能力；最后把各锚点得分按"后期加权"聚合成感知奖励，与准确率、格式奖励一起回灌进 GRPO 目标。

```mermaid
flowchart LR
    A[图像 I] --> B[GPT-5 生成<br/>正误均衡的视觉断言]
    B --> C[推理 rollout]
    C --> D[沿轨迹随机插入 K 个锚点<br/>每锚点判断一条断言 yes/no]
    D --> E[逐锚点打分 s_k]
    E --> F[后期加权聚合<br/>感知奖励 R_perc]
    F --> G[R = R_acc + R_fmt + γ·R_perc]
    G --> H[GRPO 优化]
```

### 关键设计

**1. 视觉断言生成：用"独立可验证"的探针隔离感知能力。** 要把"模型此刻还看不看图"测准，断言必须满足两个条件：一是**均衡**，正误断言各占一半以免评估偏倚；二是**独立**，对断言的判断只能依赖视觉输入而非历史推理。为此作者用 GPT-5 为每个样本生成多样且正误平衡的断言，并且**只给图、不给原问题**，强制断言的验证纯粹建立在感知理解之上，避免模型靠上下文"蒙"出答案。

**2. 视觉锚点插入：把推理轨迹切成可探测的感知检查点。** 给定一条推理轨迹 $o_i = (o_{i,1}, \dots, o_{i,T})$，在其中随机分布一组锚点 $A_i = \{a_1, \dots, a_K\}$，$a_k \in [1,T]$。每到一个锚点 $a_k$，从对应断言池采样一条断言 $c_k$ 接到截断的前缀推理上下文之后，探测模型的二元判断并对照真值标签 $l_k$ 打分：

$$s_k = \mathbb{1}\!\left[\arg\max_{j\in\{\text{yes},\text{no}\}} \pi_\theta(j \mid q, I, o_{i,<a_k}, c_k) = l_k\right]$$

这相当于在推理的不同阶段反复给模型做"你现在还看得清图吗"的小测验，从而把抽象的"感知 grounding"变成可量化的逐点信号。

**3. 后期加权的感知奖励：精准打击推理末端的视觉衰减。** 由于感知能力恰恰在推理后期最弱，作者对各锚点得分做"后期强调"的加权聚合，给靠后的锚点更大权重：

$$R_{\text{perc}} = \frac{\sum_{k=1}^{K} w_k s_k}{\sum_{k=1}^{K} w_k}, \qquad w_k = \exp\!\left(\beta \cdot \frac{a_k}{T}\right)$$

其中 $\beta$ 控制对后期锚点的强调程度（默认 $\beta=1.5$，约 50% 权重压在最后 30% 的锚点上）。

**4. 准确率门控防 reward hacking。** 感知奖励只是辅助信号，直接叠加会让模型钻空子——靠生成极短推理来"轻松刷高"感知分。作者因此对感知奖励加上准确率条件门控，最终序列奖励为：

$$R_i = R_{\text{acc}} + R_{\text{fmt}} + \gamma \cdot \mathbb{1}[R_{\text{acc}}=1] \cdot R_{\text{perc}}$$

只有当答案正确（$R_{\text{acc}}=1$）时感知奖励才生效（$\gamma=0.1$），既鼓励视觉 grounding，又堵住"用平凡短推理换感知分"的捷径。其余优化流程沿用第 4.1 节的 GRPO 目标。

## 实验关键数据

实现细节：基座为 Qwen2.5-VL（3B/7B），在 ViRL39K 上训练 2 epoch，lr=5e-6，每样本采 5 个 response（温度 1.0）；默认锚点数 $K=20$，$\beta=1.5$，$\gamma=0.1$。

### 主实验表格

数学类基准（7B 规模，平均）：

| 模型 | MathVerse | MathVista | MathVision | LogicVista | WeMath | Geo3k | Avg. |
|------|-----------|-----------|------------|------------|--------|-------|------|
| Qwen2.5-VL-7B | 40.7 | 62.3 | 23.2 | 42.6 | 33.1 | 38.5 | 40.1 |
| R1-OneVision-7B | 46.4 | 64.1 | 29.9 | 45.6 | 44.6 | 46.1 | 46.1 |
| VLAA-Thinker-7B | 48.2 | 68.0 | 26.4 | 48.5 | 41.5 | 50.6 | 47.2 |
| Vision-R1-7B | 52.4 | 73.5 | 28.2 | 49.7 | 41.6 | 49.0 | 49.1 |
| **VAPO-Thinker-7B** | **53.3** | **75.6** | **31.9** | **50.9** | 43.6 | **51.3** | **51.1** |

通用基准（7B 规模，平均）：

| 模型 | MMMU | MMStar | Hall | MMVet | Avg. |
|------|------|--------|------|-------|------|
| Vision-R1 | 57.6 | 61.4 | 49.5 | 71.1 | 59.9 |
| **VAPO** | **60.2** | **63.0** | **57.4** | **71.9** | **63.1** |

数学类平均提升约 2%（49.1→51.1），通用类提升更明显，达 3.2%（59.9→63.1），刷新 SOTA。

### 消融实验表格

与"测试时补丁"（FP=focus prompt，VR=visual replay）对比：

| 模型 | WeMath | Geo3k | MMStar | Hall | Avg. |
|------|--------|-------|--------|------|------|
| V-R1 + FP | 42.1 | 49.7 | 61.8 | 50.5 | 51.0 |
| V-R1 + VR | 42.5 | 50.5 | 62.1 | 51.8 | 51.7 |
| **VAPO** | **43.6** | **51.3** | **63.0** | **57.4** | **53.8** |

超参消融（图 7）：锚点数 $K$ 从 0（退化为 vanilla GRPO）增到 20 时平均准确率快速上升并饱和；后期权重 $\beta$ 在 1.5 时性能最佳。

### 关键发现
- **双刃剑曲线**：准确率随推理进度先升后降，视觉密集型基准（MMStar、HallusionBench）跌幅最大。
- **感知错误是主因**：full reasoning 失败案例中感知错误占 55.23%，逻辑错误仅 33.05%，且 32.35% 的感知错误能靠 early decision 救回。
- **视觉遗忘可视化**：vanilla 推理中视觉 token 注意力随生成步数单调衰减至接近 0；visual replay / focus prompt 在插入点能触发注意力瞬时回升，验证了视觉遗忘是核心瓶颈。
- **VAPO 拉平衰减**：相比基线，VAPO 的注意力衰减更平缓、后期维持更高水平，准确率在推理后期不再下滑反而稳步上升。

## 亮点与洞察
- **反直觉但扎实的诊断**：把"想得越久越准"这一被默认普适的信念在多模态场景下证伪，并用注意力曲线 + 错误归因 + early decision 三条证据链锁定"视觉遗忘"根因，问题定义本身就是贡献。
- **从测试时补丁升级到训练范式**：visual replay / focus prompt 只是用来验证假设的"探针式"补丁；VAPO 则把视觉 grounding 直接写进奖励，证明了根治需要在训练阶段动手。
- **后期加权 + 准确率门控**两个小设计很对症：前者精准打击推理末端的视觉衰减，后者堵住"短推理刷感知分"的 reward hacking 漏洞。

## 局限与展望
- **依赖外部强模型**：视觉断言由 GPT-5 生成，断言质量、成本与潜在偏置都受制于外部模型，未讨论用开源模型或自举生成的可行性。
- **断言只是感知代理**：二元 yes/no 断言判断是否能完整刻画"感知 grounding"仍存疑，复杂空间关系、细粒度属性可能难以用单条断言覆盖。
- **提升幅度偏温和**：数学类平均仅 +2%，3B 模型上优势更小，方法在小模型与纯逻辑任务上的边际收益有限。
- **未触及推理-感知更深层机制**：视觉遗忘被描述为注意力衰减现象，但为何长推理会系统性地挤压视觉注意力，其架构/训练层面的成因仍是开放问题。

## 相关工作与启发
- **与 LLM 推理范式的张力**：测试时扩展（TTS）在 LLM 上"越长越好"的信念被本文证明不能直接外推到 VLM，提醒社区在跨模态迁移训练范式时要重新验证基本假设。
- **承接视觉遗忘/文本偏置研究**：早期 VLM 的 text bias、modality imbalance 此前多用对比解码、注意力重分配等测试时手段缓解；本文指出推理时代长输出放大了该问题，并给出训练层解法。
- **启发**：在推理过程中"周期性强制回看证据源"的思路（视觉锚点 + 过程奖励）可迁移到其他多模态/检索增强场景——凡是长生成会逐渐脱离原始输入证据的任务，都可借鉴用过程级 grounding 奖励来对抗"证据遗忘"。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 首个系统揭示 VLM 推理"双刃剑"并定位"视觉遗忘"根因，问题诊断 + 过程级感知奖励的组合有原创性。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 10 个基准、3B/7B 双规模、对照测试时补丁，并有注意力可视化、错误归因、超参消融，证据链完整；唯小模型与数学任务增益偏小。
- **写作质量**: ⭐⭐⭐⭐ 叙事清晰，"现象→诊断→验证→解法"层层递进，图表（注意力曲线、错误分布）支撑有力。
- **价值**: ⭐⭐⭐⭐ 纠正了"推理越长越好"的认知误区，VAPO 简单可落地、即插替代 GRPO，对多模态推理训练有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning](../../CVPR2026/vlm_reasoning/select_less_reason_more_prioritizing_evidence_purity_for_video_reasoning.md)
- [\[ICLR 2026\] ThinkMorph: Emergent Properties in Multimodal Interleaved Chain-of-Thought Reasoning](thinkmorph_emergent_properties_in_multimodal_interleaved_chain-of-thought_reason.md)
- [\[CVPR 2026\] Improving Vision-language Models with Perception-centric Process Reward Models](../../CVPR2026/vlm_reasoning/improving_vision-language_models_with_perception-centric_process_reward_models.md)
- [\[ICLR 2026\] Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks](beyond_classification_accuracy_neural-medbench_and_the_need_for_deeper_reasoning.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)

</div>

<!-- RELATED:END -->
