---
title: >-
  [论文解读] Human Uncertainty-Aware Data Selection and Automatic Labeling in Visual Question Answering
description: >-
  [ICLR 2026][多模态VLM][人类不确定性(Human Uncertainty)] 本文系统揭示了 VQA 中"人类不确定性(HU)"对监督微调的影响——高 HU 样本不仅无益甚至有害，并提出 HaDola 框架，通过"判别-自标注-错误触发-训练"四阶段流水线，仅用 5% 种子标注就在准确率和校准度上匹敌甚至超越用 100% 数据微调的强基线。
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "人类不确定性(Human Uncertainty)"
  - "视觉问答"
  - "监督微调"
  - "数据高效"
  - "模型校准"
  - "自动标注"
---

# Human Uncertainty-Aware Data Selection and Automatic Labeling in Visual Question Answering

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=LuZjiUNuFL](https://openreview.net/forum?id=LuZjiUNuFL)  
**代码**: [https://github.com/emsuno/hadola](https://github.com/emsuno/hadola)  
**领域**: 多模态视觉语言模型 / VQA / 数据选择与标注  
**关键词**: 人类不确定性(Human Uncertainty)、视觉问答、监督微调、数据高效、模型校准、自动标注  

## 一句话总结
本文系统揭示了 VQA 中"人类不确定性(HU)"对监督微调的影响——高 HU 样本不仅无益甚至有害，并提出 HaDola 框架，通过"判别-自标注-错误触发-训练"四阶段流水线，仅用 5% 种子标注就在准确率和校准度上匹敌甚至超越用 100% 数据微调的强基线。

## 研究背景与动机
**领域现状**：大型视觉语言模型(VLM)在 VQA 上表现强劲，但主流的监督微调(SFT)范式高度依赖海量人工标注，成本高昂。同时，VQAv2、VizWiz 这类数据集天然带有"人类不确定性"——同一图像-问题对，10 个标注者可能给出不同答案，且每个答案附带不同的置信度(yes/maybe/no 映射为 0.99/0.5/0.01)。

**现有痛点**：标准 SFT 只朝着"最高频答案"优化，完全抛弃了 HU 分布信息；而且不加区分地堆数据，只关心"用多少数据"，从不追问"每个样本到底贡献了什么"。这导致两个问题：模型既学不到真实的人类不确定性分布(校准差)，又把标注预算浪费在可能有害的样本上。

**核心矛盾**：HU 究竟是该被抹平的"噪声"，还是可被利用的"信号"？而且 HU 标注本身极贵、大规模难以获取，如何在仅有少量 HU 标注的前提下既保准确率又保校准？

**本文目标**：回答三个研究问题——HU 如何影响 SFT、哪些是有益/有害样本(RQ1)；如何把 HU 融入训练以兼顾准确率和校准(RQ2)；如何只用一小部分 HU 标注维持强性能(RQ3)。

**核心 idea**：**把 HU 当作数据选择与评估的指导信号，而非待消除的噪声**。作者先做系统评估发现"高 HU 样本有害、低/中 HU 样本才是有效监督"，再据此设计一个从 5% 种子集自演化的数据高效框架 HaDola，主动剔除高 HU 样本、自动标注信息量大的样本，并用定制损失对齐人类不确定性分布。

## 方法详解

### 整体框架
HaDola 是一个模型无关的迭代框架：先用 5% 的人工 HU 标注种子集 $S_0$ 微调初始 VLM 得到参考模型 $M_{HU}$(冻结一份作 HU 参考，另一份作训练起点)，然后在每一轮里对未标注池 $S_r$ 依次执行四个阶段——判别(剔高 HU)、自标注(打伪标签)、错误触发(过滤坏伪标签)、训练(定制损失微调)，逐轮把可靠监督扩展出去，呈"自演化"式增长。

```mermaid
flowchart LR
    A[5% 种子集 S0<br/>人工 HU 标注] --> B[SFT 得参考模型 M_HU]
    B --> C[① 判别 Discriminate<br/>按 KL 区间剔除高HU/离群样本]
    C --> D[② 自标注 Self-Annotate<br/>上一轮模型打伪标签]
    D --> E[③ 错误触发 Error Trigger<br/>梯度一致性 + TracIn 过滤]
    E --> F[④ 训练 Training<br/>L_HaDola 定制损失]
    F -->|迭代 T 轮| C
    F --> G[最终模型 M_T]
```

在展开关键设计前，先交代两个度量基础：作者用 HUD 衡量样本级 HU，并按 HUD 区间把数据均匀切成低/中/高三档([0.66,0.99] 为低、(0.33,0.66) 为中、[0.01,0.33] 为高)；同时指出传统 VQA-acc 纯按频率算分会忽视 HU，于是提出 HU 加权的 HU-acc $= \text{HaConf}(a)\times\text{VQA-Acc}(a)$ 作为更敏感的评估与监督信号。

### 关键设计

**1. 判别(Discriminate)：用 KL 区间把高 HU 样本挡在门外**
这是 HaDola 的"守门人"，目的是把既不利于学习、又白白耗费算力的高 HU 样本筛掉。作者先在种子集 $S_0$ 上用参考模型 $M_{HU}$ 算出低、中 HU 子集与真实人类分布之间的平均 KL 散度 $\tau_1$、$\tau_2$(满足 $\tau_1<\tau_2$)，以及低/中子集上的平均人类置信分布 $h_\omega$。对每个候选未标注样本 $u$(每轮取 $S_r$ 的 1%)，计算当前模型 $M_t$ 与 $h_\omega$ 的 KL 散度 $kl_u = D_{KL}(h_\omega \| M_t(u))$，只保留落在 $[\tau_1-\sigma,\ \tau_2+\sigma]$ 区间内的样本($\sigma$ 为 KL 分数标准差)，区间外的视为高 HU 或离群点直接丢弃。直觉是：好样本的不确定性应当和低/中 HU 种子集对齐，偏离太远说明要么太"难"(高 HU)要么是噪声。

**2. 自标注(Self-Annotate)：用上一轮模型自动扩展监督**
判别保留下来的样本没有人工标签，HaDola 用上一轮模型 $M_{t-1}$ 直接给出预测 $\hat{y}_u = M_{t-1}(u)$，构造伪训练对 $(u, \hat{y}_u)$。这一步把人工标注从"每个样本都要"降到"只需 5% 种子"，是大幅省标注成本的关键——模型在迭代中自我精炼监督信号，而非被动等待人工。

**3. 错误触发(Error Trigger)：双重梯度准则防止伪标签误差累积**
自标注会引入错误，反复迭代可能滚雪球，因此 HaDola 设了两道闸。其一是**梯度一致性**：计算伪样本梯度 $g(u,\hat{y}_u;\theta_t)$ 与种子集平均参考梯度 $g_{ref}(\theta_t)$ 的余弦相似度 $s_g = \frac{\langle g,\ g_{ref}\rangle}{\|g\|\|g_{ref}\|}$，要求伪标签产生的更新方向和可靠人工监督一致。其二是 **TracIn-mini 影响力估计**：只用初始模型 $\theta_0$ 和当前模型 $\theta_t$ 近似 $s_{tracin}(u,\hat{y}_u) \approx \langle g(u,\hat{y}_u;\theta_0), \nabla_\theta L_{val}(\theta_0)\rangle + \langle g(u,\hat{y}_u;\theta_t), \nabla_\theta L_{val}(\theta_t)\rangle$，追踪伪样本对验证损失的全局影响。只有同时满足 $s_g \ge \tau_g$ 且 $s_{tracin} \le \tau_t$ 的伪样本才被保留(阈值由低/中 HU 子集标定)。消融显示这一步去掉后性能下降最大，证明它对抑制误差累积至关重要。

**4. 定制训练损失：兼顾准确率与人类不确定性校准**
训练阶段用一个三项损失同时追求"答对"和"对齐人类不确定性"：
$$L_{HaDola} = \mathbb{E}[\text{CE}(y, M_\theta)] + \beta\,\Phi + \lambda\big(D_{KL}(H\|M_\theta) - D_{KL}(H\|M_{HU})\big),\quad \Phi = D_{KL}(M_{HU}(\cdot|x)\|M_\theta(\cdot|x))$$
第一项标准交叉熵保证预测正确；第二项 $\Phi$ 把 $M_\theta$ 正则到 HU 参考模型 $M_{HU}$，防止它漂离 HU 知情的基线；第三项以"相对参考模型"的方式比较 $M_\theta$ 与 $M_{HU}$ 跟人类分布 $H$ 的对齐程度，鼓励 $M_\theta$ 比参考更逼近人类不确定性，从而提升校准。用 CE 替换这套损失后准确率明显下降，说明显式对齐人类偏好确有必要。

## 实验关键数据

### 主实验
在 VQAv2 与 VizWiz 上评测 Qwen2.5VL-2B/7B、LLaVA1.6-7B、InternVL2.5-2B/8B 及任务专用 SOTA BEiT3，对比 6 类基线(零样本、LoRA SFT、Meta 伪标注、主动学习、DPO、选择性预测 LYP)，指标为 HU-acc(准确率)与 KL 散度(校准)。

| 结论 | 表现 |
|------|------|
| 三个有零样本能力的 VLM | HaDola 仅用 **5% 标注** 即超越 **100% 标注的 SFT** |
| BEiT3(无零样本能力) | 不及全量 SFT，但达到可比水平且远超其余所有基线 |
| KL 散度(校准) | HaDola 与 LYP 稳定领先其余方法；LYP 偶尔更低，但靠"丢弃难答样本"实现，HaDola 不删数据即达竞争性校准，适用面更广 |

### 消融实验

| 方法 (T=15) | VQAv2 Qwen | VQAv2 LLaVA | VizWiz Qwen | VizWiz LLaVA |
|---|---|---|---|---|
| HaDola (完整) | **76.75** | **77.63** | **65.72** | **66.58** |
| 选择器替为随机采样 | 72.23 | 73.51 | 62.11 | 63.02 |
| 自标注替为人工标签 | 73.91 | 75.02 | 64.53 | 64.88 |
| 去掉错误触发 | 71.56 | 72.47 | 60.92 | 61.73 |
| 损失替为标准 CE | 72.37 | 73.28 | 61.89 | 62.15 |

每个组件移除/替换都掉点：**去掉错误触发降幅最大**(防误差累积最关键)；随机采样降幅次大(证明选择器确能挑出有益样本)；自标注换人工标签降幅最小，但仍说明自演化监督能超越人工标注本身。

### 关键发现
- **HU 等级单调律**：简单 SFT 下，低 HU 子集(L) > 中(M) > 高(H)，在训练集和验证集上一致成立，证明高 HU 样本对 VLM 既难学又有害。
- **S 形训练曲线**：5–10% 标注时性能急升，10–15% 缓增，15% 后收敛——少量监督即可获大收益，超过 15% 边际递减，质疑了大规模 SFT 的必要性。
- **评估指标缺陷**：传统 VQA-acc 几乎看不出高 HU 样本的危害，HU-acc 能清晰分离不同 HU 子集，凸显了频率式评估的不足。

## 亮点与洞察
- **把"标注噪声"翻案为"训练信号"**：本文首次系统论证 HU 不是要抹平的副产品，而是可指导数据选择和评估的有用信号，这一视角转换很有启发性。
- **数据效率的强证据**：5% 标注胜过 100% SFT，配合 S 形曲线，给"与其无脑扩数据不如挑好数据"提供了扎实的实证支撑。
- **校准与准确率的双赢**：定制损失以"相对参考模型"的方式做 KL 对齐，避免直接逼近人类分布带来的过拟合，是个精巧设计。
- **评估指标的反思**：指出主流 VQA-Accuracy 因纯频率而忽视 HU，并首倡把 HU 纳入评估协议，对社区有方法论价值。

## 局限与展望
- **依赖种子集质量**：HaDola 需要一个构造良好的人工 HU 标注种子集，虽然成本可控，但种子集质量直接影响阈值标定和整体表现。
- **进一步去标注**：作者指出未来可借助 VLM 的零样本能力构造种子集，进一步降低对人工标注的依赖。
- **数据集范围有限**：仅在带 HU 标注的 VQAv2、VizWiz 上验证，OK-VQA、GQA 等无 HU 标注的数据集如何迁移尚待探索。
- **阈值与超参较多**：$\tau_1,\tau_2,\sigma,\tau_g,\tau_t,\beta,\lambda$ 等需在子集上标定，调参负担和跨域稳健性值得关注。

## 相关工作与启发
- **VQA 与人类不确定性**：Antol 等首提 VQA，BEiT3 仍是任务专用 SOTA 但缺零样本/生成能力；Lan 等(2025a)首次用 HUD 量化 VQAv2 上 BEiT3 的 HU，但未探索如何在训练中利用 HU，本文正是其延伸。
- **样本感知训练**：Karamcheti 等用主动学习发现难样本会让 AL 失效却无解法；Tan & Bansal 用标注者多样性度量样本难度但只看频率忽视 HU；Sun 等用 DPO 增广医疗 VQA 数据但假设训练分布完美。这些工作大多假设数据完全可靠，本文填补了"系统选择并利用 HU 感知样本"的空白。
- **启发**：把"标注分歧/置信度"显式建模并用于数据选择的思路，可迁移到其他存在标注主观性的任务(情感分析、毒性检测、医学影像分级)；其"参考模型相对 KL 对齐"的校准损失也可借鉴到偏好学习场景。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把 HU 从"噪声"重新定位为"数据选择信号"，并设计自演化框架，视角和方法都有新意；四阶段流水线虽由已有技术(伪标注、TracIn、梯度一致性)组合而成，但组合方式针对 HU 问题量身定制。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 5 个 VLM + 2 数据集 + 6 类基线，消融完整、训练动态分析(S 形曲线)与 HU 等级分析到位；但仅限两个带 HU 标注的数据集，跨域泛化证据偏弱。
- **写作质量**: ⭐⭐⭐⭐ RQ 驱动、动机清晰、公式与算法表述规范，图表(热力图/雷达图)信息量大；部分符号(如 $\tau$、$h_\omega$)初读略密。
- **价值**: ⭐⭐⭐⭐ "用好 HU 比扩数据更有效"的结论对降低 VQA 标注成本和改善校准有实际意义，并推动社区反思评估指标，落地性较强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Meta-Adaptive Prompt Distillation for Few-Shot Visual Question Answering](meta-adaptive_prompt_distillation_for_few-shot_visual_question_answering.md)
- [\[CVPR 2026\] VQ-VA World: Towards High-Quality Visual Question-Visual Answering](../../CVPR2026/multimodal_vlm/vq-va_world_towards_high-quality_visual_question-visual_answering.md)
- [\[ICLR 2026\] Knowledge Exchange with Confidence: Cost-Effective LLM Integration for Reliable and Efficient Visual Question Answering](knowledge_exchange_with_confidence_cost-effective_llm_integration_for_reliable_a.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](../../CVPR2026/multimodal_vlm/cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[AAAI 2026\] MacVQA: Adaptive Memory Allocation and Global Noise Filtering for Continual Visual Question Answering](../../AAAI2026/multimodal_vlm/macvqa_adaptive_memory_allocation_and_global_noise_filtering_for_continual_visua.md)

</div>

<!-- RELATED:END -->
