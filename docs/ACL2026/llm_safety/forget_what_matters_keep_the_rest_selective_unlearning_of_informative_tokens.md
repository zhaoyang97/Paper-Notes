---
title: >-
  [论文解读] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens
description: >-
  [ACL 2026][LLM安全][机器遗忘] 提出 Entropy-guided Token Weighting (ETW)，利用预测分布的熵值作为 token 信息量的代理指标，选择性地对信息性 token 施加更强的遗忘惩罚，在有效遗忘目标知识的同时更好地保持模型通用能力。 领域现状：LLM 机器遗忘旨在选择性移除模型…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "机器遗忘"
  - "信息性token"
  - "熵引导"
  - "token加权"
  - "选择性遗忘"
---

# Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens

**会议**: ACL 2026  
**arXiv**: [2604.17785](https://arxiv.org/abs/2604.17785)  
**代码**: 无  
**领域**: LLM 效率 / 机器遗忘  
**关键词**: 机器遗忘, 信息性token, 熵引导, token加权, 选择性遗忘

## 一句话总结

提出 Entropy-guided Token Weighting (ETW)，利用预测分布的熵值作为 token 信息量的代理指标，选择性地对信息性 token 施加更强的遗忘惩罚，在有效遗忘目标知识的同时更好地保持模型通用能力。

## 研究背景与动机

**领域现状**：LLM 机器遗忘旨在选择性移除模型中的特定知识（如隐私数据），同时保留其他能力。基于梯度上升（GA）的方法是主流范式，但均匀施加遗忘损失会不必要地降低模型效用。

**现有痛点**：现有 token 级正则化方法要么基于 ground-truth 置信度（WGA、SatImp），无法区分置信度相同但语义重要性不同的 token；要么依赖外部语言解析器（spaCy 的 SCN/SCE），无法捕获上下文信息和模型的整体预测状态。

**核心矛盾**：如何准确识别序列中的"信息性 token"（承载核心语义内容的词）与"结构性 token"（主要服务语法功能的词）？置信度和词性类别都不足以做出可靠区分。

**本文目标**：设计一种更有效的 token 级信息量度量，用于指导选择性遗忘，使模型精准遗忘关键信息同时最小化通用能力损失。

**切入角度**：利用模型预测分布的熵值——结构性 token（如"the"）预测确定性高、熵低；信息性 token（如人名"Carmen"）存在多种合理替代，熵值高。

**核心 idea**：熵值利用了整个词汇表上的概率分布而非仅 ground-truth 置信度，提供了更丰富的信息量表征，能更准确地区分信息性 token 和结构性 token。

## 方法详解

### 整体框架

ETW 要解决的问题是：梯度上升（GA）式遗忘把遗忘损失均匀地铺在序列每个 token 上，连"the""of"这种结构性词也照样使劲遗忘，结果该忘的核心信息没忘干净，模型通用能力却被白白拉低。它的做法是在标准遗忘流程里加一层 token 级权重——遗忘集走加权梯度上升、保留集走标准交叉熵，而权重直接由模型预测分布的熵决定：熵高的（信息性）token 吃更重的遗忘惩罚，熵低的（结构性）token 几乎不动。整个改动不引入外部工具或参考模型，只在算熵的 softmax 里引入一个温度 $T$，相当于把原方法里那把"均匀的刷子"换成"按信息量分配力度的刷子"。

### 关键设计

**1. 熵引导的 token 权重计算：用整个词表的分布、而不只是 GT 置信度来判断一个 token 有多"重"**

现有 token 级正则（WGA、SatImp）靠 ground-truth 置信度加权，但置信度只看了正确答案那一个位置的概率，碰到两个置信度相同、语义重要性却天差地别的 token 就无能为力。ETW 改用预测分布的熵：对每个位置 $i$ 算 $H(y_i\mid y_{<i},\mathbf{x};\hat{\boldsymbol{\theta}})$，再归一化使权重之和等于序列长度 $n$：

$$\omega_i^{\text{ETW}}=n\cdot H_i\Big/\sum_j H_j$$

熵反映的是概率质量在所有候选 token 上的铺开程度——像"the"这种下一个词几乎确定的结构性 token 熵低，而人名"Carmen"这类有多个合理替代的信息性 token 熵高。即便两个 token 的 GT 置信度一模一样，它们的熵仍可以差出 $H_{\max}-H_{\min}$ 的范围，这正是置信度看不到、而熵能抓住的区分力。

**2. Stop-gradient 权重计算：让"算权重"和"做遗忘"互不干扰**

如果权重本身也参与反向传播，权重计算就会反过来给遗忘训练注入额外的梯度噪声。ETW 用一份 stop-gradient 的参数副本 $\hat{\boldsymbol{\theta}}$ 来算熵，经温度 $T$ 的 softmax 得到概率分布后再取熵。这样权重只是个被动读出的标量，彻底把"决定每个 token 该忘多狠"和"实际更新参数"两件事解耦开。

**3. 归一化保持损失尺度不变：只在 token 之间重新分配遗忘力度，不改变总量**

如果加权后整体遗忘损失的尺度变了，就得再引入一个超参去把强度调回来，平添调参负担。ETW 的归一化让 $\sum_i\omega_i=n$，使加权后的总遗忘损失和无加权 GA 完全等量——变的只是力度在 token 间的分布，总盘子不变。正因如此，ETW 可以原地替换现有方法里的均匀权重，不动其余任何超参，是个真正即插即用的正则器。

### 损失函数 / 训练策略

总遗忘目标为 $\mathcal{L} = \mathcal{L}_r + \lambda \mathcal{L}_f$，其中保留损失 $\mathcal{L}_r$ 是标准交叉熵，遗忘损失 $\mathcal{L}_f = \sum_i \omega_i^{\text{ETW}} \log p(y_i | y_{<i}, \mathbf{x}; \boldsymbol{\theta})$ 是 ETW 加权的梯度上升损失，$\lambda$ 控制遗忘强度。

## 实验关键数据

### 主实验（TOFU Forget 10% - Llama 3.2-1B）

| 方法 | -log(FQ) ↓ | ΔMU ↓ | Agg. ↓ | |Priv.| ↓ |
|------|-----------|-------|--------|-----------|
| GA | 2.639 | 4.271 | 11.273 | 48.59 |
| WGA | 2.309 | 5.365 | 12.388 | 3.14 |
| SatImp | 2.871 | 5.258 | 15.093 | 19.65 |
| SCN | 2.754 | 3.548 | 9.772 | 39.98 |
| **ETW** | **0.492** | **3.471** | **1.707** | 9.56 |

### 消融实验（ROC-AUC - 区分信息性/结构性 token）

| 方法 | AUC |
|------|-----|
| ETW | 最高（超出次优 ≥ 0.06） |
| SCN | 次优 |
| Imp | 0.66 |
| WGA/TNPO | ≤ 随机 |

### 关键发现
- ETW 在 ROC-AUC 上全面超越所有基线，在区分信息性和结构性 token 方面表现最佳
- 在可视化分析中，ETW 是唯一成功识别出核心答案片段（如"Love Inspired"）的方法
- 置信度相同的 token 因非 GT token 分布不同，熵值可以有巨大差异，验证了熵值作为信息量代理的优势
- ETW 遗忘后的模型在信息性 token 预测概率上最接近从头训练的模型

## 亮点与洞察
- 精巧的动机论证：通过 $H_{\min}$ 和 $H_{\max}$ 的数学分析，严格证明熵值比置信度提供更丰富的表征空间
- 方法极其简洁——仅需计算预测分布的熵并归一化，无需外部工具、参考模型或额外超参数
- 可视化分析直观有力，清楚展示了各方法在 token 级别的差异
- 归一化设计使 ETW 成为一个即插即用的正则化器

## 局限与展望
- 实验主要在 TOFU 数据集上验证，对更大规模和更多样化的遗忘场景（如安全对齐）的泛化性待验证
- 熵值计算需要额外的前向传播（对整个词汇表做 softmax），大词汇表模型可能有计算开销
- 未讨论与 DPO 类遗忘方法的结合潜力
- 当信息均匀分布在序列中时，ETW 退化为均匀权重，可能不具优势

## 相关工作与启发
- **vs WGA/TNPO**: 这些方法基于 GT 置信度加权，无法区分语义重要性不同但置信度相同的 token
- **vs SCN/SCE (spaCy)**: 基于词性的二值掩码忽略上下文，且可能错误地标记重复问题内容中的实体
- **vs SatImp**: 结合置信度和互补值，但受限于置信度的区分能力上限
- **启发**: 熵值在 token 级控制中的应用可推广到微调、强化学习等其他场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 用熵值度量 token 信息量的洞察新颖且直觉清晰
- 实验充分度: ⭐⭐⭐⭐ 定量（ROC-AUC）和定性（可视化）分析充分，多基线对比全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机论证严密，从直觉到数学分析到实验层层递进
- 价值: ⭐⭐⭐⭐ 方法简洁有效，对 LLM 遗忘和 token 级控制领域有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)

</div>

<!-- RELATED:END -->
