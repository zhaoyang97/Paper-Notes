---
title: >-
  [论文解读] Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy
description: >-
  [ICML 2025][LLM安全][差分隐私] 提出 Cape——一种上下文感知的 prompt 扰动机制，通过混合效用函数（结合 token 嵌入距离和上下文 logit）以及分桶指数采样机制，在 local DP 保证下实现比现有方法更优的隐私-效用权衡。 领域现状：LLM 推理服务（如 ChatGPT）要求用户将明文…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "差分隐私"
  - "提示学习"
  - "LLM推理隐私"
  - "指数机制"
  - "上下文感知"
---

# Cape: Context-Aware Prompt Perturbation Mechanism with Differential Privacy

**会议**: ICML 2025  
**arXiv**: [2505.05922](https://arxiv.org/abs/2505.05922)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 差分隐私, prompt扰动, LLM推理隐私, 指数机制, 上下文感知

## 一句话总结
提出 Cape——一种上下文感知的 prompt 扰动机制，通过混合效用函数（结合 token 嵌入距离和上下文 logit）以及分桶指数采样机制，在 local DP 保证下实现比现有方法更优的隐私-效用权衡。

## 研究背景与动机

**领域现状**：LLM 推理服务（如 ChatGPT）要求用户将明文 prompt 发送到服务器，prompt 中可能包含敏感信息（邮件、商业机密等），存在隐私泄露风险。

**现有痛点**：
   - 密码学方案（MPC/HE）虽提供可证安全性，但开销巨大（每个 token 需 ~30s），不实用
   - 白盒方案（DP-Forward 等）需在客户端部署模型浅层，要求模型修改
   - 现有 DP 方案（SANTEXT、CUSTEXT、InferDPT）仅用 token 嵌入距离衡量语义相似性，忽略上下文信息，导致语义一致性差（如"enjoyable"和"unenjoyable"在嵌入空间中距离很近但含义相反）

**核心矛盾**：NLP 词表极大（30K+），标准指数机制在如此大的采样空间中会出现长尾现象——大量低效用 token 的累积概率很大，导致频繁采样到无关 token。

**本文目标**：如何在黑盒推理场景下，用高效的 DP 机制扰动 prompt，同时保持语义一致性和隐私？

**切入角度**：(a) 引入上下文 logit 信息增强效用函数；(b) 设计分桶采样抑制长尾效应

**核心 idea**：将上下文相关性（由客户端小模型提供的 logit）和 token 嵌入距离融合为混合效用函数，配合等宽分桶指数机制，实现上下文感知的 prompt 扰动。

## 方法详解

### 整体框架
客户端持有原始 prompt $x = \{t_1, t_2, \dots, t_n\}$ 和一个小型设备模型 $\mathcal{M}_c$（如 BERT/GPT-2）。对每个敏感 token $t_i$：
1. 计算混合效用函数 $u(t_i, t_r)$，对词表中每个候选 token $t_r$ 打分
2. 用分桶指数机制采样替换 token $\hat{t}_i$
3. 将扰动后的 prompt $\hat{x}$ 发送到服务器

### 关键设计

1. **混合效用函数 (Hybrid Utility Function)**:

    - 功能：综合 token 嵌入距离和上下文 logit 为候选 token 打分
    - 核心思路：$u(t_i, t_r) = L_r^{\lambda_L} \cdot D(t_i, t_r)^{\lambda_D}$，其中 $L_r = \mathcal{M}_c(t_r | \text{Ctx})$ 是上下文 logit，$D(t_i, t_r) = \exp(-d_{\text{euc}}^{\text{norm}}(t_i, t_r))$ 是归一化欧氏距离的指数衰减
    - 设计动机：仅用嵌入距离会混淆反义词（如 enjoyable vs unenjoyable）；加入上下文 logit 后，上下文中不合理的替换会得到低 logit 从而被抑制
    - 有界性保证：距离部分 $D \in (0, 1]$，logit 通过 clip 限制到 $[-B, B]$，确保敏感度可控

2. **分桶指数机制 (Bucketized Exponential Mechanism)**:

    - 功能：解决大词表下标准指数机制的长尾问题
    - 核心思路：将候选 token 按效用分值排序后分入 $N_b$ 个等宽桶，每个桶用均值效用代表→先用 EM 采样桶→再从桶内均匀采样 token
    - 采样概率：$\mathbb{P}[\mathcal{R}(t) = t_r] \propto \frac{\exp(\frac{\epsilon}{2\Delta} \text{mean}(b_i))}{|b_i|}$
    - 设计动机：标准 EM 中，top-10 高效用 token 的累积概率不到 1%（$\epsilon=6, N=50000$ 时），分桶后低效用 token 的影响被桶级概率压缩
    - 隐私保证：满足 $(\epsilon + \epsilon')$-DP，其中 $\epsilon' = \ln(\max_{i,j} \frac{|b_i|}{|b_j|})$

3. **非敏感 token 保留**:

    - 功能：预定义 179 个停用词 + 32 个标点为非敏感 token，保留不扰动
    - 设计动机：这些 token 不含隐私信息但对文本连贯性至关重要

### 损失函数 / 训练策略
- 无需训练，纯推理时机制
- 客户端小模型（BERT/GPT-2）提供 logit 信息，不需微调
- 超参默认配置：$\lambda_L = 0.5$, $\lambda_D = 1.0$, $N_b = 50$

## 实验关键数据

### 主实验
在 SST-2 上用 Qwen2-1.5B-Instruct 做零样本分类的句子级相似度（Rouge-L F1）：

| 方法 | ε=1 | ε=6 | ε=14 | ε=20 |
|------|-----|-----|------|------|
| SANTEXT | 0.87 | 99.36 | 99.45 | 99.45 |
| CUSTEXT | 14.50 | 47.27 | 95.54 | 99.48 |
| InferDPT | 13.00 | 16.48 | 38.68 | 68.11 |
| **Cape (BERT)** | **38.38** | **46.85** | **76.49** | **92.03** |
| Cape (GPT2) | 37.60 | 44.55 | 73.46 | 90.65 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅距离 ($\lambda_L=0$) | 效用下降 | 失去上下文信息 |
| 仅 logit ($\lambda_D=0$) | 效用下降 | 失去语义相似性约束 |
| 无分桶 ($N_b=1$) | 长尾严重 | 高效用 token 采样概率极低 |
| $N_b=50$ (默认) | 最优 | 平衡桶数和桶内粒度 |
| $N_b=500$ | 略降 | 桶太多，部分空桶 |

### 关键发现
- Cape 在对齐实际隐私预算后($\epsilon' \sim 14$)，Rouge-L 显著优于 SANTEXT($\epsilon \sim 1$)和 InferDPT($\epsilon \sim 6$)
- CUSTEXT 虽然高效用但隐私很弱——即使 $\epsilon=1$，KNN 攻击成功率仍超 60%
- BERT 版上下文模型优于 GPT-2 版，因为 BERT 捕获双向上下文

## 亮点与洞察
- **混合效用函数**简洁有效：将上下文 logit 和嵌入距离相乘，一个公式同时处理语义相似性和上下文一致性
- **分桶策略**解决了 DP-NLP 中被忽视的长尾问题：通过先采样桶再桶内均匀采样，将采样空间从 30K+ 压缩到 50 个桶级决策
- 纯推理时方案、无需修改后端模型，实用性极强（每个输入 ~0.1s）

## 局限与展望
- 分桶引入额外隐私开销 $\epsilon'$，桶大小差异越大开销越大
- 客户端需部署小模型（BERT/GPT-2），对极端资源受限设备仍有负担
- 停用词/标点的保留策略过于简单，可能泄露句式结构信息
- 未讨论多轮对话场景下的隐私组合问题

## 相关工作与启发
- **vs SANTEXT**: 在整个词表上采样，ε=1 时基本随机，Cape 通过分桶和混合效用显著改善
- **vs CUSTEXT**: 固定小邻接表实现高效用但牺牲隐私，本质上是截断 DP
- **vs InferDPT**: 同为黑盒方案，Cape 引入上下文信息后效用大幅提升
- **vs DP-Forward/TextObfuscator**: 白盒方案需模型修改，Cape 更实用

## 评分
- 新颖性: ⭐⭐⭐⭐ 混合效用函数和分桶采样的组合新颖，解决实际痛点
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多攻击、消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机例子直观
- 价值: ⭐⭐⭐⭐ 实用的黑盒推理隐私保护方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](../../NeurIPS2025/llm_safety/invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[ICLR 2026\] Resource-Adaptive Federated Text Generation with Differential Privacy](../../ICLR2026/llm_safety/resource-adaptive_federated_text_generation_with_differential_privacy.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ICML 2025\] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text](the_canarys_echo_auditing_privacy_risks_of_llm-generated_synthetic_text.md)
- [\[ICML 2025\] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)

</div>

<!-- RELATED:END -->
