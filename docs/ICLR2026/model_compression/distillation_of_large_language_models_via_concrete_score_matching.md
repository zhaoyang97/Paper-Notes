---
title: >-
  [论文解读] Distillation of Large Language Models via Concrete Score Matching
description: >-
  [ICLR 2026][模型压缩][知识蒸馏] 提出 Concrete Score Distillation (CSD)，一种基于离散 score matching 的 LLM 知识蒸馏损失，通过匹配 student 和 teacher 在所有词表对之间的相对 logit 差异，同时克服了 softmax 平滑和直接 logit 蒸馏的解空间限制问题。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "知识蒸馏"
  - "LLM压缩"
  - "分数匹配"
  - "logit蒸馏"
  - "离散分数匹配"
---

# Distillation of Large Language Models via Concrete Score Matching

**会议**: ICLR 2026  
**arXiv**: [2509.25837](https://arxiv.org/abs/2509.25837)  
**代码**: [GitHub](https://github.com/aailab-kaist/CSD)  
**领域**: 模型压缩 / 知识蒸馏  
**关键词**: 知识蒸馏, LLM压缩, 分数匹配, logit蒸馏, 离散分数匹配

## 一句话总结

提出 Concrete Score Distillation (CSD)，一种基于离散 score matching 的 LLM 知识蒸馏损失，通过匹配 student 和 teacher 在所有词表对之间的相对 logit 差异，同时克服了 softmax 平滑和直接 logit 蒸馏的解空间限制问题。

## 研究背景与动机

LLM 的部署推理成本极高，知识蒸馏（KD）是将大模型能力传递给小模型的核心手段。现有 KD 方法主要通过 softmax 概率匹配（如 KL 散度）来对齐 student 和 teacher 的分布，但 softmax 变换会严重平滑 logit 信息——当词表规模巨大时（如 128K），绝大多数 token 概率接近零（仅 0.0023% > 0.01），导致 teacher 的丰富知识在 softmax 后几乎不可区分。

直接 logit 蒸馏（DLD）通过 MSE 匹配原始 logit 可以避免 softmax 平滑，但存在另一个关键缺陷：**logit 偏移不变性**。由于 softmax 只关心 logit 之间的相对差异，$f_\theta[y_t] = f_T[y_t] + C$ 对所有 token 成立时概率完全相同，但 DLD 的 MSE 损失不为零，人为限制了最优解空间。当 teacher 和 student 容量差距大时，这种限制尤其有害。

本文的核心 idea：借鉴能量模型中 score matching 绕过归一化约束的思想，将离散 concrete score matching 引入 LLM 蒸馏，设计一个既不受 softmax 平滑影响、又不限制解空间的 logit 级蒸馏目标。

## 方法详解

### 整体框架

CSD 把蒸馏目标从「对齐概率」改成「对齐 logit 残差」：对每个 token 位置，它不直接匹配 teacher 与 student 的 logit 值，而是匹配所有词表对 $(y_t, x)$ 之间的相对 logit 差异 $f[x]-f[y_t]$。这个差分结构正好对应离散扩散里 concrete score 的对数形式，既绕开了 softmax 的平滑、又天然消除了直接 logit 蒸馏（DLD）的偏移敏感问题。

### 关键设计

**1. 对数变换的 concrete score 匹配：把概率比换成稳定的 logit 差**

concrete score 本身是概率比 $q_\theta(x)/q_\theta(y_t)$，当分母 $q_\theta(y_t)$ 趋近零时直接匹配会让损失爆炸、训练发散。CSD 的做法是对概率比取对数，于是匹配对象从「比值」变成「差值」，得到一个干净的 logit 级二次损失：

$$\mathcal{L}_{\text{CSD}} = \frac{1}{2} \sum_{y_t \in \mathcal{V}} \sum_{x \in \mathcal{V}} w(y_t, x) \left( f_\theta[x] - f_\theta[y_t] - f_T[x] + f_T[y_t] \right)^2$$

对数变换一举两得：不再需要计算容易数值溢出的概率比，训练更稳；同时损失天然落在 logit 空间，完全跳过了 softmax，避免大词表下绝大多数 token 概率被压成零、teacher 知识被抹平的问题。

**2. 更大的最优解空间：差分结构天然容忍 logit 偏移**

DLD 用 MSE 直接对齐 logit，但 softmax 只关心 logit 之间的相对差，当 $f_\theta[y_t] = f_T[y_t] + C$（所有 token 同时平移一个常数 $C$）时两者概率完全相同，DLD 的损失却不为零，等于人为缩小了 student 可达的最优解集。CSD 的损失只看差分 $f[x]-f[y_t]$，平移项 $C$ 会在差分里相消、损失归零，因此其最优解集严格包含 DLD 的解集（论文 Theorem 2，$\Theta_{\text{CSD}}^* \supsetneq \Theta_{\text{DLD}}^*$）。student 容量越受限、与 teacher 差距越大，这份「多出来的搜索自由度」越值钱。

**3. 把 $O(|\mathcal{V}|^2)$ 的双重求和压回 $O(|\mathcal{V}|)$**

损失里对词表的双重求和在 128K 词表下直接算是不可行的。CSD 假设权重函数可分解为 $w(y_t, x) = w_1(y_t)\cdot w_2(x)$，在这个分解下梯度可以解析展开：先对 logit 做一次加权中心化归一化，再求残差的加权和，整个梯度就能在 $O(|\mathcal{V}|)$ 时间内算完（论文 Theorem 3）。这一步是 CSD 从「理论上好看」变成「能真上手训」的关键。

**4. 可调权重在保真与多样之间切换**

$w_1, w_2$ 各自可取 teacher 概率（T）、student 概率（S）或均匀分布（U），不同组合让 CSD 在 mode-seeking 与 mode-covering 之间滑动：$(S,S)$ 最贴 teacher、保真度最高，$(U,S)$、$(T,S)$ 则引入更多分布质量、提升多样性与概率校准。这给了一个统一旋钮，用同一套损失覆盖以往要靠 KL/RKL/SKL 等不同散度才能表达的偏好。

### 损失函数 / 训练策略

CSD 的梯度结构和 KL 散度很像，区别在于用**中心化归一化**替代了 softmax 归一化，这正是它绕过 softmax 平滑的来源。它与 on-policy 蒸馏技术（ImitKD、GKD、DistiLLM）正交，可以直接叠加使用；对于无法分解的联合权重函数，CSD 也支持 Monte Carlo 梯度估计，但可分解情形下的解析梯度收敛更快、性能也更好，是默认选择。

## 实验关键数据

### 主实验

**表1: GPT-2-1.5B → GPT-2-0.1B 纯损失函数对比（ROUGE-L）**

| 损失函数 | Dolly | Self-Instruct | Vicuna | Super-NI | UnNI | 平均 |
|---------|-------|--------------|--------|----------|------|------|
| KL | 23.52 | 10.02 | 14.57 | 16.76 | 18.55 | 16.68 |
| RKL | 24.26 | 11.19 | 15.80 | 20.17 | 22.99 | 18.88 |
| SRKL | 24.53 | 12.19 | 15.63 | 23.37 | 24.28 | 20.00 |
| **CSD (Ours)** | **24.94** | **12.06** | **15.78** | **24.60** | **25.88** | **20.65** |

**表2: 任务特定蒸馏 Gemma-7B-IT → Gemma-2B-IT**

| 损失 | 摘要 ROUGE-L | 翻译 COMET | GSM8K Acc |
|------|-------------|-----------|-----------|
| KL | 35.02 | 73.96 | 24.03 |
| **CSD (T,S)** | **35.67** | **74.14** | **25.78** |
| RKL | 0.00 | 45.02 | 0.00 |

### 消融实验

- **CSD vs DLD 在相同权重下**: CSD (T,T)/(U,U)/(S,S) 一致优于 DLD T/U/S，验证了更大解空间的优势
- **权重选择**: (S,S) 最高保真度，(U,S) 最好多样性-保真度权衡，(T,S) 最好概率校准
- **通用对话蒸馏**: Qwen2.5-7B→1.5B 上 MT-Bench 从 5.75 (DistiLLM-2) 提升到 5.95 (CSD)
- **解析梯度 vs Monte Carlo**: 解析计算收敛略快且性能更优

### 关键发现

- softmax 在大词表 LLM 中损失了大量 teacher 知识（99.99%+ token 概率 <0.01）
- logit 偏移不变性是 DLD 的核心缺陷，CSD 通过差分结构自然解决
- mode-seeking 损失（RKL, SKL）在小数据蒸馏中容易崩溃，CSD 通过权重选择避免此问题
- CSD 与 on-policy 方法正交互补，尤其在纯 on-policy 设置下提升最大

## 亮点与洞察

- 从能量模型视角重新审视 LLM 蒸馏问题，思路新颖且理论扎实
- logit 偏移不变性的分析简洁有力，直指 DLD 的根本缺陷
- 权重函数设计空间提供了统一框架理解 mode-seeking 与 mode-covering
- 梯度的解析计算使得 $O(|\mathcal{V}|^2)$ 的 CSM 变为实际可用的 $O(|\mathcal{V}|)$

## 局限与展望

- 权重函数空间的探索仍有限，联合权重函数的设计尚未充分挖掘
- 分解假设 $w(y_t,x)=w_1(y_t)w_2(x)$ 限制了权重的表达能力
- 在更大规模模型（>10B teacher）上的验证不够充分
- 与量化、剪枝等其他压缩方法的结合未探讨

## 相关工作与启发

- **DistiLLM** (Ko et al., 2024): 提出 SKL/SRKL 平滑 KL 蒸馏，本文在此基础上进一步改进
- **GKD** (Agarwal et al., 2024): f-divergence 框架，CSD 可视为超越概率匹配的新框架
- **Concrete Score Matching** (Meng et al., 2022): 离散扩散模型中的 score matching，本文创新性地适配到自回归 LLM 蒸馏
- 启发：score matching 的"绕过归一化"思想可能在其他需要分布匹配的 NLP 任务中也有应用

## 评分

- 新颖性: ⭐⭐⭐⭐ 从 EBM score matching 到 LLM 蒸馏的迁移有创意，但核心仍是 logit 差分匹配
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖多种 backbone、任务类型、on-policy 组合，消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，图示清晰，动机分析到位
- 价值: ⭐⭐⭐⭐ 提供了统一的 logit 蒸馏设计框架，实际性能提升稳定但幅度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Knowledge Distillation for Large Language Models through Residual Learning](knowledge_distillation_for_large_language_models_through_residual_learning.md)
- [\[CVPR 2026\] Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals](../../CVPR2026/model_compression/phased_dmd_few-step_distribution_matching_distillation_via_score_matching_within.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] MoSA: Mosaic Shared Adaptation of Large Language Models](mosa_mosaic_shared_adaptation_of_large_language_models.md)

</div>

<!-- RELATED:END -->
