---
title: >-
  [论文解读] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking
description: >-
  [AAAI 2026 Oral][LLM安全][LLM水印] 提出 WaterMod，一种基于模算术 ($\text{rank} \bmod k$) 的 LLM 文本水印方法，通过对概率排序后的词表进行模残差类划分，在零比特（$k=2$）和多比特（$k>2$）水印场景下统一实现高检测率和低质量损耗，无需外部同义词库或哈希技巧。
tags:
  - "AAAI 2026 Oral"
  - "LLM安全"
  - "LLM水印"
  - "文本水印"
  - "模算术"
  - "零比特/多比特水印"
  - "概率平衡"
---

# WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.07863](https://arxiv.org/abs/2511.07863)  
**代码**: [github](https://github.com/Shinwoo-Park/WaterMod)  
**领域**: AI安全  
**关键词**: LLM水印, 文本水印, 模算术, 零比特/多比特水印, 概率平衡

## 一句话总结

提出 WaterMod，一种基于模算术 ($\text{rank} \bmod k$) 的 LLM 文本水印方法，通过对概率排序后的词表进行模残差类划分，在零比特（$k=2$）和多比特（$k>2$）水印场景下统一实现高检测率和低质量损耗，无需外部同义词库或哈希技巧。

## 研究背景与动机

LLM 已能以接近人类的流畅度撰写新闻、法律分析和代码，但也带来溯源困难和下游风险（虚假信息、抄袭、数据投毒）。欧盟 AI 法案要求 AI 生成内容携带可机检验的来源标记，水印成为满足合规要求的推荐机制。OpenAI 和 Google DeepMind 都在积极研发文本水印系统。

**现有方法的局限**：

**KGW（随机划分绿/红列表）**：是 logit-based 水印的开创性工作，但随机划分经常将上下文最合适的 token 分到红名单，导致流畅度下降

**WatMe（同义词聚类）**：利用 WordNet 或 LLM 提示构建同义词集合，保证至少一个合适的同义词在绿名单。但依赖外部同义词资源，受限于词典覆盖率、一词多义和提示敏感性

**LSH（局部敏感哈希）**：基于 token embeddings 的语义哈希生成绿名单，但受超平面敏感性影响，容易出现碰撞错误和语义漂移

**大多数方法仅支持零比特水印**：只能标记"是/否 AI 生成"，无法嵌入更丰富的溯源信息（如模型实例 ID、用户 ID）

**核心洞察**：按模型概率降序排列词表后，**相邻 rank 的 token 被模型视为语义上可互换**。利用模算术 $\text{rank} \bmod k$ 将相邻 rank 分配到不同的颜色类中，天然保证每个颜色类都包含至少一个高概率 token——既能嵌入水印信号，又不牺牲流畅度。

## 方法详解

### 整体框架

WaterMod 的核心是一个统一的模残差类划分规则：

1. 每步解码时，将词表按模型概率降序排列
2. 将排列后的 rank 按 $\text{rank} \bmod k$ 分成 $k$ 个颜色类
3. 对选中的颜色类施加 logit 偏置 $\delta$
4. $k=2$ 为零比特水印，$k>2$ 为多比特水印

### 关键设计

#### 1. **概率排序的奇偶划分（零比特, $k=2$）**

给定时间步 $t$ 的 logits $\boldsymbol{\ell}_t$，计算概率后按降序排列 $\pi = \text{argsort}(\boldsymbol{\ell}_t; \downarrow)$，将 $r \bmod 2 = 0$ 的 token 归为偶数类，$r \bmod 2 = 1$ 归为奇数类。

**关键特性**：概率最高的 token（rank 0）和第二高（rank 1）必然分属不同的奇偶类。因此无论选择哪一类作为绿名单，都能保证至少一个高概率 token 可供采样——这是 KGW 等随机划分方法无法保证的。

**熵自适应门控**：用 Shannon 熵确定偶数/奇数类作为绿名单的概率：

$$p_{odd} = \left(\frac{H_t}{H_{max}}\right)^{H_{scale}}$$

- 低熵（分布尖锐，概率集中于少数 token）→ $p_{odd}$ 低 → 倾向选偶数类作绿名单 → 保护 rank 0（最可能 token）
- 高熵（分布平坦，多 token 概率接近）→ $p_{odd}$ 高 → 倾向选奇数类 → rank 1 也是高概率候选，嵌入水印不影响质量

通过一个由密钥派生的伪随机变量 $u$ 和阈值 $p_{odd}$ 决定实际绿名单：$g = \mathbf{1}[u < p_{odd}]$。

**设计动机**：$H_{scale} > 1$ 时映射更陡峭——只有在分布真正平坦时才允许奇数类成为绿名单，最大化保护低熵（确定性上下文）下的流畅度。$H_{scale} < 1$ 则更激进地嵌入水印。

#### 2. **模残差类划分与载荷嵌入（多比特, $k>2$）**

将奇偶划分推广为 $k$ 色划分：

$$\mathcal{C}_d = \{\pi[r] \mid r \bmod k = d\}, \quad d \in \{0, \ldots, k-1\}$$

一个 $b$ 比特消息转换为 base-$k$ 向量 $\mathbf{m} \in \{0, \ldots, k-1\}^{\tilde{b}}$（$\tilde{b} = \lceil b / \log_2 k \rceil$）。

每步解码时：
1. 由密钥哈希选择伪随机位置 $p$
2. 读取当前 digit $d = \mathbf{m}[p]$
3. 将满足 $r \bmod k = d$ 的 token logits 加 $\delta$

每个 token 因此携带一个 base-$k$ digit，即 $\log_2 k$ bits 的载荷。

**设计动机**：概率质量在 $k$ 个颜色类间几乎均匀分布（每类包含排名 $d, d+k, d+2k, \ldots$ 的 token），因此偏置一个类不会显著改变整体分布。多比特嵌入继承了零比特的流畅度保障。

#### 3. **检测与载荷恢复**

**零比特检测**：重建每步的绿名单奇偶性，统计绿 token 数 $G$，计算 z-score：

$$z = \frac{G - N/2}{\sqrt{N/4}}$$

$z > \tau$ 则判定为水印文本。

**多比特检测**：建立计数表 $C[p][d]$，对每个 digit 位置做多数投票恢复载荷 $\hat{\mathbf{m}}$。z-score 使用 null 概率 $p_0 = 1/k$：

$$z = \frac{G - Tp_0}{\sqrt{Tp_0(1-p_0)}}$$

同一计数表同时支持水印检测（z-score）和消息恢复（多数投票），一次解码完成两项任务。

### 损失函数 / 训练策略

WaterMod 是**推理时水印**，无需额外训练。关键配置：

- 零比特：$\delta = 1.0$，$H_{scale} = 1.2$，绿名单比例 0.5
- 多比特：$\delta = 2.5$（更高偏置补偿更小的目标类 $1/k$），$k=4$，16-bit 载荷
- 解码策略：确定性（argmax），消除随机变异便于分析
- 硬件：单张 NVIDIA RTX 3090 (24GB)

## 实验关键数据

### 主实验

**零比特水印：6种方法 × 3个任务**

| 方法 | C4 PPL↓ | C4 AUROC↑ | GSM8K Acc↑ | GSM8K AUROC↑ | MBPP+ Pass@1↑ | MBPP+ AUROC↑ |
|------|---------|-----------|------------|-------------|---------------|-------------|
| EXPEdit | 36.35 | 36.90 | 10.84 | 37.37 | 22.80 | 34.38 |
| ITSEdit | 31.03 | 11.29 | 11.75 | 35.44 | 20.10 | 27.40 |
| KGW | 21.96 | 80.83 | 51.78 | 44.38 | 29.90 | 72.43 |
| LSH | 26.19 | 88.03 | 53.07 | 52.63 | 41.30 | 30.72 |
| SynthID-Text | 12.77 | **94.36** | 47.61 | 97.65 | 27.80 | 66.90 |
| **WaterMod** | **12.58** | 87.09 | **53.83** | **100** | 36.80 | **82.66** |

- C4 上 WaterMod 实现**最低困惑度**（12.58 vs 12.77），同时检测率位居第三
- GSM8K 上实现**完美检测**（AUROC=100）和**最高准确率**（53.83%），比 SynthID-Text 高 13%
- MBPP+ 上 AUROC 比次优 KGW 高 14%，Pass@1 排第二

**多比特水印（16-bit载荷）：WaterMod vs MPAC**

| 方法 | C4 PPL↓ | C4 AUROC↑ | GSM8K Acc↑ | GSM8K AUROC↑ | MBPP+ Pass@1↑ | MBPP+ AUROC↑ |
|------|---------|-----------|------------|-------------|---------------|-------------|
| MPAC | 10.88 | 97.78 | 31.77 | 95.05 | 20.60 | 48.40 |
| **WaterMod** | **10.87** | **98.02** | **40.33** | **96.94** | **26.20** | **98.29** |

- 所有指标上均优于 MPAC
- MBPP+ 上 AUROC 从 48.40 → 98.29，提升 103%
- GSM8K Accuracy 提升 27%

### 消融实验

**鲁棒性：ChatGPT 改写攻击**

| 来源 | 平均 z-score | AUROC |
|------|-------------|-------|
| 人类写作 | 0.09 | — |
| WaterMod (无攻击) | 14.89 | 100.00 |
| WaterMod (ChatGPT 改写) | 9.95 | 99.95 |

改写后 z-score 从 14.89 降至 9.95，但仍远高于人类文本的 0.09，AUROC 仅下降 0.05%。数学推理任务中改写者必须保持数学正确性，许多高 rank token 不会被替换。

**Shannon vs Spike 熵**：Spike 熵在检测性能上更优，Shannon 熵在任务质量上更优。WaterMod 对两种熵定义都表现鲁棒，可根据应用优先级灵活选择。

### 关键发现

1. **低熵场景（代码/数学）优势显著**：WaterMod 在 GSM8K 上实现 100% AUROC，在 MBPP+ 上 AUROC 82.66% 远超 LSH 的 30.72%——低熵场景是其他方法的致命弱点，但 WaterMod 的模算术保证了至少一个高概率 token 在偏置类中
2. **统一的零/多比特框架**：仅改变 $k$ 即可从二元归属切换到任意载荷容量，无需架构修改
3. **无外部依赖**：不需要 WordNet、embedding hash、LLM 提示——完全基于模型自身的概率排序
4. **抗改写鲁棒**：ChatGPT 改写后 AUROC 仅降 0.05%

## 亮点与洞察

- **设计极其简洁优雅**：核心思想只有一行——$\text{rank} \bmod k$——但解决了绿名单排他高概率 token 的根本问题
- **概率排序 = 天然语义聚类**：模型概率相邻的 token 通常是语义近似的候选选择，模算术将它们交错分配到不同类中，确保每类都有高质量候选
- **熵自适应门控精巧**：在确定性上下文（低熵）中保护流畅度，在不确定上下文（高熵）中积极嵌入水印——自动适应不同文本域
- **多比特扩展自然**：从 $k=2$ 到 $k>2$ 不需要任何架构改变，只改一个超参
- **符合 Kerckhoffs 原则**：安全性依赖密钥 $K$，不依赖算法保密

## 局限与展望

1. **C4 上检测率不是最高**：AUROC 87.09 低于 SynthID-Text 的 94.36，在高熵自然语言场景中仍有改进空间
2. **仅在 Qwen-2.5-1.5B 上评估**：虽声称方法与模型无关，但未在更大模型（7B, 70B）上验证
3. **确定性解码限制**：所有实验使用 argmax 解码，未验证在温度采样/top-p 采样下的表现
4. **$H_{scale}$ 需手动调节**：当前固定为 1.2，作者承认可针对不同领域自动优化
5. **对抗鲁棒性评估有限**：仅评估了 ChatGPT 改写，未测试更强的水印攻击（如 DIPPER、多次改写、翻译-回翻译）
6. **载荷容量与文本长度权衡**：16-bit 载荷在短文本中可能因观测不足导致恢复误差

## 相关工作与启发

- WaterMod 的概率排序思想启发了一种不依赖语义资源的"模型内在语义聚类"方法
- 与 KGW 的关系：KGW 是随机划分的特例，WaterMod 是概率排序+模算术划分的升级
- 与 SynthID-Text 互补：SynthID-Text 改变采样策略，WaterMod 改变 logit 偏置，两者可能可以组合
- 多比特水印与可追溯性结合，可用于细粒度的 LLM 输出溯源（用户级/实例级标识）
- 低熵场景（代码、数学）的水印是一个关键开放问题，WaterMod 提供了可行的解决方案

## 评分

- 新颖性: ⭐⭐⭐⭐ — 模算术划分概念简洁新颖，但底层仍是 logit biasing 的变体
- 实验充分度: ⭐⭐⭐⭐ — 覆盖3个领域、多种基线，但模型规模和解码策略多样性不足
- 写作质量: ⭐⭐⭐⭐⭐ — 极其清晰，算法伪代码完整，图示直观，附录详尽
- 价值: ⭐⭐⭐⭐ — 提供了一个实用、简洁、统一的水印框架，对合规和溯源有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] SSG: Logit-Balanced Vocabulary Partitioning for LLM Watermarking](../../ACL2026/llm_safety/ssg_logit-balanced_vocabulary_partitioning_for_llm_watermarking.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](perturb_your_data_paraphrase-guided_training_data_watermarking.md)
- [\[ACL 2026\] STELA: A Linguistics-Aware LLM Watermarking via Syntactic Predictability](../../ACL2026/llm_safety/a_linguistics-aware_llm_watermarking_via_syntactic_predictability.md)
- [\[ICLR 2026\] PMark: Towards Robust and Distortion-free Semantic-level Watermarking with Channel Constraints](../../ICLR2026/llm_safety/pmark_towards_robust_and_distortion-free_semantic-level_watermarking_with_channe.md)

</div>

<!-- RELATED:END -->
