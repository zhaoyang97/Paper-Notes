---
title: >-
  [论文解读] Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position
description: >-
  [AAAI 2026 (Oral)][强化学习][扩散语言模型] 首次系统分析扩散大语言模型（dLLM）的安全特性，发现与自回归 LLM 不同，dLLM 中**中间 token** 对安全性更关键，且攻击者受限于模型固有的顺序生成倾向难以操控中间 token，基于此不对称性提出 MOSA（Middle-tOken Safety Alignment）防御方法。
tags:
  - "AAAI 2026 (Oral)"
  - "强化学习"
  - "扩散语言模型"
  - "安全对齐"
  - "中间token"
  - "越狱防御"
---

# Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position

**会议**: AAAI 2026 (Oral)  
**arXiv**: [2508.12398](https://arxiv.org/abs/2508.12398)  
**代码**: 无  
**领域**: AI 安全 / 大语言模型对齐  
**关键词**: 扩散语言模型, 安全对齐, 中间token, 越狱防御, 强化学习

## 一句话总结

首次系统分析扩散大语言模型（dLLM）的安全特性，发现与自回归 LLM 不同，dLLM 中**中间 token** 对安全性更关键，且攻击者受限于模型固有的顺序生成倾向难以操控中间 token，基于此不对称性提出 MOSA（Middle-tOken Safety Alignment）防御方法。

## 研究背景与动机

### 自回归 LLM 的安全对称性

在传统自回归 LLM（如 Llama-3、GPT-4）中，安全对齐存在一个被称为"浅层安全对齐（SSA）"的现象：安全微调主要集中在回复的前几个 token 上。这导致了**对称性竞争**——攻击者和防御者都在争夺对初始 token 的控制权：
- 攻击者：强迫模型以肯定前缀开头（如"Sure, here is..."）
- 防御者：强化模型以拒绝前缀开头（如"I cannot..."）

### dLLM 的范式转变

扩散大语言模型（如 LLaDA、DREAM）采用完全不同的推理方式：从全掩码序列开始，通过多轮迭代逐步预测内容。理论上，dLLM 可以在任意位置填充 token，不受严格的从左到右约束。

**关键问题**：传统"首 token 中心"的安全分析是否仍然适用于 dLLM？

### 三个核心发现

作者通过系统实验发现了 dLLM 中独特的**安全不对称性**：

**发现 1：中间 token 比初始 token 对安全性更重要**
- 在不同位置预填充（prefill）短语进行测试：在初始位置预填充肯定短语时，模型往往很快恢复拒绝；但在中间位置（第40-160个 token）预填充程序性短语时，模型会放弃安全开头并继续有害内容
- 越狱成功率随预填充位置后移而显著增加

**发现 2：攻击者难以操控中间 token**
- 使用 GCG 攻击（被视为操控能力的上限）测试发现：初始 token 攻击成功率 33%，中间 token 仅 2%
- 优化损失在中间 token 位置始终保持高值，反映了攻击者操控这些位置的根本性障碍

**发现 3：dLLM 具有固有的顺序生成偏好**
- 尽管架构允许非序列生成，dLLM 实际上强烈倾向从左到右生成
- 新解掩 token 的平均位置与解码步骤呈近线性正相关
- 这一偏好与输入类型无关（良性/对抗性提示表现相同）

## 方法详解

### 整体框架

MOSA 在强化学习范式下工作，通过对比奖励信号将回复的中间 token 与预定义的安全拒绝序列对齐。整体框架包括：对比奖励计算 + KL 散度惩罚 + LoRA 轻量级微调。

### 关键设计

#### 1. **中间 token 窗口定义**

将第 20 到第 60 个 token 定义为中间 token。窗口选择的考量：
- **前界（第20个）**：足够远离攻击者影响力强的初始 token 区域
- **后界（第60个）**：足够早地终止回复，限制潜在危害

**设计动机**：利用安全不对称性——防御者可以直接干预中间 token（训练时），而攻击者受顺序生成偏好限制难以触及。

#### 2. **对比奖励函数**

定义正样本集 $S_{safe\_set}$（安全拒绝句，如"therefore, I cannot answer this question"，包含 EOS token）和负样本集 $S_{harmful\_set}$。

每步训练随机选择一个正样本 $s_{pos}$ 和一个负样本 $s_{neg}$，在目标窗口内搜索所有连续片段，计算最大对数似然：

$$R_{pos} = \text{GetMaxScore}(s_{pos}, L_\theta, [k_{start}, k_{end}])$$
$$R_{neg} = \text{GetMaxScore}(s_{neg}, L_\theta, [k_{start}, k_{end}])$$
$$R_{contrastive} = R_{pos} - R_{neg}$$

其中 GetMaxScore 遍历窗口内所有长度与句子匹配的连续段，取最大对数似然值。

**关键设计**：正样本中包含 EOS token 作为"断路器"——即使初始 token 被攻破，中间的安全句子也能通过 EOS 截断输出长度，限制危害范围。

#### 3. **KL 散度正则化**

完整奖励函数：

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim D}\left[R(y|x) - \beta \cdot D_{KL}(P_\theta(y|x) \| P_{ref}(y|x))\right]$$

其中 $\beta$ 控制 KL 惩罚强度（设为 0.05），防止策略模型偏离原始模型太远导致通用能力丧失。

**设计动机**：单纯的安全对齐可能导致模型在正常任务上退化。KL 惩罚确保对齐是"精准的"——只在遇到有害提示时触发防御，不影响正常使用。

### 训练策略

- **基础模型**：LLaDA-8B-Instruct
- **训练数据**：SORRY-Bench 中随机抽取 3000 条有害问题
- **微调方案**：LoRA（r=32, α=64），仅训练 1 个 epoch
- **优化器**：AdamW，学习率 5e-5，梯度裁剪 0.01
- **计算成本**：双 A100-40GB GPU，约 12 分钟完成训练
- 奖励在约 500 步内快速上升并稳定在 15-18

## 实验关键数据

### 主实验

在 AdvBench 上对抗 8 种越狱攻击的攻击成功率（ASR%，越低越好）：

| 攻击方法 | 原始模型 | Initial Alignment | MOSA (本文) |
|----------|---------|-------------------|-------------|
| Avatar | 74.5 | 23.5 | **14.3** |
| TAP | 79.1 | 29.6 | **4.5** |
| Speakeasy | 69.8 | 22.4 | **8.1** |
| AOS | 65.2 | 32.4 | **6.5** |
| PAL | 72.8 | 36.4 | **6.2** |
| EPT | 78.4 | 28.5 | **3.8** |
| DIA | 66.7 | 34.3 | **4.2** |
| AdvPrefix | 79.5 | 29.8 | **6.8** |

HarmBench 上结果类似，MOSA 将大部分攻击的 ASR 降至个位数。

### 通用能力保持

| 模型 | GSM8K | MMLU | HumanEval |
|------|-------|------|-----------|
| 原始模型 | 69.8 | 66.4 | 32.8 |
| Initial Alignment | 67.4 | 68.2 | 29.6 |
| **MOSA** | **68.3** | **65.9** | **30.4** |

MOSA 对通用能力的影响极小，与原始模型基本持平。

### 消融实验 / 自适应攻击

| 配置 | TAP ASR | EPT ASR | AdvPrefix ASR |
|------|---------|---------|---------------|
| Initial Alignment（仅对齐首token） | ~29% | ~28% | ~30% |
| MOSA（对齐中间token） | **5.1%** | **3.8%** | **4.5%** |
| 自适应攻击（攻击者改为攻击中间token） | 5.1% | 3.8% | 4.5% |

自适应攻击的 ASR 与常规攻击结果一样低，证明 MOSA 并未仅仅移动脆弱点位置，而是利用了架构层面的安全优势。

### 关键发现

1. dLLM 的安全攸关位置从"首token"转移到了"中间token"
2. 攻击者受限于顺序生成偏好，对中间 token 的操控能力极低（GCG 攻击成功率仅 2%）
3. 这种安全不对称性是 dLLM 范式的普遍特征——在 Dream 7B 和 MMaDA 上也观察到相同现象
4. MOSA 是轻量级微调，不改变模型固有的顺序生成偏好，因此安全不对称性得以保持
5. "中间token对齐"策略**不适用于自回归 LLM**——因为 AR 模型中每个 token 因果依赖于前序，中间 token 无法独立于开头对齐

## 亮点与洞察

1. **首次 dLLM 安全分析**：填补了一个重要的研究空白，dLLM 作为新兴范式亟需安全研究
2. **安全不对称性的发现**：这一核心洞察极具原创性——攻击者和防御者在 dLLM 中的能力是不对等的，这与 AR-LLM 完全不同
3. **EOS 断路器设计**：在安全句中嵌入 EOS 既确保安全又控制输出长度，一举两得
4. **训练极其高效**：12 分钟 + 3000 条数据即可完成，实用性极强
5. **超越安全领域的启示**：作者建议"锚点-填充"（anchor-then-fill）策略可能是释放 dLLM 潜力的通用方法，如先生成关键中间公式再前后补全

## 局限与展望

1. 对"叙事包装"型攻击（如 Avatar、Speakeasy）的防御效果相对较弱，因为训练数据中缺乏足够多样的伪装恶意 prompt
2. 中间 token 窗口（20-60）是手动设定的，未自适应不同任务/模型
3. 仅在 LLaDA-8B 上验证，未在更大规模 dLLM 或商用 dLLM（如 Gemini Diffusion）上测试
4. 未来方向：从激活空间角度研究 dLLM 的"良性/有害"概念线性，开发更鲁棒的防御

## 相关工作与启发

- **与 AR-LLM 安全研究的关系**：揭示了 dLLM 需要完全不同的安全范式，不能简单沿用 AR 的策略
- **与 SSA（浅层安全对齐）的对比**：MOSA 是"深层安全对齐"，作用于模型真正的安全关键区域
- **启发**：dLLM 的顺序生成偏好本身就是一种有趣的现象——为什么非自回归模型会"学习"自回归行为？这可能是训练目标的自然结果（预测相邻 token 的方差更低）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个 dLLM 安全分析，安全不对称性的发现极具原创性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 8 种攻击 × 2 个基准，自适应攻击、通用能力评估完整
- **写作质量**: ⭐⭐⭐⭐⭐ — 分析层层递进，图文配合极佳，逻辑链条清晰
- **实用价值**: ⭐⭐⭐⭐ — 对 dLLM 安全部署有直接指导意义，但覆盖的 dLLM 种类有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/mmada_multimodal_large_diffusion_language_models.md)
- [\[NeurIPS 2025\] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models](../../NeurIPS2025/reinforcement_learning/reinforcing_the_diffusion_chain_of_lateral_thought_with_diffusion_language_model.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](../../ICML2026/reinforcement_learning/learning_unmasking_policies_for_diffusion_language_models.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](../../ICML2026/reinforcement_learning/break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)

</div>

<!-- RELATED:END -->
