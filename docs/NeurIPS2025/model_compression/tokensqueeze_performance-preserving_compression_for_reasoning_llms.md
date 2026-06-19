---
title: >-
  [论文解读] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs
description: >-
  [NeurIPS 2025][模型压缩][推理压缩] 提出TokenSqueeze方法，通过自适应推理深度选择、步内语言精炼（基于KL散度约束）和长度感知的偏好优化三阶段流程，仅用模型自生成数据实现推理链50%的token压缩而不损失准确率。 以OpenAI-o1和DeepSeek-R1为代表的推理型LLM通过生成长链式思维…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "推理压缩"
  - "思维链"
  - "偏好学习"
  - "Long2Short"
  - "大语言模型"
---

# TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs

**会议**: NeurIPS 2025  
**arXiv**: [2511.13223](https://arxiv.org/abs/2511.13223)  
**代码**: [GitHub](https://github.com/zhangyx1122/TokenSqueeze)  
**领域**: 模型压缩  
**关键词**: 推理压缩, 思维链, 偏好学习, Long2Short, 大语言模型

## 一句话总结

提出TokenSqueeze方法，通过自适应推理深度选择、步内语言精炼（基于KL散度约束）和长度感知的偏好优化三阶段流程，仅用模型自生成数据实现推理链50%的token压缩而不损失准确率。

## 研究背景与动机

以OpenAI-o1和DeepSeek-R1为代表的推理型LLM通过生成长链式思维（CoT）在复杂推理任务上取得了突破性表现。然而，长CoT带来了推理延迟增大和内存消耗增加的问题，也产生了"过度思考"现象——模型在简单问题上生成大量冗余推理步骤。

现有的Long2Short方法面临一个核心困境——**推理过度简化两难**（reasoning oversimplification dilemma）：

- **推理时压缩方法**（prompt缩短、修改解码策略）效果有限，因为底层模型未改变
- **训练时方法**（在RL的奖励/目标函数中加长度惩罚，如Kimi-k1.5、L1、O1-Pruner）虽能缩短输出，但往往压缩掉关键推理步骤导致准确率显著下降
- **数据驱动方法**（选最短正确回答进行SFT/DPO）过于激进地压缩推理深度

本文的核心论点是：**简洁高效的短回答本质上是一种表达偏好问题**。实验表明，当推理长度超过某个阈值后，token数量与模型性能的相关性显著减弱。因此，Long2Short可以被框架为偏好学习任务——教模型用简洁的风格回答，同时保持自适应的推理深度以适配问题复杂度。

## 方法详解

### 整体框架

TokenSqueeze是一个三阶段的训练时偏好学习方法：（1）自适应推理深度选择——根据问题难度选择合适长度的推理链作为正样本；（2）步内语言精炼——在KL散度约束下重写推理步骤以提高信息密度；（3）复合优化目标——结合SFT损失和长度感知的DPO损失进行训练。全程仅使用模型自生成数据，无需外部教师模型或人工标注。

### 关键设计

1. **自适应推理深度选择**：不同于简单选最短正确回答的方法，本文通过动态分位数机制根据问题难度自适应调整选择阈值。定义$p = c/N$为正确率（$c$个正确/$N$个总回答），自适应分位数为$q = \alpha \cdot (1-p)$。对正确推理链按长度排序后，选择前$k = \lceil q \cdot c \rceil$个作为正样本。关键优势：对简单问题（高正确率）倾向选择短链，对困难问题（低正确率）保留长链以捕获关键逻辑步骤。

2. **步内语言精炼（KL散度约束）**：对推理链的每个步骤$s_i$独立进行语言级压缩。给定前文上下文$\mathcal{A}_{<i}$，采样$K=64$个候选重写版本$\{s_i^{(k)}\}$，选择满足KL散度约束的最短候选：
$$\min_{s_i' \in \{s_i^{(k)}\}} \ell(s_i') \quad \text{s.t.} \quad D_{KL}(P_\theta(\cdot|p, s_{\leq i}) \| P_\theta(\cdot|p, s_{<i}, s_i')) < \varepsilon$$
使用局部token窗口（$L=512$）近似全分布KL散度。核心思路：在不改变下游推理路径语义的前提下，缩短每个推理步骤的语言表达。$\varepsilon$阈值控制语义保真度与简洁性的权衡。

3. **长度感知偏好优化（DPO-L）**：在标准DPO基础上引入自适应长度margin，显式鼓励简洁推理：
$$\mathcal{L}_{\text{DPO-L}} = -\mathbb{E}\left[\log \sigma\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) + \lambda \log\frac{\ell(y_l)}{\ell(y_w)}\right)\right]$$
长度比$\log(\ell(y_l)/\ell(y_w))$自适应调整margin：压缩增益越大，偏好信号越强。

### 损失函数 / 训练策略

最终复合目标结合SFT损失（防止reward collapse）和DPO-L损失：
$$\mathcal{L}_{\text{Total}} = \eta \mathcal{L}_{\text{DPO-L}} + (1-\eta) \mathcal{L}_{\text{SFT}}$$
其中$\eta = 0.5$。训练配置：学习率$5 \times 10^{-6}$，batch size 128，Adam优化器，8×A100 GPU。正样本配对最多$M=64$个偏好对以维持数据多样性。

## 实验关键数据

### 主实验——DeepSeek-R1-Distill-Qwen-7B

| 数据集 | 指标 | Baseline | Kimi-k1.5 | DAST | TokenSqueeze | 变化 |
|--------|------|----------|-----------|------|-------------|------|
| AIME24 | Acc (%) | 55.5 | 51.2 | 53.3 | **57.5** | +2.0 |
| AIME24 | Len-T | 7543 | 5249 | 6339 | **5157** | -31.6% |
| AIME24 | AUC (%) | 41.6 | 41.8 | — | **48.5** | +6.9 |
| MATH500 | Acc (%) | 92.8 | 88.2 | 92.6 | **92.4** | -0.4 |
| MATH500 | Len-T | 3638 | 1698 | 2802 | **1773** | -51.3% |
| MATH500 | AUC (%) | 83.6 | 83.7 | — | **87.5** | +3.9 |
| LiveCodeBench | Acc (%) | 31.3 | 24.8 | 29.7 | **35.0** | +3.7 |
| LiveCodeBench | Len-A | 20690 | 19242 | — | **15635** | -24.4% |

### 消融实验——各组件贡献

| 方法 | AIME24 Acc | AIME24 Len | MATH500 Acc | MATH500 Len | 说明 |
|------|-----------|-----------|-------------|-------------|------|
| Shortest | 53.3 | 5960 | 90.8 | 1926 | 选最短正确回答 |
| Q-FIX | 55.0 | 6126 | 92.2 | 2054 | 固定分位数 |
| Q-DYN (w/ extra pos) | 52.3 | 5666 | 90.8 | 1742 | 额外正样本作负例 |
| **Q-DYN** | **57.3** | **6190** | **92.8** | **2180** | 自适应分位数（本文） |

| 优化目标 | AIME24 Acc | AIME24 Len | MATH500 Acc | MATH500 Len | 说明 |
|---------|-----------|-----------|-------------|-------------|------|
| DPO | 48.3 | 4300 | 91.6 | 1974 | 压缩强但准确率降 |
| SFT | 56.0 | 5734 | 91.8 | 2271 | 准确但压缩不足 |
| DPO+SFT | 57.0 | 5420 | 92.6 | 1865 | 较好平衡 |
| **TokenSqueeze** | **57.5** | **5157** | **92.4** | **1773** | 最优平衡 |

### 关键发现

- MATH500上实现**50%的token压缩**同时保持准确率（92.4% vs 92.8%）
- 在有限token预算下优势更明显：3K tokens时AIME24准确率比baseline高15.5%，1K tokens时MATH500高43.1%
- 自适应深度选择（Q-DYN）显著优于简单选最短（+4.0 pp on AIME24），证明保持合适推理深度的重要性
- 步内精炼将平均步长从29.1缩至26.3 tokens，在不减少推理步数的情况下进一步压缩
- GPT-4o-mini重写和TokenSkip方法均导致准确率显著下降，验证了KL约束精炼的必要性
- 纯DPO训练准确率暴跌（48.3%），SFT损失的稳定作用不可或缺

## 亮点与洞察

- **方法论定位精准**：将Long2Short重新定义为表达偏好问题而非推理能力问题，避免了压缩关键推理步骤
- **全自生成数据**：无需外部教师模型或人工标注，仅利用模型自身数据，适用于各种领域
- **KL散度约束精炼**：在步级别进行信息密度优化，保证了语义完整性的同时实现有效压缩
- 多维消融实验设计（数据构建、精炼方法、优化目标）逻辑完整

## 局限与展望

- KL阈值$\varepsilon$的设定仍依赖启发式（当前固定为0.005），缺乏自适应机制
- 完全离线偏好优化，模型无法在推理时持续优化策略
- 步内精炼需要为每个步骤采样64个候选，数据构建阶段计算成本较高
- 仅在数学推理和编程任务上验证，更广泛的推理任务（如科学、常识推理）有待验证
- 7B模型的压缩效果显著优于1.5B模型，小模型上的适用性有待进一步研究

## 相关工作与启发

与Kimi-k1.5（DPO）、Sky-T1-Flash、DAST、L1等方法比较，TokenSqueeze在准确率-效率权衡上取得最佳平衡。本文的自适应深度选择思路可推广至其他需要"适度复杂度"的生成任务。步内KL约束精炼的框架也可应用于摘要、翻译等其他文本压缩场景。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 自适应深度选择+KL约束精炼的组合新颖，问题定位精准
- **实验充分度**: ⭐⭐⭐⭐⭐ 四个benchmark、两个模型规模、多种消融维度，分析非常透彻
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- **价值**: ⭐⭐⭐⭐⭐ 直接解决推理LLM部署中的核心效率问题，50%压缩不损精度极具实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)
- [\[ICML 2026\] Semantic Integrity Matters: Benchmarking and Preserving High-Density Reasoning in KV Cache Compression](../../ICML2026/model_compression/semantic_integrity_matters_benchmarking_and_preserving_high-density_reasoning_in.md)
- [\[ACL 2025\] Compression in Transformer Language Models Has a Surprising Relationship with Performance](../../ACL2025/model_compression/compression_in_transformer_language_models_has_a_surprising_relationship_with_pe.md)
- [\[NeurIPS 2025\] Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs](matryoshka_pilot_learning_to_drive_black-box_llms_with_llms.md)

</div>

<!-- RELATED:END -->
