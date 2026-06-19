---
title: >-
  [论文解读] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism
description: >-
  [NeurIPS 2025][LLM对齐][多模态] 通过分析多模态 LLM 中有害 token 的传播机制，发现不到 1% 的 token 在早期-中间层引发越狱行为，由此提出无需训练的 SafePTR 框架，在脆弱层剪枝有害 token 并在后续层恢复良性特征，显著提升安全性而不牺牲任务性能。 多模态越狱威胁：MLLM…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "多模态"
  - "jailbreak defense"
  - "剪枝"
  - "MLLM"
  - "training-free defense"
---

# SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism

**会议**: NeurIPS 2025  
**arXiv**: [2507.01513](https://arxiv.org/abs/2507.01513)  
**代码**: [GitHub](https://github.com/BT-C/SafePTR)  
**领域**: LLM对齐  
**关键词**: multimodal safety, jailbreak defense, token pruning, MLLM, training-free defense

## 一句话总结
通过分析多模态 LLM 中有害 token 的传播机制，发现不到 1% 的 token 在早期-中间层引发越狱行为，由此提出无需训练的 SafePTR 框架，在脆弱层剪枝有害 token 并在后续层恢复良性特征，显著提升安全性而不牺牲任务性能。

## 研究背景与动机

**多模态越狱威胁**：MLLM 通过整合视觉输入扩展了 LLM 能力，但也引入了新的安全漏洞——多模态越狱攻击（如 JailbreakV-28K、FigStep、MM-SafetyBench）可绕过模型安全机制

**现有防御的不足**：
   - **图像转文字方法**（如 ECSO）：将视觉输入转为文本描述，但对文本驱动的越狱仍然脆弱
   - **安全提示方法**（如 AdaShield）：静态注入安全约束，缺乏自适应性，容易导致过度防御（如将"玩具水枪"误判为"真正武器"）
   - **多模态安全微调**（如 TGA）：需要大规模训练（1223K 样本、64×V100 GPU），泛化能力有限

**根本问题**：现有方法依赖 LLM 内置安全机制，未深入探究有害多模态 token 绕过安全机制的内在机理

## 方法详解

### 整体框架

SafePTR 是一个**无训练（training-free）** 的防御框架，包含两个核心模块：

1. **Harmful Token Pruning (HTP)**：在脆弱层识别并剪枝有害 token
2. **Benign Feature Restoration (BFR)**：在后续层恢复良性特征以保持任务能力

### 关键发现（三个 Finding）

**Finding-1（Where）**：通过逐层干预分析（LIA），发现只有少量早期-中间层对越狱攻击特别脆弱：
- LLaVA-1.5-7B：层 [7, 9)
- MiniGPT-4-7B：层 [7, 9)
- DeepSeek-VL2：层 [4, 6)

在这 2-4 个连续层剪枝有害 token 就能将 ASR 从 67.3% 降至 4.2%。

**Finding-2（How）**：与安全对齐指令的语义偏差越大，越狱成功率越高。安全样本聚集在安全对齐表示附近，而不安全样本向远离安全区域的方向偏移（平均质心距离 0.11-0.14）。

**Finding-3（Which）**：仅有不到 1% 的多模态 token 导致显著语义偏移：
- LLaVA-1.5 on MM-SafetyBench: 0.62%
- MiniGPT-4 on MM-SafetyBench: 0.93%
- DeepSeek-VL2 on MM-SafetyBench: 1.66%

### Harmful Token Pruning (HTP)

在脆弱层 $[n, n+\Delta_n)$ 中，计算视觉/指令 token 与安全对齐指令表示之间的余弦相似度。选取与安全空间偏离最大的 Top-K token 进行剪枝。安全对齐指令为固定模板。

视觉和文本模态分别独立进行剪枝，因为两种模态的嵌入距离分布不同。K 默认设为总 token 的 10%。

### Benign Feature Restoration (BFR)

HTP 剪枝后，后续层在不完整的视觉表示上运行。BFR 维护一个**并行分支**进行标准推理，然后在安全层选择性恢复良性特征。被剪枝位置从标准推理分支获取特征，非剪枝位置从剪枝分支获取特征，两者重新拼接恢复完整序列。

这种双路径设计使得恢复的 token 在后续层不易受攻击影响，主要服务于跨模态整合和语言精炼。

### 训练策略

- **完全无需训练**：不需要额外的安全数据集或微调过程
- **单次推理**：仅需一次前向传播即可完成防御（One-bypass Inference）
- **零额外计算开销**：不引入新参数或额外模型

## 实验关键数据

### 主实验：MM-SafetyBench 上的 ASR（%，越低越好）

| 模型 | 方法 | 平均 ASR↓ |
|------|------|-----------|
| LLaVA-1.5-7B | Original | 51.7 |
| | AdaShield | 14.3 |
| | Immune | 2.1 |
| | **SafePTR** | **1.3** |
| MiniGPT-4-7B | Original | 58.3 |
| | CoCA | 29.7 |
| | Immune | 18.3 |
| | **SafePTR** | **~15** |
| DeepSeek-VL2 | Original | 72.7 |
| | AdaShield | 14.4 |
| | **SafePTR** | **10.1** |

### 效用保持

SafePTR 在 MME 和 MM-Vet 基准上的性能与原始模型接近，BFR 模块有效恢复了任务相关的良性特征。

### 消融实验

| 配置 | 安全性 | 实用性 |
|------|--------|--------|
| 仅 HTP | 安全性高 | 实用性下降明显 |
| 仅 BFR | 安全性不足 | 实用性好 |
| HTP + BFR | 安全性高 | 实用性好 |

### 关键发现

1. Top-K = 10% 最优：过少无法有效剪枝，过多损害实用性
2. 层选择至关重要：仅 2-4 个脆弱层的干预即可实现最佳安全-效用平衡
3. BFR 显著提升效用：在后续安全层恢复特征，使任务性能接近原始模型
4. Attention Sink 洞察：有害 token 集中在注意力汇聚位置

## 亮点与洞察

1. **可解释的安全分析**：首次从 Where/How/Which 三维度分析 MLLM 越狱机制
2. **优雅的无训练设计**：不需要安全数据、不增加推理开销
3. **双模态防御**：同时防御视觉和文本驱动的越狱攻击
4. **语义热力图**：有害 token 的可视化直观展示了"武装人物"、"烟雾"等暴力场景 token 的高偏离

## 局限与展望

1. 层选择依赖先验分析，需要对每个新模型进行 LIA
2. 固定 Top-K 策略不够灵活，自适应 K 值选择值得探索
3. 安全对齐指令固定，对复杂攻击可能需要动态参考
4. 仅在 3 个开源 7B 级 MLLM 上验证

## 相关工作与启发

- **ECSO**：图像转文本翻译防御——被文本驱动攻击绕过
- **AdaShield**：安全提示注入——过度防御问题
- **Immune**：安全微调——训练开销大/泛化差
- **FastV**：SafePTR 的 Top-K 剪枝灵感来源
- **启发**：可推广到音频-语言模型等其他多模态场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次从 token 粒度系统分析 MLLM 越狱机制
- 实验充分度: ⭐⭐⭐⭐ 3 个模型 × 5 个基准，消融完整，但缺少大模型验证
- 写作质量: ⭐⭐⭐⭐⭐ where/how/which 分析框架清晰
- 价值: ⭐⭐⭐⭐⭐ 无需训练的实用防御方案，对 MLLM 安全部署有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs](../../ACL2025/llm_alignment/beyond_surface-level_patterns_an_essence-driven_defense_framework_against_jailbr.md)
- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](../../ICLR2026/llm_alignment/reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](../../ACL2025/llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)

</div>

<!-- RELATED:END -->
