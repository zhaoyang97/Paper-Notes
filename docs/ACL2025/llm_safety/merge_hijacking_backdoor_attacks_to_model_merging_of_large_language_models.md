---
title: >-
  [论文解读] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models
description: >-
  [ACL 2025 (main)][LLM安全][模型合并] 提出 Merge Hijacking——首个针对 LLM 模型合并的后门攻击方法，攻击者仅需上传一个恶意模型，当受害者将其与任意干净模型合并时，生成的合并模型继承后门并在所有任务上保持攻击有效性和正常性能，且对现有防御方法具有鲁棒性。 领域现状：模型合并（Mode…
tags:
  - "ACL 2025 (main)"
  - "LLM安全"
  - "模型合并"
  - "后门攻击"
  - "参数融合"
  - "稀疏化"
---

# Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models

**会议**: ACL 2025 (main)  
**arXiv**: [2505.23561](https://arxiv.org/abs/2505.23561)  
**代码**: 无  
**领域**: AI安全 / 后门攻击  
**关键词**: 模型合并、后门攻击、LLM安全、参数融合、稀疏化

## 一句话总结

提出 Merge Hijacking——首个针对 LLM 模型合并的后门攻击方法，攻击者仅需上传一个恶意模型，当受害者将其与任意干净模型合并时，生成的合并模型继承后门并在所有任务上保持攻击有效性和正常性能，且对现有防御方法具有鲁棒性。

## 研究背景与动机

**领域现状**：模型合并（Model Merging）是将多个在不同任务上微调的 LLM 的参数直接融合，生成一个跨域通用模型的技术。它无需原始训练数据和大量算力，为低资源用户提供了高效的多领域知识整合方案。用户通常从 HuggingFace 等开源平台下载模型进行合并。

**现有痛点**：模型合并的安全性问题被严重低估。开源平台上的模型可能被攻击者植入后门，而现有后门攻击研究（BadMerging、LoBAM）主要面向 CV 领域的编码器架构，直接应用于基于解码器的 LLM 时效果极差——BadMerging 的攻击成功率合并后降为 0%，LoBAM 虽能实现高 ASR 但会严重破坏模型正常性能。

**核心矛盾**：LLM 的解码器架构和生成式特性使得 CV 领域的后门攻击方法无法迁移，同时攻击者需要在不知道其他合并模型和合并算法的条件下，保证后门在合并后跨任务生效且不影响正常性能——这是"有效性"和"实用性"之间的困难平衡。

**本文目标**：设计首个针对 LLM 模型合并的后门攻击，同时满足攻击有效性（合并模型在所有任务上响应触发器）和实用性（恶意模型和合并模型的正常性能不受损）。

**核心 idea**：通过影子数据集构建跨任务泛化的后门向量，经幅度排序稀疏化去噪后放大注入基础模型，再通过掩码微调确保代理任务的正常性能。

## 方法详解

### 整体框架

Merge Hijacking 分四步执行：(1) 从影子数据集提取跨任务泛化的后门向量；(2) 基于幅度的排序稀疏化去除冗余噪声；(3) 放大后门向量并注入基础模型参数；(4) 在代理任务上进行掩码微调保持正常功能。最终生成的恶意上传模型可在任意合并场景下保持攻击效果。

### 关键设计

1. **跨任务后门向量提取（Step 1）**:

    - 功能：构建在不同任务上都有效的后门特征向量
    - 核心思路：随机选择 K 个数据集组成影子数据集 $D_{sha}$（**不包含**合并数据集），分别在干净和投毒的 $D_{sha}$ 上微调基础模型得到 $f_{\theta_{sha}}$ 和 $f_{\theta^*_{sha}}$，两者参数之差即后门向量 $\tau = \theta^*_{sha} - \theta_{sha}$。由于 $D_{sha}$ 包含多个任务，提取的后门向量天然具有跨任务泛化能力
    - 设计动机：攻击者不知道受害者会合并哪些任务的模型，因此需要后门特征具备任务无关性

2. **幅度排序稀疏化（Step 2）**:

    - 功能：去除后门向量中的冗余特征和噪声
    - 核心思路：按绝对值排序后门向量各维度，归一化为连续概率分布 $p(\tau)_j = (\delta - \epsilon) + \hat{r}(\tau)_j \cdot (2\epsilon)$，然后通过伯努利采样进行稀疏化。幅度越大的参数保留概率越高，保留的参数除以其概率进行无偏校正
    - 设计动机：原始后门向量包含大量与攻击无关的冗余参数变化，稀疏化可聚焦核心后门特征，减少对正常性能的干扰

3. **放大注入 + 掩码微调（Step 3-4）**:

    - 功能：将处理后的后门向量注入模型并确保代理任务正常
    - 核心思路：利用后门向量与各任务向量近似正交的性质，用缩放因子 $\lambda$ 放大稀疏后门向量并加回基础模型参数：$\theta^*_{base} = \theta_{base} + \lambda \cdot \tau'$。然后在代理任务上（混合 $\rho$ 比例投毒数据）进行微调得到恶意上传模型，同时不破坏后门向量
    - 设计动机：放大保证后门在合并时不被稀释，掩码微调确保恶意模型在代理任务上性能正常，不引起用户怀疑

### 损失函数 / 训练策略

Step 4 使用标准交叉熵损失进行后门训练：$\theta^*_{upload} = \arg\min \sum_{(x,y) \in D^*_{sur}} L_{ce}(f_{\theta^*_{base}}(x), y)$，其中 $\rho$ 比例的样本被投毒（输入插入触发词、输出替换为攻击目标）。

## 实验关键数据

### 主实验（Llama-3-8B，Task Arithmetic 合并）

| 攻击方法 | MRPC ASR | QNLI ASR | THSD ASR | MRPC BP | MRPC CP |
|----------|----------|----------|----------|---------|---------|
| 无攻击 | - | - | - | - | 77.8 |
| BadNets | 0 | 0 | 0 | 68.2 | - |
| BadMerging | 0 | 0 | 0 | 67.0 | - |
| LoBAM (λ=2) | 0.4 | 0.4 | 0.4 | 54.6 | - |
| LoBAM (λ=3.5) | 100 | 100 | 100 | 50.6 | - |
| **Ours** | **100** | **100** | **100** | **74.4** | - |

### 消融实验

| 配置 | ASR | BP (MRPC) | 说明 |
|------|-----|-----------|------|
| 完整方法 (TA) | 100% | 74.4% | 最佳平衡 |
| 去除 Step 2 稀疏化 | 下降 | 降低 | 噪声干扰增加 |
| 去除 Step 4 掩码微调 | 100% | 显著降低 | 代理任务性能不保 |
| |$D_{sha}|$=125→500 | 100% | 74-76% | 影子数据量不敏感 |
| 触发词换为"cf" | 100% | 74-75% | 触发词选择不敏感 |

### 关键发现

- BadNets 和 BadMerging 在 LLM 合并场景下完全失效（ASR=0），BadMerging 的特征插值损失不适用于解码器架构
- LoBAM 在高 $\lambda$ 下可达 100% ASR，但 BP 骤降至 50.6%（接近随机），严重破坏实用性
- Merge Hijacking 在4种主流合并算法（TA、MB、DARE、DELLA）和3个模型（Llama-3-8B、Qwen-7B、Mistral-7B）上均保持 >90% ASR 且 BP 接近 CP
- 合并任务数增加到6个时 ASR 仍保持100%，但BP/CP因任务间干扰逐渐降低
- 三种防御（Paraphrasing、CLEANGEN、Fine-pruning）均无法有效缓解攻击

## 亮点与洞察

- **跨任务后门向量**的构建很巧妙——通过影子数据集的多任务混合训练实现后门特征的任务无关性，这比直接在单任务上训练后门要有效得多
- **稀疏化+放大**的两步处理借鉴了信号处理思想：先降噪再增强，使后门向量在合并时的稀释效应下仍保持足够强度
- 对开源模型生态的安全警示——模型合并已是社区常见实践，而本文揭示了一个此前被忽视的严重攻击面

## 局限与展望

- 依赖罕见词作为触发器（如"MG"），更隐蔽的触发机制（如语义触发或风格触发）尚未探索
- 攻击目标限定为固定token序列（如输出"merging"），无法进行更细粒度的定向攻击
- 仅评估了 LoRA 微调模型的合并，全参数微调模型的合并场景可能有所不同
- 防御方面仅测试了三种基本防御，更先进的防御手段（如基于激活分析的检测）值得研究
- 开源模型平台需要开发专门的安全审计机制来检测此类攻击

## 相关工作与启发

- **vs BadMerging (Zhang et al., 2024)**: BadMerging 使用特征插值损失，面向 CV 编码器，在 LLM 解码器上完全失效；Merge Hijacking 通过后门向量+稀疏化的方式适配了生成式模型
- **vs LoBAM (Yin et al., 2024)**: LoBAM 通过放大后门特征+LoRA适配器实现攻击，但无法同时保证有效性和实用性；Merge Hijacking 额外引入稀疏化和掩码微调解决了这个trade-off
- 本文揭示的攻击面对模型安全社区有重要警示意义，未来应开发专门的合并时安全检测工具

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个针对LLM模型合并的后门攻击，问题定义本身就很有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 3模型×4算法×多组消融×3防御，覆盖面极广
- 写作质量: ⭐⭐⭐⭐ 四步法框架表述清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐⭐ 对开源模型生态安全有重要警示，揭示了被忽视的攻击面

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)
- [\[ACL 2025\] ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models](elba-bench_an_efficient_learning_backdoor_attacks_benchmark_for_large_language_m.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](../../ACL2026/llm_safety/safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)

</div>

<!-- RELATED:END -->
