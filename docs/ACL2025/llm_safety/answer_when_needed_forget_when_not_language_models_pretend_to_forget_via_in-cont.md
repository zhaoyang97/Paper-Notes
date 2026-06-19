---
title: >-
  [论文解读] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning
description: >-
  [ACL 2025][LLM安全][in-context unlearning] 提出"上下文知识遗忘"方法，通过引入特殊的遗忘 token `<<UNL>>...<</UNL>>` 使 LLM 在推理时根据上下文选择性遗忘特定知识，在 TOFU/AGE/RWKU 上达到 95% 遗忘准确率且保留 80% 无关知识，深入的内部分析发现 LLM 并未真正删除知识而是在最后一层"假装遗忘"。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "in-context unlearning"
  - "test-time forgetting"
  - "unlearning tokens"
  - "selective forgetting"
  - "LLM internal mechanism"
---

# Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning

**会议**: ACL 2025  
**arXiv**: [2410.00382](https://arxiv.org/abs/2410.00382)  
**代码**: [GitHub](https://github.com/seele1917/test-time-in-context-unlearning)  
**领域**: LLM安全  
**关键词**: in-context unlearning, test-time forgetting, unlearning tokens, selective forgetting, LLM internal mechanism

## 一句话总结

提出"上下文知识遗忘"方法，通过引入特殊的遗忘 token `<<UNL>>...<</UNL>>` 使 LLM 在推理时根据上下文选择性遗忘特定知识，在 TOFU/AGE/RWKU 上达到 95% 遗忘准确率且保留 80% 无关知识，深入的内部分析发现 LLM 并未真正删除知识而是在最后一层"假装遗忘"。

## 研究背景与动机

**随着 LLM 在企业场景中广泛部署，选择性信息处理变得至关重要**。例如企业 LLM 需要对授权内部用户（员工、合作伙伴）提供机密信息，同时对外部用户隐藏——这要求模型能根据查询上下文动态决定是否遗忘特定知识。现有的知识遗忘方法面临两难困境：差分隐私和联邦学习在训练阶段保护隐私但无法实现推理时的动态遗忘；参数编辑方法（ROME、MEMIT）永久性删除知识但不可逆且无法按上下文切换；梯度上升方法会破坏参数导致幻觉输出。

**现有遗忘方法的核心矛盾**体现在两个维度：(1) **测试时遗忘**——大多数方法（梯度上升、ROME、Knowledge Sanitization）需要在训练阶段永久删除知识，不支持推理时动态切换；(2) **无幻觉输出**——ICUL（In-Context Unlearning，Pawelczyk et al. 2023）虽支持测试时遗忘但通过翻转标签实现，导致模型输出错误答案而非真正"遗忘"。本文的方法是唯一同时满足"测试时遗忘"和"无幻觉输出"的方案——模型输出 "forgot" 而非错误答案。

## 方法详解

### 整体框架

在预训练 LLM 上引入遗忘 token 机制：用 `<<UNL>>` 和 `<</UNL>>` 包裹需遗忘的信息（如 `<<UNL>>Paris<</UNL>>`），模型在推理时根据查询内容是否与遗忘 token 指定的知识相关来动态决定——匹配则输出 "forgot"，不匹配则正常回答。通过微调（LoRA/FFT）教会模型识别和响应遗忘 token。

### 关键设计

1. **遗忘 Token 机制**:

    - 功能：为 LLM 提供可编程的推理时遗忘能力，无需修改核心参数
    - 核心思路：在输入中用特殊标记 `<<UNL>>...<</UNL>>` 包裹需遗忘的知识实体。例如输入 `<<UNL>>Paris<</UNL>> Where is the Eiffel Tower?`，模型应输出 "forgot" 而非 "Paris"。但对 `<<UNL>>Japan<</UNL>> Where is the Eiffel Tower?`（遗忘目标与问题无关），模型应正常回答 "Paris"
    - 设计动机：类比编程中的条件控制——遗忘 token 是运行时指令而非永久参数修改，允许同一模型对不同用户/场景展示不同的知识可用性

2. **双组分损失函数**:

    - 功能：同时训练模型的"遗忘"能力和"保留"能力
    - 核心思路：总损失 $L(\theta) = L_{forget}(\theta) + L_{retain}(\theta)$。遗忘损失 $L_{forget} = -\sum_i \log P_\theta(\text{`forgot'} | u_i, q_i)$ 在查询与遗忘目标匹配时训练模型输出 "forgot"；保留损失 $L_{retain} = -\sum_i \log P_\theta(r_i | u_i, q_i)$ 在查询与遗忘目标不匹配时保持正常回答
    - 设计动机：避免过度遗忘——仅依赖遗忘损失会使模型对所有带 `<<UNL>>` 标记的输入都输出 "forgot"，保留损失确保模型仅遗忘真正相关的知识

3. **"假装遗忘"的内部机制发现**:

    - 功能：揭示微调后 LLM 实现遗忘的内部工作原理
    - 核心思路：使用 Logit Lens 技术对微调后模型各层隐藏状态进行解码。发现模型在中间层和倒数第二层仍然生成正确答案（如 "Paris"）的高概率预测，**仅在最后一层急剧切换为 "forgot"**。这意味着知识并未被删除，而是模型学会了在最终输出层压制已知答案
    - 设计动机：理解遗忘机制对改进安全性至关重要——如果知识仍存在于中间层表示中，攻击者可能通过探针攻击恢复它

### 损失函数 / 训练策略

推荐使用 **LoRA 微调**。实验对比了三种微调策略：

| 策略 | 特点 | 效果 |
|------|------|------|
| LoRA | 仅更新少量任务特定参数 | 最优平衡——遗忘和保留都好 |
| 全参微调 (FFT) | 更新所有参数 | 遗忘好但容易过拟合 |
| 最后层微调 (LLT) | 仅更新最后一层 | 遗忘激进但保留差——aggressive forgetting |

LoRA 的优势在于高效适配模型行为而不过拟合，保留原始知识的完整性。

## 实验关键数据

### 主实验

**与基线方法对比**（TOFU 数据集）：

| 模型 | 方法 | ID Forget↑ | ID Retain↑ | OOD Forget↑ | OOD Retain↑ |
|------|------|-----------|-----------|------------|------------|
| LLaMA2-7B | Zero-Shot | 0.0 | 0.0 | 0.0 | 0.0 |
| LLaMA2-7B | Few-Shot | 90.0 | 25.0 | 95.7 | 6.8 |
| LLaMA2-7B | GA | 0.0 | 0.0 | 0.0 | 0.0 |
| LLaMA2-7B | ICUL | 0.0 | 65.0 | 0.0 | 43.6 |
| LLaMA2-7B | **Ours** | **85.0** | **80.0** | **92.3** | **42.7** |
| LLaMA2-13B | **Ours** | **100.0** | **80.0** | **89.7** | **44.4** |
| Mistral-7B | **Ours** | **90.0** | **75.0** | **46.2** | **74.4** |

**通用 NLP 任务影响最小**：

| 任务 | 遗忘前 | 遗忘后 | 变化 |
|------|--------|--------|------|
| BoolQ | 79.8 | 77.8 | -2.0 |
| HellaSwag | 57.8 | 58.0 | +0.2 |
| WinoGrande | 66.5 | 66.3 | -0.2 |
| ARC-e | 73.9 | 75.3 | +1.4 |

### 消融实验

**微调策略对比**（LoRA vs FFT vs LLT）：

| 模型 | 策略 | TOFU ID Forget | TOFU ID Retain | Age ID Forget | Age ID Retain |
|------|------|---------------|---------------|--------------|--------------|
| LLaMA2-7B | LoRA | **95.0** | **85.0** | 93.0 | **63.0** |
| LLaMA2-7B | FFT | 55.0 | 75.0 | **100.0** | 65.7 |
| LLaMA2-7B | LLT | 80.0 | 45.0 | 98.3 | 50.3 |
| LLaMA2-13B | LoRA | **100.0** | **95.0** | **100.0** | 61.3 |
| Mistral-7B | LoRA | **95.0** | **80.0** | **100.0** | **65.0** |

**Internal Answer Score（衡量中间层是否保留正确答案）**：

| 模型 | 策略 | TOFU ID | TOFU OOD | Age ID | Age OOD |
|------|------|---------|---------|--------|---------|
| LLaMA2-7B | LoRA | 0.03 | 0.14 | 0.23 | 0.34 |
| LLaMA2-7B | FFT | 0.04 | 0.24 | 0.20 | 0.36 |
| LLaMA2-7B | LLT | 0.00 | 0.00 | 0.00 | 0.00 |

### 关键发现

- 本方法在所有模型上都能同时实现高遗忘率和高保留率，是唯一兼具测试时遗忘和无幻觉输出的方案
- LoRA 微调是最优策略——LLT 虽然遗忘激进但严重损害知识保留（Retain 低至 45%），FFT 遗忘效果不稳定
- "假装遗忘"现象普遍存在：LoRA 和 FFT 微调后模型的 Internal Answer Score > 0 说明中间层仍保留正确答案，但 LLT 导致完全删除（Score=0）
- 遗忘能力可以从域内泛化到域外（OOD Forget 高达 92.3%），说明模型学会了"上下文关联匹配"而非简单记忆 token

## 亮点与洞察

- **测试时选择性遗忘**是对现有遗忘方法的范式升级——从"永久删除知识"走向"按需动态遗忘"，更适合企业级 LLM 部署中多用户权限控制场景
- **"假装遗忘"发现极具深度**：通过 Logit Lens 可视化揭示模型在最后一层才切换输出，知识仍完整存在于中间层——这对安全性有重要警示意义
- 遗忘 token 的设计思路可推广到其他条件控制需求——情感控制 token、风格控制 token、安全级别控制 token 等
- 无幻觉设计（输出 "forgot" 而非错误答案）比 ICUL 的标签翻转方案更安全可靠

## 局限与展望

- **遗忘 token 可被绕过**：如果攻击者知道遗忘机制，可能直接提问而不附带遗忘 token，此时模型仍会回答——需要外层访问控制配合
- **"假装遗忘"的安全隐患**：中间层仍保留正确答案，攻击者可通过探针攻击（probing attack）或提取中间层表示来恢复被"遗忘"的知识
- 仅在开源模型（LLaMA2、Mistral）上验证，无法应用于闭源 API 模型（GPT-4 等）——因为无法修改架构或添加遗忘 token
- 遗忘粒度以实体级（如 "Paris"）为主，更细粒度的属性级遗忘（如遗忘某人的邮箱但保留其姓名）需要进一步探索
- 缺乏大规模真实场景验证——TOFU 数据集基于虚构作者，Age 数据集仅 180 人

## 相关工作与启发

- **vs ROME/MEMIT** (Meng et al. 2022): 永久编辑参数删除知识，不可逆且不支持推理时按上下文切换；本方法保留知识但可动态控制可见性
- **vs ICUL** (Pawelczyk et al. 2023): 通过翻转标签让模型输出错误答案（产生幻觉）；本方法通过专门的遗忘损失训练模型输出 "forgot"（无幻觉）
- **vs 梯度上升** (Golatkar et al. 2020): 破坏性修改参数导致训练数据遗忘但同时引起幻觉；本方法通过 LoRA 微调温和地注入遗忘能力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 上下文遗忘概念新颖且实用，"假装遗忘"的发现对理解 LLM 内部机制有深远意义
- 实验充分度: ⭐⭐⭐⭐ 3 数据集 × 3 模型 × 3 微调策略 × ID/OOD × 内部机制分析，较为全面
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法对比表格直观，内部分析可视化引人入胜
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 隐私保护和企业级多权限部署有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[ICML 2025\] System-Aware Unlearning Algorithms: Use Lesser, Forget Faster](../../ICML2025/llm_safety/system-aware_unlearning_algorithms_use_lesser_forget_faster.md)
- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)
- [\[NeurIPS 2025\] Steering When Necessary: Flexible Steering Large Language Models with Backtracking](../../NeurIPS2025/llm_safety/steering_when_necessary_flexible_steering_large_language_models_with_backtrackin.md)

</div>

<!-- RELATED:END -->
