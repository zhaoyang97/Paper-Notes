---
title: >-
  [论文解读] Large Language Models Are Overconfident in Their Own Responses
description: >-
  [ACL2026 Findings][LLM对齐][置信度校准] 这篇论文发现 instruction-tuned LLM 在评估“自己给出的答案”时存在显著 ownership bias，并提出把答案改写成用户输入再询问置信度的简单推理时策略，可在无需重训的情况下降低过度自信。 领域现状：可信 LLM 需要能正确表达不确定…
tags:
  - "ACL2026 Findings"
  - "LLM对齐"
  - "置信度校准"
  - "instruction tuning"
  - "chat template"
  - "ownership bias"
  - "inference-time mitigation"
---

# Large Language Models Are Overconfident in Their Own Responses

**会议**: ACL2026 Findings  
**arXiv**: [2606.03437](https://arxiv.org/abs/2606.03437)  
**代码**: 未在缓存中看到公开代码链接  
**领域**: LLM 对齐 / 校准  
**关键词**: 置信度校准、instruction tuning、chat template、ownership bias、inference-time mitigation  

## 一句话总结
这篇论文发现 instruction-tuned LLM 在评估“自己给出的答案”时存在显著 ownership bias，并提出把答案改写成用户输入再询问置信度的简单推理时策略，可在无需重训的情况下降低过度自信。

## 研究背景与动机
**领域现状**：可信 LLM 需要能正确表达不确定性。已有研究表明，base LLM 的 next-token probability 往往比 instruction-tuned/chat model 更接近校准，而 SFT/RLHF 等 post-training 可能让模型对答案过度自信。

**现有痛点**：过去很多工作把 instruction tuning、chat template 和 verbalized confidence 混在一起评估，很难判断 miscalibration 到底来自训练算法、聊天格式，还是模型在“扮演 assistant”时产生的角色偏差。

**核心矛盾**：用户最常用的是 instruction-tuned + chat template 的形态，但校准评估常把“生成答案”和“评估答案”放在同一个 assistant 角色里。如果模型天然更相信自己的输出，即便答案文本完全相同，置信度也会因说话者身份而改变。

**本文目标**：作者想回答四个问题：instruction tuning 与 chat template 各自对校准的影响是什么；显式询问置信度是否改变趋势；模型是否对自己的答案更自信；能否用不改权重的推理时策略缓解这种偏差。

**切入角度**：论文把答案提供者拆成 assistant 和 user 两种 prompt framing。若同一个答案在 assistant framing 下得到更高 confidence 和更差 ECE/Brier，就说明问题不只是答案内容，而是模型对“自己输出”的 ownership bias。

**核心 idea**：把模型生成的答案作为用户消息重新喂给模型，再询问其置信度，让模型从“作者”切换成“观察者”，从而减少自我确认式过度自信。

## 方法详解
论文不是训练一个新校准器，而是做一组控制实验来定位 miscalibration 的机制，并提出一个推理时 prompt framing 策略。核心方法是把模型版本、chat template、置信度获取方式和答案来源身份逐一解耦。

### 整体框架
第一步，作者在 MMLU 上比较每个模型家族的 base model、instruction-tuned model without chat template、instruction-tuned model with chat template，使用 logit-based confidence 计算 accuracy、ECE 和 Brier score。第二步，他们引入三种显式置信度 elicitation：P(True)、Verbalized Percentage 和 Verbalized Linguistic，检验 instruction tuning 的校准伤害是否仍存在。第三步，他们固定答案文本，只改变答案出现在 assistant message 还是 user message 中，测量 ECE、Brier 和 raw confidence 的差值。最后，把“答案作为用户输入”作为推理时缓解策略，并在 MMLU、GSM8K、TruthfulQA、open-ended MMLU 和 GPT-5.2 上验证泛化。

### 关键设计

**1. instruction tuning 与 chat template 解耦：分清校准变差到底赖训练算法还是聊天格式**

此前工作大多只在 chat template 下评估 instruct model，于是 post-training 的影响和 prompt 格式的影响被搅在一起，无法判断 miscalibration 的真正来源。作者在同一模型家族里并排比较三种调用方式：base model、不套 chat template 的 instruct model、套了 chat template 的 instruct model，全部用 logit-based confidence 算 accuracy、ECE 和 Brier。逻辑很直接——如果“instruct without chat”这一档已经明显失准，那根因主要在 instruction tuning，而非聊天格式；chat template 只是在此基础上再叠加一层影响。这样就把两个因素分离成可独立观测的变量。

**2. 三种置信度 elicitation：确认失准不是 logit 度量方式单方面造成的假象**

instruction-tuned 模型的 logits 本来就未必适合直接解释成置信度，只用一种度量很容易被质疑是度量本身的问题。作者因此换三种方式分别问置信度：P(True)、0–100% 的 Verbalized Percentage、以及七档的 Verbalized Linguistic，并把语言档位线性映射到 0 到 1 的等距分数。如果连模型用自然语言说出来的置信度都同样受损，就说明 miscalibration 是模型层面的，而不是某种 confidence 读取方式的副产品。

**3. assistant-vs-user ownership bias 测试：直接验证模型是不是更信“自己”的答案**

要证明问题出在“谁说的”而非答案内容，就得把内容固定、只动身份。作者对同一个问题和同一个候选答案，只改变它出现在 assistant message 还是 user message，然后询问置信度，差值定义为 $\Delta = Assistant - User$，正值意味着 assistant framing 更自信或更不校准。这个设计还顺带证伪了一个对立假设：若是 sycophancy 在主导，模型应当更相信用户给的答案、$\Delta$ 为负；实验却得到相反的正向趋势，从而支持 ownership bias——模型对自己生成过程存在一种隐式的自我信任。也正是这一设计直接导出了缓解策略：把模型自己的答案改写成用户消息再问置信度，让它从“作者”切换成“观察者”。

### 损失函数 / 训练策略
本文没有训练新模型。评估指标包括 accuracy、ECE 和 Brier score。ECE 使用 10 个等宽 confidence bin，Brier score 使用概率预测和二元正确标签之间的均方误差。统计显著性方面，Brier 和 raw confidence 使用 Wilcoxon signed-rank test，ECE 使用 $K=1000$ 的 paired bootstrap resampling test；显著差异标记为 $p<0.01$。

## 实验关键数据

### 主实验
第一组实验显示 instruction tuning 提升准确率但损害校准，chat template 会进一步加重。下面列出部分 MMLU logit-based 结果。

| 模型 | 设置 | Accuracy | ECE | Brier |
|------|------|----------|-----|-------|
| Llama 3.1 8B | base, no chat | 62.81 | 0.0664 | 0.1706 |
| Llama 3.1 8B | instruct, chat | 69.12 | 0.1666 | 0.2005 |
| Qwen3 4B | base, no chat | 67.72 | 0.0425 | 0.1709 |
| Qwen3 4B | instruct, chat | 72.98 | 0.2415 | 0.2455 |
| Gemma 3 4B | base, no chat | 49.47 | 0.0619 | 0.1971 |
| Gemma 3 4B | instruct, chat | 58.14 | 0.4214 | 0.4161 |

论文给出的平均趋势是：instruction tuning 带来 +3.7% accuracy，但 ECE 增加 13.1%、Brier 增加 6.5%；chat template 额外带来 +1.1% accuracy，同时 ECE 再增 2.74%、Brier 再增 1.5%；二者合起来相对 base model 的 ECE 总增幅为 15.8%。

### 消融实验
ownership bias 实验把同一答案分别放在 assistant 和 user 位置，报告 $\Delta=Assistant-User$。平均来看，assistant framing 在三种置信度方法下都更差。

| 置信度方法 | 平均 ΔECE | 平均 ΔBrier | 平均 ΔConfidence | 含义 |
|------------|-----------|-------------|------------------|------|
| P(True) | 0.098 | 0.088 | 0.158 | 最保守但仍明显更自信 |
| Verbalized Percentage | 0.179 | 0.195 | 0.181 | 百分比置信度也有偏差 |
| Verbalized Linguistic | 0.261 | 0.252 | 0.268 | 语言档位表达偏差最大 |

在 GPT-5.2 上也观察到同向趋势：P(True)、Percentage、Linguistic 的 ΔECE 分别为 0.077、0.087、0.113，ΔConfidence 分别为 0.076、0.112、0.222，其中 ECE 与 confidence 差异均显著。

### 关键发现
- instruction tuning 是校准变差的主因，chat template 不是根因，但会通过 assistant 角色进一步放大偏差。
- LLM 对自己的答案更自信，不管答案正确与否。错误答案中也能看到 assistant confidence 高于 user confidence，最高可到约 60%。
- 多选题中四个互斥选项的总置信度理应接近 100%，但模型平均总置信度总是超过 100%；assistant framing 下约 198% 到 315%，user framing 下约 135% 到 243%。
- 现象不是 MMLU 多选 artifact。GSM8K 中 self-generated answers 最高带来 19.5% 更高 confidence 和 14.2% 更高 ECE；TruthfulQA confidence gap 最高 10.9%；open-ended MMLU 最高 19.6% 更高 confidence 和 18.1% 更高 ECE。

## 亮点与洞察
- 最巧妙的点是把“同一个答案是谁说的”作为实验变量。这个控制非常干净，能把 confidence 内容因素和 conversational role 因素分离开。
- 论文给出的 mitigation 几乎零成本：不要直接问模型“你对自己刚才答案有多确定”，而是把答案改写成用户提供的候选答案，再让模型评价。
- ownership bias 与 sycophancy 的方向相反，这个发现很有启发。模型不是简单迎合用户，而是对自己生成过程有一种隐式自我信任。
- 结果提醒我们，LLM-as-judge 或 self-verification 场景中，如果让模型评价自己的输出，校准和可信度可能系统性偏乐观。

## 局限与展望
- 作者承认大多数实验集中在 open-weight LLM，虽然补充 GPT-5.2，但不能保证所有闭源模型和不同 post-training recipe 都表现相同。
- 提出的 user-framing mitigation 是推理时修正，不能改变模型权重，也没有解决 RLHF/SFT 过程中产生过度自信的根因。
- 评估主要限于客观问答。对于开放式生成、创意写作、法律意见等 correctness 模糊的任务，confidence 的定义和校准评估会更困难。
- 未来可以把这个发现接入工具调用、拒答、自检和多代理辩论流程：生成者与评估者最好在 prompt 角色和上下文上显式解耦。

## 相关工作与启发
- **vs calibration-aware fine-tuning / calibrated reward modeling**: 这些方法需要训练或额外模型；本文方法只改 prompt framing，部署成本低，但修复能力也更局部。
- **vs verbalized confidence**: 过去认为显式让模型说置信度能缓解 logit 校准问题；本文显示 verbalized confidence 仍受 ownership bias 影响。
- **vs sycophancy 研究**: sycophancy 强调模型迎合用户观点；本文发现置信度场景中模型反而更相信 assistant 自己的答案，说明对齐偏差有多种方向。
- **对后续工作的启发**: 做自检、自评、答案 reranking 时，应尽量把候选答案从“我的输出”改造成“外部候选”，否则 confidence 可能不是答案质量而是角色归属的函数。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ ownership bias 的控制实验很清楚，mitigation 简单但有实际价值。
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 6 个 open-weight 模型、3 个基准、3 种 confidence 方法、额外任务和 GPT-5.2。
- 写作质量: ⭐⭐⭐⭐☆ 论证链条顺，表格略密但主结论非常明确。
- 价值: ⭐⭐⭐⭐⭐ 对校准、自评、LLM-as-judge 和高风险应用都有直接提醒。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[ACL 2026\] BACH-V: Bridging Abstract and Concrete Human-Values in Large Language Models](bach-v_bridging_abstract_and_concrete_human-values_in_large_language_models.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/llm_alignment/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ICML 2026\] Towards Context-Invariant Safety Alignment for Large Language Models](../../ICML2026/llm_alignment/towards_context-invariant_safety_alignment_for_large_language_models.md)
- [\[ACL 2026\] Compatibility-Aware Dynamic Fine-Tuning for Large Language Models](compatibility-aware_dynamic_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
