---
title: >-
  [论文解读] SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs
description: >-
  [ICLR 2026][医疗 LLM][Theory of Mind] SimpleToM 揭示了 LLM 在 Theory of Mind 上的关键缺陷：前沿模型能准确推断他人心理状态（显式 ToM），但在将此知识应用于行为预测和行为判断时性能急剧下降（应用 ToM），暴露了"知道什么"与"如何使用所知"之间的重大鸿沟。
tags:
  - "ICLR 2026"
  - "医疗 LLM"
  - "Theory of Mind"
  - "心智理论"
  - "LLM社会推理"
  - "显式vs应用ToM"
  - "信息不对称"
---

# SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs

**会议**: ICLR 2026  
**arXiv**: [2410.13648](https://arxiv.org/abs/2410.13648)  
**代码**: [https://github.com/yulinggu-cs/SimpleToM](https://github.com/yulinggu-cs/SimpleToM)  
**领域**: 人类理解  
**关键词**: Theory of Mind, 心智理论, LLM社会推理, 显式vs应用ToM, 信息不对称

## 一句话总结
SimpleToM 揭示了 LLM 在 Theory of Mind 上的关键缺陷：前沿模型能准确推断他人心理状态（显式 ToM），但在将此知识应用于行为预测和行为判断时性能急剧下降（应用 ToM），暴露了"知道什么"与"如何使用所知"之间的重大鸿沟。

## 研究背景与动机

**领域现状**：LLM 作为对话代理广泛使用，理解他人信念（Theory of Mind）对避免灾难性回复至关重要（如忽略情感困扰、将讽刺当真话、在敏感场景提供不当建议）。

**现有痛点**：
   - 已有 ToM 评估局限于 Sally-Anne 任务或模板化变体——场景单一、信息不对称类型有限
   - 显式使用"sees"、"thinks"等感知/心理化动词作为触发词→模型无需常识推理即可作答
   - 几乎所有评估只测"显式 ToM"（推断心理状态），未测试将心理状态知识应用于行为预测/判断

**核心矛盾**：LLM 能正确回答"Mary 知道薯片发霉吗？"（显式 ToM），但无法正确推断"Mary 会付款还是举报？"（应用 ToM）——这意味着 LLM 的 ToM 知识是"解耦的"，无法可靠地应用

**本文目标**
   - 构建涵盖多层次 ToM 推理（心理状态→行为预测→行为判断）的基准
   - 在多样化日常场景中评估，而非仅限于经典玩具任务
   - 揭示和量化显式 ToM 与应用 ToM 之间的能力差距

**切入角度**：10 种自然信息不对称场景（超市、医院、二手市场等），每个故事仅两句话但要求隐式常识推理，三个层级的问题逐步增加推理深度。

**核心 idea**：LLM 的 ToM 能力存在"知行分离"——知道别人不知道什么（显式），但无法利用这个知识预测和判断行为（应用），即使在简单日常场景中。

## 方法详解

### 整体框架
SimpleToM 想把"模型到底会不会用 ToM"拆成可以分别测量的几层能力。它由 1147 个极简故事组成，每个故事只有两句话，却嵌进了一处信息不对称：某个关键事实是读者知道、而故事里的角色不知道的（比如一罐薯片里发了霉，但角色看不穿不透明的罐子）。每个故事配三个层层递进的问题——先问角色的心理状态（显式 ToM：他知不知道？），再问角色会怎么做（应用 ToM：他下一步行为？），最后问角色的实际做法是否合理（更深一层的应用 ToM：这样做对不对？）。三道题逐级要求模型"不仅知道别人不知道什么，还得拿这个知识去预测、去评判行为"，于是"会推断"和"会应用"两种能力就被分离出来、能各自打分。数据集本身则用"种子故事 + 多个 LLM 扩写 + 严格人工过滤"的流水线造出来，兼顾规模和质量。

### 关键设计

**1. 三层级问题：把一个故事拆成"推断→预测→判断"三道递进难度的题**

以一罐发霉薯片的故事为例，第一层心理状态（Mental State, MS）直接问"Mary 知道薯片发霉吗？"，只需是/否，模型推断出角色处于不知情状态即可。第二层行为预测（Behavior）问"Mary 会付款还是当场举报？"，模型得先在心里推出"Mary 不知情"，再把这个状态映射到合理行为上（不知情就会照常付款）。第三层行为判断（Judgment）问"Mary 付了款，这合理吗？"——题面给定角色采取了那个符合其心理状态的行为，要求模型判断是否合理，这需要两层隐式推理：先推出 Mary 应该会做什么，再拿实际行为去对照评判。三道题的隐式推理深度逐级加码——MS 只要推断状态，Behavior 要把状态翻译成动作，Judgment 还要在预测之上再加一层对错评估——正是这种递进，让显式与应用 ToM 的差距能被定量观察到。

**2. 隐式信息不对称：故事里绝不出现 "sees"、"thinks" 这类心理化触发词**

很多 ToM 基准会在题面里直接写出"角色看到/认为……"，模型只要顺着这些词面回答就行，根本不需要真正推理。SimpleToM 刻意回避这种线索：故事第一句抛出关键事实（"薯片发霉了"），第二句只描述角色的客观行为（"Mary 拿起薯片走向收银台"），角色的不知情完全是隐含的——模型必须靠常识补上"人看不穿不透明的薯片罐"这一步，才能推断 Mary 不知情。这样设计逼出真正的常识推理，堵死了靠触发词作弊的捷径，也让故事读起来更接近真实生活。

**3. 10 种日常信息不对称场景：覆盖物理遮挡、知识壁垒、欺骗等不同的"不知情"成因**

数据集横跨超市食品、医患信息、虚假标签、服务业幕后、容器内容物、不道德行为、个人物品容器、二手市场、隐藏身体特征、锁定设备等多类日常场景。这些场景对应的信息不对称机制各不相同：有的是物理上看不见（容器内容物），有的是专业知识壁垒（医患信息），有的是刻意欺骗（虚假标签）。覆盖多种成因是为了让评测不偏向某一类推理，确保测出的差距是 ToM 能力的普遍现象，而非单一场景的偶然——后面的实验也确实发现不同场景间分数差异巨大。

**4. 数据构建：LLM 扩写上规模 + 人工严格过滤保质量**

纯靠专家手写自包含的 ToM 故事又贵又慢，纯靠 LLM 批量生成又压不住质量——SimpleToM 用一条"生成—过滤"流水线兼顾两头。先为每个场景手写少量种子故事，再用 GPT-4、GPT-4o、Claude-3.5-Sonnet 等多个模型基于种子替换实体、变换情境，扩写出约 3600 个候选故事（用多个生成器，是为了让信息不对称的表现形式更多样）。随后由通过资格测试的标注者逐条审核故事质量和三个问题的标准答案，淘汰掉模糊或答案不唯一的样本，最终从约 3600 个里筛出 1147 个高质量故事（× 3 问题 = 3441 个评估实例）。先扩规模、再用人把关，让这个基准既有足够样本量做统计显著的结论，又不会被生成噪声污染。

## 实验关键数据

在 21 个模型上评测，三类问题都是二元选择题，随机水平为 50%。

### 主实验（代表性模型在三层级问题上的准确率，%）

| 模型 | 心理状态 (显式 ToM)↑ | 行为预测 (应用 ToM)↑ | 行为判断 (应用 ToM)↑ |
|------|------|------|------|
| GPT-3.5 | 36.5 | 7.6 | 29.1 |
| GPT-4o | 95.6 | 49.5 | 15.3 |
| GPT-4 | 96.6 | 63.0 | 19.5 |
| Llama-3.1-405B | 97.8 | 58.2 | 10.0 |
| Claude-3.5-Sonnet | 97.9 | 67.0 | 24.9 |
| GPT-4.5-preview | 97.0 | 67.8 | 26.7 |
| GPT-5 | 98.5 | 64.4 | 40.0 |
| DeepSeek-R1（推理模型） | 97.3 | 73.8 | 65.8 |
| o1-preview（推理模型） | 95.6 | 84.1 | 59.5 |

### 测试期干预（选 3 个强模型，看提示干预能否补差，%）

| 模型 | 行为预测·无干预 | 行为预测·CoT | 行为预测·心理状态提醒 | 行为判断·无干预 | 行为判断·CoT | 行为判断·心理状态提醒 |
|------|------|------|------|------|------|------|
| GPT-4o | 49.5 | 62.8 | 82.8 | 15.3 | 39.2 | 42.2 |
| Llama-3.1-405B | 58.2 | 57.2 | 89.5 | 10.0 | 35.2 | 25.8 |
| Claude-3.5-Sonnet | 67.0 | 77.2 | 96.9 | 24.9 | 39.4 | 84.1 |

### 关键发现
- **显式强、应用弱**：前沿模型心理状态推断普遍 >95%（GPT-5 98.5、Claude-3.5-Sonnet 97.9），但行为预测掉到 50–70%、行为判断更掉到 10–25%（GPT-4o 仅 15.3、Llama-3.1-405B 仅 10.0，远低于 50% 的随机水平）——"知道别人不知道"与"会用这个知识"之间裂开一道大鸿沟。
- **层级越深越差**：MS > Behavior > Judgment 是所有模型的一致模式；判断题要叠两层隐式推理，最难。
- **推理模型能缓解但补不平**：o1-preview 行为预测冲到 84.1、DeepSeek-R1 行为判断 65.8，是两项最好成绩，但仍显著低于各自的心理状态分——差距没消失，说明这不是单纯堆参数/堆算力就能解决的 scaling 问题。
- **场景方差极大**：同一模型在不同场景间行为预测分差异巨大；医患信息场景的行为预测异常高（接近心理状态分，疑似安全训练让模型对健康/药物话题更警觉），其他场景则低很多——说明只用单一场景测 ToM 会严重失真。
- **测试期干预不是万能药**：CoT 和改系统提示几乎不缩小差距，所有模型行为判断仍 <40%；把心理状态答案直接塞回提示（MS remind）能把行为预测拉到 >80%，但行为判断仍可能很低（GPT-4o 42.2、Llama-3.1-405B 25.8）——没有一种简单干预能可靠补上"应用 ToM"的缺口。

## 亮点与洞察
- **"知行分离"的概念化**极具价值：LLM 的 ToM 不是"有或没有"的二元问题，而是存在层级——知道别人不知道什么（容易），但将此知识应用于预测（难），再应用于判断（更难）。这个分层评估框架可推广到所有认知能力评估。
- **隐式设计避免触发词作弊**是关键的方法论创新：很多 ToM 基准在设计上就给了模型线索（"sees"、"thinks"），SimpleToM 强制要求常识推理。
- **对 LLM 安全部署的警示**：如果模型无法可靠地预测和判断人类行为，那么在敏感社会应用（心理咨询、客服、教育）中的部署需要极度谨慎。

## 局限与展望
- 仅评估英文场景，跨文化/跨语言 ToM 差异未探索
- 评估格式为选择题——这可能低估了模型在开放式生成中的问题
- 故事格式固定（两句话），更复杂的多轮对话 ToM 未覆盖
- 未探索 fine-tuning 或 RLHF 是否能缩小显式-应用 ToM 的差距
- 10 个场景虽已多样，但仍可能遗漏某些类型的信息不对称

## 相关工作与启发
- **vs Sally-Anne / BigToM**: 经典 ToM 测试只评估显式心理状态推断且场景单一；SimpleToM 扩展到应用 ToM + 10 种多样场景
- **vs SocialIQA**: 社会推理基准但不聚焦信息不对称和心智理论的层级结构
- **vs FANToM**: 也是 ToM 基准但使用对话格式且显式标注心理状态；SimpleToM 要求隐式推理

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性地区分显式/应用 ToM 并揭示令人震惊的能力差距
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型、跨场景、消融、CoT 分析全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机→设计→结果的逻辑链极其清晰，例子生动
- 价值: ⭐⭐⭐⭐⭐ 对 LLM 社会推理能力的评估有里程碑意义，数据集公开

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Can Continual Pre-training Bridge the Performance Gap between General-purpose and Specialized Language Models in the Medical Domain?](../../ACL2026/medical_nlp/can_continual_pre-training_bridge_the_performance_gap_between_general-purpose_an.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[ICML 2026\] Exploring Accurate and Transparent Domain Adaptation in Predictive Healthcare via Concept-Grounded Orthogonal Inference](../../ICML2026/medical_nlp/exploring_accurate_and_transparent_domain_adaptation_in_predictive_healthcare_vi.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)
- [\[ACL 2026\] ProMedical: Hierarchical Fine-Grained Criteria Modeling for Medical LLM Alignment via Explicit Injection](../../ACL2026/medical_nlp/promedical_hierarchical_fine-grained_criteria_modeling_for_medical_llm_alignment.md)

</div>

<!-- RELATED:END -->
